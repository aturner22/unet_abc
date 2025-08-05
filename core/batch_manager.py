import psutil
import torch


class MemorySafeBatchManager:
    def __init__(self, device: torch.device, config=None):
        self.device = device
        self.config = config
        self.global_batch_size = None
        self.calibrated = False
        self.runtime_batch_size = None
        self.measured_memory_per_sample = None

    def get_memory_stats(self):
        if torch.cuda.is_available() and self.device.type == "cuda":
            allocated = torch.cuda.memory_allocated(self.device)
            total = torch.cuda.get_device_properties(self.device).total_memory
            return allocated, total, total - allocated
        else:
            mem = psutil.virtual_memory()
            return mem.used, mem.total, mem.available

    def measure_memory_per_sample(
        self,
        model: torch.nn.Module,
        sample_input: torch.Tensor,
        sample_time: torch.Tensor,
    ) -> int:
        if self.measured_memory_per_sample is not None:
            return self.measured_memory_per_sample

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        memory_before = self.get_memory_stats()[0]

        try:
            single_input = sample_input[:1]
            single_time = sample_time[:1]

            with torch.no_grad():
                output = model(single_input, single_time)
                K, V, H, W = output.shape

                reshaped = output.view(K, -1)
                temp_storage = output.clone()
                diff_computation = torch.abs(output - temp_storage)  # noqa: F841
                memory_peak = self.get_memory_stats()[0]

                if K > 1:
                    n_pairs = (
                        self.config.memory_management["batch_size_limits"][
                            "memory_test_pairs"
                        ]
                        if self.config
                        else 500
                    )
                    n_pairs = min(n_pairs, K * 4)
                    sample_pairs = torch.randint(0, K, (n_pairs,), device=output.device)
                    pairwise_test = (
                        reshaped[sample_pairs[: len(sample_pairs) // 2]]
                        - reshaped[sample_pairs[len(sample_pairs) // 2 :]]
                    )
                    _ = torch.abs(pairwise_test).mean()

                memory_after = self.get_memory_stats()[0]

            memory_used = max(memory_peak, memory_after) - memory_before
            self.measured_memory_per_sample = max(memory_used, 10 * 1024 * 1024)

            return self.measured_memory_per_sample

        except Exception:
            if self.config:
                mem_config = self.config.memory_management.get(
                    "sample_memory_estimates", {}
                )
                fallback = (
                    (
                        mem_config.get("cpu_mb_per_sample", 50)
                        if self.device.type == "cpu"
                        else mem_config.get("gpu_mb_per_sample", 100)
                    )
                    * 1024
                    * 1024
                )
            else:
                fallback = (50 if self.device.type == "cpu" else 100) * 1024 * 1024

            self.measured_memory_per_sample = fallback
            return fallback

    def test_batch_size(
        self,
        batch_size: int,
        model: torch.nn.Module,
        sample_input: torch.Tensor,
        sample_time: torch.Tensor,
    ) -> bool:
        try:
            if batch_size <= sample_input.shape[0]:
                test_input = sample_input[:batch_size]
                test_time = sample_time[:batch_size]
            else:
                repeats = (
                    batch_size + sample_input.shape[0] - 1
                ) // sample_input.shape[0]
                test_input = sample_input.repeat(repeats, 1, 1, 1)[:batch_size]
                test_time = sample_time.repeat(repeats, 1)[:batch_size]

            with torch.no_grad():
                output = model(test_input, test_time)
                K, V, H, W = output.shape
                reshaped = output.view(K, -1)
                temp_storage = output.clone()
                diff_computation = torch.abs(output - temp_storage)

                if K > 1:
                    sample_pairs = torch.randint(
                        0, K, (min(500, K * 4),), device=output.device
                    )
                    pairwise_test = (
                        reshaped[sample_pairs[: len(sample_pairs) // 2]]
                        - reshaped[sample_pairs[len(sample_pairs) // 2 :]]
                    )
                    _ = torch.abs(pairwise_test).mean()

                accumulated_memory = [output, temp_storage, diff_computation]
                if len(accumulated_memory) > 2:
                    _ = torch.stack(accumulated_memory[:2])

            del test_input, test_time, output, reshaped, temp_storage, diff_computation
            return True

        except (RuntimeError, MemoryError, OSError) as e:
            error_msg = str(e).lower()
            if (
                "out of memory" in error_msg
                or "memory" in error_msg
                or isinstance(e, (MemoryError, OSError))
            ):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return False
            else:
                raise e

    def find_max_batch_size(
        self,
        model: torch.nn.Module,
        sample_input: torch.Tensor,
        sample_time: torch.Tensor,
        max_search: int = 512,
    ) -> int:
        if self.calibrated:
            return self.global_batch_size

        measured_memory_per_sample = self.measure_memory_per_sample(
            model, sample_input, sample_time
        )
        allocated, total_mem, free_mem = self.get_memory_stats()

        if self.config and "memory_management" in self.config.__dict__:
            mem_config = self.config.memory_management.get("utilization_ratios", {})
            if self.device.type == "cpu":
                memory_ratio = mem_config.get("cpu_free_memory_ratio", 0.3)
                min_batch_size = mem_config.get("cpu_min_batch_size", 8)
            else:
                memory_ratio = mem_config.get("gpu_free_memory_ratio", 0.6)
                min_batch_size = mem_config.get("gpu_min_batch_size", 16)
        else:
            memory_ratio = 0.3 if self.device.type == "cpu" else 0.6
            min_batch_size = 8 if self.device.type == "cpu" else 16

        memory_based_limit = int(free_mem * memory_ratio / measured_memory_per_sample)
        high = min(max_search, max(min_batch_size, memory_based_limit))

        low = 1
        best_working = 1

        while low <= high:
            mid = (low + high) // 2
            if self.test_batch_size(mid, model, sample_input, sample_time):
                best_working = mid
                low = mid + 1
            else:
                high = mid - 1

        allocated, total_mem, free_mem = self.get_memory_stats()
        memory_pressure = allocated / total_mem

        if self.config and "memory_management" in self.config.__dict__:
            margin_config = self.config.memory_management.get("safety_margins", {})
            if self.device.type == "cpu":
                if memory_pressure < 0.3:
                    safety_margin = margin_config.get("cpu_low_memory", 0.8)
                elif memory_pressure < 0.6:
                    safety_margin = margin_config.get("cpu_medium_memory", 0.65)
                else:
                    safety_margin = margin_config.get("cpu_high_memory", 0.5)
            else:
                if memory_pressure < 0.4:
                    safety_margin = margin_config.get("gpu_low_memory", 0.9)
                elif memory_pressure < 0.7:
                    safety_margin = margin_config.get("gpu_medium_memory", 0.8)
                else:
                    safety_margin = margin_config.get("gpu_high_memory", 0.7)
        else:
            if self.device.type == "cpu":
                safety_margin = (
                    0.8
                    if memory_pressure < 0.3
                    else (0.65 if memory_pressure < 0.6 else 0.5)
                )
            else:
                safety_margin = (
                    0.9
                    if memory_pressure < 0.4
                    else (0.8 if memory_pressure < 0.7 else 0.7)
                )

        safe_batch_size = max(1, int(best_working * safety_margin))
        self.global_batch_size = safe_batch_size
        self.calibrated = True

        return safe_batch_size

    def get_batch_size(self) -> int:
        if not self.calibrated:
            raise RuntimeError("Must calibrate batch size first")
        return self.runtime_batch_size or self.global_batch_size

    def reduce_batch_size(self):
        current = self.runtime_batch_size or self.global_batch_size

        if self.config and "memory_management" in self.config.__dict__:
            reduction_config = self.config.memory_management.get("batch_reduction", {})
            large_threshold = reduction_config.get("large_batch_threshold", 16)
            medium_threshold = reduction_config.get("medium_batch_threshold", 4)
            large_reduction_factor = reduction_config.get("large_reduction_factor", 4)
            medium_reduction_factor = reduction_config.get("medium_reduction_factor", 2)
            min_reduction_size = reduction_config.get("min_reduction_size", 4)
        else:
            large_threshold, medium_threshold = 16, 4
            large_reduction_factor, medium_reduction_factor = 4, 2
            min_reduction_size = 4

        if current > large_threshold:
            new_size = max(min_reduction_size, current // large_reduction_factor)
        elif current > medium_threshold:
            new_size = max(2, current // medium_reduction_factor)
        elif current > 1:
            new_size = 1
        else:
            new_size = 1

        self.runtime_batch_size = new_size
        return new_size

    def check_memory_pressure(self) -> bool:
        if torch.cuda.is_available():
            allocated, total, _ = self.get_memory_stats()
            usage_ratio = allocated / total

            threshold = (
                self.config.memory_management.get("memory_pressure_threshold", 0.85)
                if self.config and "memory_management" in self.config.__dict__
                else 0.85
            )

            return usage_ratio > threshold
        return False
