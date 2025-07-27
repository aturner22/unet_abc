import os
import numpy as np
import torch
from typing import Any
from tqdm import tqdm
from core.evaluation import (
    compute_rank_histogram,
    compute_mean_absolute_error,
    compute_ensemble_spread,
)
from core.simple_crps import compute_crps_for_proposal

INITIAL_ALPHA_RANGE = (0.05, 1.5)
PROPOSAL_SCALE = 0.05
MIN_ALPHA = 1e-4
ADAPT_EVERY = 5
ADAPT_FACTOR = 0.85
EPS_ENERGY = 1e-12
CHECKPOINT_FILE = "gibbs_checkpoint_step.npz"


class MemorySafeBatchManager:
    """
    Memory-safe batch size finder through actual testing, not guessing.
    Finds maximum safe batch size through binary search with real memory tests.
    """
    
    def __init__(self, device: torch.device):
        self.device = device
        self.global_batch_size = None  # Cache the one good batch size
        self.calibrated = False
        self.runtime_batch_size = None  # Can be reduced during runtime if needed
        
    def get_memory_stats(self):
        """Get current memory usage (GPU or CPU)."""
        if torch.cuda.is_available() and self.device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(self.device)
            total = torch.cuda.get_device_properties(self.device).total_memory
            free = total - allocated
            return allocated, total, free
        else:
            import psutil
            mem = psutil.virtual_memory()
            # For CPU, we use available memory as our working space
            return mem.used, mem.total, mem.available
    
    def test_batch_size(self, batch_size: int, model: torch.nn.Module, 
                       sample_input: torch.Tensor, sample_time: torch.Tensor) -> bool:
        """Test if a specific batch size works by actually running it."""
        try:
            # Create test batch of the proposed size - ensure we test with realistic load
            if batch_size <= sample_input.shape[0]:
                test_input = sample_input[:batch_size]
                test_time = sample_time[:batch_size]
            else:
                # Repeat samples to get the target batch size
                repeats = (batch_size + sample_input.shape[0] - 1) // sample_input.shape[0]
                test_input = sample_input.repeat(repeats, 1, 1, 1)[:batch_size]
                test_time = sample_time.repeat(repeats, 1)[:batch_size]
            
            # More comprehensive test that simulates actual workload memory pattern
            with torch.no_grad():
                # Simulate the actual inference workload
                output = model(test_input, test_time)
                
                # Simulate CRPS computation memory pattern:
                K, V, H, W = output.shape[0], output.shape[1], output.shape[2], output.shape[3]
                reshaped = output.view(K, -1)  # Flatten for CRPS
                
                # Simulate multiple operations that happen during proposal evaluation
                temp_storage = output.clone()  # Simulate keeping best proposal
                diff_computation = torch.abs(output - temp_storage)  # Simulate CRPS computation
                
                # Simulate the memory-intensive operations
                if K > 1:
                    # Test both small and larger tensor operations
                    sample_pairs = torch.randint(0, K, (min(500, K*4),), device=output.device)
                    pairwise_test = reshaped[sample_pairs[:len(sample_pairs)//2]] - reshaped[sample_pairs[len(sample_pairs)//2:]]
                    _ = torch.abs(pairwise_test).mean()  # Force computation
                
                # Test peak memory usage by keeping multiple tensors
                accumulated_memory = [output, temp_storage, diff_computation]
                if len(accumulated_memory) > 2:
                    _ = torch.stack(accumulated_memory[:2])  # Simulate tensor operations
                
            # If we get here, it worked
            del test_input, test_time, output, reshaped, temp_storage, diff_computation
            return True
            
        except (RuntimeError, MemoryError, OSError) as e:
            # OSError can be thrown on CPU when system runs out of memory
            error_msg = str(e).lower()
            if "out of memory" in error_msg or "memory" in error_msg or isinstance(e, (MemoryError, OSError)):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return False
            else:
                raise e
    
    def find_max_batch_size(self, model: torch.nn.Module, 
                           sample_input: torch.Tensor, sample_time: torch.Tensor,
                           max_search: int = 512) -> int:
        """
        Binary search to find maximum safe batch size through actual testing.
        ONLY CALIBRATES ONCE - reuses result for entire run.
        """
        if self.calibrated:
            return self.global_batch_size
            
        _, total_mem, free_mem = self.get_memory_stats()
        print(f"[MemorySafe] One-time calibration (Free: {free_mem/1e9:.1f}GB/{total_mem/1e9:.1f}GB)")
        
        # Adaptive search bounds based on available memory
        allocated, total_mem, free_mem = self.get_memory_stats()
        
        if self.device.type == 'cpu':
            # For CPU: estimate batch size based on available system memory
            # Assume each sample needs ~50MB (rough estimate for typical model)
            estimated_sample_size = 50 * 1024 * 1024  # 50MB per sample
            memory_based_limit = int(free_mem * 0.3 / estimated_sample_size)  # Use 30% of free memory
            high = min(max_search, max(8, memory_based_limit))  # At least 8, but memory-aware
        else:
            # For GPU: use GPU memory more aggressively since it's dedicated
            estimated_sample_size = 100 * 1024 * 1024  # 100MB per sample for GPU
            memory_based_limit = int(free_mem * 0.6 / estimated_sample_size)  # Use 60% of free GPU memory
            high = min(max_search, max(16, memory_based_limit))  # At least 16 for GPU
        
        low = 1
        best_working = 1
        print(f"[MemorySafe] Search range: {low}-{high} (Free: {free_mem/1e9:.1f}GB)")
        
        while low <= high:
            mid = (low + high) // 2
            print(f"[MemorySafe] Testing batch size: {mid}", end=" ")
            
            if self.test_batch_size(mid, model, sample_input, sample_time):
                best_working = mid
                low = mid + 1
                print("✓")
            else:
                high = mid - 1
                print("✗")
        
        # Apply adaptive safety margin based on how much memory we have
        allocated, total_mem, free_mem = self.get_memory_stats()
        memory_pressure = allocated / total_mem
        
        if self.device.type == 'cpu':
            # CPU: More aggressive if we have lots of free memory
            if memory_pressure < 0.3:  # Low pressure
                safety_margin = 0.8
            elif memory_pressure < 0.6:  # Medium pressure
                safety_margin = 0.65
            else:  # High pressure
                safety_margin = 0.5
        else:
            # GPU: Dedicated memory, can be more aggressive
            if memory_pressure < 0.4:  # Low pressure
                safety_margin = 0.9
            elif memory_pressure < 0.7:  # Medium pressure
                safety_margin = 0.8
            else:  # High pressure
                safety_margin = 0.7
        
        safe_batch_size = max(1, int(best_working * safety_margin))
        print(f"[MemorySafe] Memory pressure: {memory_pressure:.1%}, safety margin: {safety_margin:.1%}")
        
        # Cache globally for entire run
        self.global_batch_size = safe_batch_size
        self.calibrated = True
        print(f"[MemorySafe] Calibrated: {best_working} → {safe_batch_size} (safety margin: {safety_margin:.1%})")
        
        return safe_batch_size
    
    def get_batch_size(self) -> int:
        """Get current batch size (must be calibrated first)."""
        if not self.calibrated:
            raise RuntimeError("Must calibrate batch size first")
        return self.runtime_batch_size or self.global_batch_size
    
    def reduce_batch_size(self):
        """Reduce batch size due to runtime OOM."""
        current = self.runtime_batch_size or self.global_batch_size
        
        # More aggressive reduction strategy for persistent OOM
        if current > 16:
            new_size = max(4, current // 4)  # Reduce by 4x for large batches
        elif current > 4:
            new_size = max(2, current // 2)  # Reduce by 2x for medium batches
        elif current > 1:
            new_size = 1                     # Down to minimum
        else:
            # Already at minimum - this suggests fundamental memory issue
            print(f"[MemorySafe] WARNING: Already at batch size 1, cannot reduce further")
            new_size = 1
            
        self.runtime_batch_size = new_size
        print(f"[MemorySafe] Aggressive batch size reduction due to OOM: {current} → {new_size}")
        return new_size
    
    def check_memory_pressure(self) -> bool:
        """Check if we're approaching memory limits."""
        if torch.cuda.is_available():
            allocated, total, _ = self.get_memory_stats()
            usage_ratio = allocated / total
            if usage_ratio > 0.85:  # If using >85% of GPU memory
                print(f"[MemorySafe] High memory pressure: {usage_ratio:.1%} used")
                return True
        return False


def generate_joint_rfp(
    reference_tensor: torch.Tensor,
    alpha_matrix: torch.Tensor,
    batch_size: int,
    ensemble_size: int,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    P = alpha_matrix.shape[0]
    T, V = reference_tensor.shape[0], reference_tensor.shape[1]
    idx1 = torch.randint(0, T, (batch_size, ensemble_size), device=device, generator=generator)
    idx2 = torch.randint(0, T, (batch_size, ensemble_size), device=device, generator=generator)
    diff = reference_tensor[idx1] - reference_tensor[idx2]
    energy = torch.sqrt(diff.pow(2).mean(dim=(2, 3, 4), keepdim=True).clamp_min(EPS_ENERGY))
    diff_norm = diff / energy
    perturb = alpha_matrix.reshape(P, 1, 1, V, 1, 1) * diff_norm.unsqueeze(0)
    return perturb


# CRPS computation now handled by simple_crps.py


def batched_forward_proposals(
    *,
    model: torch.nn.Module,
    previous_fields: torch.Tensor,
    current_fields: torch.Tensor,
    time_normalised: torch.Tensor,
    reference_tensor: torch.Tensor,
    alpha_matrix: torch.Tensor,
    ensemble_size: int,
    device: torch.device,
    buffers: dict,
    batch_manager: MemorySafeBatchManager,
    generator: torch.Generator,
) -> tuple[torch.Tensor, list[float]]:
    N, C, H, W = previous_fields.shape
    V = current_fields.shape[1]
    P = alpha_matrix.shape[0]

    current_slice = previous_fields[:, :V]
    past_slice = previous_fields[:, V : 2 * V]
    static_slice = previous_fields[:, -2:]

    perturb = generate_joint_rfp(
        reference_tensor,
        alpha_matrix,
        batch_size=N,
        ensemble_size=ensemble_size,
        device=device,
        generator=generator,
    )

    curr_base = current_slice.unsqueeze(1).expand(-1, ensemble_size, -1, -1, -1)
    past_base = past_slice.unsqueeze(1).expand(-1, ensemble_size, -1, -1, -1)
    stat_base = static_slice.unsqueeze(1).expand(-1, ensemble_size, -1, -1, -1)

    # Calibrate batch size only once (on first call)
    if not batch_manager.calibrated:
        # Create sample input for one-time batch size testing
        test_n = min(N, 2)  # Use minimal samples for testing
        sample_input = torch.cat([curr_base[:test_n], past_base[:test_n], stat_base[:test_n]], dim=2)
        sample_input = sample_input.view(test_n * ensemble_size, C, H, W)
        sample_time = time_normalised[:test_n].view(-1, 1).expand(-1, ensemble_size).reshape(-1, 1)
        
        # One-time calibration
        batch_manager.find_max_batch_size(
            model=model,
            sample_input=sample_input, 
            sample_time=sample_time,
            max_search=128  # Faster search
        )
    
    # Use cached batch size (very fast)
    optimal_batch_size = batch_manager.get_batch_size()

    joint_scores: list[float] = []
    best_proposal_output: torch.Tensor | None = None
    best_score = float("inf")
    best_proposal_idx = -1

    for p in range(P):
        buffers["curr"][:N].copy_(curr_base)
        buffers["curr"][:N].add_(perturb[p])
        buffers["past"][:N].copy_(past_base)
        buffers["stat"][:N].copy_(stat_base)

        full_input = torch.cat([buffers["curr"][:N], buffers["past"][:N], buffers["stat"][:N]], dim=2)
        full_input = full_input.view(N * ensemble_size, C, H, W)
        full_time = time_normalised.view(-1, 1).expand(-1, ensemble_size).reshape(-1, 1)

        step = optimal_batch_size * ensemble_size
        out_chunks = []
        start = 0
        while start < N * ensemble_size:
            end = min(start + step, N * ensemble_size)
            
            # Retry loop for handling OOM with exponential backoff
            retry_count = 0
            max_retries = 5
            
            while retry_count <= max_retries:
                try:
                    with torch.no_grad():
                        y = model(full_input[start:end], full_time[start:end])
                    out_chunks.append(y)
                    start = end
                    break  # Success, exit retry loop
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower() and retry_count < max_retries:
                        retry_count += 1
                        torch.cuda.empty_cache()
                        
                        # Reduce batch size more aggressively with each retry
                        optimal_batch_size = batch_manager.reduce_batch_size()
                        step = optimal_batch_size * ensemble_size
                        end = min(start + step, N * ensemble_size)
                        
                        print(f"[MemorySafe] OOM retry {retry_count}/{max_retries}, new batch size: {optimal_batch_size}")
                        
                        # If we're down to batch size 1 and still failing, something is seriously wrong
                        if optimal_batch_size == 1 and retry_count >= 3:
                            raise RuntimeError(f"Persistent OOM even with batch size 1. Available memory may be insufficient.") from e
                    else:
                        raise e
            
            if retry_count > max_retries:
                raise RuntimeError(f"Failed to process batch after {max_retries} retries with batch size reductions")

        y_full = torch.cat(out_chunks, dim=0).view(N, ensemble_size, V, H, W)
        proposal_output = y_full.permute(1, 0, 2, 3, 4)

        # Compute CRPS score immediately
        joint_score = compute_crps_for_proposal(proposal_output, current_fields, V)
        joint_scores.append(joint_score)

        # Only keep this proposal if it's the best so far
        if joint_score < best_score:
            best_score = joint_score
            best_proposal_idx = p
            
            # Delete previous best to free memory
            if best_proposal_output is not None:
                del best_proposal_output
            
            # Store new best (only one proposal in memory at a time)
            best_proposal_output = proposal_output.clone()
        
        # IMMEDIATELY delete current proposal after evaluation
        del proposal_output, y_full, out_chunks, full_input, full_time
        
        # Aggressive cleanup after each proposal
        torch.cuda.empty_cache()
        
        print(f"  Proposal {p+1}/{P}: CRPS={joint_score:.4f} {'(NEW BEST)' if p == best_proposal_idx else ''}")

    print(f"  → Best proposal: {best_proposal_idx+1} with CRPS={best_score:.4f}")
    return best_proposal_output, joint_scores


def run_gibbs_abc_rfp(
    *,
    model: torch.nn.Module,
    batches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ensemble_size: int,
    n_steps: int,
    n_proposals: int,
    num_variables: int,
    variable_names: list[str],
    reference_mmap: np.memmap,
    result_directory: str,
    log_diagnostics: bool = True,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    ref_full = torch.from_numpy(np.array(reference_mmap, copy=True)).to(device)

    prev_all = torch.cat([b[0] for b in batches], dim=0).to(device)
    curr_all = torch.cat([b[1] for b in batches], dim=0).to(device)
    time_all = torch.cat([b[2] for b in batches], dim=0).to(device)
    N, _, H, W = prev_all.shape  # C not used after buffers creation
    V = curr_all.shape[1]

    buffers = {
        "curr": torch.empty((N, ensemble_size, V, H, W), device=device),
        "past": torch.empty((N, ensemble_size, V, H, W), device=device),
        "stat": torch.empty((N, ensemble_size, 2, H, W), device=device),
    }

    # Initialize memory-safe batch manager
    batch_manager = MemorySafeBatchManager(device)
    
    # Device warm-up
    with torch.no_grad():
        _ = model(prev_all[:1], time_all[:1])

    posterior_samples = np.zeros((n_steps, num_variables, 1), dtype=np.float32)
    posterior_crps = np.zeros((n_steps, num_variables), dtype=np.float32)
    step_mean_crps = np.zeros(n_steps, dtype=np.float32)

    rank_histograms = [[] for _ in range(num_variables)]
    ensemble_spread_records = [[] for _ in range(num_variables)]
    mean_absolute_error_records = [[] for _ in range(num_variables)]

    current_alpha = np.random.uniform(*INITIAL_ALPHA_RANGE, size=(num_variables, 1))
    proposal_std = np.full((num_variables, 1), PROPOSAL_SCALE, dtype=np.float32)

    rng = np.random.default_rng()
    torch_gen = torch.Generator(device=device)

    ckpt_path = os.path.join(result_directory, CHECKPOINT_FILE)
    start_step = 0
    if os.path.exists(ckpt_path):
        ck = np.load(ckpt_path, allow_pickle=True)
        print(f"[checkpoint] Resuming from step {ck['step']+1}")
        posterior_samples[: ck["step"] + 1] = ck["posterior_samples"]
        posterior_crps[: ck["step"] + 1] = ck["posterior_crps"]
        step_mean_crps[: ck["step"] + 1] = ck["step_mean_crps"]
        current_alpha = ck["last_alpha"]
        start_step = int(ck["step"]) + 1
        del ck

    for s in tqdm(range(start_step, n_steps), desc="Gibbs", position=0):
        print(f"\n[Gibbs step {s+1}/{n_steps}]")
        if s and (s % ADAPT_EVERY == 0):
            proposal_std *= ADAPT_FACTOR
            print(f"[adapt] proposal σ -> {proposal_std.mean():.3f}")

        for v in range(num_variables):
            proposals_v = np.clip(
                rng.normal(loc=current_alpha[v], scale=proposal_std[v], size=(n_proposals, 1)),
                MIN_ALPHA,
                None,
            )
            alpha_mat = np.repeat(current_alpha.squeeze(-1)[None, :], n_proposals, axis=0)
            alpha_mat[:, v] = proposals_v.squeeze(-1)
            alpha_tensor = torch.tensor(alpha_mat, device=device, dtype=torch.float32)

            print(f" {variable_names[v]:5s} | Processing {n_proposals} proposals...")

            torch_gen.manual_seed(int(rng.integers(0, 2**31 - 1)))
            best_ensemble, joint_scores = batched_forward_proposals(
                model=model,
                previous_fields=prev_all,
                current_fields=curr_all,
                time_normalised=time_all,
                reference_tensor=ref_full,
                alpha_matrix=alpha_tensor,
                ensemble_size=ensemble_size,
                device=device,
                buffers=buffers,
                batch_manager=batch_manager,
                generator=torch_gen,
            )

            joint_scores = np.array(joint_scores)
            best_idx = int(joint_scores.argmin())
            current_alpha[v] = proposals_v[best_idx]
            posterior_samples[s, v] = current_alpha[v]
            posterior_crps[s, v] = joint_scores[best_idx]

            print(f" {variable_names[v]:5s} α*={current_alpha[v,0]:.3f}  jointCRPS={joint_scores[best_idx]: .4f}")

            if log_diagnostics:
                for j in range(num_variables):
                    if j == v:
                        spread_val = compute_ensemble_spread(best_ensemble[:, :, j].cpu())
                        mae_val = compute_mean_absolute_error(best_ensemble[:, :, j].cpu(), curr_all[:, j].cpu())
                        ensemble_spread_records[j].append(spread_val)
                        mean_absolute_error_records[j].append(mae_val)
                        ranks = compute_rank_histogram(best_ensemble[:, :, j], curr_all[:, j], ensemble_size)
                        rank_histograms[j].extend(ranks.tolist())

            del best_ensemble
            torch.cuda.empty_cache()

        step_mean_crps[s] = posterior_crps[s].mean()
        print(f"⇒ mean joint CRPS (all vars) = {step_mean_crps[s]:.4f}")

        np.savez_compressed(
            ckpt_path,
            step=s,
            posterior_samples=posterior_samples,
            posterior_crps=posterior_crps,
            step_mean_crps=step_mean_crps,
            last_alpha=current_alpha,
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    return {
        "posterior_samples": posterior_samples,
        "posterior_crps": posterior_crps,
        "posterior_mean": posterior_samples.mean(axis=0),
        "posterior_variance": posterior_samples.var(axis=0),
        "rank_histograms": rank_histograms,
        "ensemble_mae": np.array(mean_absolute_error_records, dtype=np.float32),
        "ensemble_spread": np.array(ensemble_spread_records, dtype=np.float32),
        "step_mean_crps": step_mean_crps,
    }
