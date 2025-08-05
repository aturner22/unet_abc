import os
import numpy as np
import torch
import logging
from typing import Any
from tqdm import tqdm
from .evaluation import (
    compute_rank_histogram,
    compute_mean_absolute_error,
    compute_ensemble_spread,
)
from .scoring import get_scoring_function
from .batch_manager import MemorySafeBatchManager


def generate_joint_rfp(
    reference_tensor: torch.Tensor,
    alpha_matrix: torch.Tensor,
    batch_size: int,
    ensemble_size: int,
    device: torch.device,
    generator: torch.Generator,
    eps_energy: float,
) -> torch.Tensor:
    P = alpha_matrix.shape[0]
    T, V = reference_tensor.shape[0], reference_tensor.shape[1]
    idx1 = torch.randint(
        0, T, (batch_size, ensemble_size), device=device, generator=generator
    )
    idx2 = torch.randint(
        0, T, (batch_size, ensemble_size), device=device, generator=generator
    )
    diff = reference_tensor[idx1] - reference_tensor[idx2]
    energy = torch.sqrt(
        diff.pow(2).mean(dim=(2, 3, 4), keepdim=True).clamp_min(eps_energy)
    )
    diff_norm = diff / energy
    perturb = alpha_matrix.reshape(P, 1, 1, V, 1, 1) * diff_norm.unsqueeze(0)
    return perturb


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
    scoring_fn,
    config,
    logger: logging.Logger,
    eps_energy: float,
) -> tuple[torch.Tensor, list[float]]:
    N, C, H, W = previous_fields.shape
    V = current_fields.shape[1]
    P = alpha_matrix.shape[0]
    logger.debug("Evaluating batched proposals for %d perturbation(s)", P)

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
        eps_energy=eps_energy,
    )

    curr_base = current_slice.unsqueeze(1).expand(-1, ensemble_size, -1, -1, -1)
    past_base = past_slice.unsqueeze(1).expand(-1, ensemble_size, -1, -1, -1)
    stat_base = static_slice.unsqueeze(1).expand(-1, ensemble_size, -1, -1, -1)

    if not batch_manager.calibrated:
        test_n = min(N, 2)
        sample_input = torch.cat(
            [curr_base[:test_n], past_base[:test_n], stat_base[:test_n]], dim=2
        )
        sample_input = sample_input.view(test_n * ensemble_size, C, H, W)
        logger.debug("Calibrating batch size for input shape: %s", sample_input.shape)
        sample_time = (
            time_normalised[:test_n]
            .view(-1, 1)
            .expand(-1, ensemble_size)
            .reshape(-1, 1)
        )

        allocated, total_mem, free_mem = batch_manager.get_memory_stats()
        logger.info(
            f"Memory calibration (Free: {free_mem / 1e9:.1f}GB/{total_mem / 1e9:.1f}GB)"
        )

        batch_manager.find_max_batch_size(
            model=model,
            sample_input=sample_input,
            sample_time=sample_time,
            max_search=128,
        )

    optimal_batch_size = batch_manager.get_batch_size()
    logger.debug("Determined optimal batch size: %d", optimal_batch_size)
    joint_scores: list[float] = []
    best_proposal_output: torch.Tensor | None = None
    best_score = float("inf")

    for p in range(P):
        buffers["curr"][:N].copy_(curr_base)
        buffers["curr"][:N].add_(perturb[p])
        buffers["past"][:N].copy_(past_base)
        buffers["stat"][:N].copy_(stat_base)

        full_input = torch.cat(
            [buffers["curr"][:N], buffers["past"][:N], buffers["stat"][:N]], dim=2
        )
        full_input = full_input.view(N * ensemble_size, C, H, W)
        full_time = time_normalised.view(-1, 1).expand(-1, ensemble_size).reshape(-1, 1)

        step = optimal_batch_size * ensemble_size
        out_chunks = []
        start = 0

        while start < N * ensemble_size:
            end = min(start + step, N * ensemble_size)
            retry_count = 0
            max_retries = (
                config.memory_management.get("max_retries", 5)
                if hasattr(config, "memory_management")
                else 5
            )

            while retry_count <= max_retries:
                try:
                    with torch.no_grad():
                        y = model(full_input[start:end], full_time[start:end])
                    out_chunks.append(y)
                    start = end
                    break

                except RuntimeError as e:
                    if "out of memory" in str(e).lower() and retry_count < max_retries:
                        retry_count += 1
                        torch.cuda.empty_cache()
                        optimal_batch_size = batch_manager.reduce_batch_size()
                        step = optimal_batch_size * ensemble_size
                        end = min(start + step, N * ensemble_size)

                        logger.warning(
                            f"OOM retry {retry_count}/{max_retries}, batch size: {optimal_batch_size}"
                        )

                        retry_threshold = (
                            config.memory_management.get(
                                "retry_at_batch_one_threshold", 3
                            )
                            if hasattr(config, "memory_management")
                            else 3
                        )
                        if optimal_batch_size == 1 and retry_count >= retry_threshold:
                            raise RuntimeError(
                                "Persistent OOM even with batch size 1. "
                                "Available memory may be insufficient."
                            ) from e
                    else:
                        raise e

            if retry_count > max_retries:
                raise RuntimeError(
                    f"Failed to process batch after {max_retries} retries "
                    "with batch size reductions"
                )

        y_full = torch.cat(out_chunks, dim=0).view(N, ensemble_size, V, H, W)

        # REMOVE ONCE FIXED
        if torch.isnan(y_full).any():
            logger.error("NaNs detected in model outputs before permutation")
            logger.debug(
                "Offending output tensor stats: min=%.4e, max=%.4e, mean=%.4e",
                y_full.min().item(),
                y_full.max().item(),
                y_full.mean().item(),
            )
            raise ValueError("NaNs in model outputs (pre-permutation)")

        if torch.isinf(y_full).any():
            logger.error("Infs detected in model outputs before permutation")
            raise ValueError("Infs in model outputs (pre-permutation)")
        ###

        proposal_output = y_full.permute(1, 0, 2, 3, 4)
        ###
        if torch.isnan(proposal_output).any():
            logger.error("NaNs in permuted proposal_output")
            raise ValueError("NaNs in proposal_output")

        if torch.isinf(proposal_output).any():
            logger.error("Infs in permuted proposal_output")
            raise ValueError("Infs in proposal_output")

        joint_score = scoring_fn(proposal_output, current_fields, V)
        joint_scores.append(joint_score)

        if joint_score < best_score:
            best_score = joint_score
            if best_proposal_output is not None:
                del best_proposal_output
            best_proposal_output = proposal_output.clone()
        elif best_proposal_output is None:
            # If no proposal has been better, keep the first one to avoid None
            best_proposal_output = proposal_output.clone()
            best_score = joint_score

        del proposal_output, y_full, out_chunks, full_input, full_time
        torch.cuda.empty_cache()

    return best_proposal_output, joint_scores


def resample_temporal_batches(
    full_dataset,
    sample_size: int,
    device: torch.device,
    num_variables: int,
    max_horizon: int,
    latitude,
    longitude,
    step_seed: int = None,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    if step_seed is not None:
        np.random.seed(step_seed)

    subset_indices = np.random.choice(
        len(full_dataset), size=sample_size, replace=False
    )
    dataset = torch.utils.data.Subset(full_dataset, subset_indices)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    batches = []
    for previous_fields, current_fields, valid_time in loader:
        previous_fields = previous_fields.to(device)
        current_fields = current_fields.view(
            -1, num_variables, len(latitude), len(longitude)
        ).to(device)
        time_normalised = (
            torch.tensor([valid_time[0]], dtype=torch.float32, device=device)
            / max_horizon
        )
        batches.append((previous_fields, current_fields, time_normalised))

    return batches


def run_gibbs_abc_rfp(
    *,
    model: torch.nn.Module,
    config,
    batches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    full_dataset=None,
    sample_size: int = None,
    ensemble_size: int,
    n_steps: int,
    n_proposals: int,
    num_variables: int,
    variable_names: list[str],
    reference_mmap: np.memmap,
    result_directory: str,
    log_diagnostics: bool = True,
    resample_temporal: bool = False,
    score_function: str,
    logger: logging.Logger,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    ref_full = torch.from_numpy(np.array(reference_mmap, copy=True)).to(device)

    scoring_fn = get_scoring_function(score_function)
    logger.info(f"Using {score_function.upper()} as discrepancy measure")

    if resample_temporal:
        if full_dataset is None or sample_size is None:
            raise ValueError(
                "resample_temporal=True requires full_dataset and sample_size"
            )
        logger.info(f"Temporal resampling enabled: {sample_size} samples per step")

        sample_batch = next(
            iter(torch.utils.data.DataLoader(full_dataset, batch_size=1))
        )
        _, current_sample, _ = sample_batch
        V = current_sample.shape[1] if len(current_sample.shape) > 1 else 5
        H = current_sample.shape[2] if len(current_sample.shape) > 2 else 64
        W = current_sample.shape[3] if len(current_sample.shape) > 3 else 64

        latitude = list(range(H))
        longitude = list(range(W))
        max_horizon = 240
        num_variables = V

        initial_batches = resample_temporal_batches(
            full_dataset,
            sample_size,
            device,
            num_variables,
            max_horizon,
            latitude,
            longitude,
            step_seed=42,
        )
        prev_all = torch.cat([b[0] for b in initial_batches], dim=0).to(device)
        curr_all = torch.cat([b[1] for b in initial_batches], dim=0).to(device)
        time_all = torch.cat([b[2] for b in initial_batches], dim=0).to(device)
        N = sample_size
    else:
        if batches is None:
            raise ValueError("batches required when resample_temporal=False")
        prev_all = torch.cat([b[0] for b in batches], dim=0).to(device)
        curr_all = torch.cat([b[1] for b in batches], dim=0).to(device)
        time_all = torch.cat([b[2] for b in batches], dim=0).to(device)
        N = len(batches)

    _, _, H, W = prev_all.shape
    V = curr_all.shape[1]

    buffers = {
        "curr": torch.empty((N, ensemble_size, V, H, W), device=device),
        "past": torch.empty((N, ensemble_size, V, H, W), device=device),
        "stat": torch.empty((N, ensemble_size, 2, H, W), device=device),
    }

    batch_manager = MemorySafeBatchManager(device, config)

    with torch.no_grad():
        _ = model(prev_all[:1], time_all[:1])

    posterior_samples = np.zeros((n_steps, num_variables, 1), dtype=np.float32)
    posterior_scores = np.zeros((n_steps, num_variables), dtype=np.float32)
    step_mean_scores = np.zeros(n_steps, dtype=np.float32)

    rank_histograms = [[] for _ in range(num_variables)]
    ensemble_spread_records = [[] for _ in range(num_variables)]
    mean_absolute_error_records = [[] for _ in range(num_variables)]

    current_alpha = np.random.uniform(
        *config.initial_alpha_range, size=(num_variables, 1)
    )
    proposal_std = np.full((num_variables, 1), config.proposal_scale, dtype=np.float32)

    rng = np.random.default_rng()
    torch_gen = torch.Generator(device=device)

    ckpt_path = os.path.join(result_directory, config.checkpoint_file)
    start_step = 0
    if os.path.exists(ckpt_path):
        ck = np.load(ckpt_path, allow_pickle=True)
        logger.info(f"Resuming from step {ck['step'] + 1}")
        posterior_samples[: ck["step"] + 1] = ck["posterior_samples"]
        posterior_scores[: ck["step"] + 1] = ck["posterior_scores"]
        step_mean_scores[: ck["step"] + 1] = ck["step_mean_scores"]
        current_alpha = ck["last_alpha"]
        start_step = int(ck["step"]) + 1
        del ck

    for s in tqdm(range(start_step, n_steps), desc="Gibbs Steps", position=0):
        logger.debug("Beginning Gibbs iteration s = %d", s)
        if resample_temporal:
            step_seed = 1000 + s
            current_batches = resample_temporal_batches(
                full_dataset,
                sample_size,
                device,
                num_variables,
                max_horizon,
                latitude,
                longitude,
                step_seed,
            )
            prev_all = torch.cat([b[0] for b in current_batches], dim=0).to(device)
            curr_all = torch.cat([b[1] for b in current_batches], dim=0).to(device)
            time_all = torch.cat([b[2] for b in current_batches], dim=0).to(device)

        if s and (s % config.adapt_every == 0) and (s < config.adapt_stop):
            proposal_std *= config.adapt_factor
            logger.debug(f"Proposal variance adapted: {proposal_std.mean():.3f}")
        elif s == config.adapt_stop:
            logger.info(
                f"Adaptation stopped at step {s}, fixed at {proposal_std.mean():.3f}"
            )

        for v in range(num_variables):
            proposals_v = np.clip(
                rng.normal(
                    loc=current_alpha[v], scale=proposal_std[v], size=(n_proposals, 1)
                ),
                config.min_alpha,
                None,
            )
            alpha_mat = np.repeat(
                current_alpha.squeeze(-1)[None, :], n_proposals, axis=0
            )
            alpha_mat[:, v] = proposals_v.squeeze(-1)
            alpha_tensor = torch.tensor(alpha_mat, device=device, dtype=torch.float32)

            logger.debug("Sampling proposals for variable index v = %d", v)
            logger.debug("Current alpha_v: %.5f", current_alpha[v, 0])
            logger.debug(
                "Proposal standard deviation alpha_v: %.5f", proposal_std[v, 0]
            )
            logger.debug("Generated proposals: %s", proposals_v.squeeze().tolist())

            torch_gen.manual_seed(int(rng.integers(0, 2**31 - 1)))
            logger.debug("previous_fields.shape[1] = %d", prev_all.shape[1])
            logger.debug("V = %d", V)
            logger.debug("Expected: 2V + S = %d", 2 * V + 2)

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
                scoring_fn=scoring_fn,
                config=config,
                logger=logger,
                eps_energy=config.eps_energy,
            )

            joint_scores = np.array(joint_scores)
            best_idx = int(joint_scores.argmin())
            current_alpha[v] = proposals_v[best_idx]
            posterior_samples[s, v] = current_alpha[v]
            posterior_scores[s, v] = joint_scores[best_idx]
            logger.debug(
                "Scores for proposals (variable %d): %s", v, joint_scores.tolist()
            )
            logger.debug(
                "Selected best proposal index: %d with score: %.5f",
                best_idx,
                joint_scores[best_idx],
            )
            logger.debug("Updated alpha [%d] = %.5f", v, current_alpha[v, 0])

            if log_diagnostics:
                for j in range(num_variables):
                    if j == v:
                        spread_val = compute_ensemble_spread(
                            best_ensemble[:, :, j].cpu()
                        )
                        mae_val = compute_mean_absolute_error(
                            best_ensemble[:, :, j].cpu(), curr_all[:, j].cpu()
                        )
                        ensemble_spread_records[j].append(spread_val)
                        mean_absolute_error_records[j].append(mae_val)
                        ranks = compute_rank_histogram(
                            best_ensemble[:, :, j], curr_all[:, j], ensemble_size
                        )
                        rank_histograms[j].extend(ranks.tolist())

            del best_ensemble
            torch.cuda.empty_cache()

        step_mean_scores[s] = posterior_scores[s].mean()

        logger.info(
            f"Step {s + 1}/{n_steps}: mean {score_function.upper()}={step_mean_scores[s]:.4f}"
        )

        np.savez_compressed(
            ckpt_path,
            step=s,
            posterior_samples=posterior_samples,
            posterior_scores=posterior_scores,
            step_mean_scores=step_mean_scores,
            last_alpha=current_alpha,
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    logger.debug(
        "Final posterior mean: %s", posterior_samples.mean(axis=0).squeeze().tolist()
    )
    logger.debug(
        "Final posterior variance: %s", posterior_samples.var(axis=0).squeeze().tolist()
    )

    logger.info("ABC-Gibbs sampling completed")

    return {
        "posterior_samples": posterior_samples,
        "posterior_scores": posterior_scores,
        "posterior_mean": posterior_samples.mean(axis=0),
        "posterior_variance": posterior_samples.var(axis=0),
        "rank_histograms": rank_histograms,
        "ensemble_mae": np.array(mean_absolute_error_records, dtype=np.float32),
        "ensemble_spread": np.array(ensemble_spread_records, dtype=np.float32),
        "step_mean_scores": step_mean_scores,
    }
