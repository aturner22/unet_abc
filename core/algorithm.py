import os
import numpy as np
import torch
import logging
from tqdm import tqdm
from .scoring import get_scoring_function
import scipy.stats
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
    offset = torch.randint(
        1, T, (batch_size, ensemble_size), device=device, generator=generator
    )
    idx2 = (idx1 + offset) % T
    diff = reference_tensor[idx1] - reference_tensor[idx2]
    energy = torch.sqrt(
        diff.pow(2).mean(dim=(2, 3, 4), keepdim=True) + eps_energy
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

    current_slice = previous_fields[:, :V]
    past_slice = previous_fields[:, V: 2 * V]
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

        _, total_mem, free_mem = batch_manager.get_memory_stats()
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
    joint_scores: list[float] = []
    first_output = None

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
                                "Persistent OOM even with batch size 1. Available memory may be insufficient."
                            ) from e
                    else:
                        raise e

            if retry_count > max_retries:
                raise RuntimeError(
                    f"Failed to process batch after {max_retries} retries with batch size reductions"
                )

        y_full = torch.cat(out_chunks, dim=0).view(N, ensemble_size, V, H, W)
        proposal_output = y_full.permute(1, 0, 2, 3, 4)
        joint_scores.append(scoring_fn(proposal_output, current_fields, V))

        if P == 1 and first_output is None:
            first_output = proposal_output.clone()

        logger.debug("Joint score for alpha[%d]: %.4f", p, joint_scores[-1])

        del proposal_output, y_full, out_chunks, full_input, full_time
        torch.cuda.empty_cache()

    return (first_output if P == 1 else None), joint_scores


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

    subset_indices = np.random.choice(len(full_dataset), size=sample_size, replace=False)
    dataset = torch.utils.data.Subset(full_dataset, subset_indices)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    batches = []
    for previous_fields, current_fields, valid_time in loader:
        previous_fields = previous_fields.to(device)
        current_fields = current_fields.view(
            -1, num_variables, len(latitude), len(longitude)
        ).to(device)
        time_normalised = (
            torch.tensor([valid_time[0]], dtype=torch.float32, device=device) / max_horizon
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
) -> dict:
    """
    Modes (config.inference_mode):
      - "smc"         : SMC-ABC with RW-normal proposals, exponential prior, Gaussian score-kernel.
      - "abc_gibbs"   : ABC-Gibbs using conditional-prior reference table + argmin inside ε-quantile.
      - "greedy"      : Greedy optimiser (deterministic argmin) over RW-normal proposals.
    """

    device = next(model.parameters()).device
    ref_full = torch.from_numpy(np.array(reference_mmap, copy=True)).to(device)

    scoring_fn = get_scoring_function(score_function)
    logger.info(f"Using {score_function.upper()} as discrepancy measure")

    inference_mode = getattr(config, "inference_mode", "abc_gibbs").lower()
    assert inference_mode in {"smc_gibbs", "conditional_gibbs", "greedy"}, (
        "config.inference_mode must be one of {'smc_gibbs','conditional_gibbs','greedy'}"
    )
    logger.info(f"Inference mode: {inference_mode}")

    # Data/batch materialisation
    if resample_temporal:
        if full_dataset is None or sample_size is None:
            raise ValueError("resample_temporal=True requires full_dataset and sample_size")
        logger.info(f"Temporal resampling enabled: {sample_size} samples per step")

        sample_batch = next(iter(torch.utils.data.DataLoader(full_dataset, batch_size=1)))
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

    # Warm model
    with torch.no_grad():
        _ = model(prev_all[:1], time_all[:1])

    posterior_samples = np.zeros((n_steps, num_variables, 1), dtype=np.float32)
    posterior_scores = np.zeros((n_steps, num_variables), dtype=np.float32)
    step_mean_scores = np.zeros(n_steps, dtype=np.float32)

    current_alpha = np.random.uniform(*getattr(config, "initial_alpha_range", (0.05, 1.0)),
                                      size=(num_variables, 1))
    proposal_std = np.full((num_variables, 1), getattr(config, "proposal_scale", 0.05), dtype=np.float32)

    rng = np.random.default_rng()
    torch_gen = torch.Generator(device=device)

    GAMMA_SHAPE = 2.0 
    GAMMA_SCALE = 0.13  
    ckpt_path = os.path.join(result_directory, getattr(config, "checkpoint_file", "gibbs_checkpoint_step.npz"))
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

        # Adapt RW step size (where relevant)
        if s and (s % getattr(config, "adapt_every", 5) == 0) and (s < getattr(config, "adapt_stop", 30)):
            proposal_std *= getattr(config, "adapt_factor", 0.85)
            logger.debug(f"Proposal variance adapted: {proposal_std.mean():.3f}")
        elif s == getattr(config, "adapt_stop", 30):
            logger.info(f"Adaptation stopped at step {s}, fixed at {proposal_std.mean():.3f}")

        for v in tqdm(range(num_variables), leave=False):
            if inference_mode == "conditional_gibbs":
                proposals_v = np.random.gamma(shape=GAMMA_SHAPE, scale=GAMMA_SCALE, size=(n_proposals, 1))
                min_alpha = float(getattr(config, "min_alpha", 1e-4))
                proposals_v = np.clip(proposals_v, min_alpha, None)

            else:
                proposals_v = np.clip(
                    rng.normal(
                        loc=current_alpha[v], scale=proposal_std[v], size=(n_proposals, 1)
                    ),
                    getattr(config, "min_alpha", 1e-4),
                    None,
                )
            alpha_mat = np.repeat(current_alpha.squeeze(-1)[None, :], n_proposals, axis=0)
            alpha_mat[:, v] = proposals_v.squeeze(-1)
            alpha_tensor = torch.tensor(alpha_mat, device=device, dtype=torch.float32)

            _, joint_scores = batched_forward_proposals(
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
                eps_energy=getattr(config, "eps_energy", 1e-6),
            )

            joint_scores = np.asarray(joint_scores, dtype=np.float64)
            a = proposals_v.squeeze(-1)

            if inference_mode == "conditional_gibbs":
                # --- ABC-Gibbs: argmin inside epsilon-quantile of scores ---
                if getattr(config, "adaptive_epsilon", False):
                    eps_j = max(
                        getattr(config, "min_epsilon", 1e-3),
                        np.quantile(joint_scores, getattr(config, "epsilon_quantile", 0.3)),
                    )
                else:
                    eps_j = float(getattr(config, "abc_epsilon", 0.02))

                acceptable = np.where(joint_scores <= eps_j)[0]
                if acceptable.size == 0:
                    sel_idx = int(np.argmin(joint_scores))
                else:
                    sel_idx = acceptable[int(np.argmin(joint_scores[acceptable]))]

                current_alpha[v] = a[sel_idx]
                posterior_samples[s, v] = current_alpha[v]
                posterior_scores[s, v] = joint_scores[sel_idx]

            elif inference_mode == "smc_gibbs":
                # --- SMC-ABC: importance weights with exponential prior and Gaussian kernel ---
                if getattr(config, "adaptive_epsilon", False):
                    config.abc_epsilon = max(
                        getattr(config, "min_epsilon", 1e-3),
                        np.quantile(joint_scores, getattr(config, "epsilon_quantile", 0.3)),
                    )
                eps = float(getattr(config, "abc_epsilon", 0.02))

                # Proposal q density (RW normal)
                q_density = scipy.stats.norm.pdf(
                    a,
                    loc=float(current_alpha[v, 0]),
                    scale=float(proposal_std[v, 0]),
                )
                q_density = np.clip(q_density, 1e-32, None)

                # Exponential prior density π(a) = λ e^{-λ a}, a>=0
                prior_density = scipy.stats.gamma.pdf(a, a=GAMMA_SHAPE, scale=GAMMA_SCALE)
                prior_density = np.clip(prior_density, 1e-300, None)

                # ABC kernel on the proper score (Gaussian in discrepancy)
                lik_weights = np.exp(-(joint_scores ** 2) / (2.0 * (eps ** 2)))
                lik_weights = np.clip(lik_weights, 1e-300, None)

                weights = lik_weights * (prior_density / q_density)
                wsum = weights.sum()
                if not np.isfinite(wsum) or wsum <= 0:
                    finite = np.isfinite(weights) & (weights > 0)
                    weights = (
                        finite.astype(float) / finite.sum()
                        if finite.any()
                        else np.ones_like(weights) / len(weights)
                    )
                else:
                    weights /= wsum

                sel_idx = int(rng.choice(len(weights), p=weights))
                current_alpha[v] = a[sel_idx]
                posterior_samples[s, v] = current_alpha[v]
                posterior_scores[s, v] = joint_scores[sel_idx]

            else:  # greedy
                # --- Greedy optimiser: deterministic argmin over RW-normal proposals ---
                tol = float(getattr(config, "argmin_tolerance", 0.0))
                m = np.min(joint_scores)
                if tol > 0:
                    candidates = np.where(joint_scores <= m + tol)[0]
                    sel_idx = int(rng.choice(candidates)) if candidates.size else int(np.argmin(joint_scores))
                else:
                    sel_idx = int(np.argmin(joint_scores))

                current_alpha[v] = a[sel_idx]
                posterior_samples[s, v] = current_alpha[v]
                posterior_scores[s, v] = joint_scores[sel_idx]

            torch.cuda.empty_cache()

        step_mean_scores[s] = posterior_scores[s].mean()
        logger.info(f"Step {s + 1}/{n_steps}: mean {score_function.upper()}={step_mean_scores[s]:.4f}")

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

    logger.info("ABC-Gibbs sampling completed")

    return {
        "posterior_samples": posterior_samples,
        "posterior_scores": posterior_scores,
    }
