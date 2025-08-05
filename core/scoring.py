import torch
from typing import Callable


def safe_crps_computation(ensemble: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    K = ensemble.shape[0]
    target_expanded = target.unsqueeze(0)
    absolute_error = torch.abs(ensemble - target_expanded).mean(dim=0)

    if K <= 10:
        ensemble_flat = ensemble.view(K, -1)
        pairwise_diffs = torch.abs(
            ensemble_flat.unsqueeze(1) - ensemble_flat.unsqueeze(0)
        )
        pairwise_mean = pairwise_diffs.mean(dim=(0, 1)).view_as(target)
    else:
        ensemble_flat = ensemble.view(K, -1)
        n_samples = 500
        idx1 = torch.randint(0, K, (n_samples,), device=ensemble.device)
        idx2 = torch.randint(0, K, (n_samples,), device=ensemble.device)
        sampled_diffs = torch.abs(ensemble_flat[idx1] - ensemble_flat[idx2])
        pairwise_mean = sampled_diffs.mean(dim=0).view_as(target)

    return absolute_error - 0.5 * pairwise_mean


def multivariate_energy_score(
    ensemble: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    K, N = ensemble.shape[:2]
    ensemble_flat = ensemble.view(K, N, -1)
    target_flat = target.view(N, -1)

    target_expanded = target_flat.unsqueeze(0)
    forecast_obs_distances = torch.norm(ensemble_flat - target_expanded, dim=-1)

    first_term = forecast_obs_distances.mean(dim=0)

    if K <= 10:
        ensemble_expanded_1 = ensemble_flat.unsqueeze(1)
        ensemble_expanded_2 = ensemble_flat.unsqueeze(0)
        pairwise_distances = torch.norm(
            ensemble_expanded_1 - ensemble_expanded_2, dim=-1
        )
        second_term = pairwise_distances.mean(dim=(0, 1))
    else:
        n_samples = min(256, K * (K - 1) // 2)
        idx1 = torch.randint(0, K, (n_samples,), device=ensemble.device)
        idx2 = torch.randint(0, K, (n_samples,), device=ensemble.device)
        
        # Compute distances in smaller chunks to avoid OOM
        chunk_size = 10
        distance_chunks = []
        for i in range(0, n_samples, chunk_size):
            end_idx = min(i + chunk_size, n_samples)
            chunk_idx1 = idx1[i:end_idx]
            chunk_idx2 = idx2[i:end_idx]
            diffs = ensemble_flat[chunk_idx1] - ensemble_flat[chunk_idx2]
            distances = torch.norm(diffs, dim=-1)
            distance_chunks.append(distances)
        
        sampled_distances = torch.cat(distance_chunks, dim=0)
        second_term = sampled_distances.mean(dim=0)

    energy = first_term - 0.5 * second_term
    return energy


def compute_crps_for_proposal(
    ensemble_output: torch.Tensor, target: torch.Tensor, num_variables: int
) -> float:
    crps_values = []
    for j in range(num_variables):
        crps_pj = safe_crps_computation(
            ensemble_output[:, :, j].contiguous(), target[:, j].contiguous()
        ).mean()
        crps_values.append(crps_pj)
    return torch.stack(crps_values).mean().item()


def compute_energy_score_for_proposal(
    ensemble_output: torch.Tensor, target: torch.Tensor, num_variables: int
) -> float:
    energy_scores = multivariate_energy_score(ensemble_output, target)
    return energy_scores.mean().item()


def get_scoring_function(
    score_name: str,
) -> Callable[[torch.Tensor, torch.Tensor, int], float]:
    scoring_functions = {
        "crps": compute_crps_for_proposal,
        "energy": compute_energy_score_for_proposal,
    }

    if score_name not in scoring_functions:
        available = list(scoring_functions.keys())
        raise ValueError(
            f"Unknown scoring function '{score_name}'. Available: {available}"
        )

    return scoring_functions[score_name]


def compute_score_for_proposal(
    ensemble_output: torch.Tensor,
    target: torch.Tensor,
    num_variables: int,
    score_function: str,
) -> float:
    scoring_fn = get_scoring_function(score_function)
    return scoring_fn(ensemble_output, target, num_variables)
