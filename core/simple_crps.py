"""
Simple, safe CRPS computation that avoids quadratic memory allocation.
Only fixes the CRPS bottleneck without complex memory management.
"""

import torch


def safe_crps_computation(ensemble: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Safe CRPS computation that uses sampling to avoid quadratic memory usage.
    
    Args:
        ensemble: [K, ...] where K is ensemble size
        target: [...] target values
    
    Returns:
        CRPS values of same shape as target
    """
    K = ensemble.shape[0]
    target_expanded = target.unsqueeze(0)  # [1, ...]
    
    # Exact absolute error term - this is always safe
    absolute_error = torch.abs(ensemble - target_expanded).mean(dim=0)
    
    # For pairwise term, use sampling to avoid K²×spatial memory allocation
    if K <= 10:  # Small ensembles - use exact computation
        ensemble_flat = ensemble.view(K, -1)
        pairwise_diffs = torch.abs(ensemble_flat.unsqueeze(1) - ensemble_flat.unsqueeze(0))
        pairwise_mean = pairwise_diffs.mean(dim=(0, 1)).view_as(target)
    else:
        # Large ensembles - use random sampling approximation
        ensemble_flat = ensemble.view(K, -1)  # [K, spatial]
        
        # Sample 500 random pairs (much smaller than K²)
        n_samples = 500
        idx1 = torch.randint(0, K, (n_samples,), device=ensemble.device)
        idx2 = torch.randint(0, K, (n_samples,), device=ensemble.device)
        
        # Compute sampled pairwise differences
        sampled_diffs = torch.abs(ensemble_flat[idx1] - ensemble_flat[idx2])  # [500, spatial]
        pairwise_mean = sampled_diffs.mean(dim=0).view_as(target)
    
    return absolute_error - 0.5 * pairwise_mean


def compute_crps_for_proposal(
    ensemble_output: torch.Tensor,  # [K,N,V,H,W]
    target: torch.Tensor,          # [N,V,H,W]
    num_variables: int,
) -> float:
    """Simple CRPS computation with memory-safe implementation."""
    
    crps_values = []
    for j in range(num_variables):
        crps_pj = safe_crps_computation(
            ensemble_output[:, :, j].contiguous(),
            target[:, j].contiguous(),
        ).mean()
        crps_values.append(crps_pj)
    
    return torch.stack(crps_values).mean().item()