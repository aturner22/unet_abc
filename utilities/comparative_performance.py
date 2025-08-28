#!/usr/bin/env python3
"""
Comparative Performance Evaluation Script - Batched Version

Compares ABC-calibrated RFP forecasts against multiple benchmarks:
- Persistence forecasting (deterministic)
- Climatological forecasting (probabilistic)
- AR(1) per grid point (probabilistic)
- Deterministic model + Gaussian noise (probabilistic)

Computes CRPS, Energy Score, Spread, and MAE with efficient batching.
"""

import argparse
import json
import logging
import sys
import warnings
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import scipy.stats

# Suppress warnings
warnings.filterwarnings('ignore')

# Add current directory to path for imports
sys.path.append('.')

# Import core modules
from core.config import Config
from core.scoring import compute_crps_for_proposal, compute_energy_score_for_proposal
from core.io_utils import load_model_and_test_data
from core.algorithm import generate_joint_rfp


# DEFAULT_T_STEPS removed - was causing parameter confusion
DEFAULT_VARIABLE_NAMES = ["z500", "t850", "t2m", "u10", "v10"]

# Prior distribution parameters (same as used in ABC algorithm)
GAMMA_SHAPE = 2.0
GAMMA_SCALE = 0.13

# MCMC burn-in periods
BURN_IN_DEFAULT = 20  # Default burn-in for most algorithms
BURN_IN_GREEDY = 40   # Extended burn-in for greedy algorithm


def get_burn_in_period(result_path: Path) -> int:
    """Determine appropriate burn-in period based on algorithm type."""
    path_str = str(result_path).lower()
    if 'greedy' in path_str:
        return BURN_IN_GREEDY
    else:
        return BURN_IN_DEFAULT

# Uncalibrated RFP methods - two different baselines
def get_uncalibrated_rfp_alpha_prior(num_variables: int, seed: int = 42) -> np.ndarray:
    """
    Generate uncalibrated RFP alpha parameters by drawing from the prior distribution.
    This provides a proper baseline to demonstrate the impact of ABC calibration.
    
    Args:
        num_variables: Number of atmospheric variables
        seed: Random seed for reproducibility
        
    Returns:
        Array of alpha parameters drawn from Gamma(shape=2.0, scale=0.13) prior
    """
    rng = np.random.RandomState(seed)
    alpha_prior = rng.gamma(shape=GAMMA_SHAPE, scale=GAMMA_SCALE, size=num_variables)
    return alpha_prior


def get_uncalibrated_rfp_alpha_ones(num_variables: int) -> np.ndarray:
    """
    Generate uncalibrated RFP alpha parameters with all values set to 1.0.
    This provides a simple baseline where all variables have equal perturbation scaling.
    
    Args:
        num_variables: Number of atmospheric variables
        
    Returns:
        Array of alpha parameters all set to 1.0
    """
    return np.ones(num_variables)


# Legacy function for backward compatibility
def get_uncalibrated_rfp_alpha(num_variables: int, seed: int = 42) -> np.ndarray:
    """Legacy function - redirects to prior-based uncalibrated method."""
    return get_uncalibrated_rfp_alpha_prior(num_variables, seed)


def setup_logging():
    """Configure logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(__name__)


def load_posterior_data(result_path: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
    """Load posterior samples and scores from result directory."""
    samples_path = result_path / "posterior_samples.npy"
    scores_path = result_path / "posterior_scores.npy"
    checkpoint_path = result_path / "gibbs_checkpoint_step.npz"

    # Try to load from regular result files first
    if samples_path.exists() and scores_path.exists():
        samples = np.load(samples_path) 
        scores = np.load(scores_path)
        data_source = "regular results"
    elif checkpoint_path.exists():
        # Fallback to checkpoint file
        checkpoint = np.load(checkpoint_path, allow_pickle=True)
        
        # Extract data from checkpoint
        samples = checkpoint["posterior_samples"]
        scores = checkpoint["posterior_scores"]
        completed_step = int(checkpoint["step"])
        
        # Only use data up to the completed step (inclusive)
        samples = samples[:completed_step + 1]
        scores = scores[:completed_step + 1]
        data_source = "checkpoint"
    else:
        return None, None, None
    
    return samples, scores, data_source


def get_posterior_alpha(samples: np.ndarray, mode: str = "mean", burn_in: int = 20) -> np.ndarray:
    """Extract alpha values from posterior samples with burn-in period.
    
    IMPORTANT: This preserves the joint posterior structure.
    - 'mean': Takes the empirical mean of the joint posterior 
    - 'sample': Takes a single joint sample (preserves correlations)
    """
    # Apply burn-in period
    samples_burned = samples[burn_in:] if samples.shape[0] > burn_in else samples
    
    if mode == "mean":
        # Empirical mean of joint posterior (still valid)
        return samples_burned.mean(axis=0).squeeze()  # Shape: (n_variables,)
    elif mode == "sample":
        # JOINT sample from posterior (preserves parameter correlations)
        random_step = np.random.randint(0, samples_burned.shape[0])
        return samples_burned[random_step].squeeze()  # Shape: (n_variables,)
    else:
        raise ValueError(f"Mode must be 'mean' or 'sample', got {mode}")




def generate_rfp_forecasts_batched(
    model: torch.nn.Module,
    test_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    alpha_values: np.ndarray,
    reference_tensor: torch.Tensor,
    ensemble_size: int,
    device: torch.device,
    logger: logging.Logger,
    batch_size: int = 8
) -> np.ndarray:
    """Generate RFP forecasts for all test samples with efficient batching."""
    
    logger.info(f"Generating RFP forecasts for {len(test_data)} samples with batch_size={batch_size}")
    
    n_test = len(test_data)
    if n_test == 0:
        return np.array([])
    
    all_forecasts = []
    
    # Process in batches for memory efficiency
    for batch_start in tqdm(range(0, n_test, batch_size), desc="RFP forecast batches"):
        batch_end = min(batch_start + batch_size, n_test)
        batch_data = test_data[batch_start:batch_end]
        current_batch_size = len(batch_data)
        
        # Stack batch data
        batch_prev = torch.stack([prev for prev, _, _ in batch_data]).to(device)  # (B, C, H, W)
        batch_curr = torch.stack([curr for _, curr, _ in batch_data]).to(device)  # (B, V, H, W) 
        batch_time = torch.stack([time for _, _, time in batch_data]).to(device)  # (B,)
        
        V = batch_curr.shape[1]
        
        # Generate RFP perturbations for entire batch
        alpha_matrix = torch.tensor(alpha_values, device=device, dtype=torch.float32)
        alpha_matrix = alpha_matrix.unsqueeze(0).repeat(ensemble_size, 1)  # (E, n_variables)
        
        # Generate perturbations for the batch
        batch_perturbations = []
        for i in range(current_batch_size):
            perturb = generate_joint_rfp(
                reference_tensor=reference_tensor,
                alpha_matrix=alpha_matrix,
                batch_size=1,  # Each test sample is batch_size=1
                ensemble_size=ensemble_size,
                device=device,
                generator=torch.Generator(device=device),  # Match training: no seed for random perturbations
                eps_energy=1e-6
            )
            # perturb shape: (E, 1, E, V, H, W) -> use different alpha for each ensemble member
            perturb_diag = torch.stack([perturb[e, 0, e] for e in range(ensemble_size)])
            batch_perturbations.append(perturb_diag)
        
        batch_perturbations = torch.stack(batch_perturbations)  # (B, E, V, H, W)
        
        # Generate ensemble forecasts efficiently
        batch_forecasts = []
        
        # Process ensemble members in smaller groups to manage memory
        ensemble_batch_size = min(ensemble_size, 10)  # Process 10 ensemble members at once
        
        for ens_start in range(0, ensemble_size, ensemble_batch_size):
            ens_end = min(ens_start + ensemble_batch_size, ensemble_size)
            
            # Prepare inputs for this ensemble batch
            ens_inputs = []
            ens_times = []
            
            for b in range(current_batch_size):
                for e in range(ens_start, ens_end):
                    # Add perturbation to current fields
                    perturbed_curr = batch_curr[b] + batch_perturbations[b, e]  # (V, H, W)
                    
                    # Reconstruct model input
                    past_slice = batch_prev[b, V:2*V]
                    static_slice = batch_prev[b, -2:]
                    model_input = torch.cat([perturbed_curr, past_slice, static_slice], dim=0)  # (C, H, W)
                    
                    ens_inputs.append(model_input)
                    ens_times.append(batch_time[b])
            
            # Stack and run model on ensemble batch
            if ens_inputs:
                ens_input_tensor = torch.stack(ens_inputs)  # (B*E_sub, C, H, W)
                ens_time_tensor = torch.stack(ens_times)    # (B*E_sub,)
                
                with torch.no_grad():
                    ens_forecasts = model(ens_input_tensor, ens_time_tensor)  # (B*E_sub, V, H, W)
                
                # Reshape back to (B, E_sub, V, H, W)
                ens_forecasts = ens_forecasts.view(current_batch_size, ens_end - ens_start, V, 
                                                 ens_forecasts.shape[-2], ens_forecasts.shape[-1])
                batch_forecasts.append(ens_forecasts.cpu())
        
        # Concatenate ensemble forecasts: (B, E, V, H, W)
        if batch_forecasts:
            batch_result = torch.cat(batch_forecasts, dim=1)  # (B, E, V, H, W)
            all_forecasts.append(batch_result.numpy())
        
        # Memory management
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Stack all batch results: (n_test, ensemble_size, V, H, W)
    if all_forecasts:
        final_forecasts = np.concatenate(all_forecasts, axis=0)
        # Add singleton dimension: (n_test, ensemble_size, 1, V, H, W)
        final_forecasts = final_forecasts[:, :, np.newaxis, :, :, :]
    else:
        final_forecasts = np.array([])
    
    logger.info(f"Generated forecasts shape: {final_forecasts.shape}")
    return final_forecasts


def generate_rfp_forecasts_legacy(
    model: torch.nn.Module,
    test_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    alpha_values: np.ndarray,
    reference_tensor: torch.Tensor,
    ensemble_size: int,
    device: torch.device,
    logger: logging.Logger
) -> np.ndarray:
    """Generate RFP forecasts - original implementation for compatibility."""
    
    logger.info(f"Generating RFP forecasts for {len(test_data)} samples")
    
    forecasts = []
    
    for i, (prev_fields, curr_fields, time_norm) in enumerate(tqdm(test_data, desc="RFP forecasts")):
        batch_size = prev_fields.shape[0]
        V = curr_fields.shape[1]
        
        # Generate RFP perturbations - alpha_matrix needs shape (n_proposals, n_variables)
        alpha_matrix = torch.tensor(alpha_values, device=device, dtype=torch.float32).unsqueeze(0)  # (1, n_variables)
        # Repeat for ensemble_size proposals
        alpha_matrix = alpha_matrix.repeat(ensemble_size, 1)  # (ensemble_size, n_variables)
        
        perturb = generate_joint_rfp(
            reference_tensor=reference_tensor,
            alpha_matrix=alpha_matrix,
            batch_size=batch_size,
            ensemble_size=ensemble_size,
            device=device,
            generator=torch.Generator(device=device),  # Match training: no seed for random perturbations
            eps_energy=1e-6
        )
        
        # Generate ensemble forecasts
        ensemble_forecasts = []
        for e in range(ensemble_size):
            # FIXED: Add perturbation to INPUT current fields (not target fields)
            # This matches training approach in core/algorithm.py where perturbations are added to current_slice
            # perturb shape: (P, batch_size, ensemble_size, V, H, W)
            current_slice = prev_fields[:, :V]  # Extract current fields from input
            perturbed_current = current_slice + perturb[e, 0, e]  # Add perturbation to input current fields
            
            # Reconstruct input for model
            past_slice = prev_fields[:, V:2*V] 
            static_slice = prev_fields[:, -2:]
            
            # Use perturbed current fields as input
            model_input = torch.cat([perturbed_current, past_slice, static_slice], dim=1)
            
            with torch.no_grad():
                forecast = model(model_input, time_norm)
            
            ensemble_forecasts.append(forecast.cpu().numpy())
        
        forecasts.append(np.array(ensemble_forecasts))  # Shape: (ensemble_size, batch_size, V, H, W)
        
        # Memory management
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Stack all forecasts: (n_test, ensemble_size, 1, V, H, W)
    all_forecasts = np.array(forecasts)
    
    logger.info(f"Generated forecasts shape: {all_forecasts.shape}")
    return all_forecasts


def generate_persistence_forecasts(test_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> np.ndarray:
    """Generate persistence forecasts (use most recent observation) - vectorized."""
    
    # Extract all prev_fields and stack them
    all_prev_fields = [prev_fields for prev_fields, _, _ in test_data]
    
    if not all_prev_fields:
        return np.array([])
    
    # Get dimensions from first sample
    sample_prev, sample_curr, _ = test_data[0]
    V = sample_curr.shape[1]  # Number of variables in current fields (5)
    
    # Stack all previous fields: (n_test, C, H, W) where C=12
    prev_batch = torch.stack(all_prev_fields)
    
    # Handle case where prev_batch has extra batch dimension
    if prev_batch.ndim == 5:  # (n_test, 1, C, H, W)
        prev_batch = prev_batch.squeeze(1)  # (n_test, C, H, W)
    
    # Extract most recent observations (current slice)
    current_observations = prev_batch[:, :V]  # (n_test, V, H, W)
    
    # Debug shapes (uncomment for debugging)
    # logger = logging.getLogger(__name__)
    # logger.info(f"Persistence: prev_batch {prev_batch.shape}, V={V}, current_obs {current_observations.shape}")
    
    # Convert to numpy and add ensemble dimension: (n_test, 1, V, H, W)
    forecasts = current_observations.cpu().numpy()  # (n_test, V, H, W)
    return forecasts[:, np.newaxis, :, :, :]  # (n_test, 1, V, H, W)


def compute_climatology_stats(full_dataset, variable_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:  # noqa: ARG001
    """Compute climatological mean and std from full dataset."""
    logger = logging.getLogger(__name__)
    logger.info("Computing climatological statistics from full dataset")
    
    all_observations = []
    
    # Sample from full dataset to estimate climatology
    n_samples = min(1000, len(full_dataset))
    indices = np.random.choice(len(full_dataset), n_samples, replace=False)
    
    for idx in tqdm(indices, desc="Computing climatology"):
        try:
            _, current_fields, _ = full_dataset[idx]
            all_observations.append(current_fields.cpu().numpy())
        except Exception as e:
            logger.warning(f"Failed to load sample {idx}: {e}")
            continue
    
    if len(all_observations) == 0:
        raise ValueError("Could not load any observations for climatology")
    
    # Stack and compute statistics
    obs_array = np.array(all_observations)  # Shape: (n_samples, V, H, W)
    clim_mean = obs_array.mean(axis=0)  # Shape: (V, H, W)
    clim_std = obs_array.std(axis=0)    # Shape: (V, H, W)
    
    logger.info(f"Climatology computed from {len(all_observations)} samples")
    return clim_mean, clim_std


def generate_climatological_forecasts(
    test_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], 
    clim_mean: np.ndarray, 
    clim_std: np.ndarray,
    ensemble_size: int
) -> np.ndarray:
    """Generate climatological forecasts from normal distribution - vectorized."""
    n_test = len(test_data)
    
    if n_test == 0:
        return np.array([])
    
    # Generate all samples at once: (n_test, ensemble_size, V, H, W)
    shape = (n_test, ensemble_size) + clim_mean.shape
    forecasts = np.random.normal(
        loc=clim_mean[np.newaxis, np.newaxis, :],  # Broadcast over test and ensemble dims
        scale=clim_std[np.newaxis, np.newaxis, :],
        size=shape
    )
    
    return forecasts


def fit_ar1_models(full_dataset, variable_names: List[str], n_samples: int = 500) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:  # noqa: ARG001
    """Fit AR(1) models for each grid point and variable using OLS."""
    logger = logging.getLogger(__name__)
    logger.info(f"Fitting AR(1) models using {n_samples} samples")
    
    # Collect sequential pairs for AR(1) fitting
    observations = []
    indices = np.random.choice(len(full_dataset) - 1, n_samples, replace=False)
    
    for idx in tqdm(indices, desc="Collecting AR(1) data"):
        try:
            _, obs_t, _ = full_dataset[idx]
            _, obs_t1, _ = full_dataset[idx + 1]
            observations.append((obs_t.cpu().numpy(), obs_t1.cpu().numpy()))
        except Exception as e:
            logger.warning(f"Failed to load AR(1) pair {idx}: {e}")
            continue
    
    if len(observations) == 0:
        raise ValueError("Could not collect AR(1) training data")
    
    # Stack observations
    obs_t_array = np.array([obs[0] for obs in observations])    # Shape: (n_samples, V, H, W)
    obs_t1_array = np.array([obs[1] for obs in observations])   # Shape: (n_samples, V, H, W)
    
    V, H, W = obs_t_array.shape[1:]
    
    # Fit AR(1): X_{t+1} = φ * X_t + ε, ε ~ N(0, σ²)
    # OLS: φ = Σ(X_t * X_{t+1}) / Σ(X_t²)
    
    ar_coeffs = np.zeros((V, H, W))
    residual_vars = np.zeros((V, H, W))
    intercepts = np.zeros((V, H, W))
    
    logger.info("Fitting AR(1) parameters per grid point...")
    
    for v in range(V):
        for h in range(H):
            for w in range(W):
                x_t = obs_t_array[:, v, h, w]
                x_t1 = obs_t1_array[:, v, h, w]
                
                # OLS estimation with intercept: x_{t+1} = α + φ * x_t + ε
                X = np.column_stack([np.ones(len(x_t)), x_t])
                try:
                    params = np.linalg.lstsq(X, x_t1, rcond=None)[0]
                    intercepts[v, h, w] = params[0]
                    ar_coeffs[v, h, w] = params[1]
                    
                    # Compute residual variance
                    predicted = intercepts[v, h, w] + ar_coeffs[v, h, w] * x_t
                    residuals = x_t1 - predicted
                    residual_vars[v, h, w] = np.var(residuals)
                except Exception:
                    # Fallback to simple moments
                    intercepts[v, h, w] = 0
                    ar_coeffs[v, h, w] = np.corrcoef(x_t, x_t1)[0, 1] if len(x_t) > 1 else 0
                    residual_vars[v, h, w] = np.var(x_t1) * (1 - ar_coeffs[v, h, w]**2)
    
    logger.info("AR(1) model fitting completed")
    return ar_coeffs, intercepts, residual_vars


def generate_ar1_forecasts(
    test_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ar_coeffs: np.ndarray,
    intercepts: np.ndarray, 
    residual_vars: np.ndarray,
    ensemble_size: int
) -> np.ndarray:
    """Generate AR(1) forecasts - vectorized."""
    
    if not test_data:
        return np.array([])
    
    # Extract all current observations
    all_prev_fields = [prev_fields for prev_fields, _, _ in test_data]
    prev_batch = torch.stack(all_prev_fields)  # (n_test, C, H, W) or (n_test, 1, C, H, W)
    
    # Handle case where prev_batch has extra batch dimension
    if prev_batch.ndim == 5:  # (n_test, 1, C, H, W)
        prev_batch = prev_batch.squeeze(1)  # (n_test, C, H, W)
    
    # Get V from current fields (more reliable than assuming channel structure)
    _, sample_curr, _ = test_data[0]
    V = sample_curr.shape[1]  # Number of variables (5)
    current_obs_batch = prev_batch[:, :V].cpu().numpy()  # (n_test, V, H, W)
    
    n_test = len(test_data)
    
    # AR(1) forecast: X_{t+1} = α + φ * X_t + ε - vectorized over all samples
    # Broadcast operations: (n_test, V, H, W)
    ar1_forecasts = intercepts[np.newaxis, :, :, :] + ar_coeffs[np.newaxis, :, :, :] * current_obs_batch
    
    # Generate noise for all samples and ensemble members: (n_test, ensemble_size, V, H, W)
    noise_shape = (n_test, ensemble_size) + residual_vars.shape
    noise = np.random.normal(0, np.sqrt(residual_vars[np.newaxis, np.newaxis, :]), size=noise_shape)
    
    # Add ensemble dimension to AR forecasts and add noise
    ar1_forecasts = ar1_forecasts[:, np.newaxis, :, :, :] + noise  # (n_test, ensemble_size, V, H, W)
    
    return ar1_forecasts


def generate_gaussian_noise_forecasts(
    model: torch.nn.Module,
    test_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    empirical_stds: np.ndarray,
    ensemble_size: int,
    device: torch.device,
    batch_size: int = 16
) -> np.ndarray:
    """Generate deterministic model + Gaussian noise forecasts - batched."""
    
    if not test_data:
        return np.array([])
    
    n_test = len(test_data)
    all_forecasts = []
    
    # Process in batches for efficiency
    for batch_start in tqdm(range(0, n_test, batch_size), desc="Deterministic forecasts"):
        batch_end = min(batch_start + batch_size, n_test)
        batch_data = test_data[batch_start:batch_end]
        current_batch_size = len(batch_data)
        
        # Stack batch inputs
        batch_prev = torch.stack([prev for prev, _, _ in batch_data]).to(device)  # (B, C, H, W) or (B, 1, C, H, W)
        batch_time = torch.stack([time for _, _, time in batch_data]).to(device)  # (B,)
        
        # Handle case where batch_prev has extra batch dimension  
        if batch_prev.ndim == 5:  # (B, 1, C, H, W)
            batch_prev = batch_prev.squeeze(1)  # (B, C, H, W)
        
        # Get deterministic predictions for entire batch
        with torch.no_grad():
            batch_det_forecasts = model(batch_prev, batch_time).cpu().numpy()  # (B, V, H, W)
        
        # Generate noise for ensemble: (B, ensemble_size, V, H, W)
        noise_shape = (current_batch_size, ensemble_size) + empirical_stds.shape
        noise = np.random.normal(0, empirical_stds[np.newaxis, np.newaxis, :], size=noise_shape)
        
        # Add noise to deterministic forecasts
        batch_ensemble = batch_det_forecasts[:, np.newaxis, :, :, :] + noise  # (B, E, V, H, W)
        
        all_forecasts.append(batch_ensemble)
        
        # Memory management
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Concatenate all batches: (n_test, ensemble_size, V, H, W)
    return np.concatenate(all_forecasts, axis=0) if all_forecasts else np.array([])


def compute_empirical_stds(full_dataset, model: torch.nn.Module, device: torch.device, n_samples: int = 200) -> np.ndarray:
    """Compute empirical standard deviations from model residuals."""
    logger = logging.getLogger(__name__)
    logger.info(f"Computing empirical stds from {n_samples} samples")
    
    residuals = []
    indices = np.random.choice(len(full_dataset), n_samples, replace=False)
    
    model.eval()
    for idx in tqdm(indices, desc="Computing residuals"):
        try:
            prev_fields, curr_fields, time_norm = full_dataset[idx]
            prev_fields = prev_fields.unsqueeze(0).to(device)
            curr_fields = curr_fields.unsqueeze(0).to(device)
            time_norm = torch.tensor([time_norm], dtype=torch.float32, device=device)
            
            with torch.no_grad():
                prediction = model(prev_fields, time_norm)
                residual = (prediction - curr_fields).cpu().numpy().squeeze(0)
                residuals.append(residual)
        except Exception as e:
            logger.warning(f"Failed to process sample {idx}: {e}")
            continue
    
    if len(residuals) == 0:
        raise ValueError("Could not compute any residuals")
    
    residual_array = np.array(residuals)  # Shape: (n_samples, V, H, W)
    empirical_stds = np.std(residual_array, axis=0)  # Shape: (V, H, W)
    
    logger.info("Empirical standard deviations computed")
    return empirical_stds


def compute_metrics(forecasts: np.ndarray, targets: np.ndarray, variable_names: List[str]) -> Dict[str, float]:  # noqa: ARG001
    """Compute performance metrics for forecasts."""
    
    # Debug shapes (uncomment for debugging)
    # logger = logging.getLogger(__name__)
    # logger.info(f"Computing metrics: forecasts shape {forecasts.shape}, targets shape {targets.shape}")
    
    if forecasts.ndim == 4:  # Deterministic case: (n_test, V, H, W)
        # Convert to ensemble format by repeating
        forecasts = forecasts[:, np.newaxis, :]  # (n_test, 1, V, H, W)
    
    # Ensure targets match forecast dimensions - targets should be (n_test, V, H, W)
    if targets.ndim == 4 and targets.shape[1] > forecasts.shape[2]:  # targets: (n_test, C, H, W)
        # Extract only the variables that match forecasts
        V_forecast = forecasts.shape[2] if forecasts.ndim >= 3 else forecasts.shape[-3]
        targets = targets[:, :V_forecast]  # Take first V variables
    
    if targets.ndim == 3:  # (n_test, V, H, W) -> add ensemble dim
        targets = targets[:, np.newaxis, :]  # (n_test, 1, V, H, W)
    
    # Handle the forecast shape - remove singleton batch dimension if present
    if forecasts.ndim == 6:  # (n_test, ensemble_size, 1, V, H, W) 
        forecasts = forecasts.squeeze(2)  # (n_test, ensemble_size, V, H, W)
    
    # Verify shapes are compatible
    if forecasts.shape[0] != targets.shape[0]:
        raise ValueError(f"Sample count mismatch: forecasts {forecasts.shape[0]} vs targets {targets.shape[0]}")
    if forecasts.ndim >= 3 and targets.ndim >= 3 and forecasts.shape[2] != targets.shape[2]:
        raise ValueError(f"Variable count mismatch: forecasts {forecasts.shape[2]} vs targets {targets.shape[2]}")
    
    n_test = forecasts.shape[0]
    
    # Convert to torch tensors for metric computation
    forecast_tensor = torch.tensor(forecasts, dtype=torch.float32)  # (n_test, ensemble_size, V, H, W)
    target_tensor = torch.tensor(targets.squeeze(1), dtype=torch.float32)  # (n_test, V, H, W)
    
    metrics = {}
    
    # Reshape for scoring functions: (ensemble_size, n_test, V, H, W)
    forecast_reshaped = forecast_tensor.transpose(0, 1)
    
    # CRPS - average across all dimensions
    try:
        crps_total = 0
        for t in range(n_test):
            crps_t = compute_crps_for_proposal(
                forecast_reshaped[:, t:t+1], target_tensor[t:t+1], forecasts.shape[2]
            )
            crps_total += crps_t
        metrics["crps"] = crps_total / n_test
    except Exception as e:
        logging.warning(f"CRPS computation failed: {e}")
        metrics["crps"] = float('nan')
    
    # Energy Score - average across time
    try:
        energy_total = 0
        for t in range(n_test):
            energy_t = compute_energy_score_for_proposal(
                forecast_reshaped[:, t:t+1], target_tensor[t:t+1], forecasts.shape[2]
            )
            energy_total += energy_t
        metrics["energy_score"] = energy_total / n_test
    except Exception as e:
        logging.warning(f"Energy score computation failed: {e}")
        metrics["energy_score"] = float('nan')
    
    # MAE (ensemble mean vs truth)
    ensemble_mean = forecasts.mean(axis=1)  # (n_test, V, H, W)
    mae = np.abs(ensemble_mean - targets.squeeze(1)).mean()
    metrics["mae"] = float(mae)
    
    # Spread (ensemble standard deviation)
    ensemble_std = forecasts.std(axis=1)  # (n_test, V, H, W)
    spread = ensemble_std.mean()
    metrics["spread"] = float(spread)
    
    return metrics


def process_single_result(result_path: Path, config_path: str, n_test_samples: int, logger: logging.Logger, force_regenerate: bool = False) -> bool:
    """Process a single result directory for performance evaluation."""
    
    logger.info("Processing: %s", result_path.name)
    
    # Check if statistics already exist
    stats_file = result_path / "statistics.json"
    if not force_regenerate and stats_file.exists():
        logger.info("  Statistics already exist, skipping")
        return True
    
    # Load posterior data
    samples, _, data_source = load_posterior_data(result_path)
    if samples is None:
        logger.warning("  No posterior data found")
        return False
    
    # Determine appropriate burn-in period based on algorithm type
    burn_in = get_burn_in_period(result_path)
    logger.info(f"  Loaded posterior data from {data_source} (burn-in: {burn_in})")
    
    try:
        # Load configuration and model
        config = Config(config_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and test data
        loader, model, _, _, _ = load_model_and_test_data(config, device, 42)
        
        # Convert loader to list for easier indexing
        test_data = []
        for prev_fields, curr_fields, time_norm in loader:
            test_data.append((
                prev_fields.to(device),
                curr_fields.to(device),
                time_norm.to(device)
            ))
        
        # Limit to manageable size with RANDOM sampling to match ABC training
        max_test_samples = min(n_test_samples, len(test_data))
        # Use random sampling like ABC training does (resample_temporal=True)
        np.random.seed(42)  # Reproducible random sampling
        subset_indices = np.random.choice(len(test_data), size=max_test_samples, replace=False)
        test_subset = [test_data[i] for i in sorted(subset_indices)]
        n_test = len(test_subset)
        
        # Extract ground truth targets
        targets = np.array([curr_fields.cpu().numpy() for _, curr_fields, _ in test_subset])
        
        # Load reference tensor for RFP
        ref_path = config.data_directory / "z500_t850_t2m_u10_v10_standardized.npy"
        if ref_path.exists():
            reference_tensor = torch.from_numpy(np.load(ref_path, mmap_mode="r")).to(device)
        else:
            logger.warning("Reference tensor not found, using subset of test data")
            reference_tensor = torch.cat([curr_fields for _, curr_fields, _ in test_data[:1000]], dim=0)
        
        ensemble_size = config.ensemble_size
        variable_names = config.variable_names
        
        logger.info(f"  Running benchmarks on {n_test} test samples")
        
        # Initialize results dictionary
        all_results = {}
        
        # 1. RFP with posterior mean  
        logger.info("  Computing RFP (posterior mean)...")
        alpha_mean = get_posterior_alpha(samples, mode="mean", burn_in=burn_in)
        rfp_mean_forecasts = generate_rfp_forecasts_legacy(
            model, test_subset, alpha_mean, reference_tensor, 
            ensemble_size, device, logger
        )
        all_results["rfp_posterior_mean"] = compute_metrics(
            rfp_mean_forecasts, targets, variable_names
        )
        
        # 2. RFP with posterior sample
        logger.info("  Computing RFP (posterior sample)...")
        alpha_sample = get_posterior_alpha(samples, mode="sample", burn_in=burn_in)
        rfp_sample_forecasts = generate_rfp_forecasts_legacy(
            model, test_subset, alpha_sample, reference_tensor,
            ensemble_size, device, logger
        )
        all_results["rfp_posterior_sample"] = compute_metrics(
            rfp_sample_forecasts, targets, variable_names
        )
        
        # 3. RFP with uncalibrated parameters (baseline 1) - all alphas = 1.0
        logger.info("  Computing RFP (uncalibrated baseline - alpha=1.0)...")
        alpha_ones = get_uncalibrated_rfp_alpha_ones(len(variable_names))
        rfp_ones_forecasts = generate_rfp_forecasts_legacy(
            model, test_subset, alpha_ones, reference_tensor,
            ensemble_size, device, logger
        )
        all_results["rfp_uncalibrated_ones"] = compute_metrics(
            rfp_ones_forecasts, targets, variable_names
        )
        
        # 4. RFP with uncalibrated parameters (baseline 2) - drawn from prior
        logger.info("  Computing RFP (uncalibrated baseline - prior samples)...")
        alpha_prior = get_uncalibrated_rfp_alpha_prior(len(variable_names))
        rfp_prior_forecasts = generate_rfp_forecasts_legacy(
            model, test_subset, alpha_prior, reference_tensor,
            ensemble_size, device, logger
        )
        all_results["rfp_uncalibrated_prior"] = compute_metrics(
            rfp_prior_forecasts, targets, variable_names
        )
        
        # 5. Persistence forecasting
        logger.info("  Computing persistence forecasts...")
        persistence_forecasts = generate_persistence_forecasts(test_subset)
        all_results["persistence"] = compute_metrics(
            persistence_forecasts, targets, variable_names
        )
        
        # 6. Climatological forecasting
        logger.info("  Computing climatological forecasts...")
        full_dataset = loader.dataset.dataset if hasattr(loader.dataset, 'dataset') else loader.dataset
        clim_mean, clim_std = compute_climatology_stats(full_dataset, variable_names)
        clim_forecasts = generate_climatological_forecasts(
            test_subset, clim_mean, clim_std, ensemble_size
        )
        all_results["climatology"] = compute_metrics(
            clim_forecasts, targets, variable_names
        )
        
        # 7. AR(1) forecasting
        logger.info("  Computing AR(1) forecasts...")
        ar_coeffs, intercepts, residual_vars = fit_ar1_models(full_dataset, variable_names)
        ar1_forecasts = generate_ar1_forecasts(
            test_subset, ar_coeffs, intercepts, residual_vars, ensemble_size
        )
        all_results["ar1"] = compute_metrics(
            ar1_forecasts, targets, variable_names
        )
        
        # 8. Deterministic + Gaussian noise
        logger.info("  Computing deterministic + Gaussian noise...")
        empirical_stds = compute_empirical_stds(full_dataset, model, device)
        gaussian_forecasts = generate_gaussian_noise_forecasts(
            model, test_subset, empirical_stds, ensemble_size, device, batch_size=16
        )
        all_results["deterministic_gaussian"] = compute_metrics(
            gaussian_forecasts, targets, variable_names
        )
        
        # Save results
        with open(stats_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        logger.info(f"  ✓ Saved performance statistics to {stats_file}")
        return True
        
    except Exception as e:
        logger.error(f"  ✗ Failed to process {result_path.name}: {e}")
        return False


def process_all_results(results_directory: Path, config_path: str, n_test_samples: int, force_regenerate: bool = False):
    """Process all result directories for performance evaluation."""
    logger = logging.getLogger(__name__)
    
    if not results_directory.exists():
        raise FileNotFoundError(f"Results directory not found: {results_directory}")
    
    # Find all potential result directories
    result_dirs = [d for d in results_directory.iterdir() 
                   if d.is_dir() and not d.name.startswith('.')]
    
    if not result_dirs:
        logger.info(f"No result directories found in: {results_directory}")
        return
    
    logger.info(f"Scanning {len(result_dirs)} directories in: {results_directory}")
    
    processed = 0
    skipped = 0
    failed = 0
    
    for result_path in sorted(result_dirs):
        # Check if data exists
        samples, _, data_source = load_posterior_data(result_path)
        if samples is None:
            continue  # Skip directories without data
        
        # Check if statistics already exist
        stats_file = result_path / "statistics.json"
        if not force_regenerate and stats_file.exists():
            skipped += 1
            continue
        
        logger.info("Processing: %s (%s)", result_path.name, data_source)
        
        if process_single_result(result_path, config_path, n_test_samples, logger, force_regenerate):
            processed += 1
        else:
            failed += 1
    
    logger.info("\nSummary:")
    logger.info("  Processed: %d", processed)
    logger.info("  Skipped (already have statistics): %d", skipped)
    logger.info("  Failed: %d", failed)


def main():
    parser = argparse.ArgumentParser(description="Comparative performance evaluation")
    parser.add_argument("result_directory", type=str, nargs='?', default="./results",
                       help="Path to single result directory OR results root directory (default: ./results)")
    parser.add_argument("--config", type=str, default="config.json",
                       help="Path to configuration file (default: config.json)")
    parser.add_argument("--n-samples", type=int, default=256,
                       help="Number of test samples to evaluate (default: 256)")
    parser.add_argument("--all", action="store_true", 
                       help="Process all result directories in the specified path")
    parser.add_argument("--force", action="store_true",
                       help="Regenerate statistics even if they already exist")
    
    args = parser.parse_args()
    
    logger = setup_logging()
    result_path = Path(args.result_directory)
    
    if args.all:
        # Batch processing mode
        process_all_results(result_path, args.config, args.n_samples, force_regenerate=args.force)
        return
    
    # Single directory mode
    if not result_path.exists():
        raise FileNotFoundError(f"Directory not found: {result_path}")

    # Check if this looks like a results root directory
    if result_path.is_dir():
        subdirs_with_data = []
        for subdir in result_path.iterdir():
            if subdir.is_dir():
                samples, _, _ = load_posterior_data(subdir)
                if samples is not None:
                    subdirs_with_data.append(subdir.name)
        
        if len(subdirs_with_data) > 1:
            logger.info("Found %d result directories in %s:", len(subdirs_with_data), result_path)
            for dirname in sorted(subdirs_with_data):
                logger.info("  - %s", dirname)
            logger.info("\nTo process all directories, use: --all")
            logger.info("To process a specific directory, provide its full path.")
            return

    # Process single directory
    success = process_single_result(result_path, args.config, args.n_samples, logger, args.force)
    if success:
        logger.info("✓ Performance evaluation completed successfully")
    else:
        logger.error("✗ Performance evaluation failed")


if __name__ == "__main__":
    main()