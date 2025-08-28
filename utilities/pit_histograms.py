#!/usr/bin/env python3
"""
PIT Histogram Analysis Script

Computes Probability Integral Transform (PIT) histograms for each forecasting method
from comparative_performance.py. PIT values are computed using empirical CDFs from
ensemble forecasts and aggregated across all samples, grid points, and variables.
"""

import argparse
import json
import logging
import sys
import warnings
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import scipy.stats
from typing import Dict, List, Tuple, Optional

# Suppress warnings
warnings.filterwarnings('ignore')

# Add current directory to path for imports
sys.path.append('.')

# Import core modules
from core.config import Config
from core.io_utils import load_model_and_test_data
from core.algorithm import generate_joint_rfp

# Constants
PIT_HISTOGRAM_BINS = 10
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
    """Extract alpha values from posterior samples with burn-in period."""
    # Apply burn-in period
    samples_burned = samples[burn_in:] if samples.shape[0] > burn_in else samples
    
    if mode == "mean":
        return samples_burned.mean(axis=0).squeeze()  # Shape: (n_variables,)
    elif mode == "sample":
        # Random sample from posterior (after burn-in)
        random_step = np.random.randint(0, samples_burned.shape[0])
        return samples_burned[random_step].squeeze()  # Shape: (n_variables,)
    else:
        raise ValueError(f"Mode must be 'mean' or 'sample', got {mode}")


def compute_pit_values_empirical(forecasts: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """
    Compute PIT values using empirical CDF from ensemble forecasts.
    
    Args:
        forecasts: Ensemble forecasts shape (n_test, ensemble_size, V, H, W) or (n_test, V, H, W) for deterministic
        targets: Ground truth targets shape (n_test, V, H, W)
        
    Returns:
        PIT values flattened across all dimensions
    """
    
    if forecasts.ndim == 4:  # Deterministic case: (n_test, V, H, W)
        # For deterministic forecasts, PIT is 0 if target < forecast, 1 if target >= forecast
        pit_values = (targets >= forecasts).astype(float)
        return pit_values.flatten()
    
    # Ensemble case: (n_test, ensemble_size, V, H, W)
    n_test, ensemble_size, V, H, W = forecasts.shape
    
    # Flatten spatial and variable dimensions for easier processing
    forecasts_flat = forecasts.reshape(n_test, ensemble_size, -1)  # (n_test, ensemble_size, V*H*W)
    targets_flat = targets.reshape(n_test, -1)  # (n_test, V*H*W)
    
    pit_values = []
    
    for t in range(n_test):
        for vhw in range(V * H * W):
            # Get ensemble forecast and target for this time and location
            ensemble = forecasts_flat[t, :, vhw]  # (ensemble_size,)
            observation = targets_flat[t, vhw]    # scalar
            
            # Compute empirical CDF at the observation
            # PIT = (# ensemble members <= observation) / ensemble_size
            pit = np.mean(ensemble <= observation)
            pit_values.append(pit)
    
    return np.array(pit_values)


def generate_rfp_forecasts_legacy(
    model: torch.nn.Module,
    test_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    alpha_values: np.ndarray,
    reference_tensor: torch.Tensor,
    ensemble_size: int,
    device: torch.device,
    logger: logging.Logger
) -> np.ndarray:
    """Generate RFP forecasts - adapted from comparative_performance.py."""
    
    logger.info(f"Generating RFP forecasts for {len(test_data)} samples")
    
    forecasts = []
    
    for i, (prev_fields, curr_fields, time_norm) in enumerate(tqdm(test_data, desc="RFP forecasts")):
        batch_size = prev_fields.shape[0]
        V = curr_fields.shape[1]
        
        # Generate RFP perturbations
        alpha_matrix = torch.tensor(alpha_values, device=device, dtype=torch.float32).unsqueeze(0)
        alpha_matrix = alpha_matrix.repeat(ensemble_size, 1)
        
        perturb = generate_joint_rfp(
            reference_tensor=reference_tensor,
            alpha_matrix=alpha_matrix,
            batch_size=batch_size,
            ensemble_size=ensemble_size,
            device=device,
            generator=torch.Generator(device=device).manual_seed(42 + i),
            eps_energy=1e-6
        )
        
        # Generate ensemble forecasts
        ensemble_forecasts = []
        for e in range(ensemble_size):
            # FIXED: Add perturbation to INPUT current fields (not target fields)
            # This matches training approach in core/algorithm.py where perturbations are added to current_slice
            current_slice = prev_fields[:, :V]  # Extract current fields from input
            perturbed_current = current_slice + perturb[0, 0, e]  # Add perturbation to input current fields
            
            # Reconstruct input for model
            past_slice = prev_fields[:, V:2*V] 
            static_slice = prev_fields[:, -2:]
            model_input = torch.cat([perturbed_current, past_slice, static_slice], dim=1)
            
            with torch.no_grad():
                forecast = model(model_input, time_norm)
            
            ensemble_forecasts.append(forecast.cpu().numpy())
        
        forecasts.append(np.array(ensemble_forecasts))
        
        # Memory management
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Stack all forecasts: (n_test, ensemble_size, 1, V, H, W)
    all_forecasts = np.array(forecasts)
    logger.info(f"Generated forecasts shape: {all_forecasts.shape}")
    return all_forecasts


def generate_persistence_forecasts(test_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> np.ndarray:
    """Generate persistence forecasts (use most recent observation)."""
    
    all_prev_fields = [prev_fields for prev_fields, _, _ in test_data]
    
    if not all_prev_fields:
        return np.array([])
    
    # Get dimensions from first sample
    sample_prev, sample_curr, _ = test_data[0]
    V = sample_curr.shape[1]
    
    # Stack all previous fields
    prev_batch = torch.stack(all_prev_fields)
    
    # Handle case where prev_batch has extra batch dimension
    if prev_batch.ndim == 5:
        prev_batch = prev_batch.squeeze(1)
    
    # Extract most recent observations (current slice)
    current_observations = prev_batch[:, :V]
    
    # Convert to numpy: (n_test, V, H, W)
    forecasts = current_observations.cpu().numpy()
    return forecasts


def compute_climatology_stats(full_dataset, variable_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
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
    obs_array = np.array(all_observations)
    clim_mean = obs_array.mean(axis=0)
    clim_std = obs_array.std(axis=0)
    
    logger.info(f"Climatology computed from {len(all_observations)} samples")
    return clim_mean, clim_std


def generate_climatological_forecasts(
    test_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], 
    clim_mean: np.ndarray, 
    clim_std: np.ndarray,
    ensemble_size: int
) -> np.ndarray:
    """Generate climatological forecasts from normal distribution."""
    n_test = len(test_data)
    
    if n_test == 0:
        return np.array([])
    
    # Generate all samples at once: (n_test, ensemble_size, V, H, W)
    shape = (n_test, ensemble_size) + clim_mean.shape
    forecasts = np.random.normal(
        loc=clim_mean[np.newaxis, np.newaxis, :],
        scale=clim_std[np.newaxis, np.newaxis, :],
        size=shape
    )
    
    return forecasts


def fit_ar1_models(full_dataset, variable_names: List[str], n_samples: int = 500) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    obs_t_array = np.array([obs[0] for obs in observations])
    obs_t1_array = np.array([obs[1] for obs in observations])
    
    V, H, W = obs_t_array.shape[1:]
    
    # Fit AR(1): X_{t+1} = φ * X_t + ε, ε ~ N(0, σ²)
    ar_coeffs = np.zeros((V, H, W))
    residual_vars = np.zeros((V, H, W))
    intercepts = np.zeros((V, H, W))
    
    logger.info("Fitting AR(1) parameters per grid point...")
    
    for v in range(V):
        for h in range(H):
            for w in range(W):
                x_t = obs_t_array[:, v, h, w]
                x_t1 = obs_t1_array[:, v, h, w]
                
                # OLS estimation with intercept
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
    """Generate AR(1) forecasts."""
    
    if not test_data:
        return np.array([])
    
    # Extract all current observations
    all_prev_fields = [prev_fields for prev_fields, _, _ in test_data]
    prev_batch = torch.stack(all_prev_fields)
    
    # Handle case where prev_batch has extra batch dimension
    if prev_batch.ndim == 5:
        prev_batch = prev_batch.squeeze(1)
    
    # Get V from current fields
    _, sample_curr, _ = test_data[0]
    V = sample_curr.shape[1]
    current_obs_batch = prev_batch[:, :V].cpu().numpy()
    
    n_test = len(test_data)
    
    # AR(1) forecast: X_{t+1} = α + φ * X_t + ε
    ar1_forecasts = intercepts[np.newaxis, :, :, :] + ar_coeffs[np.newaxis, :, :, :] * current_obs_batch
    
    # Generate noise for all samples and ensemble members
    noise_shape = (n_test, ensemble_size) + residual_vars.shape
    noise = np.random.normal(0, np.sqrt(residual_vars[np.newaxis, np.newaxis, :]), size=noise_shape)
    
    # Add ensemble dimension to AR forecasts and add noise
    ar1_forecasts = ar1_forecasts[:, np.newaxis, :, :, :] + noise
    
    return ar1_forecasts


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
    
    residual_array = np.array(residuals)
    empirical_stds = np.std(residual_array, axis=0)
    
    logger.info("Empirical standard deviations computed")
    return empirical_stds


def generate_gaussian_noise_forecasts(
    model: torch.nn.Module,
    test_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    empirical_stds: np.ndarray,
    ensemble_size: int,
    device: torch.device,
    batch_size: int = 16
) -> np.ndarray:
    """Generate deterministic model + Gaussian noise forecasts."""
    
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
        batch_prev = torch.stack([prev for prev, _, _ in batch_data]).to(device)
        batch_time = torch.stack([time for _, _, time in batch_data]).to(device)
        
        # Handle case where batch_prev has extra batch dimension  
        if batch_prev.ndim == 5:
            batch_prev = batch_prev.squeeze(1)
        
        # Get deterministic predictions for entire batch
        with torch.no_grad():
            batch_det_forecasts = model(batch_prev, batch_time).cpu().numpy()
        
        # Generate noise for ensemble
        noise_shape = (current_batch_size, ensemble_size) + empirical_stds.shape
        noise = np.random.normal(0, empirical_stds[np.newaxis, np.newaxis, :], size=noise_shape)
        
        # Add noise to deterministic forecasts
        batch_ensemble = batch_det_forecasts[:, np.newaxis, :, :, :] + noise
        
        all_forecasts.append(batch_ensemble)
        
        # Memory management
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return np.concatenate(all_forecasts, axis=0) if all_forecasts else np.array([])


def create_pit_histogram_plot(pit_data: Dict[str, np.ndarray], output_path: Path, n_bins: int = PIT_HISTOGRAM_BINS):
    """Create and save PIT histogram plots."""
    
    # Set up the plot - now need 3x3 to accommodate 8 methods
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()
    
    # Method names and colors
    method_names = {
        'rfp_posterior_mean': 'RFP (Posterior Mean)',
        'rfp_posterior_sample': 'RFP (Posterior Sample)',
        'rfp_uncalibrated_ones': 'RFP (α=1.0)',
        'rfp_uncalibrated_prior': 'RFP (Prior)',
        'persistence': 'Persistence',
        'climatology': 'Climatology',
        'ar1': 'AR(1)',
        'deterministic_gaussian': 'Deterministic + Gaussian'
    }
    
    colors = ['blue', 'red', 'darkgreen', 'green', 'orange', 'purple', 'brown', 'gray']
    
    # Plot histogram for each method
    for i, (method_key, method_name) in enumerate(method_names.items()):
        if method_key in pit_data and len(pit_data[method_key]) > 0:
            ax = axes[i]
            
            # Create histogram
            counts, bin_edges = np.histogram(pit_data[method_key], bins=n_bins, range=(0, 1))
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Plot bars
            ax.bar(bin_centers, counts, width=1.0/n_bins, alpha=0.7, 
                  color=colors[i], edgecolor='black', linewidth=0.5)
            
            # Add uniform reference line
            expected_count = len(pit_data[method_key]) / n_bins
            ax.axhline(y=expected_count, color='red', linestyle='--', alpha=0.8, 
                      label=f'Uniform (n={expected_count:.0f})')
            
            # Formatting
            ax.set_title(method_name, fontsize=12, fontweight='bold')
            ax.set_xlabel('PIT Value', fontsize=10)
            ax.set_ylabel('Count', fontsize=10)
            ax.set_xlim(0, 1)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            
            # Add statistics
            pit_values = pit_data[method_key]
            n_samples = len(pit_values)
            mean_pit = np.mean(pit_values)
            ax.text(0.02, 0.98, f'n={n_samples:,}\nmean={mean_pit:.3f}', 
                   transform=ax.transAxes, fontsize=8, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            # No data for this method
            axes[i].text(0.5, 0.5, f'{method_names[method_key]}\nNo Data', 
                        ha='center', va='center', transform=axes[i].transAxes, fontsize=12)
            axes[i].set_xlim(0, 1)
    
    # Hide unused subplots (we have 7 methods in a 3x3 grid, so hide the last 2)
    for j in range(len(method_names), len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    
    # Save as PNG and PDF
    png_path = output_path.with_suffix('.png')
    pdf_path = output_path.with_suffix('.pdf')
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    
    return png_path, pdf_path


def process_single_result(result_path: Path, config_path: str, n_test_samples: int, logger: logging.Logger, force_regenerate: bool = False) -> bool:
    """Process a single result directory for PIT histogram analysis."""
    
    logger.info("Processing: %s", result_path.name)
    
    # Check if PIT data already exists
    pit_png = result_path / "pit_histograms.png"
    pit_pdf = result_path / "pit_histograms.pdf"
    pit_data_file = result_path / "pit_data.json"
    
    if not force_regenerate and all(f.exists() for f in [pit_png, pit_pdf, pit_data_file]):
        logger.info("  PIT histograms already exist, skipping")
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
        
        logger.info(f"  Computing PIT histograms on {n_test} test samples")
        
        # Initialize PIT data dictionary
        pit_data = {}
        
        # 1. RFP with posterior mean  
        logger.info("  Computing PIT for RFP (posterior mean)...")
        alpha_mean = get_posterior_alpha(samples, mode="mean", burn_in=burn_in)
        rfp_mean_forecasts = generate_rfp_forecasts_legacy(
            model, test_subset, alpha_mean, reference_tensor, 
            ensemble_size, device, logger
        )
        if rfp_mean_forecasts.size > 0:
            pit_data["rfp_posterior_mean"] = compute_pit_values_empirical(
                rfp_mean_forecasts.squeeze(2), targets  # Remove singleton dimension
            )
        
        # 2. RFP with posterior sample
        logger.info("  Computing PIT for RFP (posterior sample)...")
        alpha_sample = get_posterior_alpha(samples, mode="sample", burn_in=burn_in)
        rfp_sample_forecasts = generate_rfp_forecasts_legacy(
            model, test_subset, alpha_sample, reference_tensor,
            ensemble_size, device, logger
        )
        if rfp_sample_forecasts.size > 0:
            pit_data["rfp_posterior_sample"] = compute_pit_values_empirical(
                rfp_sample_forecasts.squeeze(2), targets
            )
        
        # 3. RFP with uncalibrated parameters (baseline 1) - all alphas = 1.0
        logger.info("  Computing PIT for RFP (uncalibrated baseline - alpha=1.0)...")
        alpha_ones = get_uncalibrated_rfp_alpha_ones(len(variable_names))
        rfp_ones_forecasts = generate_rfp_forecasts_legacy(
            model, test_subset, alpha_ones, reference_tensor,
            ensemble_size, device, logger
        )
        if rfp_ones_forecasts.size > 0:
            pit_data["rfp_uncalibrated_ones"] = compute_pit_values_empirical(
                rfp_ones_forecasts.squeeze(2), targets
            )
        
        # 4. RFP with uncalibrated parameters (baseline 2) - drawn from prior
        logger.info("  Computing PIT for RFP (uncalibrated baseline - prior samples)...")
        alpha_prior = get_uncalibrated_rfp_alpha_prior(len(variable_names))
        rfp_prior_forecasts = generate_rfp_forecasts_legacy(
            model, test_subset, alpha_prior, reference_tensor,
            ensemble_size, device, logger
        )
        if rfp_prior_forecasts.size > 0:
            pit_data["rfp_uncalibrated_prior"] = compute_pit_values_empirical(
                rfp_prior_forecasts.squeeze(2), targets
            )
        
        # 5. Persistence forecasting
        logger.info("  Computing PIT for persistence forecasts...")
        persistence_forecasts = generate_persistence_forecasts(test_subset)
        if persistence_forecasts.size > 0:
            pit_data["persistence"] = compute_pit_values_empirical(persistence_forecasts, targets)
        
        # 5. Climatological forecasting
        logger.info("  Computing PIT for climatological forecasts...")
        full_dataset = loader.dataset.dataset if hasattr(loader.dataset, 'dataset') else loader.dataset
        clim_mean, clim_std = compute_climatology_stats(full_dataset, variable_names)
        clim_forecasts = generate_climatological_forecasts(
            test_subset, clim_mean, clim_std, ensemble_size
        )
        if clim_forecasts.size > 0:
            pit_data["climatology"] = compute_pit_values_empirical(clim_forecasts, targets)
        
        # 6. AR(1) forecasting
        logger.info("  Computing PIT for AR(1) forecasts...")
        ar_coeffs, intercepts, residual_vars = fit_ar1_models(full_dataset, variable_names)
        ar1_forecasts = generate_ar1_forecasts(
            test_subset, ar_coeffs, intercepts, residual_vars, ensemble_size
        )
        if ar1_forecasts.size > 0:
            pit_data["ar1"] = compute_pit_values_empirical(ar1_forecasts, targets)
        
        # 7. Deterministic + Gaussian noise
        logger.info("  Computing PIT for deterministic + Gaussian noise...")
        empirical_stds = compute_empirical_stds(full_dataset, model, device)
        gaussian_forecasts = generate_gaussian_noise_forecasts(
            model, test_subset, empirical_stds, ensemble_size, device, batch_size=16
        )
        if gaussian_forecasts.size > 0:
            pit_data["deterministic_gaussian"] = compute_pit_values_empirical(gaussian_forecasts, targets)
        
        # Create and save PIT histogram plots
        png_path, pdf_path = create_pit_histogram_plot(pit_data, result_path / "pit_histograms")
        
        # Save raw PIT data for further analysis
        pit_data_serializable = {k: v.tolist() for k, v in pit_data.items() if len(v) > 0}
        with open(pit_data_file, 'w') as f:
            json.dump(pit_data_serializable, f, indent=2)
        
        logger.info(f"  ✓ Saved PIT histograms to {png_path} and {pdf_path}")
        logger.info(f"  ✓ Saved raw PIT data to {pit_data_file}")
        return True
        
    except Exception as e:
        logger.error(f"  ✗ Failed to process {result_path.name}: {e}")
        return False


def process_all_results(results_directory: Path, config_path: str, n_test_samples: int, force_regenerate: bool = False):
    """Process all result directories for PIT histogram analysis."""
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
        
        # Check if PIT histograms already exist
        pit_png = result_path / "pit_histograms.png"
        pit_pdf = result_path / "pit_histograms.pdf"
        pit_data_file = result_path / "pit_data.json"
        
        if not force_regenerate and all(f.exists() for f in [pit_png, pit_pdf, pit_data_file]):
            skipped += 1
            continue
        
        logger.info("Processing: %s (%s)", result_path.name, data_source)
        
        if process_single_result(result_path, config_path, n_test_samples, logger, force_regenerate):
            processed += 1
        else:
            failed += 1
    
    logger.info("\nSummary:")
    logger.info("  Processed: %d", processed)
    logger.info("  Skipped (already have PIT histograms): %d", skipped)
    logger.info("  Failed: %d", failed)


def main():
    parser = argparse.ArgumentParser(description="PIT histogram analysis for forecasting methods")
    parser.add_argument("result_directory", type=str, nargs='?', default="./results",
                       help="Path to single result directory OR results root directory (default: ./results)")
    parser.add_argument("--config", type=str, default="config.json",
                       help="Path to configuration file (default: config.json)")
    parser.add_argument("--n-samples", type=int, default=256,
                       help="Number of test samples to evaluate (default: 256)")
    parser.add_argument("--all", action="store_true", 
                       help="Process all result directories in the specified path")
    parser.add_argument("--force", action="store_true",
                       help="Regenerate PIT histograms even if they already exist")
    
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
        logger.info("✓ PIT histogram analysis completed successfully")
    else:
        logger.error("✗ PIT histogram analysis failed")


if __name__ == "__main__":
    main()