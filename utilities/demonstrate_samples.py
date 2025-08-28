#!/usr/bin/env python3
"""
Spatial Forecast Demonstration Script

This script generates comprehensive spatial visualizations of probabilistic weather forecasts
to demonstrate the performance of ABC-calibrated RFP (Random Field Perturbation) methods.

Key Features:
- Generates probabilistic forecasts using ABC-calibrated parameters
- Creates spatial maps showing ensemble statistics and individual members
- Compares forecasts against ground truth with error analysis
- Computes spatial CRPS for forecast quality assessment
- Produces publication-quality plots in both PNG and PDF formats
- Uses cartopy for meteorological standard map projections

Generated Visualizations (each as separate PNG and PDF files):
1. Individual Variable Analysis (per sample):
   - {variable}_sample_{idx:03d}_ensemble_mean: Average forecast across ensemble
   - {variable}_sample_{idx:03d}_ensemble_std: Forecast uncertainty (spread)
   - {variable}_sample_{idx:03d}_member_{n:02d}: Individual ensemble realizations (n=1-3)
   - {variable}_sample_{idx:03d}_ground_truth: Observed atmospheric fields
   - {variable}_sample_{idx:03d}_error: Difference between ensemble mean and truth
   - {variable}_sample_{idx:03d}_sample_mean: RFP posterior sample mean

2. 2x2 Grid Analysis (per method, variable, and sample):
   - 2x2_spatial_{method}_{variable}_sample_{idx:03d}: Combined 2x2 plot showing:
     * Top-left: Spatial score (CRPS/Energy Score as appropriate)
     * Top-right: Ensemble mean
     * Bottom-left: Ensemble standard deviation
     * Bottom-right: Ground truth

3. Spatial Metric Analysis:
   - spatial_crps_averaged_{method}: CRPS at each grid point averaged across all variables
   - spatial_mae_averaged_{method}: MAE at each grid point averaged across all variables
   - spatial_energy_averaged_{method}: Energy Score at each grid point averaged across all variables
   - Shows overall spatial forecast skill patterns by method

Usage Examples:
    # Basic usage - single variable and sample
    python demonstrate_samples.py results/my_abc_run --variables z500 --n-samples 1
    
    # Multiple variables and samples
    python demonstrate_samples.py results/my_abc_run --variables z500 t2m u10 --n-samples 3
    
    # All variables with verbose output
    python demonstrate_samples.py results/my_abc_run --all --verbose
    
    # Overwrite existing output
    python demonstrate_samples.py results/my_abc_run --all --force

Requirements:
- Completed ABC experiment with checkpoint file
- Access to reference tensor data (z500_t850_t2m_u10_v10_standardized.npy)
- Sufficient computational resources for ensemble generation

Output Structure:
results/experiment_name/samples/
├── {variable}_sample_{idx:03d}_ensemble_mean.png/pdf      # Ensemble mean forecast
├── {variable}_sample_{idx:03d}_ensemble_std.png/pdf       # Ensemble spread
├── {variable}_sample_{idx:03d}_member_{n:02d}.png/pdf     # Individual members
├── {variable}_sample_{idx:03d}_ground_truth.png/pdf       # Observations
├── {variable}_sample_{idx:03d}_error.png/pdf              # Forecast errors
├── {variable}_sample_{idx:03d}_sample_mean.png/pdf        # Sample mean
├── 2x2_spatial_{method}_{variable}_sample_{idx:03d}.png/pdf # 2x2 combined analysis
├── spatial_crps_averaged_{method}.png/pdf                 # Averaged spatial CRPS
├── spatial_mae_averaged_{method}.png/pdf                  # Averaged spatial MAE
├── spatial_energy_averaged_{method}.png/pdf               # Averaged spatial Energy Score
└── baseline_ensemble_samples/                             # Baseline method samples
    └── ensemble_samples_{baseline_method}_{variable}.pdf  # 6 representative members
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
from typing import Dict, List, Tuple
import scipy.stats

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import seaborn as sns

warnings.filterwarnings('ignore')

sys.path.append('.')


from core.config import Config
from core.scoring import compute_crps_for_proposal, compute_energy_score_for_proposal
from core.io_utils import load_model_and_test_data
from core.algorithm import generate_joint_rfp

# Import utilities
from utilities.comparative_performance import (
    load_posterior_data, get_posterior_alpha, generate_rfp_forecasts_batched,
    generate_persistence_forecasts, generate_climatological_forecasts,
    generate_ar1_forecasts, generate_gaussian_noise_forecasts
)

DEFAULT_VARIABLE_NAMES = ["z500", "t850", "t2m", "u10", "v10"]
DEFAULT_DPI = 300

# Prior distribution parameters (same as used in ABC algorithm)
GAMMA_SHAPE = 2.0
GAMMA_SCALE = 0.13

# Uncalibrated RFP method - draws alpha parameters from prior distribution
def compute_empirical_residual_std(model, validation_data, device, logger, max_samples=100):
    """Compute empirical standard deviations from model residuals on validation data."""
    # Use at least 50 samples for reliable statistics, but cap at max_samples
    num_samples = min(max(50, len(validation_data) // 2), max_samples, len(validation_data))
    logger.info(f"Computing empirical residual standard deviations from {num_samples} validation samples")
    
    residuals = []
    model.eval()
    
    with torch.no_grad():
        for i, (prev_fields, target_fields, static_fields) in enumerate(validation_data):
            if i >= num_samples:
                break
                
            prev_fields = prev_fields.to(device)
            target_fields = target_fields.to(device) 
            static_fields = static_fields.to(device)
            
            # Debug shapes for first sample only
            if i == 0:
                logger.info(f"Debug shapes - prev_fields: {prev_fields.shape}, target_fields: {target_fields.shape}, static_fields: {static_fields.shape}")
            
            # Generate deterministic forecast
            deterministic_forecast = model(prev_fields.unsqueeze(0), static_fields.unsqueeze(0))  # [1, T*V, H, W]
            
            if i == 0:
                logger.info(f"Debug shapes - deterministic_forecast: {deterministic_forecast.shape}")
            
            # The target has shape [V, H, W] (variables, height, width)
            # The forecast has shape [1, T*V, H, W] where T*V is the flattened time-variable dimension
            # We need to extract the right forecast variables that correspond to target_fields
            
            # For now, let's use a simple approach - just take the first V channels of the forecast
            V, H, W = target_fields.shape
            forecast_for_target = deterministic_forecast[0, :V, :, :]  # [V, H, W] - first V channels
            
            if i == 0:
                logger.info(f"Debug shapes - forecast_for_target: {forecast_for_target.shape}, target_fields: {target_fields.shape}")
            
            # Compute residuals: forecast - truth
            residual = forecast_for_target - target_fields  # [V, H, W]
            residuals.append(residual.cpu().numpy())
    
    # Stack all residuals and compute empirical standard deviation
    residuals = np.stack(residuals, axis=0)  # [n_samples, V, H, W]
    empirical_std = np.std(residuals, axis=0)  # [V, H, W]
    
    logger.info(f"Computed empirical std with shape: {empirical_std.shape} from {len(residuals)} samples")
    
    # Log ranges for each variable with mean
    var_names = ['z500', 't850', 't2m', 'u10', 'v10']
    ranges = []
    for i in range(empirical_std.shape[0]):
        min_val = np.min(empirical_std[i])
        max_val = np.max(empirical_std[i])  
        mean_val = np.mean(empirical_std[i])
        ranges.append(f"{var_names[i]}: {min_val:.4f}-{max_val:.4f} (mean: {mean_val:.4f})")
    logger.info(f"Empirical std ranges: {ranges}")
    
    # Add a small minimum standard deviation to avoid zero variance (especially for good models)
    min_std = 0.01  # Small non-zero minimum
    empirical_std = np.maximum(empirical_std, min_std)
    logger.info(f"Applied minimum std of {min_std} to avoid zero variance")
    
    return empirical_std


def generate_simple_gaussian_noise_forecasts(model, test_data, empirical_std, ensemble_size, device, logger):
    """Generate deterministic forecast + spatially independent empirical Gaussian noise ensemble."""
    forecasts = []
    
    logger.info(f"Generating deterministic + independent Gaussian noise forecasts using empirical residual std")
    
    for prev_fields, target_fields, static_fields in test_data:
        prev_fields = prev_fields.to(device)
        static_fields = static_fields.to(device)
        
        # Generate deterministic forecast
        with torch.no_grad():
            deterministic_forecast = model(prev_fields.unsqueeze(0), static_fields.unsqueeze(0))  # [1, T*V, H, W]
        
        # Convert to [T, V, H, W] format
        T_total, H, W = deterministic_forecast.shape[1], deterministic_forecast.shape[2], deterministic_forecast.shape[3]
        V = 5  # Number of variables
        T_per_var = T_total // V
        
        det_forecast = deterministic_forecast.reshape(1, T_per_var, V, H, W)  # [1, T, V, H, W]
        
        # Generate ensemble by adding spatially independent Gaussian noise calibrated to empirical residuals
        ensemble_forecasts = []
        for _ in range(ensemble_size):
            # Generate independent Gaussian noise with empirical standard deviations
            noise = np.random.normal(0, empirical_std)  # [V, H, W] - spatially independent per variable
            noise_tensor = torch.from_numpy(noise).float().to(device)
            
            # Add noise to deterministic forecast
            noisy_forecast = det_forecast + noise_tensor.unsqueeze(0).unsqueeze(0)  # Broadcast to [1, T, V, H, W]
            ensemble_forecasts.append(noisy_forecast.cpu().numpy())
        
        # Stack ensemble: [ensemble_size, 1, T, V, H, W]  
        ensemble_array = np.stack(ensemble_forecasts, axis=0)
        forecasts.append(ensemble_array[:, 0])  # Remove batch dimension -> [ensemble_size, T, V, H, W]
    
    # Convert to expected format: [n_samples, ensemble_size, T, V, H, W]
    return np.array(forecasts)


def get_uncalibrated_rfp_alpha(num_variables: int, seed: int = 42) -> np.ndarray:
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

# Variable display information
VARIABLE_INFO = {
    "z500": {"name": "Geopotential Height 500hPa", "units": "m", "cmap": "RdBu_r"},
    "t850": {"name": "Temperature 850hPa", "units": "K", "cmap": "RdYlBu_r"},
    "t2m": {"name": "2m Temperature", "units": "K", "cmap": "RdYlBu_r"},
    "u10": {"name": "10m U-wind", "units": "m/s", "cmap": "RdBu_r"},
    "v10": {"name": "10m V-wind", "units": "m/s", "cmap": "RdBu_r"}
}


def setup_logging():
    """Configure logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(__name__)


def setup_plot_style():
    """Configure matplotlib and seaborn for academic plots."""
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'legend.title_fontsize': 10,
        'figure.dpi': DEFAULT_DPI,
        'savefig.dpi': DEFAULT_DPI,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1
    })


def get_coordinate_grids(tensor_shape):
    """Create coordinate grids for spatial plotting."""
    # Assuming global grid - adjust these ranges based on your actual data
    # Standard global grid: -90 to 90 lat, -180 to 180 lon
    _, _, H, W = tensor_shape
    
    lat = np.linspace(90, -90, H)  # North to South
    lon = np.linspace(-180, 180, W)  # West to East
    
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    
    return lat_grid, lon_grid, lat, lon


def create_spatial_subplot(fig, rows, cols, index, projection=ccrs.PlateCarree()):
    """Create a cartopy subplot with standard features."""
    ax = fig.add_subplot(rows, cols, index, projection=projection)
    
    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, alpha=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.6)
    ax.add_feature(cfeature.OCEAN, alpha=0.3, color='lightblue')
    ax.add_feature(cfeature.LAND, alpha=0.2, color='lightgray')
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.6)
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()
    
    return ax


def create_single_spatial_plot(
    data: np.ndarray,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    title: str,
    var_units: str,
    cmap: str,
    output_path: Path,
    logger: logging.Logger
):
    """Create a single spatial plot."""
    
    # Create figure with single subplot
    fig = plt.figure(figsize=(10, 6))
    ax = create_spatial_subplot(fig, 1, 1, 1)
    
    # Determine color limits
    if "Error" in title or "Difference" in title:
        vmax = np.percentile(np.abs(data), 95)
        vmin = -vmax
    else:
        vmin, vmax = np.percentile(data, [5, 95])
    
    # Create contour plot
    contour = ax.contourf(
        lon_grid, lat_grid, data,
        levels=20, cmap=cmap, vmin=vmin, vmax=vmax,
        transform=ccrs.PlateCarree(), extend='both'
    )
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(contour, ax=ax, shrink=0.8, aspect=30, pad=0.1)
    cbar.set_label(var_units, fontsize=10)
    
    plt.tight_layout()
    
    # Save plots
    plt.savefig(f"{output_path}.png", dpi=DEFAULT_DPI, bbox_inches='tight')
    plt.savefig(f"{output_path}.pdf", dpi=DEFAULT_DPI, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved plot: {output_path}")


def plot_variable_spatial_analysis(
    forecasts_dict: Dict[str, np.ndarray],
    targets: np.ndarray,
    variable_idx: int,
    variable_name: str,
    sample_idx: int,
    output_dir: Path,
    logger: logging.Logger
):
    """Create separate spatial analysis plots for one variable."""
    
    # Get variable info
    var_info = VARIABLE_INFO[variable_name]
    var_display_name = var_info["name"]
    var_units = var_info["units"]
    var_cmap = var_info["cmap"]
    
    logger.info(f"Creating spatial analysis for {var_display_name} (sample {sample_idx})")
    
    # Extract data for this variable and sample
    target_field = targets[sample_idx, variable_idx, :, :]  # [H, W]
    
    # Get coordinate grids
    lat_grid, lon_grid, lat, lon = get_coordinate_grids(targets.shape)
    
    # Create base filename
    base_name = f"{variable_name}_sample_{sample_idx:03d}"
    
    # RFP ensemble mean and std
    if 'rfp_posterior_mean' in forecasts_dict:
        rfp_forecasts = forecasts_dict['rfp_posterior_mean'][sample_idx]  # [ensemble_size, time_steps, variables, height, width]
        # Take the first (and likely only) time step
        rfp_var_forecasts = rfp_forecasts[:, 0, variable_idx, :, :]  # [ensemble_size, height, width]
        rfp_mean = np.mean(rfp_var_forecasts, axis=0)  # [height, width]
        rfp_std = np.std(rfp_var_forecasts, axis=0)   # [height, width]
        
        # Create individual plots
        create_single_spatial_plot(
            rfp_mean, lat_grid, lon_grid,
            f"RFP Ensemble Mean - {var_display_name}",
            var_units, var_cmap,
            output_dir / f"{base_name}_ensemble_mean",
            logger
        )
        
        create_single_spatial_plot(
            rfp_std, lat_grid, lon_grid,
            f"RFP Ensemble Std - {var_display_name}",
            var_units, "viridis",
            output_dir / f"{base_name}_ensemble_std",
            logger
        )
        
        # Add 3 individual ensemble members  
        for i in range(min(3, rfp_forecasts.shape[0])):
            member_field = rfp_forecasts[i, 0, variable_idx, :, :]  # [height, width]
            create_single_spatial_plot(
                member_field, lat_grid, lon_grid,
                f"RFP Member {i+1} - {var_display_name}",
                var_units, var_cmap,
                output_dir / f"{base_name}_member_{i+1:02d}",
                logger
            )
        
        # Forecast error
        error_field = rfp_mean - target_field
        create_single_spatial_plot(
            error_field, lat_grid, lon_grid,
            f"Forecast Error (Mean - Truth) - {var_display_name}",
            var_units, "RdBu_r",
            output_dir / f"{base_name}_error",
            logger
        )
    
    # Ground truth
    create_single_spatial_plot(
        target_field, lat_grid, lon_grid,
        f"Ground Truth - {var_display_name}",
        var_units, var_cmap,
        output_dir / f"{base_name}_ground_truth",
        logger
    )
    
    # RFP posterior sample plots if available
    if 'rfp_posterior_sample' in forecasts_dict:
        rfp_sample_forecasts = forecasts_dict['rfp_posterior_sample'][sample_idx]
        rfp_sample_var = rfp_sample_forecasts[:, 0, variable_idx, :, :]
        rfp_sample_mean = np.mean(rfp_sample_var, axis=0)
        
        create_single_spatial_plot(
            rfp_sample_mean, lat_grid, lon_grid,
            f"RFP Sample Mean - {var_display_name}",
            var_units, var_cmap,
            output_dir / f"{base_name}_sample_mean",
            logger
        )
    
    # RFP uncalibrated plots if available
    if 'rfp_uncalibrated' in forecasts_dict:
        rfp_uncalibrated_forecasts = forecasts_dict['rfp_uncalibrated'][sample_idx]
        rfp_uncalibrated_var = rfp_uncalibrated_forecasts[:, 0, variable_idx, :, :]
        rfp_uncalibrated_mean = np.mean(rfp_uncalibrated_var, axis=0)
        
        create_single_spatial_plot(
            rfp_uncalibrated_mean, lat_grid, lon_grid,
            f"RFP Uncalibrated Mean - {var_display_name}",
            var_units, var_cmap,
            output_dir / f"{base_name}_uncalibrated_mean",
            logger
        )


def compute_spatial_crps_simple(forecasts: np.ndarray, targets: np.ndarray, variable_idx: int) -> np.ndarray:
    """Compute simplified spatial CRPS for a variable using empirical CRPS formula."""
    n_samples, ensemble_size, _, _, H, W = forecasts.shape
    
    # For simplicity, just compute CRPS for the first sample
    sample_idx = 0
    forecast_sample = forecasts[sample_idx, :, 0, variable_idx, :, :]  # [ensemble_size, H, W]
    target_sample = targets[sample_idx, variable_idx, :, :]  # [H, W]
    
    # Simplified CRPS approximation: mean absolute error between sorted forecast and target
    spatial_crps = np.zeros((H, W))
    
    # Vectorized computation for efficiency
    sorted_forecasts = np.sort(forecast_sample, axis=0)  # [ensemble_size, H, W]
    
    for i in range(H):
        for j in range(W):
            # Empirical CRPS approximation
            target_val = target_sample[i, j]
            forecast_vals = sorted_forecasts[:, i, j]
            
            # Simple CRPS approximation
            crps = np.mean(np.abs(forecast_vals - target_val)) - 0.5 * np.mean(np.abs(forecast_vals[:, None] - forecast_vals[None, :]))
            spatial_crps[i, j] = crps
    
    return spatial_crps


def compute_spatial_mae(forecasts: np.ndarray, targets: np.ndarray, variable_idx: int) -> np.ndarray:
    """Compute spatial MAE (Mean Absolute Error) for a variable."""
    n_samples, ensemble_size, _, _, H, W = forecasts.shape
    
    # Use first sample for demonstration (could be extended to average across samples)
    sample_idx = 0
    forecast_sample = forecasts[sample_idx, :, 0, variable_idx, :, :]  # [ensemble_size, H, W]
    target_sample = targets[sample_idx, variable_idx, :, :]  # [H, W]
    
    # Compute ensemble mean
    forecast_mean = np.mean(forecast_sample, axis=0)  # [H, W]
    
    # MAE between ensemble mean and target
    spatial_mae = np.abs(forecast_mean - target_sample)  # [H, W]
    
    return spatial_mae


def compute_spatial_energy_score(forecasts: np.ndarray, targets: np.ndarray, variable_idx: int) -> np.ndarray:
    """Compute spatial Energy Score for a variable using simplified 2-norm distance."""
    n_samples, ensemble_size, _, _, H, W = forecasts.shape
    
    # Use first sample for demonstration
    sample_idx = 0
    forecast_sample = forecasts[sample_idx, :, 0, variable_idx, :, :]  # [ensemble_size, H, W]
    target_sample = targets[sample_idx, variable_idx, :, :]  # [H, W]
    
    spatial_energy = np.zeros((H, W))
    
    # Compute Energy Score at each grid point
    for i in range(H):
        for j in range(W):
            forecast_vals = forecast_sample[:, i, j]  # [ensemble_size]
            target_val = target_sample[i, j]
            
            # First term: mean distance between forecasts and target
            first_term = np.mean(np.abs(forecast_vals - target_val))
            
            # Second term: mean pairwise distance between forecasts (divided by 2)
            if ensemble_size > 1:
                pairwise_diffs = forecast_vals[:, None] - forecast_vals[None, :]
                second_term = 0.5 * np.mean(np.abs(pairwise_diffs))
            else:
                second_term = 0.0
                
            spatial_energy[i, j] = first_term - second_term
    
    return spatial_energy


def plot_spatial_mae_comparison(
    forecasts_dict: Dict[str, np.ndarray],
    targets: np.ndarray,
    variable_names: List[str],
    output_dir: Path,
    logger: logging.Logger
):
    """Create spatial MAE plots averaged across all variables for each method."""
    
    logger.info("Computing spatial MAE averaged across all variables...")
    
    # Get coordinate grids
    lat_grid, lon_grid, lat, lon = get_coordinate_grids(targets.shape)
    
    for method_name, forecasts in forecasts_dict.items():
        logger.info(f"Computing averaged MAE for {method_name}")
        
        # Compute MAE for all variables and average
        all_variable_mae = []
        
        for var_idx, var_name in enumerate(variable_names):
            logger.info(f"  Computing MAE for {method_name} - {var_name}")
            spatial_mae = compute_spatial_mae(forecasts, targets, var_idx)
            all_variable_mae.append(spatial_mae)
        
        # Average MAE across all variables
        averaged_mae = np.mean(all_variable_mae, axis=0)  # Shape: [H, W]
        
        # Create averaged MAE plot
        create_single_spatial_plot(
            averaged_mae, lat_grid, lon_grid,
            f"Spatial MAE (Averaged) - {method_name}",
            "MAE", "Reds",
            output_dir / f"spatial_mae_averaged_{method_name}",
            logger
        )


def plot_spatial_energy_score_comparison(
    forecasts_dict: Dict[str, np.ndarray],
    targets: np.ndarray,
    variable_names: List[str],
    output_dir: Path,
    logger: logging.Logger
):
    """Create spatial Energy Score plots averaged across all variables for each method."""
    
    logger.info("Computing spatial Energy Score averaged across all variables...")
    
    # Get coordinate grids
    lat_grid, lon_grid, lat, lon = get_coordinate_grids(targets.shape)
    
    for method_name, forecasts in forecasts_dict.items():
        logger.info(f"Computing averaged Energy Score for {method_name}")
        
        # Compute Energy Score for all variables and average
        all_variable_energy = []
        
        for var_idx, var_name in enumerate(variable_names):
            logger.info(f"  Computing Energy Score for {method_name} - {var_name}")
            spatial_energy = compute_spatial_energy_score(forecasts, targets, var_idx)
            all_variable_energy.append(spatial_energy)
        
        # Average Energy Score across all variables
        averaged_energy = np.mean(all_variable_energy, axis=0)  # Shape: [H, W]
        
        # Create averaged Energy Score plot
        create_single_spatial_plot(
            averaged_energy, lat_grid, lon_grid,
            f"Spatial Energy Score (Averaged) - {method_name}",
            "Energy Score", "plasma",
            output_dir / f"spatial_energy_averaged_{method_name}",
            logger
        )


def create_2x2_spatial_plot(
    spatial_score: np.ndarray,
    ensemble_mean: np.ndarray,
    ensemble_std: np.ndarray,
    ground_truth: np.ndarray,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    method_name: str,
    score_type: str,
    variable_name: str,
    sample_idx: int,
    output_path: Path,
    logger: logging.Logger
):
    """Create a 2x2 grid plot showing spatial score, ensemble mean, ensemble std, and ground truth."""
    
    # Create figure with 2x2 subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Define plot data and properties
    plots_config = [
        {
            'data': spatial_score,
            'title': f'{score_type} - {method_name}',
            'units': score_type,
            'cmap': 'viridis' if score_type == 'CRPS' else 'plasma',
            'pos': 1
        },
        {
            'data': ensemble_mean,
            'title': f'Ensemble Mean - {method_name}',
            'units': VARIABLE_INFO[variable_name]['units'],
            'cmap': VARIABLE_INFO[variable_name]['cmap'],
            'pos': 2
        },
        {
            'data': ensemble_std,
            'title': f'Ensemble Std - {method_name}',
            'units': VARIABLE_INFO[variable_name]['units'],
            'cmap': 'viridis',
            'pos': 3
        },
        {
            'data': ground_truth,
            'title': f'Ground Truth - {variable_name}',
            'units': VARIABLE_INFO[variable_name]['units'],
            'cmap': VARIABLE_INFO[variable_name]['cmap'],
            'pos': 4
        }
    ]
    
    for plot_info in plots_config:
        ax = create_spatial_subplot(fig, 2, 2, plot_info['pos'])
        
        # Determine color limits
        if "Error" in plot_info['title'] or "Difference" in plot_info['title']:
            vmax = np.percentile(np.abs(plot_info['data']), 95)
            vmin = -vmax
        else:
            vmin, vmax = np.percentile(plot_info['data'], [5, 95])
        
        # Create contour plot
        contour = ax.contourf(
            lon_grid, lat_grid, plot_info['data'],
            levels=20, cmap=plot_info['cmap'], vmin=vmin, vmax=vmax,
            transform=ccrs.PlateCarree(), extend='both'
        )
        
        ax.set_title(plot_info['title'], fontsize=11, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(contour, ax=ax, shrink=0.6, aspect=20, pad=0.05)
        cbar.set_label(plot_info['units'], fontsize=9)
        cbar.ax.tick_params(labelsize=8)
    
    # Add main title
    fig.suptitle(f'{variable_name} - {method_name}', 
                 fontsize=14, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    
    # Save plots
    plt.savefig(f"{output_path}.png", dpi=DEFAULT_DPI, bbox_inches='tight')
    plt.savefig(f"{output_path}.pdf", dpi=DEFAULT_DPI, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved 2x2 spatial plot: {output_path}")


def create_persistence_comparison_plot(
    forecast_sample: np.ndarray,
    target_field: np.ndarray,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    var_name: str,
    var_info: dict,
    method_dir: Path,
    logger: logging.Logger
):
    """Create persistence-specific plot showing t-1 and t side by side."""
    
    # For persistence: all ensemble members are identical, so use first member
    persistence_field = forecast_sample[0, :, :]  # [H, W]
    
    # Create figure with 2 plots side by side
    fig = plt.figure(figsize=(16, 6))
    
    # Left: t-1 (persistence = current observation, which should equal target for perfect persistence)  
    ax1 = create_spatial_subplot(fig, 1, 2, 1)
    vmin, vmax = np.percentile(target_field, [5, 95])
    contour1 = ax1.contourf(
        lon_grid, lat_grid, persistence_field,
        levels=20, cmap=var_info['cmap'], vmin=vmin, vmax=vmax,
        transform=ccrs.PlateCarree(), extend='both'
    )
    ax1.set_title(f'Persistence Forecast (t) - {var_info["name"]}', fontsize=12, fontweight='bold')
    
    # Right: t (target/truth)
    ax2 = create_spatial_subplot(fig, 1, 2, 2)
    contour2 = ax2.contourf(
        lon_grid, lat_grid, target_field,
        levels=20, cmap=var_info['cmap'], vmin=vmin, vmax=vmax,
        transform=ccrs.PlateCarree(), extend='both'
    )
    ax2.set_title(f'Ground Truth (t) - {var_info["name"]}', fontsize=12, fontweight='bold')
    
    # Add colorbars
    cbar1 = plt.colorbar(contour1, ax=ax1, shrink=0.6, aspect=20, pad=0.05)
    cbar1.set_label(var_info['units'], fontsize=9)
    cbar2 = plt.colorbar(contour2, ax=ax2, shrink=0.6, aspect=20, pad=0.05)
    cbar2.set_label(var_info['units'], fontsize=9)
    
    plt.tight_layout()
    
    # Save plots
    output_path = method_dir / f"persistence_comparison_{var_name}"
    plt.savefig(f"{output_path}.png", dpi=DEFAULT_DPI, bbox_inches='tight')
    plt.savefig(f"{output_path}.pdf", dpi=DEFAULT_DPI, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved persistence comparison plot: {output_path}")


def create_climatology_ensemble_plot(
    forecast_sample: np.ndarray,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    var_name: str,
    var_info: dict,
    method_dir: Path,
    logger: logging.Logger
):
    """Create climatology-specific plot showing 3 ensemble members horizontally."""
    
    # Select 3 representative ensemble members
    n_members = min(3, forecast_sample.shape[0])
    member_indices = np.linspace(0, forecast_sample.shape[0]-1, n_members, dtype=int)
    
    # Create figure with 3 plots horizontally
    fig = plt.figure(figsize=(24, 6))
    
    # Determine consistent color scale across all members
    all_data = forecast_sample[member_indices, :, :]
    vmin, vmax = np.percentile(all_data, [5, 95])
    
    for i, member_idx in enumerate(member_indices):
        ax = create_spatial_subplot(fig, 1, n_members, i+1)
        
        member_field = forecast_sample[member_idx, :, :]
        contour = ax.contourf(
            lon_grid, lat_grid, member_field,
            levels=20, cmap=var_info['cmap'], vmin=vmin, vmax=vmax,
            transform=ccrs.PlateCarree(), extend='both'
        )
        
        ax.set_title(f'Climatology Member {member_idx+1} - {var_info["name"]}', 
                    fontsize=12, fontweight='bold')
        
        # Add colorbar only to the rightmost plot
        if i == n_members - 1:
            cbar = plt.colorbar(contour, ax=ax, shrink=0.6, aspect=20, pad=0.05)
            cbar.set_label(var_info['units'], fontsize=9)
    
    plt.tight_layout()
    
    # Save plots
    output_path = method_dir / f"climatology_ensemble_{var_name}"
    plt.savefig(f"{output_path}.png", dpi=DEFAULT_DPI, bbox_inches='tight')
    plt.savefig(f"{output_path}.pdf", dpi=DEFAULT_DPI, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved climatology ensemble plot: {output_path}")


def create_deterministic_gaussian_2x2_plot(
    forecast_sample: np.ndarray,
    target_field: np.ndarray,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    var_name: str,
    var_info: dict,
    method_dir: Path,
    logger: logging.Logger,
    use_crps: bool = False
):
    """Create deterministic+gaussian-specific 2x2 plot: 2 members (top) + energy score + ensemble std (bottom)."""
    
    # Select 2 representative ensemble members
    member_indices = [0, forecast_sample.shape[0]//2] if forecast_sample.shape[0] > 1 else [0, 0]
    
    # Compute ensemble statistics
    ensemble_std = np.std(forecast_sample, axis=0)
    
    # Compute spatial score
    if use_crps:
        # Simplified CRPS computation for single sample
        spatial_score = np.mean(np.abs(forecast_sample - target_field[np.newaxis, :, :]), axis=0)
        score_name = 'CRPS'
        score_cmap = 'viridis'
    else:
        # Energy score computation
        first_term = np.mean(np.abs(forecast_sample - target_field[np.newaxis, :, :]), axis=0)
        if forecast_sample.shape[0] > 1:
            pairwise_diffs = forecast_sample[:, np.newaxis, :, :] - forecast_sample[np.newaxis, :, :, :]
            second_term = 0.5 * np.mean(np.abs(pairwise_diffs), axis=(0, 1))
        else:
            second_term = 0.0
        spatial_score = first_term - second_term
        score_name = 'Energy Score'
        score_cmap = 'plasma'
    
    # Create 2x2 figure
    fig = plt.figure(figsize=(16, 12))
    
    # Define color limits for ensemble members
    all_member_data = forecast_sample[member_indices, :, :]
    vmin_members, vmax_members = np.percentile(all_member_data, [5, 95])
    
    plots_config = [
        {
            'data': forecast_sample[member_indices[0], :, :],
            'title': f'Member {member_indices[0]+1} - {var_info["name"]}',
            'units': var_info['units'],
            'cmap': var_info['cmap'],
            'pos': 1,
            'vmin': vmin_members,
            'vmax': vmax_members
        },
        {
            'data': forecast_sample[member_indices[1], :, :],
            'title': f'Member {member_indices[1]+1} - {var_info["name"]}',
            'units': var_info['units'],
            'cmap': var_info['cmap'],
            'pos': 2,
            'vmin': vmin_members,
            'vmax': vmax_members
        },
        {
            'data': spatial_score,
            'title': f'{score_name} - {var_info["name"]}',
            'units': score_name,
            'cmap': score_cmap,
            'pos': 3,
            'vmin': None,
            'vmax': None
        },
        {
            'data': ensemble_std,
            'title': f'Ensemble Std - {var_info["name"]}',
            'units': var_info['units'],
            'cmap': 'viridis',
            'pos': 4,
            'vmin': None,
            'vmax': None
        }
    ]
    
    for plot_info in plots_config:
        ax = create_spatial_subplot(fig, 2, 2, plot_info['pos'])
        
        # Determine color limits
        if plot_info['vmin'] is not None and plot_info['vmax'] is not None:
            vmin, vmax = plot_info['vmin'], plot_info['vmax']
        else:
            vmin, vmax = np.percentile(plot_info['data'], [5, 95])
        
        # Create contour plot
        contour = ax.contourf(
            lon_grid, lat_grid, plot_info['data'],
            levels=20, cmap=plot_info['cmap'], vmin=vmin, vmax=vmax,
            transform=ccrs.PlateCarree(), extend='both'
        )
        
        ax.set_title(plot_info['title'], fontsize=11, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(contour, ax=ax, shrink=0.6, aspect=20, pad=0.05)
        cbar.set_label(plot_info['units'], fontsize=9)
        cbar.ax.tick_params(labelsize=8)
    
    # Add main title
    fig.suptitle(f'Deterministic + Gaussian - {var_info["name"]}', 
                 fontsize=14, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    
    # Save plots
    output_path = method_dir / f"deterministic_gaussian_2x2_{var_name}"
    plt.savefig(f"{output_path}.png", dpi=DEFAULT_DPI, bbox_inches='tight')
    plt.savefig(f"{output_path}.pdf", dpi=DEFAULT_DPI, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved deterministic+gaussian 2x2 plot: {output_path}")


def create_baseline_method_plots(
    forecasts_dict: Dict[str, np.ndarray],
    targets: np.ndarray,
    variable_names: List[str],
    output_dir: Path,
    logger: logging.Logger,
    config: Config,
    sample_idx: int = 0,
    n_samples: int = 1
):
    """Create comprehensive spatial plots for baseline methods in a dedicated directory."""
    
    logger.info("Creating comprehensive baseline method plots...")
    
    # Get coordinate grids
    lat_grid, lon_grid, lat, lon = get_coordinate_grids(targets.shape)
    
    # Define baseline methods to plot
    baseline_methods = {k: v for k, v in forecasts_dict.items() 
                       if any(baseline in k.lower() for baseline in 
                             ['uncalibrated', 'persistence', 'climatology', 'gaussian'])}
    
    if not baseline_methods:
        logger.warning("No baseline methods found for plotting")
        return
    
    # Determine scoring rule
    scoring_rule = getattr(config, 'scoring_rule', 'energy_score').lower()
    if 'crps' in scoring_rule:
        score_type = 'CRPS'
        use_crps = True
    else:
        score_type = 'Energy Score'
        use_crps = False
    
    for method_name, forecasts in baseline_methods.items():
        logger.info(f"Creating plots for baseline method: {method_name}")
        
        # Create method-specific directory
        method_dir = output_dir / method_name
        method_dir.mkdir(exist_ok=True)
        
        for var_idx, var_name in enumerate(variable_names):
            # Get variable info
            var_info = VARIABLE_INFO[var_name]
            var_display_name = var_info["name"]
            var_units = var_info["units"]
            var_cmap = var_info["cmap"]
            
            for sample_idx in range(min(n_samples, forecasts.shape[0])):
                logger.info(f"  Creating plots for {method_name} - {var_name} - Sample {sample_idx}")
                
                # Extract data for this variable and sample
                target_field = targets[sample_idx, var_idx, :, :]  # [H, W]
                forecast_sample = forecasts[sample_idx, :, 0, var_idx, :, :]  # [ensemble_size, H, W]
                ensemble_mean = np.mean(forecast_sample, axis=0)  # [H, W]
                ensemble_std = np.std(forecast_sample, axis=0)   # [H, W]
                
                # Fix ensemble std for deterministic methods
                if method_name.lower() == 'persistence':
                    # Persistence is deterministic - ensemble std should be exactly zero
                    if np.allclose(forecast_sample[0], forecast_sample[-1], atol=1e-5):
                        logger.info(f"DEBUG: Setting persistence ensemble std to zero (was: {ensemble_std.min():.6f} - {ensemble_std.max():.6f})")
                        ensemble_std = np.zeros_like(ensemble_std)
                    
                    # Persistence: show t-1 and t side by side
                    create_persistence_comparison_plot(
                        forecast_sample, target_field, lat_grid, lon_grid,
                        var_name, var_info, method_dir, logger
                    )
                elif method_name.lower() == 'climatology':
                    # Climatology: show 3 ensemble members horizontally
                    create_climatology_ensemble_plot(
                        forecast_sample, lat_grid, lon_grid,
                        var_name, var_info, method_dir, logger
                    )
                elif method_name.lower() == 'deterministic_gaussian':
                    # Deterministic+Gaussian: use standard 2x2 layout same as other baselines
                    # Compute spatial score for 2x2 plot
                    if use_crps:
                        spatial_score = np.mean(np.abs(forecast_sample - target_field[np.newaxis, :, :]), axis=0)
                        score_type = 'CRPS'
                    else:
                        # Energy score computation
                        first_term = np.mean(np.abs(forecast_sample - target_field[np.newaxis, :, :]), axis=0)
                        if forecast_sample.shape[0] > 1:
                            pairwise_diffs = forecast_sample[:, np.newaxis, :, :] - forecast_sample[np.newaxis, :, :, :]
                            second_term = 0.5 * np.mean(np.abs(pairwise_diffs), axis=(0, 1))
                        else:
                            second_term = 0.0
                        spatial_score = first_term - second_term
                        score_type = 'Energy Score'
                    
                    # Use standard 2x2 plot layout
                    output_path = method_dir / f"2x2_{var_name}"
                    create_2x2_spatial_plot(
                        spatial_score, ensemble_mean, ensemble_std, target_field,
                        lat_grid, lon_grid, method_name, score_type, var_name, sample_idx,
                        output_path, logger
                    )
                
                # Create standard ensemble mean plot
                create_single_spatial_plot(
                    ensemble_mean, lat_grid, lon_grid,
                    f"{method_name} Ensemble Mean - {var_display_name}",
                    var_units, var_cmap,
                    method_dir / f"{var_name}_ensemble_mean",
                    logger
                )
                
                # Create ensemble std plot
                create_single_spatial_plot(
                    ensemble_std, lat_grid, lon_grid,
                    f"{method_name} Ensemble Std - {var_display_name}",
                    var_units, "viridis",
                    method_dir / f"{var_name}_ensemble_std",
                    logger
                )
                
                # Create error plot
                error_field = ensemble_mean - target_field
                create_single_spatial_plot(
                    error_field, lat_grid, lon_grid,
                    f"{method_name} Forecast Error - {var_display_name}",
                    var_units, "RdBu_r",
                    method_dir / f"{var_name}_error",
                    logger
                )
                
                # Create MAE plot
                mae_field = np.abs(ensemble_mean - target_field)
                create_single_spatial_plot(
                    mae_field, lat_grid, lon_grid,
                    f"{method_name} MAE - {var_display_name}",
                    "MAE", "Reds",
                    method_dir / f"{var_name}_mae",
                    logger
                )
                
                # Compute and create spatial score plot
                if use_crps:
                    spatial_score = compute_spatial_crps_simple(
                        forecasts[sample_idx:sample_idx+1], 
                        targets[sample_idx:sample_idx+1], 
                        var_idx
                    )
                else:
                    spatial_score = compute_spatial_energy_score(
                        forecasts[sample_idx:sample_idx+1], 
                        targets[sample_idx:sample_idx+1], 
                        var_idx
                    )
                
                create_single_spatial_plot(
                    spatial_score, lat_grid, lon_grid,
                    f"{method_name} {score_type} - {var_display_name}",
                    score_type, 'viridis' if use_crps else 'plasma',
                    method_dir / f"{var_name}_{score_type.lower().replace(' ', '_')}",
                    logger
                )
                
                # Create 2x2 grid plot (standard version)
                if method_name.lower() != 'deterministic_gaussian':  # Skip for deterministic_gaussian as it has custom version
                    output_path = method_dir / f"2x2_{var_name}"
                    create_2x2_spatial_plot(
                        spatial_score, ensemble_mean, ensemble_std, target_field,
                        lat_grid, lon_grid, method_name, score_type, var_name, sample_idx,
                        output_path, logger
                    )
                
                # Create ensemble member samples (first 3 members) - only if not specialized method
                if method_name.lower() not in ['persistence', 'climatology']:
                    for member_idx in range(min(3, forecast_sample.shape[0])):
                        member_field = forecast_sample[member_idx, :, :]
                        create_single_spatial_plot(
                            member_field, lat_grid, lon_grid,
                            f"{method_name} Member {member_idx+1} - {var_display_name}",
                            var_units, var_cmap,
                            method_dir / f"{var_name}_member_{member_idx+1:02d}",
                            logger
                        )
    
    logger.info(f"✓ Baseline method plots saved to: {output_dir}")


def create_baseline_ensemble_samples(
    forecasts_dict: Dict[str, np.ndarray],
    targets: np.ndarray,
    variable_names: List[str],
    output_dir: Path,
    logger: logging.Logger,
    sample_idx: int = 0,
    n_members: int = 6
):
    """Create representative ensemble samples for baseline methods as PDFs."""
    
    logger.info("Creating baseline ensemble samples...")
    
    # Create baseline samples directory
    baseline_dir = output_dir / "baseline_ensemble_samples"
    baseline_dir.mkdir(exist_ok=True)
    
    # Get coordinate grids
    lat_grid, lon_grid, lat, lon = get_coordinate_grids(targets.shape)
    
    # Define baseline methods (exclude ABC calibrated methods)
    baseline_methods = {k: v for k, v in forecasts_dict.items() 
                       if any(baseline in k.lower() for baseline in 
                             ['uncalibrated', 'persistence', 'climatology', 'gaussian'])}
    
    if not baseline_methods:
        logger.warning("No baseline methods found for ensemble sampling")
        return
    
    for method_name, forecasts in baseline_methods.items():
        logger.info(f"Creating ensemble samples for {method_name}")
        
        for var_idx, var_name in enumerate(variable_names):
            # Get variable info
            var_info = VARIABLE_INFO[var_name]
            var_display_name = var_info["name"]
            var_units = var_info["units"]
            var_cmap = var_info["cmap"]
            
            # Extract ensemble members for this variable and sample
            forecast_sample = forecasts[sample_idx, :, 0, var_idx, :, :]  # [ensemble_size, H, W]
            target_field = targets[sample_idx, var_idx, :, :]  # [H, W]
            
            # Select representative members (evenly spaced through ensemble)
            ensemble_size = forecast_sample.shape[0]
            member_indices = np.linspace(0, ensemble_size-1, n_members, dtype=int)
            
            # Create figure with 3x2 grid for 6 members
            fig = plt.figure(figsize=(16, 12))
            
            for i, member_idx in enumerate(member_indices):
                ax = create_spatial_subplot(fig, 3, 2, i+1)
                
                member_field = forecast_sample[member_idx, :, :]
                
                # Determine color limits based on ground truth
                vmin, vmax = np.percentile(target_field, [5, 95])
                
                # Create contour plot
                contour = ax.contourf(
                    lon_grid, lat_grid, member_field,
                    levels=20, cmap=var_cmap, vmin=vmin, vmax=vmax,
                    transform=ccrs.PlateCarree(), extend='both'
                )
                
                ax.set_title(f'Member {member_idx+1:02d}', fontsize=11, fontweight='bold')
                
                # Add colorbar only for the right column
                if (i+1) % 2 == 0:
                    cbar = plt.colorbar(contour, ax=ax, shrink=0.6, aspect=20, pad=0.05)
                    cbar.set_label(var_units, fontsize=9)
                    cbar.ax.tick_params(labelsize=8)
            
            # Add main title
            fig.suptitle(f'{var_display_name} - {method_name} - Ensemble Members', 
                        fontsize=14, fontweight='bold', y=0.95)
            
            plt.tight_layout()
            
            # Save only as PDF for baseline samples
            output_path = baseline_dir / f"ensemble_samples_{method_name}_{var_name}"
            plt.savefig(f"{output_path}.pdf", dpi=DEFAULT_DPI, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved baseline ensemble samples: {output_path}.pdf")


def plot_2x2_spatial_analysis(
    forecasts_dict: Dict[str, np.ndarray],
    targets: np.ndarray,
    variable_names: List[str],
    output_dir: Path,
    logger: logging.Logger,
    config: Config,
    n_samples: int = 1
):
    """Create 2x2 spatial analysis plots for each method and variable."""
    
    logger.info("Creating 2x2 spatial analysis plots...")
    
    # Get coordinate grids
    lat_grid, lon_grid, lat, lon = get_coordinate_grids(targets.shape)
    
    # Determine scoring rule from experiment configuration
    scoring_rule = getattr(config, 'scoring_rule', 'energy_score').lower()
    if 'crps' in scoring_rule:
        score_type = 'CRPS'
        use_crps = True
    else:
        score_type = 'Energy Score'
        use_crps = False
    
    logger.info(f"Using scoring rule from config: {score_type}")
    
    for method_name, forecasts in forecasts_dict.items():
        logger.info(f"Creating 2x2 plots for {method_name}")
        
        for sample_idx in range(min(n_samples, forecasts.shape[0])):
            for var_idx, var_name in enumerate(variable_names):
                logger.info(f"  Processing {method_name} - {var_name} - Sample {sample_idx}")
                
                # Extract data for this variable and sample
                target_field = targets[sample_idx, var_idx, :, :]  # [H, W]
                forecast_sample = forecasts[sample_idx, :, 0, var_idx, :, :]  # [ensemble_size, H, W]
                ensemble_mean = np.mean(forecast_sample, axis=0)  # [H, W]
                ensemble_std = np.std(forecast_sample, axis=0)   # [H, W]
                
                # Compute spatial score based on experiment configuration
                if use_crps:
                    spatial_score = compute_spatial_crps_simple(
                        forecasts[sample_idx:sample_idx+1], 
                        targets[sample_idx:sample_idx+1], 
                        var_idx
                    )
                else:
                    spatial_score = compute_spatial_energy_score(
                        forecasts[sample_idx:sample_idx+1], 
                        targets[sample_idx:sample_idx+1], 
                        var_idx
                    )
                
                # Create 2x2 plot
                output_path = output_dir / f"2x2_spatial_{method_name}_{var_name}_sample_{sample_idx:03d}"
                create_2x2_spatial_plot(
                    spatial_score, ensemble_mean, ensemble_std, target_field,
                    lat_grid, lon_grid, method_name, score_type, var_name, sample_idx,
                    output_path, logger
                )


def plot_spatial_crps_comparison(
    forecasts_dict: Dict[str, np.ndarray],
    targets: np.ndarray,
    variable_names: List[str],
    output_dir: Path,
    logger: logging.Logger
):
    """Create spatial CRPS plots averaged across all variables for each method."""
    
    logger.info("Computing spatial CRPS averaged across all variables...")
    
    # Get coordinate grids
    lat_grid, lon_grid, lat, lon = get_coordinate_grids(targets.shape)
    
    for method_name, forecasts in forecasts_dict.items():
        logger.info(f"Computing averaged CRPS for {method_name}")
        
        # Compute CRPS for all variables and average
        all_variable_crps = []
        
        for var_idx, var_name in enumerate(variable_names):
            logger.info(f"  Computing CRPS for {method_name} - {var_name}")
            spatial_crps = compute_spatial_crps_simple(forecasts, targets, var_idx)
            all_variable_crps.append(spatial_crps)
        
        # Average CRPS across all variables
        averaged_crps = np.mean(all_variable_crps, axis=0)  # Shape: [H, W]
        
        # Create averaged CRPS plot
        create_single_spatial_plot(
            averaged_crps, lat_grid, lon_grid,
            f"Spatial CRPS (Averaged) - {method_name}",
            "CRPS", "viridis",
            output_dir / f"spatial_crps_averaged_{method_name}",
            logger
        )


def generate_sample_forecasts(
    result_path: Path,
    config: Config,
    model: torch.nn.Module,
    test_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    device: torch.device,
    logger: logging.Logger,
    n_samples: int = 1
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Generate forecast samples using all methods."""
    
    logger.info(f"Generating forecasts for {n_samples} samples...")
    
    # Limit test data to n_samples with RANDOM sampling to match ABC training
    # Use random sampling like ABC training does (resample_temporal=True)
    np.random.seed(42)  # Reproducible random sampling
    subset_indices = np.random.choice(len(test_data), size=min(n_samples, len(test_data)), replace=False)
    sample_data = [test_data[i] for i in sorted(subset_indices)]
    
    # Load posterior data
    samples, scores, data_source = load_posterior_data(result_path)
    if samples is None:
        raise ValueError(f"Could not load posterior data from {result_path}")
    
    logger.info(f"Loaded posterior data from {data_source}")
    
    # Get alpha values
    alpha_mean = get_posterior_alpha(samples, mode="mean")
    alpha_sample = get_posterior_alpha(samples, mode="sample")
    
    # Load reference tensor for RFP
    from pathlib import Path as PathLib
    ref_path = PathLib(config.data_directory) / "z500_t850_t2m_u10_v10_standardized.npy"
    if ref_path.exists():
        reference_tensor = torch.from_numpy(np.load(ref_path, mmap_mode="r")).to(device)
        logger.info(f"Loaded reference tensor with shape: {reference_tensor.shape}")
    else:
        logger.warning("Reference tensor not found, using fallback")
        # Fallback: reshape the input data to the expected format
        # Convert from [T*V, H, W] to [T, V, H, W]
        first_sample = sample_data[0][0]  # [12, 32, 64]
        T_total, H, W = first_sample.shape
        V = len(config.variable_names)  # 5
        T_per_var = T_total // V  # Should be ~2-3 timesteps per variable
        
        # This is a fallback - ideally should use the proper reference tensor
        reference_tensor = first_sample.reshape(T_per_var, V, H, W).to(device)
        logger.warning(f"Using fallback reference tensor with shape: {reference_tensor.shape}")
    ensemble_size = config.ensemble_size
    
    forecasts_dict = {}
    targets = []
    
    # Extract targets
    for _, target, _ in sample_data:
        targets.append(target.cpu().numpy())
    targets = np.array(targets)  # [n_samples, H, W, n_vars]
    
    # Generate RFP forecasts (posterior mean)
    logger.info("Generating RFP forecasts (posterior mean)...")
    rfp_mean_forecasts = generate_rfp_forecasts_batched(
        model, sample_data, alpha_mean, reference_tensor,
        ensemble_size, device, logger, batch_size=2
    )
    forecasts_dict['rfp_posterior_mean'] = rfp_mean_forecasts
    
    # Generate RFP forecasts (posterior sample)
    logger.info("Generating RFP forecasts (posterior sample)...")
    rfp_sample_forecasts = generate_rfp_forecasts_batched(
        model, sample_data, alpha_sample, reference_tensor,
        ensemble_size, device, logger, batch_size=2
    )
    forecasts_dict['rfp_posterior_sample'] = rfp_sample_forecasts
    
    # Generate RFP forecasts (uncalibrated baseline) - drawn from prior
    logger.info("Generating RFP forecasts (uncalibrated baseline - prior samples)...")
    alpha_uncalibrated = get_uncalibrated_rfp_alpha(len(config.variable_names))
    rfp_uncalibrated_forecasts = generate_rfp_forecasts_batched(
        model, sample_data, alpha_uncalibrated, reference_tensor,
        ensemble_size, device, logger, batch_size=2
    )
    forecasts_dict['rfp_uncalibrated'] = rfp_uncalibrated_forecasts
    
    # Generate baseline forecasts
    # Note: AR(1) forecasts require pre-fitted parameters, skipping for this demo
    logger.info("Skipping AR(1) forecasts (requires pre-fitted parameters)")
    # ar1_forecasts = generate_ar1_forecasts(sample_data, ar_coeffs, intercepts, residual_vars, ensemble_size)
    # forecasts_dict['ar1'] = ar1_forecasts
    
    # Generate persistence forecasts
    logger.info("Generating persistence forecasts...")
    persistence_raw = generate_persistence_forecasts(sample_data)
    # Reshape from (n_samples, 1, V, H, W) to (n_samples, ensemble_size, T, V, H, W)
    # For persistence, replicate the single forecast across ensemble_size
    persistence_forecasts = np.expand_dims(persistence_raw, axis=2)  # Add time dimension: (n_samples, 1, 1, V, H, W)
    persistence_forecasts = np.repeat(persistence_forecasts, ensemble_size, axis=1)  # Replicate across ensemble
    forecasts_dict['persistence'] = persistence_forecasts
    
    # Generate climatological forecasts (need to compute climatology from data)
    logger.info("Generating climatological forecasts...")
    # For demo purposes, use simple climatology (zeros with some std)
    # In production, this should be computed from full training dataset
    clim_mean = np.zeros((5, 32, 64))  # [variables, H, W]
    clim_std = np.ones((5, 32, 64)) * 0.5  # Small std for demo
    climatology_raw = generate_climatological_forecasts(sample_data, clim_mean, clim_std, ensemble_size)
    # Reshape from (n_samples, ensemble_size, V, H, W) to (n_samples, ensemble_size, T, V, H, W)
    climatology_forecasts = np.expand_dims(climatology_raw, axis=2)  # Add time dimension
    forecasts_dict['climatology'] = climatology_forecasts
    
    # Generate deterministic + Gaussian noise forecasts (using empirical residuals from validation data)
    logger.info("Computing empirical residual statistics for deterministic + Gaussian noise forecasts...")
    # Use a subset of test data as "validation" to compute empirical residuals
    # In production, this should use separate validation data
    validation_subset = sample_data[:min(50, len(sample_data))]  # Use subset to compute residuals
    empirical_std = compute_empirical_residual_std(model, validation_subset, device, logger, max_samples=50)
    
    logger.info("Generating deterministic + independent Gaussian noise forecasts...")
    gaussian_forecasts = generate_simple_gaussian_noise_forecasts(model, sample_data, empirical_std, ensemble_size, device, logger)
    forecasts_dict['deterministic_gaussian'] = gaussian_forecasts
    
    return forecasts_dict, targets


def process_single_directory(result_path, n_samples=1, variables=None, force=False, verbose=False, force_baselines=False):
    """Process a single result directory."""
    if variables is None:
        variables = DEFAULT_VARIABLE_NAMES
        
    logger = setup_logging()
    if verbose:
        logger.setLevel(logging.DEBUG)
    
    setup_plot_style()
    
    # Validate paths
    result_path = Path(result_path)
    if not result_path.exists():
        logger.error(f"Result path does not exist: {result_path}")
        return False
    
    # Check if plots already exist (unless forcing regeneration)
    output_dir = result_path / "samples"
    if not force and output_dir.exists():
        existing_plots = list(output_dir.glob("*.png"))
        if existing_plots:
            logger.info(f"Plots already exist in {output_dir}. Use --force to regenerate.")
            return False
    
    try:
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
        
        # Generate plots for this directory
        generate_forecast_demonstrations(
            result_path=result_path,
            output_dir=output_dir,
            n_samples=n_samples,
            variables=variables,
            logger=logger,
            force_baselines=force_baselines
        )
        return True
    except Exception as e:
        logger.error(f"Failed to process {result_path}: {e}")
        return False


def process_all_results(root_path, n_samples=1, variables=None, force=False, verbose=False, force_baselines=False):
    """Process all result directories found in the root path."""
    if variables is None:
        variables = DEFAULT_VARIABLE_NAMES
        
    logger = setup_logging()
    if verbose:
        logger.setLevel(logging.DEBUG)
    
    root_path = Path(root_path)
    if not root_path.exists():
        logger.error(f"Root path does not exist: {root_path}")
        return
    
    # Find result directories (those with config.json and posterior data)
    result_dirs = []
    for item in root_path.iterdir():
        if item.is_dir() and (item / "config.json").exists():
            # Check for posterior samples or checkpoint
            has_data = (
                (item / "posterior_samples.npy").exists() or
                any(item.glob("*checkpoint*.npz")) or
                (item / "samples").exists()
            )
            if has_data:
                result_dirs.append(item)
    
    if not result_dirs:
        logger.error(f"No result directories found in: {root_path}")
        return
    
    logger.info(f"Found {len(result_dirs)} result directories to process")
    
    processed = 0
    skipped = 0
    failed = 0
    
    for result_path in sorted(result_dirs):
        logger.info(f"Processing: {result_path.name}")
        
        if process_single_directory(result_path, n_samples, variables, force, verbose, force_baselines):
            processed += 1
            logger.info(f"  ✓ Created plots for {result_path.name}")
        else:
            if not force and (result_path / "samples").exists():
                skipped += 1
                logger.info(f"  - Skipped {result_path.name} (plots exist, use --force)")
            else:
                failed += 1
                logger.error(f"  ✗ Failed to create plots for {result_path.name}")
    
    logger.info(f"\nSummary:")
    logger.info(f"  Processed: {processed}")
    logger.info(f"  Skipped: {skipped}")
    logger.info(f"  Failed: {failed}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Generate spatial forecast demonstration plots"
    )
    parser.add_argument(
        "result_directory",
        type=str,
        nargs='?',
        default="./results",
        help="Path to single result directory OR results root directory (default: ./results)"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1,
        help="Number of test samples to process (default: 1)"
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        default=DEFAULT_VARIABLE_NAMES,
        help="Variables to process"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all result directories in the specified path"
    )
    parser.add_argument(
        "--force",
        action="store_true", 
        help="Regenerate plots even if they already exist"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--force-baselines",
        action="store_true",
        help="Force regeneration of baseline method plots even if they exist"
    )
    
    args = parser.parse_args()
    
    result_path = Path(args.result_directory)
    
    if args.all:
        # Batch processing mode - process all directories
        process_all_results(
            root_path=result_path,
            n_samples=args.n_samples,
            variables=args.variables,
            force=args.force,
            verbose=args.verbose,
            force_baselines=args.force_baselines
        )
        return
    
    # Single directory mode
    if not result_path.exists():
        print(f"Error: Directory not found: {result_path}")
        sys.exit(1)
    
    # Check if this looks like a results root directory
    if result_path.is_dir() and not (result_path / "config.json").exists():
        subdirs_with_data = []
        for item in result_path.iterdir():
            if item.is_dir() and (item / "config.json").exists():
                subdirs_with_data.append(item)
        
        if len(subdirs_with_data) > 1:
            print(f"Found {len(subdirs_with_data)} result directories.")
            print("Use --all to process all directories, or specify a single directory path.")
            print("Available directories:")
            for subdir in sorted(subdirs_with_data):
                print(f"  {subdir.name}")
            sys.exit(1)
    
    # Process single directory
    success = process_single_directory(
        result_path=result_path,
        n_samples=args.n_samples,
        variables=args.variables,
        force=args.force,
        verbose=args.verbose,
        force_baselines=args.force_baselines
    )
    
    if not success:
        sys.exit(1)


def generate_forecast_demonstrations(result_path, output_dir, n_samples=1, variables=None, logger=None, force_baselines=False):
    """Generate forecast demonstration plots for a result directory."""
    if variables is None:
        variables = DEFAULT_VARIABLE_NAMES
    if logger is None:
        logger = setup_logging()
        
    try:
        # Load configuration
        config_path = result_path / "config.json"
        if not config_path.exists():
            logger.error(f"Config file not found: {config_path}")
            return False
        
        config = Config(config_path)
        logger.info(f"Loaded config: {config.name}")
        
        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Load model and test data
        logger.info("Loading model and test data...")
        loader, model, _, _, _ = load_model_and_test_data(config, device, 42)
        model.eval()
        
        # Convert loader to list for easier indexing
        test_data = []
        for batch in loader:
            prev_fields, target_fields, static_fields = batch
            for i in range(prev_fields.size(0)):
                test_data.append((
                    prev_fields[i].cpu(),
                    target_fields[i].cpu(), 
                    static_fields[i].cpu()
                ))
        
        logger.info(f"Loaded {len(test_data)} test samples")
        
        # Generate forecasts
        forecasts_dict, targets = generate_sample_forecasts(
            result_path, config, model, test_data, device, logger, n_samples
        )
        
        # Create individual variable spatial plots
        for sample_idx in range(n_samples):
            for var_idx, var_name in enumerate(variables):
                plot_variable_spatial_analysis(
                    forecasts_dict, targets, var_idx, var_name,
                    sample_idx, output_dir, logger
                )
        
        # Create spatial metric comparisons
        plot_spatial_crps_comparison(
            forecasts_dict, targets, variables, output_dir, logger
        )
        
        # Create 2x2 spatial analysis plots
        plot_2x2_spatial_analysis(
            forecasts_dict, targets, variables, output_dir, logger, config, n_samples
        )
        
        # Create baseline ensemble samples (only create once, in global results directory)
        baseline_output_dir = Path("results/baseline_method_plots")
        baseline_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Only generate baseline plots if they don't exist or if forced
        baseline_plots_exist = any(baseline_output_dir.glob("*.pdf"))
        if not baseline_plots_exist or force_baselines:
            logger.info("Creating baseline method sample plots...")
            create_baseline_method_plots(
                forecasts_dict, targets, variables, baseline_output_dir, logger, config
            )
        else:
            logger.info("Baseline method plots already exist, skipping...")
        
        # Also create baseline ensemble samples in experiment directory
        create_baseline_ensemble_samples(
            forecasts_dict, targets, variables, output_dir, logger
        )
        
        logger.info("✓ Spatial forecast demonstration completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Failed to generate demonstrations: {e}")
        return False


if __name__ == "__main__":
    main()
