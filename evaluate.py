import argparse
import datetime as dt
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
from matplotlib.patches import Rectangle
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

# Set academic plotting style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Times New Roman',
    'mathtext.it': 'Times New Roman:italic',
    'mathtext.bf': 'Times New Roman:bold',
})

def _utc_timestamp(directory_name: str) -> dt.datetime | None:
    match = re.compile(r"_(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)$").search(
        directory_name
    )
    if match is None:
        return None
    try:
        return dt.datetime.strptime(match.group(1), "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return None


def _latest_result_directory(base_path: Path) -> Path:
    candidates: Iterable[Path] = (d for d in base_path.iterdir() if d.is_dir())
    try:
        latest_path = max(
            candidates,
            key=lambda p: _utc_timestamp(p.name)
            or dt.datetime.utcfromtimestamp(p.stat().st_mtime),
        )
    except ValueError:
        raise FileNotFoundError(f"No result directories found under {base_path}")
    return latest_path


def _setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(__name__)


def _load_results(directory: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    results_file = directory / "results.npz"
    if not results_file.exists():
        raise FileNotFoundError(results_file)
    config_file = directory / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(config_file)

    results = dict(np.load(results_file, allow_pickle=True))
    with open(config_file, "r", encoding="utf-8") as handle:
        config_data = json.load(handle)
    return results, config_data


def print_posterior_summary(results: Dict[str, Any], config: Dict[str, Any], 
                          output_path: Path, logger: logging.Logger) -> None:
    """Print comprehensive posterior summary statistics."""
    logger.info("=== Posterior Summary Statistics ===")
    
    posterior_samples = results["posterior_samples"]
    variable_names = config.get("variable_names", [f"var_{i}" for i in range(posterior_samples.shape[1])])
    
    # Basic statistics
    means = np.mean(posterior_samples, axis=0).flatten()
    stds = np.std(posterior_samples, axis=0).flatten()
    medians = np.median(posterior_samples, axis=0).flatten()
    q25 = np.percentile(posterior_samples, 25, axis=0).flatten()
    q75 = np.percentile(posterior_samples, 75, axis=0).flatten()
    
    logger.info(f"Number of samples: {posterior_samples.shape[0]}")
    logger.info(f"Number of variables: {len(variable_names)}")
    
    for i, var_name in enumerate(variable_names):
        logger.info(f"{var_name:>6}: μ={means[i]:.4f} ± {stds[i]:.4f} "
                   f"[{q25[i]:.4f}, {q75[i]:.4f}] med={medians[i]:.4f}")
    
    # Effective sample size estimation
    def autocorr_func_1d(x, norm=True):
        x = np.atleast_1d(x).astype(np.float64)
        if len(x.shape) != 1:
            raise ValueError("invalid dimensions for 1D autocorrelation function")
        n = len(x)
        x = x - np.mean(x)
        c0 = np.dot(x, x) / float(n)
        acf = np.correlate(x, x, 'full')[n-1:]
        acf = acf / float(n - np.arange(len(acf)))
        acf = acf / acf[0] if norm else acf / c0
        return acf

    def auto_window(taus, c):
        m = np.arange(len(taus)) < c * taus
        if np.any(m):
            return np.argmin(m)
        return len(taus) - 1

    def autocorr_new(y, c=5.0):
        f = autocorr_func_1d(y)
        taus = 2.0 * np.cumsum(f) - 1.0
        window = auto_window(taus, c)
        return taus[window]

    logger.info("\n=== Convergence Diagnostics ===")
    for i, var_name in enumerate(variable_names):
        chain = posterior_samples[:, i, 0]
        try:
            tau = autocorr_new(chain)
            eff_samples = len(chain) / (2 * tau + 1)
            logger.info(f"{var_name:>6}: τ={tau:.2f}, N_eff={eff_samples:.0f}")
        except:
            logger.info(f"{var_name:>6}: τ computation failed")


def generate_parameter_traces(results: Dict[str, Any], config: Dict[str, Any], 
                            output_path: Path, logger: logging.Logger) -> None:
    """Generate individual parameter trace plots."""
    logger.info("Generating parameter trace plots...")
    
    posterior_samples = results["posterior_samples"]
    variable_names = config.get("variable_names", [f"var_{i}" for i in range(posterior_samples.shape[1])])
    n_vars = len(variable_names)
    
    colors = sns.color_palette("husl", n_vars)
    
    # Create individual plots for each variable
    for i, (var_name, color) in enumerate(zip(variable_names, colors)):
        fig, ax = plt.subplots(1, 1, figsize=(10, 4), constrained_layout=True)
        
        chain = posterior_samples[:, i, 0]
        ax.plot(chain, color=color, linewidth=1.0, alpha=0.8)
        ax.set_ylabel(f'α({var_name})', fontsize=12, fontweight='bold')
        ax.set_xlabel('Iteration', fontsize=11)
        ax.set_title(f'Parameter Trace: {var_name.upper()}', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add running mean
        window = max(10, len(chain) // 20)
        running_mean = pd.Series(chain).rolling(window, center=True).mean()
        ax.plot(running_mean, color='red', linewidth=2.0, alpha=0.8, 
               linestyle='--', label='Running mean')
        ax.legend(frameon=False, loc='upper right')
        
        # Add statistics text
        stats_text = f'Mean: {chain.mean():.3f}\nStd: {chain.std():.3f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
               facecolor="white", alpha=0.8), fontsize=10)
        
        plt.savefig(output_path / f'trace_{var_name.lower()}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Also create a combined overview plot
    fig, axes = plt.subplots(n_vars, 1, figsize=(10, 2.5 * n_vars), 
                            constrained_layout=True)
    if n_vars == 1:
        axes = [axes]
    
    for i, (ax, var_name, color) in enumerate(zip(axes, variable_names, colors)):
        chain = posterior_samples[:, i, 0]
        ax.plot(chain, color=color, linewidth=0.8, alpha=0.8)
        ax.set_ylabel(f'α({var_name})', fontsize=11)
        ax.set_xlabel('Iteration' if i == n_vars-1 else '')
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add running mean
        window = max(10, len(chain) // 20)
        running_mean = pd.Series(chain).rolling(window, center=True).mean()
        ax.plot(running_mean, color='red', linewidth=1.5, alpha=0.7, 
               linestyle='--', label='Running mean')
        if i == 0:
            ax.legend(frameon=False)
    
    plt.suptitle('Parameter Traces Overview', fontsize=14, fontweight='bold')
    plt.savefig(output_path / 'parameter_traces_overview.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_rank_histograms(results: Dict[str, Any], config: Dict[str, Any], 
                           output_path: Path, logger: logging.Logger) -> None:
    """Generate individual rank histograms for ensemble calibration assessment."""
    logger.info("Generating rank histogram plots...")
    
    rank_histograms = results.get("rank_histograms", [])
    variable_names = config.get("variable_names", [f"var_{i}" for i in range(len(rank_histograms))])
    ensemble_size = config.get("ensemble_size", 50)
    
    if len(rank_histograms) == 0:
        logger.warning("No rank histogram data found")
        return
    
    colors = sns.color_palette("Set2", len(variable_names))
    
    # Create individual rank histogram plots
    for i, (var_name, color) in enumerate(zip(variable_names, colors)):
        fig, ax = plt.subplots(1, 1, figsize=(8, 5), constrained_layout=True)
        
        # The rank_histograms now contain proper integer ranks (0 to ensemble_size)
        raw_ranks = np.array(rank_histograms[i])
        
        if len(raw_ranks) == 0:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14)
            ax.set_title(f'Rank Histogram: {var_name.upper()}')
            plt.savefig(output_path / f'rank_histogram_{var_name.lower()}.png', dpi=300, bbox_inches='tight')
            plt.close()
            continue
        
        # Sample a subset for computational efficiency if needed
        max_samples = 50000  # Limit for visualization
        if len(raw_ranks) > max_samples:
            indices = np.random.choice(len(raw_ranks), max_samples, replace=False)
            integer_ranks = raw_ranks[indices]
        else:
            integer_ranks = raw_ranks
            
        # Ensure ranks are in valid range (should already be correct from core function)
        integer_ranks = np.clip(integer_ranks, 0, ensemble_size).astype(int)
        
        # Create histogram
        bins = np.arange(0, ensemble_size + 2) - 0.5
        counts, _ = np.histogram(integer_ranks, bins=bins)
        
        # Plot histogram bars
        bin_centers = np.arange(0, ensemble_size + 1)
        ax.bar(bin_centers, counts, color=color, alpha=0.7, edgecolor='white', 
               linewidth=0.5)
        
        # Expected uniform line
        expected = len(integer_ranks) / (ensemble_size + 1)
        ax.axhline(y=expected, color='red', linestyle='--', linewidth=2, 
                  alpha=0.8, label='Uniform expectation')
        
        ax.set_title(f'Rank Histogram: {var_name.upper()}', fontsize=13, fontweight='bold')
        ax.set_xlabel('Rank', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add calibration statistics
        chi2_stat = np.sum((counts - expected)**2 / expected)
        p_bins = len(bin_centers)
        stats_text = f'χ² = {chi2_stat:.2f}\nBins: {p_bins}\nSamples: {len(integer_ranks)}'
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8), 
               fontsize=10)
        
        plt.savefig(output_path / f'rank_histogram_{var_name.lower()}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Also create a combined overview plot
    n_vars = len(variable_names)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    axes = axes.flatten()
    
    for i, (var_name, color) in enumerate(zip(variable_names, colors)):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # The rank_histograms now contain proper integer ranks (0 to ensemble_size)
        raw_ranks = np.array(rank_histograms[i])
        
        if len(raw_ranks) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                   transform=ax.transAxes)
            ax.set_title(var_name)
            continue
        
        # Sample a subset for computational efficiency if needed
        max_samples = 50000  # Limit for visualization
        if len(raw_ranks) > max_samples:
            indices = np.random.choice(len(raw_ranks), max_samples, replace=False)
            integer_ranks = raw_ranks[indices]
        else:
            integer_ranks = raw_ranks
            
        # Ensure ranks are in valid range (should already be correct from core function)
        integer_ranks = np.clip(integer_ranks, 0, ensemble_size).astype(int)
        
        # Create histogram
        bins = np.arange(0, ensemble_size + 2) - 0.5
        counts, _ = np.histogram(integer_ranks, bins=bins)
        
        # Plot histogram bars
        bin_centers = np.arange(0, ensemble_size + 1)
        ax.bar(bin_centers, counts, color=color, alpha=0.7, edgecolor='white', 
               linewidth=0.5)
        
        # Expected uniform line
        expected = len(integer_ranks) / (ensemble_size + 1)
        ax.axhline(y=expected, color='red', linestyle='--', linewidth=2, 
                  alpha=0.8, label='Uniform expectation')
        
        # Confidence bands for uniform distribution
        # 95% confidence interval for multinomial distribution
        se = np.sqrt(expected * (1 - 1/(ensemble_size + 1)))
        ci = 1.96 * se
        ax.axhline(y=expected + ci, color='red', linestyle=':', alpha=0.6)
        ax.axhline(y=expected - ci, color='red', linestyle=':', alpha=0.6)
        
        # Chi-square test
        expected_counts = np.full(ensemble_size + 1, expected)
        chi2_stat = np.sum((counts - expected_counts)**2 / expected_counts)
        p_value = 1 - stats.chi2.cdf(chi2_stat, ensemble_size)
        
        ax.set_title(f'{var_name} (χ²={chi2_stat:.2f}, p={p_value:.3f})')
        ax.set_xlabel('Rank')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Set reasonable y-limits
        ax.set_ylim(0, max(counts) * 1.1)
        
        if i == 0:
            ax.legend(frameon=False)
    
    # Remove unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].remove()
    
    plt.suptitle('Rank Histograms for Ensemble Calibration', fontsize=14, fontweight='bold')
    plt.savefig(output_path / 'rank_histograms.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_score_evolution(results: Dict[str, Any], config: Dict[str, Any], 
                           output_path: Path, logger: logging.Logger) -> None:
    """Generate score evolution plots."""
    logger.info("Generating score evolution plots...")
    
    step_mean_scores = results.get("step_mean_scores", [])
    score_function = config.get("score_function", "CRPS").upper()
    
    if len(step_mean_scores) == 0:
        logger.warning("No score evolution data found")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)
    
    # Main score evolution
    ax.plot(step_mean_scores, color='steelblue', linewidth=2, alpha=0.8)
    
    # Add smoothed trend
    if len(step_mean_scores) > 10:
        window = max(5, len(step_mean_scores) // 20)
        smoothed = pd.Series(step_mean_scores).rolling(window, center=True).mean()
        ax.plot(smoothed, color='red', linewidth=2, linestyle='--', 
               alpha=0.8, label='Trend')
        ax.legend(frameon=False)
    
    ax.set_xlabel('Gibbs Step')
    ax.set_ylabel(f'Mean {score_function} Score')
    ax.set_title(f'{score_function} Score Evolution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add burn-in indicator if applicable
    burnin_pct = 0.2
    burnin_step = int(len(step_mean_scores) * burnin_pct)
    ax.axvline(x=burnin_step, color='orange', linestyle=':', alpha=0.7, 
              label=f'Burn-in ({burnin_pct:.0%})')
    ax.legend(frameon=False)
    
    plt.savefig(output_path / 'score_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_posterior_diagnostics(results: Dict[str, Any], config: Dict[str, Any], 
                                 output_path: Path, logger: logging.Logger) -> None:
    """Generate comprehensive posterior diagnostics."""
    logger.info("Generating posterior diagnostic plots...")
    
    posterior_samples = results["posterior_samples"]
    variable_names = config.get("variable_names", [f"var_{i}" for i in range(posterior_samples.shape[1])])
    n_vars = len(variable_names)
    
    # Corner plot for joint posteriors
    fig, axes = plt.subplots(n_vars, n_vars, figsize=(12, 12), constrained_layout=True)
    
    colors = sns.color_palette("husl", n_vars)
    
    for i in range(n_vars):
        for j in range(n_vars):
            ax = axes[i, j]
            
            if i == j:  # Diagonal: marginal distributions
                chain = posterior_samples[:, i, 0]
                ax.hist(chain, bins=30, density=True, alpha=0.7, 
                       color=colors[i], edgecolor='white')
                ax.set_ylabel('Density' if j == 0 else '')
                ax.set_title(variable_names[i] if i == 0 else '')
                
            elif i > j:  # Lower triangle: scatter plots
                x_chain = posterior_samples[:, j, 0]
                y_chain = posterior_samples[:, i, 0]
                ax.scatter(x_chain, y_chain, alpha=0.5, s=2, color=colors[i])
                ax.set_xlabel(variable_names[j] if i == n_vars-1 else '')
                ax.set_ylabel(variable_names[i] if j == 0 else '')
                
            else:  # Upper triangle: remove
                ax.remove()
            
            if i <= j and (i, j) != (0, 0):
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
    
    plt.suptitle('Posterior Marginals and Correlations', fontsize=14, fontweight='bold')
    plt.savefig(output_path / 'posterior_diagnostics.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_ensemble_spatial_plots(results: Dict[str, Any], config: Dict[str, Any], 
                                   output_path: Path, logger: logging.Logger) -> None:
    """Generate operational ensemble forecast spatial plots using RFP method."""
    logger.info("Generating operational RFP ensemble forecast plots...")
    
    try:
        _generate_operational_rfp_forecasts(results, config, output_path, logger)
    except Exception as e:
        logger.warning(f"Failed to generate operational forecasts: {e}")
        logger.info("Falling back to framework explanation...")
        _generate_ensemble_plot_explanation(results, config, output_path, logger)


def _generate_operational_rfp_forecasts(results: Dict[str, Any], config: Dict[str, Any], 
                                       output_path: Path, logger: logging.Logger) -> None:
    """Generate operational RFP ensemble forecasts using trained model and real ERA5 data."""
    import torch
    import pandas as pd
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from core.algorithm import generate_joint_rfp
    from core.config import Config
    from core.io_utils import load_model_and_test_data
    
    logger.info("Loading trained model and ERA5 data...")
    
    # Create temporary config file and Config object
    temp_config_path = output_path / 'temp_config.json'
    with open(temp_config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    temp_config = Config(temp_config_path)
    
    # Set required paths 
    temp_config.data_directory = Path('./data')
    temp_config.result_directory = output_path
    temp_config.sample_size = 5  # Use small sample for demonstration
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model and test data
    loader, model, lat, lon, _ = load_model_and_test_data(temp_config, device, 42)
    
    # Load posterior samples
    posterior_samples = results["posterior_samples"]
    posterior_mean = posterior_samples.mean(axis=0)
    variable_names = config.get("variable_names", ["z500", "t850", "t2m", "u10", "v10"])
    
    logger.info(f"Using posterior mean: {posterior_mean.squeeze()}")
    
    # Load reference data for RFP perturbations
    reference_path = temp_config.data_directory / "z500_t850_t2m_u10_v10_1979-2018_5.625deg.npy"
    reference_shape = (len(pd.date_range('1979-01-01', '2018-12-31 23:00', freq='1h')), 
                      len(variable_names), len(lat), len(lon))
    reference_data = np.memmap(reference_path, dtype='float32', mode='r', shape=reference_shape)
    reference_tensor = torch.from_numpy(reference_data[:1000].copy()).to(device)  # Use subset and copy
    
    # Get a sample batch from the loader
    sample_batch = next(iter(loader))
    previous_fields, current_fields, lead_time = sample_batch
    previous_fields = previous_fields.to(device)
    current_fields = current_fields.to(device)
    
    logger.info(f"Sample shapes - Previous: {previous_fields.shape}, Current: {current_fields.shape}")
    
    # Create ensemble forecast
    ensemble_size = 50
    batch_size = current_fields.shape[0]
    n_vars = current_fields.shape[1]
    
    # Generate RFP perturbations using posterior mean
    alpha_matrix = torch.tensor(posterior_mean.squeeze(), device=device).unsqueeze(0)
    
    generator = torch.Generator(device=device)
    generator.manual_seed(42)
    
    logger.info("Generating RFP perturbations...")
    perturb = generate_joint_rfp(
        reference_tensor=reference_tensor,
        alpha_matrix=alpha_matrix,
        batch_size=batch_size,
        ensemble_size=ensemble_size,
        device=device,
        generator=generator,
        eps_energy=1e-6
    )
    
    # Apply perturbations to create ensemble inputs
    current_expanded = current_fields.unsqueeze(1).expand(-1, ensemble_size, -1, -1, -1)
    ensemble_inputs = current_expanded + perturb[0]  # [batch, ensemble, vars, H, W]
    
    # Create perturbed previous fields for model input
    previous_expanded = previous_fields.unsqueeze(1).expand(-1, ensemble_size, -1, -1, -1).clone()
    n_met_vars = n_vars * 2  # current + past meteorological variables
    previous_expanded[:, :, :n_met_vars] += perturb[0].repeat(1, 1, 2, 1, 1)  # Perturb met vars only
    
    logger.info("Running ensemble through trained model...")
    
    # Reshape for model forward pass
    ensemble_model_input = previous_expanded.view(batch_size * ensemble_size, -1, len(lat), len(lon))
    time_input = lead_time.expand(ensemble_size).view(-1, 1)
    
    # Forward pass through model
    with torch.no_grad():
        ensemble_forecasts = model(ensemble_model_input, time_input)
    
    # Reshape back to ensemble format
    ensemble_forecasts = ensemble_forecasts.view(batch_size, ensemble_size, n_vars, len(lat), len(lon))
    
    # Load normalization factors for denormalization
    with open(temp_config.data_directory / "norm_factors.json", 'r') as f:
        norm_stats = json.load(f)
    
    mean_data = torch.tensor([norm_stats[var]['mean'] for var in variable_names], device=device)
    std_data = torch.tensor([norm_stats[var]['std'] for var in variable_names], device=device)
    
    # Denormalize forecasts and truth
    ensemble_denorm = ensemble_forecasts * std_data[None, None, :, None, None] + mean_data[None, None, :, None, None]
    truth_denorm = current_fields * std_data[None, :, None, None] + mean_data[None, :, None, None]
    
    # Compute ensemble statistics
    ensemble_mean = ensemble_denorm.mean(dim=1)  # [batch, vars, H, W]
    ensemble_std = ensemble_denorm.std(dim=1)    # [batch, vars, H, W]
    
    # Select variable for plotting (T2M)
    var_idx = variable_names.index('t2m') if 't2m' in variable_names else 0
    var_name = variable_names[var_idx]
    
    # Convert to numpy for plotting
    truth = truth_denorm[0, var_idx].cpu().numpy()
    ens_mean = ensemble_mean[0, var_idx].cpu().numpy()
    ens_std = ensemble_std[0, var_idx].cpu().numpy()
    member1 = ensemble_denorm[0, 0, var_idx].cpu().numpy()
    member2 = ensemble_denorm[0, 1, var_idx].cpu().numpy()
    
    logger.info(f"Forecast statistics - Mean: {ens_mean.mean():.2f}, Std: {ens_std.mean():.2f}")
    
    # Set appropriate color ranges for T2M
    if var_name == 't2m':
        cmap = 'RdYlBu_r'
        units = 'K'
        vmin = min(truth.min(), ens_mean.min()) - 2
        vmax = max(truth.max(), ens_mean.max()) + 2
    else:
        cmap = 'RdBu_r'
        units = ''
        vmin = min(truth.min(), ens_mean.min())
        vmax = max(truth.max(), ens_mean.max())
    
    # Create individual spatial plots
    spatial_plots_data = [
        (truth, 'truth_target', f'Truth/Target ({var_name.upper()})', vmin, vmax, cmap),
        (ens_mean, 'ensemble_mean', f'Ensemble Mean Forecast ({var_name.upper()})', vmin, vmax, cmap),
        (ens_std, 'ensemble_spread', f'Ensemble Spread ({var_name.upper()})', 0, ens_std.max(), 'Reds'),
        (member1, 'member_1', f'Ensemble Member #1 ({var_name.upper()})', vmin, vmax, cmap),
        (member2, 'member_2', f'Ensemble Member #2 ({var_name.upper()})', vmin, vmax, cmap)
    ]
    
    for data, filename, title, v_min, v_max, colormap in spatial_plots_data:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8), 
                              subplot_kw={'projection': ccrs.PlateCarree()})
        
        # Add map features
        ax.add_feature(cfeature.COASTLINE, alpha=0.8)
        ax.add_feature(cfeature.BORDERS, alpha=0.5)
        ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
        ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.3)
        
        # Plot data
        im = ax.pcolormesh(lon, lat, data, transform=ccrs.PlateCarree(),
                          cmap=colormap, vmin=v_min, vmax=v_max, shading='auto')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_global()
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.08, shrink=0.8)
        if 'spread' in filename:
            cbar.set_label(f'Std Dev ({units})' if units else 'Std Dev', fontsize=12)
        else:
            cbar.set_label(f'{units}' if units else '', fontsize=12)
            
        # Add statistics text
        if 'spread' not in filename:
            stats_text = f'Min: {data.min():.1f}\nMax: {data.max():.1f}\nMean: {data.mean():.1f}'
        else:
            stats_text = f'Min: {data.min():.2f}\nMax: {data.max():.2f}\nMean: {data.mean():.2f}'
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
               facecolor="white", alpha=0.9), fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_path / f'spatial_{filename}_{var_name.lower()}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create parameter distribution plot with KDE smoothing
    from scipy.stats import gaussian_kde
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)
    
    colors = sns.color_palette("husl", len(variable_names))
    for v, (var_name_loop, color) in enumerate(zip(variable_names, colors)):
        chain = posterior_samples[:, v, 0]
        
        # Create histogram for background
        ax.hist(chain, bins=20, alpha=0.3, color=color, density=True, histtype='stepfilled')
        
        # Add KDE smooth curve
        if len(chain) > 1:
            kde = gaussian_kde(chain)
            x_range = np.linspace(chain.min() - 0.1 * (chain.max() - chain.min()), 
                                 chain.max() + 0.1 * (chain.max() - chain.min()), 200)
            kde_values = kde(x_range)
            ax.plot(x_range, kde_values, color=color, linewidth=2.5, 
                   label=var_name_loop, alpha=0.9)
        
        # Mark posterior mean
        ax.axvline(posterior_mean[v, 0], color=color, linestyle='--', 
                  alpha=0.8, linewidth=2)
    
    ax.set_title('ABC Posterior Parameters Distribution (KDE Smoothed)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Parameter Value (α)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.legend(frameon=False, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.savefig(output_path / 'abc_posterior_parameters.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a combined overview plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), 
                           subplot_kw={'projection': ccrs.PlateCarree()})
    
    plots_data = [
        (truth, 'Truth (Target)', vmin, vmax),
        (ens_mean, 'Ensemble Mean', vmin, vmax),
        (ens_std, 'Ensemble Spread', 0, ens_std.max()),
        (member1, 'Member #1', vmin, vmax),
        (member2, 'Member #2', vmin, vmax)
    ]
    
    for i, (data, title, v_min, v_max) in enumerate(plots_data):
        row, col = i // 3, i % 3
        ax = axes[row, col]
        
        # Add map features
        ax.add_feature(cfeature.COASTLINE, alpha=0.8)
        ax.add_feature(cfeature.BORDERS, alpha=0.5)
        ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
        ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.3)
        
        # Plot data
        if i == 2:  # Std dev uses different colormap
            im = ax.pcolormesh(lon, lat, data, transform=ccrs.PlateCarree(),
                             cmap='Reds', vmin=v_min, vmax=v_max, shading='auto')
        else:
            im = ax.pcolormesh(lon, lat, data, transform=ccrs.PlateCarree(),
                             cmap=cmap, vmin=v_min, vmax=v_max, shading='auto')
        
        ax.set_title(f'{title}', fontsize=12, fontweight='bold')
        ax.set_global()
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
        if i == 2:
            cbar.set_label('Std Dev (K)' if var_name == 't2m' else 'Std Dev')
        else:
            cbar.set_label(f'{units}' if units else '')
    
    # Parameter distribution in last subplot
    ax = axes[1, 2]
    ax.remove()
    ax = fig.add_subplot(2, 3, 6)
    
    for v, (var_name_loop, color) in enumerate(zip(variable_names, colors)):
        chain = posterior_samples[:, v, 0]
        ax.hist(chain, bins=15, alpha=0.7, color=color, 
               label=var_name_loop, density=True, histtype='stepfilled')
        # Mark posterior mean
        ax.axvline(posterior_mean[v, 0], color=color, linestyle='--', alpha=0.8)
    
    ax.set_title('ABC Posterior Parameters', fontweight='bold')
    ax.set_xlabel('Parameter Value (α)')
    ax.set_ylabel('Density')
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'RFP Ensemble Forecasts Overview - {var_name.upper()}', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path / 'ensemble_spatial_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Generated operational RFP ensemble forecast plots using trained model and ERA5 data")


def _generate_ensemble_plot_explanation(results: Dict[str, Any], config: Dict[str, Any], 
                                       output_path: Path, logger: logging.Logger) -> None:
    """Generate a professional explanation of ensemble forecast visualization."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    
    # Get basic information
    posterior_samples = results["posterior_samples"]
    variable_names = config.get("variable_names", ["z500", "t850", "t2m", "u10", "v10"])
    n_vars = len(variable_names)
    n_samples = posterior_samples.shape[0]
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), constrained_layout=True)
    
    # Create informative subplot explanations
    explanations = [
        ("Truth/Verification", "Ground truth observations\nfrom reanalysis data"),
        ("Ensemble Mean", "Average of all ensemble\nmembers from ABC posterior"),
        ("Ensemble Spread", "Standard deviation showing\nforecast uncertainty"),
        ("Individual Member #1", "Single ensemble member\nfrom posterior sample α₁"),
        ("Individual Member #2", "Single ensemble member\nfrom posterior sample α₂"),  
        ("Parameter Distribution", f"ABC posterior samples\n{n_samples} total samples")
    ]
    
    colors = sns.color_palette("husl", n_vars)
    
    for i, (title, explanation) in enumerate(explanations):
        ax = axes[i//3, i%3]
        
        if i < 5:  # Spatial plot placeholders
            # Add a simple cartopy map outline
            ax.remove()
            ax = fig.add_subplot(2, 3, i+1, projection=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE, alpha=0.7)
            ax.add_feature(cfeature.BORDERS, alpha=0.5)
            ax.set_global()
            
            # Add explanatory text
            ax.text(0.5, 0.5, f'{title}\n\n{explanation}', 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
            
        else:  # Parameter distribution plot
            # Show actual posterior distributions
            for v, (var_name, color) in enumerate(zip(variable_names, colors)):
                chain = posterior_samples[:, v, 0]
                ax.hist(chain, bins=20, alpha=0.7, color=color, 
                       label=var_name, density=True, histtype='stepfilled')
            
            ax.set_title('ABC Posterior Parameters', fontweight='bold', fontsize=14)
            ax.set_xlabel('Parameter Value (α)')
            ax.set_ylabel('Density')
            ax.legend(frameon=False)
            ax.grid(True, alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
    
    # Add overall title and information
    plt.suptitle('Ensemble Forecast Spatial Visualization Framework\n' +
                f'ABC-RFP Method with {n_samples} Posterior Samples', 
                fontsize=16, fontweight='bold')
    
    # Add methodology text at bottom
    fig.text(0.5, 0.02, 
            'Real implementation would use: (1) Trained neural network model, ' +
            '(2) ABC posterior samples α to generate RFP perturbations, ' +
            '(3) Forward pass through model to create ensemble forecasts, ' +
            '(4) Cartopy visualization with proper geographic coordinates',
            ha='center', va='bottom', fontsize=10, style='italic',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.3))
    
    plt.savefig(output_path / 'ensemble_spatial_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Generated ensemble spatial plot framework visualization")


def generate_method_comparison(results: Dict[str, Any], config: Dict[str, Any], 
                             output_path: Path, logger: logging.Logger) -> None:
    """Generate comparison plots with baseline methods if evaluation data exists."""
    logger.info("Generating method comparison plots...")
    
    eval_dir = output_path / "evaluation"
    if not eval_dir.exists():
        logger.warning("No evaluation directory found - skipping method comparison")
        return
    
    try:
        # Load evaluation results
        metrics_file = eval_dir / "verification_metrics.npy"
        pit_file = eval_dir / "pit_histograms.npy" 
        bias_file = eval_dir / "bias_rmse_spread.npy"
        
        if metrics_file.exists():
            metrics = np.load(metrics_file, allow_pickle=True).item()
            _plot_method_metrics_comparison(metrics, output_path, logger)
            
        if pit_file.exists() and bias_file.exists():
            pit_data = np.load(pit_file, allow_pickle=True).item()
            bias_data = np.load(bias_file, allow_pickle=True).item()
            variable_names = config.get("variable_names", ["z500", "t850", "t2m", "u10", "v10"])
            
            _plot_pit_comparison(pit_data, output_path, logger)
            _plot_bias_rmse_spread_matrices(bias_data, variable_names, output_path, logger)
            
    except Exception as e:
        logger.warning(f"Could not generate method comparison plots: {e}")
        

def _plot_method_metrics_comparison(metrics: Dict, output_path: Path, logger: logging.Logger) -> None:
    """Plot comparison of different methods across metrics."""
    methods = list(metrics.keys())
    metric_names = list(next(iter(metrics.values())).keys())
    
    # Create comparison plot
    n_metrics = len(metric_names)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), constrained_layout=True)
    axes = axes.flatten()
    
    colors = sns.color_palette("Set2", len(methods))
    
    for i, metric in enumerate(metric_names[:min(6, n_metrics)]):
        ax = axes[i]
        
        values = [metrics[method][metric] for method in methods]
        bars = ax.bar(methods, values, color=colors, alpha=0.7, edgecolor='white')
        
        ax.set_title(f'{metric.upper()}', fontweight='bold')
        ax.set_ylabel('Score')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Remove unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].remove()
    
    plt.suptitle('Method Comparison - Verification Metrics', fontsize=16, fontweight='bold')
    plt.savefig(output_path / 'method_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def _plot_pit_comparison(pit_data: Dict, output_path: Path, logger: logging.Logger) -> None:
    """Plot PIT histograms for different methods."""
    methods = list(pit_data.keys())
    n_methods = len(methods)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), constrained_layout=True)
    axes = axes.flatten()
    
    colors = sns.color_palette("Set2", n_methods)
    
    for i, (method, color) in enumerate(zip(methods[:min(6, n_methods)], colors)):
        ax = axes[i]
        counts = pit_data[method]
        probs = counts / counts.sum()
        
        bars = ax.bar(range(len(probs)), probs, color=color, alpha=0.7, edgecolor='white')
        
        # Expected uniform line
        expected = 1.0 / len(probs)
        ax.axhline(y=expected, color='red', linestyle='--', linewidth=2, alpha=0.8)
        
        ax.set_title(f'{method.replace("_", " ").title()}', fontweight='bold')
        ax.set_xlabel('Rank')
        ax.set_ylabel('Probability')
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Remove unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].remove()
    
    plt.suptitle('PIT Histograms - Method Comparison', fontsize=16, fontweight='bold')
    plt.savefig(output_path / 'pit_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def _plot_bias_rmse_spread_matrices(data: Dict, variable_names: list, output_path: Path, logger: logging.Logger) -> None:
    """Plot bias, RMSE, and spread matrices."""
    bias_data = data['bias']
    rmse_data = data['rmse'] 
    spread_data = data['spread']
    
    methods = list(bias_data.keys())
    n_vars = len(variable_names)
    
    # Create matrix plots
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)
    
    # Bias matrix
    bias_matrix = np.array([bias_data[m].numpy() for m in methods])
    im1 = axes[0].imshow(bias_matrix, cmap='RdBu_r', aspect='auto')
    axes[0].set_title('Bias', fontweight='bold')
    axes[0].set_yticks(range(len(methods)))
    axes[0].set_yticklabels([m.replace('_', ' ').title() for m in methods])
    axes[0].set_xticks(range(n_vars))
    axes[0].set_xticklabels(variable_names)
    plt.colorbar(im1, ax=axes[0])
    
    # RMSE matrix
    rmse_matrix = np.array([rmse_data[m].numpy() for m in methods])
    im2 = axes[1].imshow(rmse_matrix, cmap='viridis', aspect='auto')
    axes[1].set_title('RMSE', fontweight='bold')
    axes[1].set_yticks(range(len(methods)))
    axes[1].set_yticklabels([m.replace('_', ' ').title() for m in methods])
    axes[1].set_xticks(range(n_vars))
    axes[1].set_xticklabels(variable_names)
    plt.colorbar(im2, ax=axes[1])
    
    # Spread matrix
    spread_matrix = np.array([spread_data[m].numpy() for m in methods])
    im3 = axes[2].imshow(spread_matrix, cmap='plasma', aspect='auto')
    axes[2].set_title('Spread', fontweight='bold')
    axes[2].set_yticks(range(len(methods)))
    axes[2].set_yticklabels([m.replace('_', ' ').title() for m in methods])
    axes[2].set_xticks(range(n_vars))
    axes[2].set_xticklabels(variable_names)
    plt.colorbar(im3, ax=axes[2])
    
    # Spread/RMSE ratio
    ratio_matrix = spread_matrix / np.maximum(rmse_matrix, 1e-8)
    im4 = axes[3].imshow(ratio_matrix, cmap='magma', aspect='auto')
    axes[3].set_title('Spread/RMSE Ratio', fontweight='bold')
    axes[3].set_yticks(range(len(methods)))
    axes[3].set_yticklabels([m.replace('_', ' ').title() for m in methods])
    axes[3].set_xticks(range(n_vars))
    axes[3].set_xticklabels(variable_names)
    plt.colorbar(im4, ax=axes[3])
    
    plt.suptitle('Verification Matrices', fontsize=16, fontweight='bold')
    plt.savefig(output_path / 'verification_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_convergence_diagnostics(results: Dict[str, Any], config: Dict[str, Any], 
                                   output_path: Path, logger: logging.Logger) -> None:
    """Generate proper convergence diagnostics."""
    logger.info("Generating convergence diagnostic plots...")
    
    posterior_samples = results["posterior_samples"]
    variable_names = config.get("variable_names", [f"var_{i}" for i in range(posterior_samples.shape[1])])
    n_vars = len(variable_names)
    
    fig, axes = plt.subplots(2, n_vars, figsize=(4 * n_vars, 8), 
                            constrained_layout=True)
    if n_vars == 1:
        axes = axes.reshape(-1, 1)
    
    colors = sns.color_palette("husl", n_vars)
    
    for i, (var_name, color) in enumerate(zip(variable_names, colors)):
        chain = posterior_samples[:, i, 0]
        
        # Cumulative mean plot
        ax1 = axes[0, i]
        cumsum = np.cumsum(chain)
        cummean = cumsum / np.arange(1, len(chain) + 1)
        ax1.plot(cummean, color=color, linewidth=1.5)
        ax1.axhline(y=np.mean(chain), color='red', linestyle='--', alpha=0.7,
                   label='Final mean')
        ax1.set_title(f'{var_name} - Cumulative Mean')
        ax1.set_ylabel('Cumulative Mean')
        ax1.grid(True, alpha=0.3)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        if i == 0:
            ax1.legend(frameon=False)
        
        # Autocorrelation plot
        ax2 = axes[1, i]
        try:
            # Simple autocorrelation computation
            def autocorr(x, max_lags=min(50, len(chain)//4)):
                x = x - np.mean(x)
                autocorrs = np.correlate(x, x, mode='full')
                autocorrs = autocorrs[autocorrs.size//2:]
                autocorrs = autocorrs / autocorrs[0]
                return autocorrs[:max_lags]
            
            lags = range(min(50, len(chain)//4))
            acf = autocorr(chain)
            ax2.plot(lags, acf, color=color, linewidth=1.5)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax2.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, 
                       label='Threshold')
            ax2.set_title(f'{var_name} - Autocorrelation')
            ax2.set_xlabel('Lag')
            ax2.set_ylabel('Autocorrelation')
            ax2.grid(True, alpha=0.3)
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            if i == 0:
                ax2.legend(frameon=False)
        except:
            ax2.text(0.5, 0.5, 'ACF computation failed', ha='center', va='center',
                    transform=ax2.transAxes)
    
    plt.suptitle('Convergence Diagnostics', fontsize=14, fontweight='bold')
    plt.savefig(output_path / 'convergence_diagnostics.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_comprehensive_summary(results: Dict[str, Any], config: Dict[str, Any], 
                                 output_path: Path, logger: logging.Logger) -> None:
    """Generate a comprehensive summary figure."""
    logger.info("Generating comprehensive summary...")
    
    # Create a summary dashboard
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Parameter summary table
    ax1 = fig.add_subplot(gs[0, :2])
    posterior_samples = results["posterior_samples"]
    variable_names = config.get("variable_names", [f"var_{i}" for i in range(posterior_samples.shape[1])])
    
    means = np.mean(posterior_samples, axis=0).flatten()
    stds = np.std(posterior_samples, axis=0).flatten()
    
    table_data = []
    for i, var_name in enumerate(variable_names):
        table_data.append([var_name, f"{means[i]:.4f}", f"±{stds[i]:.4f}"])
    
    table = ax1.table(cellText=table_data,
                     colLabels=['Variable', 'Mean', 'Std Dev'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax1.axis('off')
    ax1.set_title('Posterior Summary', fontweight='bold')
    
    # Score evolution mini-plot
    ax2 = fig.add_subplot(gs[0, 2:])
    step_mean_scores = results.get("step_mean_scores", [])
    if len(step_mean_scores) > 0:
        ax2.plot(step_mean_scores, color='steelblue', linewidth=1.5)
        ax2.set_title('Score Evolution')
        ax2.set_xlabel('Step')
        ax2.grid(True, alpha=0.3)
    
    # Parameter traces (compact)
    ax3 = fig.add_subplot(gs[1, :])
    colors = sns.color_palette("husl", len(variable_names))
    for i, (var_name, color) in enumerate(zip(variable_names, colors)):
        chain = posterior_samples[:, i, 0]
        ax3.plot(chain, color=color, alpha=0.7, linewidth=1, label=var_name)
    ax3.set_title('Parameter Traces')
    ax3.set_xlabel('Iteration')
    ax3.legend(ncol=len(variable_names), frameon=False, loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # Posterior distributions
    ax4 = fig.add_subplot(gs[2, :])
    for i, (var_name, color) in enumerate(zip(variable_names, colors)):
        chain = posterior_samples[:, i, 0]
        ax4.hist(chain, bins=30, alpha=0.6, density=True, color=color, 
                label=var_name, histtype='stepfilled')
    ax4.set_title('Posterior Distributions')
    ax4.set_xlabel('Parameter Value')
    ax4.set_ylabel('Density')
    ax4.legend(frameon=False)
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('ABC-RFP Analysis Summary', fontsize=16, fontweight='bold')
    plt.savefig(output_path / 'comprehensive_summary.png', dpi=300, bbox_inches='tight')
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser("Evaluation of results")
    parser.add_argument(
        "specific_result_directory",
        nargs="?",
        default=None,
        help="Path to the result directory (defaults to the most recent run)",
    )
    args = parser.parse_args()
    logger = _setup_logging()

    base_path = Path("./results").expanduser()
    if not base_path.exists():
        raise FileNotFoundError(base_path)

    result_path = (
        Path(args.specific_result_directory).expanduser().resolve()
        if args.specific_result_directory is not None
        else _latest_result_directory(base_path)
    )

    logger.info("Evaluating directory: %s", result_path)

    results, config = _load_results(result_path)

    # Generate all analysis plots
    print_posterior_summary(results, config, result_path, logger)
    generate_parameter_traces(results, config, result_path, logger)
    generate_rank_histograms(results, config, result_path, logger)
    generate_score_evolution(results, config, result_path, logger)
    generate_posterior_diagnostics(results, config, result_path, logger)
    generate_convergence_diagnostics(results, config, result_path, logger)
    generate_ensemble_spatial_plots(results, config, result_path, logger)
    generate_method_comparison(results, config, result_path, logger)
    generate_comprehensive_summary(results, config, result_path, logger)

    logger.info("Evaluation completed; outputs written to %s", result_path)


if __name__ == "__main__":
    main()