import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from matplotlib.colors import LinearSegmentedColormap

# Variable names and styling
DEFAULT_VARIABLE_NAMES = ["z500", "t850", "t2m", "u10", "v10"]
DEFAULT_DPI = 300

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

# Academic paper color palette
COLORS = sns.color_palette("Set2", n_colors=5)
SCORE_COLOR = "#2E3440"

def setup_plot_style():
    """Configure seaborn and matplotlib for academic plots."""
    sns.set_theme(style="ticks", context="paper")
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'legend.title_fontsize': 10,
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5
    })


def fit_gamma_density(samples: np.ndarray, support_points: np.ndarray) -> np.ndarray:
    """
    Fit a gamma distribution to positive samples and evaluate density.
    
    Args:
        samples: Sample values (must be positive)
        support_points: Points at which to evaluate the density
        
    Returns:
        Density values at support points
    """
    # Ensure all samples are positive
    positive_samples = samples[samples > 0]
    
    if len(positive_samples) < 10:  # Fallback to basic approach if too few samples
        return np.histogram(positive_samples, bins=50, density=True)[0]
    
    try:
        # Fit gamma distribution using method of moments/MLE
        # scipy.stats.gamma uses shape (a) and scale parameters
        shape, loc, scale = stats.gamma.fit(positive_samples, floc=0)  # Fix location at 0
        
        # Evaluate fitted gamma density at support points
        density = stats.gamma.pdf(support_points, shape, loc=loc, scale=scale)
        
        # If fit failed (e.g., very poor fit), fall back to KDE with positive support
        if np.any(~np.isfinite(density)) or shape <= 0 or scale <= 0:
            raise ValueError("Gamma fit failed")
            
        return density
        
    except (ValueError, RuntimeError):
        # Fallback to bounded KDE (restrict to positive support)
        from scipy.stats import gaussian_kde
        
        try:
            kde = gaussian_kde(positive_samples)
            # Evaluate only at positive support points
            positive_support = support_points[support_points > 0]
            if len(positive_support) > 0:
                density_positive = kde(positive_support)
                # Extend with zeros for non-positive points
                density = np.zeros_like(support_points)
                density[support_points > 0] = density_positive
            else:
                density = np.zeros_like(support_points)
                
            return density
            
        except:
            # Ultimate fallback: normalized histogram
            hist, bins = np.histogram(positive_samples, bins=len(support_points)//2, density=True)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            return np.interp(support_points, bin_centers, hist, left=0, right=0)


def plot_joint_posteriors(samples: np.ndarray, result_path: Path, variable_names: list = None, burn_in: int = 20):
    """
    Create a 5x5 matrix of 2D histograms showing joint posterior distributions
    between all pairs of atmospheric variable scale parameters.
    
    Args:
        samples: Posterior samples array of shape (T, P, 1) where T is iterations, P is parameters
        result_path: Path to save the joint posterior plot
        variable_names: List of variable names for labeling
        burn_in: Number of initial samples to discard as burn-in
    """
    if variable_names is None:
        variable_names = DEFAULT_VARIABLE_NAMES
    
    T, P, _ = samples.shape
    if P != 5:
        print(f"Warning: Expected 5 parameters, got {P}. Skipping joint posterior plot.")
        return
    
    # Apply burn-in
    samples_burned = samples[burn_in:] if T > burn_in else samples
    T_burned = samples_burned.shape[0]
    
    # Extract parameter samples (remove singleton dimension)
    param_samples = samples_burned[:, :, 0]  # Shape: (T_burned, 5)
    
    # Create custom colormap for 2D histograms
    colors = ['white', '#f0f0f0', '#d0d0d0', '#a0a0a0', '#707070', '#404040', '#202020']
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('custom_gray', colors, N=n_bins)
    
    # Create 5x5 subplot grid
    fig, axes = plt.subplots(5, 5, figsize=(12, 12))
    fig.suptitle('Joint Posterior Distributions', fontsize=14, fontweight='bold', y=0.95)
    
    for i in range(5):
        for j in range(5):
            ax = axes[i, j]
            
            if i == j:
                # Diagonal: 1D marginal distribution
                param_i = param_samples[:, i]
                
                # Create histogram
                ax.hist(param_i, bins=30, density=True, alpha=0.7, 
                       color=COLORS[i], edgecolor='none')
                
                # Add gamma fit overlay
                x_min, x_max = max(0, param_i.min() * 0.9), param_i.max() * 1.1
                x_support = np.linspace(x_min, x_max, 100)
                density = fit_gamma_density(param_i, x_support)
                ax.plot(x_support, density, color=COLORS[i], linewidth=2, alpha=0.8)
                
                ax.set_xlim(left=0)
                
                # Labels only on edges
                if i == 4:  # Bottom row
                    ax.set_xlabel(f'α_{variable_names[j]}', fontsize=10)
                if j == 0:  # Left column
                    ax.set_ylabel('Density', fontsize=10)
                    
            else:
                # Off-diagonal: 2D joint distribution
                param_i = param_samples[:, i]
                param_j = param_samples[:, j]
                
                # Create 2D histogram
                hist, xedges, yedges = np.histogram2d(param_j, param_i, bins=25, density=True)
                
                # Plot as image
                im = ax.imshow(hist.T, origin='lower', aspect='auto', cmap=cmap,
                              extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
                
                # Add contour lines for better visualization
                X, Y = np.meshgrid((xedges[:-1] + xedges[1:])/2, (yedges[:-1] + yedges[1:])/2)
                ax.contour(X, Y, hist.T, levels=5, colors='black', alpha=0.3, linewidths=0.5)
                
                # Set limits to show positive values only
                ax.set_xlim(left=0)
                ax.set_ylim(bottom=0)
                
                # Labels only on edges
                if i == 4:  # Bottom row
                    ax.set_xlabel(f'α_{variable_names[j]}', fontsize=10)
                if j == 0:  # Left column
                    ax.set_ylabel(f'α_{variable_names[i]}', fontsize=10)
            
            # Remove tick labels from interior plots
            if i != 4:
                ax.set_xticklabels([])
            if j != 0:
                ax.set_yticklabels([])
                
            # Reduce tick label size
            ax.tick_params(labelsize=8)
            
            # Add subtle grid for better readability
            ax.grid(True, alpha=0.2, linewidth=0.5)
            
    # Adjust spacing between subplots
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)  # Make room for suptitle
    
    # Save the plot
    fig.savefig(result_path / "joint_posteriors.png", dpi=DEFAULT_DPI, bbox_inches='tight')
    fig.savefig(result_path / "joint_posteriors.pdf", bbox_inches='tight')
    plt.close(fig)
    
    print(f"  ✓ Created joint posterior plot with {T_burned} samples (burn-in: {burn_in})")


def load_posterior_data(result_path: Path):
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


def has_existing_plots(result_path: Path) -> bool:
    """Check if plots already exist in the result directory."""
    required_files = [
        "trace_all.png",
        "posterior_combined.png", 
        "score_trajectory.png",
        "posterior_summary_combined.png",
        "joint_posteriors.png"
    ]
    
    for filename in required_files:
        if not (result_path / filename).exists():
            return False
    
    # Check for at least one KDE plot
    kde_files = list(result_path.glob("kde_*.png"))
    return len(kde_files) > 0


def create_plots_for_directory(result_path: Path, variable_names: list = None) -> bool:
    """Create plots for a single result directory. Returns True if successful."""
    if variable_names is None:
        variable_names = DEFAULT_VARIABLE_NAMES
    
    try:
        # Load data
        samples, scores, data_source = load_posterior_data(result_path)
        if samples is None:
            return False
        
        T, P, _ = samples.shape
        if len(variable_names) != P:
            print(f"Warning: {result_path.name} has {P} variables, expected {len(variable_names)}")
            return False

        # Determine appropriate burn-in period based on algorithm type
        burn_in = get_burn_in_period(result_path)

        setup_plot_style()

        # Combined trace plot - clean and minimal
        fig, ax = plt.subplots(figsize=(7, 4))
        for p in range(P):
            trace = samples[:, p, 0]
            ax.plot(np.arange(T), trace, color=COLORS[p], label=variable_names[p], 
                    linewidth=1.0, alpha=0.8)
        
        ax.set_xlabel("Iteration")
        ax.set_ylabel(r"$\alpha$")
        ax.legend(frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.1))
        sns.despine(ax=ax)
        plt.tight_layout()
        fig.savefig(result_path / "trace_all.png", dpi=DEFAULT_DPI, bbox_inches='tight')
        fig.savefig(result_path / "trace_all.pdf", bbox_inches='tight')
        plt.close(fig)

        # Individual posterior distributions - using gamma distribution for positive support
        for p in range(P):
            name = variable_names[p]
            trace = samples[:, p, 0]
            
            # Apply burn-in period
            trace_burned = trace[burn_in:] if len(trace) > burn_in else trace

            fig, ax = plt.subplots(figsize=(4, 3))
            
            # Clean histogram (use burned-in samples)
            sns.histplot(trace_burned, ax=ax, kde=False, bins=25, stat="density", 
                        color=COLORS[p], alpha=0.3, edgecolor='none')
            
            # Gamma distribution fit overlay (appropriate for positive parameters)
            x_min, x_max = max(0, trace_burned.min() * 0.9), trace_burned.max() * 1.1
            x_support = np.linspace(x_min, x_max, 200)
            density = fit_gamma_density(trace_burned, x_support)
            
            ax.plot(x_support, density, color=COLORS[p], linewidth=2, label='Gamma fit')
            
            ax.set_xlabel(r"$\alpha$"+f"_{variable_names[p]}")
            ax.set_ylabel("Density")
            ax.set_xlim(left=0)  # Ensure positive support is shown
            sns.despine(ax=ax)
            
            plt.tight_layout()
            fig.savefig(result_path / f"kde_{name}.png", dpi=DEFAULT_DPI, bbox_inches='tight')
            fig.savefig(result_path / f"kde_{name}.pdf", bbox_inches='tight')
            plt.close(fig)

        # Combined density plot - all variables on one graph using gamma fits
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # Determine common x-axis range across all variables (with burn-in)
        all_traces_burned = [samples[burn_in:, p, 0] if samples.shape[0] > burn_in else samples[:, p, 0] for p in range(P)]
        x_min = max(0, min(trace.min() for trace in all_traces_burned) * 0.9)
        x_max = max(trace.max() for trace in all_traces_burned) * 1.1
        x_support = np.linspace(x_min, x_max, 200)
        
        for p in range(P):
            trace = samples[:, p, 0]
            trace_burned = trace[burn_in:] if len(trace) > burn_in else trace
            density = fit_gamma_density(trace_burned, x_support)
            ax.plot(x_support, density, color=COLORS[p], linewidth=2, 
                   label=variable_names[p], alpha=0.8)
        
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel("Density")
        ax.set_xlim(left=0)  # Ensure positive support is shown
        ax.legend(frameon=False, loc='upper right')
        sns.despine(ax=ax)
        
        plt.tight_layout()
        fig.savefig(result_path / "posterior_combined.png", dpi=DEFAULT_DPI, bbox_inches='tight')
        fig.savefig(result_path / "posterior_combined.pdf", bbox_inches='tight')
        plt.close(fig)

        # Score trajectory - clean line plot
        mean_scores = scores.mean(axis=1)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        
        ax.plot(np.arange(T), mean_scores, color=SCORE_COLOR, linewidth=1.2)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Score")
        
        # Add subtle grid
        ax.grid(True, alpha=0.3, linewidth=0.5)
        sns.despine(ax=ax)
        
        plt.tight_layout()
        fig.savefig(result_path / "score_trajectory.png", dpi=DEFAULT_DPI, bbox_inches='tight')
        fig.savefig(result_path / "score_trajectory.pdf", bbox_inches='tight')
        plt.close(fig)

        # Three-subplot combined summary plot - wide layout for page width
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
        
        # Subplot 1: Combined posterior densities
        # Determine common x-axis range across all variables (with burn-in)
        all_traces_burned = [samples[burn_in:, p, 0] if samples.shape[0] > burn_in else samples[:, p, 0] for p in range(P)]
        x_min = max(0, min(trace.min() for trace in all_traces_burned) * 0.9)
        x_max = max(trace.max() for trace in all_traces_burned) * 1.1
        x_support = np.linspace(x_min, x_max, 200)
        
        # Plot gamma density fits
        for p in range(P):
            trace = samples[:, p, 0]
            trace_burned = trace[burn_in:] if len(trace) > burn_in else trace
            density = fit_gamma_density(trace_burned, x_support)
            ax1.plot(x_support, density, color=COLORS[p], linewidth=2, 
                   label=variable_names[p], alpha=0.8)
        
        ax1.set_xlabel(r"$\alpha$")
        ax1.set_ylabel("Density")
        ax1.set_xlim(left=0)
        ax1.legend(frameon=False, loc='upper right', fontsize=8)
        ax1.set_title("Posterior Distributions", fontweight='bold')
        sns.despine(ax=ax1)
        
        # Subplot 2: Score trajectory
        mean_scores = scores.mean(axis=1)
        ax2.plot(np.arange(T), mean_scores, color=SCORE_COLOR, linewidth=1.2)
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Score")
        ax2.set_title("Score Trajectory", fontweight='bold')
        ax2.grid(True, alpha=0.3, linewidth=0.5)
        sns.despine(ax=ax2)
        
        # Subplot 3: Parameter traces
        for p in range(P):
            trace = samples[:, p, 0]
            ax3.plot(np.arange(T), trace, color=COLORS[p], label=variable_names[p], 
                    linewidth=1.0, alpha=0.8)
        
        ax3.set_xlabel("Iteration")
        ax3.set_ylabel(r"$\alpha$")
        ax3.set_title("Parameter Traces", fontweight='bold')
        ax3.legend(frameon=False, fontsize=8, loc='upper right')
        sns.despine(ax=ax3)
        
        plt.tight_layout()
        fig.savefig(result_path / "posterior_summary_combined.png", dpi=DEFAULT_DPI, bbox_inches='tight')
        fig.savefig(result_path / "posterior_summary_combined.pdf", bbox_inches='tight')
        plt.close(fig)

        # Joint posterior distributions (5x5 matrix plot)
        if P == 5:  # Only create joint posteriors for 5-parameter case
            plot_joint_posteriors(samples, result_path, variable_names, burn_in)
        
        return True
        
    except Exception as e:
        print(f"Error creating plots for {result_path.name}: {e}")
        return False


def process_all_results(results_directory: Path, force_regenerate: bool = False):
    """Process all result directories to create missing plots."""
    if not results_directory.exists():
        raise FileNotFoundError(f"Results directory not found: {results_directory}")
    
    # Find all potential result directories
    result_dirs = [d for d in results_directory.iterdir() 
                   if d.is_dir() and not d.name.startswith('.')]
    
    if not result_dirs:
        print(f"No result directories found in: {results_directory}")
        return
    
    print(f"Scanning {len(result_dirs)} directories in: {results_directory}")
    
    processed = 0
    skipped = 0
    failed = 0
    
    for result_path in sorted(result_dirs):
        # Check if data exists
        samples, scores, data_source = load_posterior_data(result_path)
        if samples is None:
            continue  # Skip directories without data
        
        # Check if plots already exist (unless forcing regeneration)
        if not force_regenerate and has_existing_plots(result_path):
            skipped += 1
            continue
        
        print(f"Processing: {result_path.name} ({data_source})")
        
        if create_plots_for_directory(result_path):
            processed += 1
            print(f"  ✓ Created plots for {result_path.name}")
        else:
            failed += 1
            print(f"  ✗ Failed to create plots for {result_path.name}")
    
    print(f"\nSummary:")
    print(f"  Processed: {processed}")
    print(f"  Skipped (already have plots): {skipped}")
    print(f"  Failed: {failed}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate ABC posterior samples")
    parser.add_argument("result_directory", type=str, nargs='?', default="./results",
                       help="Path to single result directory OR results root directory (default: ./results)")
    parser.add_argument("--all", action="store_true", 
                       help="Process all result directories in the specified path")
    parser.add_argument("--force", action="store_true",
                       help="Regenerate plots even if they already exist")
    args = parser.parse_args()

    result_path = Path(args.result_directory)
    
    if args.all:
        # Batch processing mode
        process_all_results(result_path, force_regenerate=args.force)
        return
    
    # Single directory mode (original functionality)
    if not result_path.exists():
        raise FileNotFoundError(f"Directory not found: {result_path}")

    # Check if this looks like a results root directory (has multiple subdirs with data)
    if result_path.is_dir():
        subdirs_with_data = []
        for subdir in result_path.iterdir():
            if subdir.is_dir():
                samples, scores, data_source = load_posterior_data(subdir)
                if samples is not None:
                    subdirs_with_data.append(subdir.name)
        
        if len(subdirs_with_data) > 1:
            print(f"Found {len(subdirs_with_data)} result directories in {result_path}:")
            for dirname in sorted(subdirs_with_data):
                print(f"  - {dirname}")
            print("\nTo process all directories, use: --all")
            print("To process a specific directory, provide its full path.")
            return

    # Load data for single directory
    samples, scores, data_source = load_posterior_data(result_path)
    if samples is None:
        available_files = list(result_path.glob("*"))
        raise FileNotFoundError(
            f"Neither result files nor checkpoint found in {result_path}. "
            f"Available files: {[f.name for f in available_files]}"
        )
    
    print(f"Processing: {result_path.name} ({data_source})")
    
    # Create plots
    if create_plots_for_directory(result_path):
        T, P, _ = samples.shape
        total_plots = P + 4  # Individual KDE plots + trace + combined posterior + score + summary combined
        if P == 5:  # Add joint posterior plot for 5-parameter case
            total_plots += 1
        total_files = total_plots * 2  # PNG and PDF for each plot
        print(f"✓ Created {total_plots} plots ({total_files} files) in: {result_path}")
        print(f"Formats: PNG (high-res) and PDF (vector)")
        print("Files created:")
        print("  - trace_all.png/pdf (combined α traces)")
        print("  - posterior_combined.png/pdf (all densities overlaid)")
        for name in DEFAULT_VARIABLE_NAMES[:P]:
            print(f"  - kde_{name}.png/pdf (individual posterior)")
        print("  - score_trajectory.png/pdf (convergence)")
        print("  - posterior_summary_combined.png/pdf (three-panel summary)")
        if P == 5:
            print("  - joint_posteriors.png/pdf (5x5 joint distribution matrix)")
    else:
        print("✗ Failed to create plots")

if __name__ == "__main__":
    main()
