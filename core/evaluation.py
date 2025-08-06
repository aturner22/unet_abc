import numpy as np
import matplotlib.pyplot as plt
import torch
import cartopy.crs as ccrs
import json
import logging
from pathlib import Path
from typing import Dict, Any


def validate_tensor_shapes(ensemble: torch.Tensor, target: torch.Tensor):
    if ensemble.dim() < 2:
        raise ValueError("Ensemble tensor must be at least 2-dimensional")
    if ensemble.shape[1:] != target.shape:
        raise ValueError(f"Incompatible shapes: {ensemble.shape[1:]} vs {target.shape}")


def print_posterior_summary_console(
    posterior_mean, posterior_variance, variable_names, parameter_labels
):
    print("\nPosterior parameter moments:")
    print("---------------------------")
    for variable_index, variable in enumerate(variable_names):
        print(f"{variable}:")
        for parameter_index, label in enumerate(parameter_labels):
            mu = posterior_mean[variable_index, parameter_index]
            sigma_square = posterior_variance[variable_index, parameter_index]
            print(f"  {label}: mu = {mu:+.4f}, sigma_square = {sigma_square:.4e}")


def continuous_ranked_probability_score(
    ensemble: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    validate_tensor_shapes(ensemble, target)
    absolute_error = torch.abs(ensemble - target.unsqueeze(0)).mean(dim=0)
    pairwise = torch.abs(ensemble.unsqueeze(0) - ensemble.unsqueeze(1)).mean(dim=(0, 1))
    return absolute_error - 0.5 * pairwise


def compute_rank_histogram(
    ensemble: torch.Tensor, target: torch.Tensor, ensemble_size: int
) -> torch.Tensor:
    validate_tensor_shapes(ensemble, target)
    with torch.no_grad():
        # Compute ranks: number of ensemble members below each target value
        rank_counts = (ensemble < target.unsqueeze(0)).sum(dim=0)
    # Return integer ranks (0 to ensemble_size), not normalized values
    ranks = rank_counts.view(-1).int().cpu().numpy()
    return ranks


def compute_mean_absolute_error(ensemble: torch.Tensor, target: torch.Tensor) -> float:
    validate_tensor_shapes(ensemble, target)
    return torch.abs(ensemble.mean(dim=0) - target).mean().item()


def compute_ensemble_spread(ensemble: torch.Tensor) -> float:
    if ensemble.dim() < 2:
        raise ValueError("Ensemble tensor must be at least 2-dimensional")
    return ensemble.std(dim=0).mean().item()


# Plotting functions
def produce_trace_and_histogram_plots(
    samples: np.ndarray, output_directory: Path, variable_names, parameter_labels
):
    num_variables = samples.shape[1]
    parameter_dim = samples.shape[2]

    for parameter_index in range(parameter_dim):
        plt.figure(figsize=(10, 6))
        for variable_index in range(num_variables):
            plt.plot(
                samples[:, variable_index, parameter_index],
                label=variable_names[variable_index],
            )
        plt.title(f"Trace {parameter_labels[parameter_index]}")
        plt.xlabel("Gibbs iteration")
        plt.ylabel(parameter_labels[parameter_index])
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_directory / f"trace_{parameter_labels[parameter_index]}.png")
        plt.close()

        plt.figure(figsize=(12, 6))
        for variable_index in range(num_variables):
            plt.hist(
                samples[:, variable_index, parameter_index],
                bins=30,
                alpha=0.6,
                label=variable_names[variable_index],
                density=True,
            )
        plt.title(f"Posterior {parameter_labels[parameter_index]}")
        plt.xlabel(parameter_labels[parameter_index])
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_directory / f"hist_{parameter_labels[parameter_index]}.png")
        plt.close()


def produce_rank_histograms(
    histograms, output_directory: Path, variable_names, ensemble_size: int
):
    for variable_index, ranks in enumerate(histograms):
        plt.figure(figsize=(8, 4))
        plt.hist(ranks, bins=ensemble_size, alpha=0.7, density=True)
        plt.axhline(1.0, color="red", linestyle="--", alpha=0.8)
        plt.title(f"Rank histogram: {variable_names[variable_index]}")
        plt.xlabel("Rank")
        plt.ylabel("Density")
        plt.tight_layout()
        plt.savefig(output_directory / f"ranks_{variable_names[variable_index]}.png")
        plt.close()


def plot_score_trace(
    step_mean_scores: np.ndarray, output_directory: Path, score_function: str
):
    plt.figure(figsize=(10, 6))
    plt.plot(step_mean_scores)
    plt.title(f"Mean {score_function.upper()} Score Evolution")
    plt.xlabel("Gibbs iteration")
    plt.ylabel(f"Mean {score_function.upper()}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_directory / f"{score_function}_evolution.png")
    plt.close()


def plot_field(
    field: torch.Tensor,
    latitude: np.ndarray,
    longitude: np.ndarray,
    title: str,
    output_path: Path,
    variable_name: str = "",
):
    plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    field_np = field.cpu().numpy() if isinstance(field, torch.Tensor) else field

    contour = ax.contourf(
        longitude,
        latitude,
        field_np,
        levels=20,
        transform=ccrs.PlateCarree(),
        cmap="RdBu_r",
    )
    ax.coastlines()
    ax.gridlines(draw_labels=True)

    plt.colorbar(contour, ax=ax, shrink=0.6)
    plt.title(f"{title} {variable_name}")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def print_posterior_summary(
    results: Dict[str, Any],
    config_data: Dict[str, Any],
    output_path: Path,
    logger: logging.Logger,
):
    variable_names = config_data["variable_names"]
    parameter_labels = ["alpha_scale"]

    posterior_mean = results["posterior_mean"]
    posterior_variance = results["posterior_variance"]

    logger.info("Generating posterior parameter summary")

    # Create summary text file
    summary_file = output_path / "posterior_summary.txt"
    with open(summary_file, "w") as f:
        f.write("Posterior Parameter Summary\n")
        f.write("=" * 50 + "\n\n")

        for variable_index, variable in enumerate(variable_names):
            f.write(f"{variable}:\n")
            for parameter_index, label in enumerate(parameter_labels):
                mu = posterior_mean[variable_index, parameter_index]
                sigma_square = posterior_variance[variable_index, parameter_index]
                f.write(
                    f"  {label}: mu = {mu:+.4f}, sigma_square = {sigma_square:.4e}\n"
                )
            f.write("\n")

    # Also print to console
    print_posterior_summary_console(
        posterior_mean, posterior_variance, variable_names, parameter_labels
    )

    logger.info(f"Posterior summary saved to {summary_file}")


def generate_trace_plots(
    results: Dict[str, Any],
    config_data: Dict[str, Any],
    output_path: Path,
    logger: logging.Logger,
):
    variable_names = config_data["variable_names"]
    parameter_labels = ["alpha_scale"]

    logger.info("Generating trace and histogram plots")
    produce_trace_and_histogram_plots(
        results["posterior_samples"], output_path, variable_names, parameter_labels
    )
    logger.info("Trace plots saved")


def generate_rank_histograms(
    results: Dict[str, Any],
    config_data: Dict[str, Any],
    output_path: Path,
    logger: logging.Logger,
):
    variable_names = config_data["variable_names"]
    ensemble_size = config_data["ensemble_size"]

    logger.info("Generating rank histograms")
    produce_rank_histograms(
        results["rank_histograms"], output_path, variable_names, ensemble_size
    )
    logger.info("Rank histograms saved")


def generate_score_evolution(
    results: Dict[str, Any],
    config_data: Dict[str, Any],
    output_path: Path,
    logger: logging.Logger,
):
    score_function = config_data["score_function"]

    logger.info("Generating score evolution plot")
    plot_score_trace(results["step_mean_scores"], output_path, score_function)
    logger.info("Score evolution plot saved")


def generate_evaluation_metrics(
    results: Dict[str, Any],
    config_data: Dict[str, Any],
    output_path: Path,
    logger: logging.Logger,
):
    logger.info("Computing evaluation metrics")

    variable_names = config_data["variable_names"]
    n_steps = config_data["n_gibbs_steps"]
    score_function = config_data["score_function"]

    metrics = {
        "run_configuration": {
            "score_function": score_function,
            "n_gibbs_steps": n_steps,
            "ensemble_size": config_data["ensemble_size"],
            "n_proposals_per_variable": config_data["n_proposals_per_variable"],
            "temporal_resampling": config_data.get("temporal_resampling", False),
        },
        "posterior_statistics": {},
        "convergence_diagnostics": {},
        "scoring_evolution": {
            "final_mean_score": float(results["step_mean_scores"][-1]),
            "initial_mean_score": float(results["step_mean_scores"][0]),
            "score_improvement": float(
                results["step_mean_scores"][0] - results["step_mean_scores"][-1]
            ),
        },
    }

    for i, var_name in enumerate(variable_names):
        samples = results["posterior_samples"][:, i, 0]
        metrics["posterior_statistics"][var_name] = {
            "mean": float(np.mean(samples)),
            "std": float(np.std(samples)),
            "median": float(np.median(samples)),
            "q025": float(np.percentile(samples, 2.5)),
            "q975": float(np.percentile(samples, 97.5)),
            "effective_sample_size": int(len(samples)),
        }

        half_point = len(samples) // 2
        first_half_mean = np.mean(samples[:half_point])
        second_half_mean = np.mean(samples[half_point:])
        metrics["convergence_diagnostics"][var_name] = {
            "first_half_mean": float(first_half_mean),
            "second_half_mean": float(second_half_mean),
            "mean_difference": float(abs(first_half_mean - second_half_mean)),
        }

    metrics_file = output_path / "evaluation_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Evaluation metrics saved to {metrics_file}")


def generate_comprehensive_analysis(
    results: Dict[str, Any],
    config_data: Dict[str, Any],
    output_path: Path,
    logger: logging.Logger,
):
    """Generate comprehensive post-evaluation analysis with advanced verification metrics."""
    logger.info("Generating comprehensive post-evaluation analysis")

    variable_names = config_data["variable_names"]
    ensemble_size = config_data.get("ensemble_size", 10)
    n_steps = results["posterior_samples"].shape[0]

    verification_metrics = compute_verification_metrics(results, variable_names)

    analysis_file = output_path / "comprehensive_analysis.txt"
    with open(analysis_file, "w") as f:
        f.write("Comprehensive Post-Evaluation Analysis\n")
        f.write("=" * 50 + "\n\n")

        f.write("Dataset Information:\n")
        f.write(f"  Variables: {', '.join(variable_names)}\n")
        f.write(f"  Ensemble size: {ensemble_size}\n")
        f.write(f"  Gibbs steps: {n_steps}\n")
        f.write(f"  Score function: {config_data.get('score_function', 'unknown')}\n\n")

        f.write("Posterior Parameter Statistics:\n")
        f.write("-" * 30 + "\n")
        for i, var in enumerate(variable_names):
            mean_alpha = (
                results["posterior_mean"][i, 0]
                if results["posterior_mean"].ndim > 1
                else results["posterior_mean"][i]
            )
            var_alpha = (
                results["posterior_variance"][i, 0]
                if results["posterior_variance"].ndim > 1
                else results["posterior_variance"][i]
            )
            f.write(f"{var}:\n")
            f.write(f"  Alpha scale: μ = {mean_alpha:.4f}, σ² = {var_alpha:.6f}\n")
        f.write("\n")

        f.write("Verification Metrics:\n")
        f.write("-" * 20 + "\n")
        for metric, values in verification_metrics.items():
            f.write(f"{metric}:\n")
            if isinstance(values, dict):
                for var, val in values.items():
                    f.write(f"  {var}: {val:.6f}\n")
            else:
                f.write(f"  Overall: {values:.6f}\n")
        f.write("\n")

        f.write("Convergence Diagnostics:\n")
        f.write("-" * 25 + "\n")
        if "posterior_scores" in results and results["posterior_scores"].size > 0:
            scores = results["posterior_scores"]
            if scores.ndim > 1:
                final_scores = scores[-1, :]
                initial_scores = scores[0, :]
                improvement = np.mean(initial_scores - final_scores)
                f.write(f"Score improvement: {improvement:.6f}\n")
                f.write(f"Final mean score: {np.mean(final_scores):.6f}\n")
                f.write(f"Score convergence: {'Good' if improvement > 0 else 'Poor'}\n")
        f.write("\n")

        f.write("Ensemble Quality Assessment:\n")
        f.write("-" * 30 + "\n")
        if "ensemble_mae" in results and "ensemble_spread" in results:
            mae = results["ensemble_mae"]
            spread = results["ensemble_spread"]
            if mae.size > 0 and spread.size > 0:
                spread_skill_ratio = np.mean(spread / (mae + 1e-8))
                f.write(f"Spread-skill ratio: {spread_skill_ratio:.4f}\n")
                f.write(
                    "  (Ideal value ~1.0, >1 = overdispersed, <1 = underdispersed)\n"
                )

                reliability = "Good" if 0.8 <= spread_skill_ratio <= 1.2 else "Poor"
                f.write(f"Ensemble reliability: {reliability}\n")

    generate_diagnostic_plots(results, variable_names, output_path, logger)

    logger.info(f"Comprehensive analysis saved to {analysis_file}")


def compute_verification_metrics(
    results: Dict[str, Any], variable_names: list
) -> Dict[str, Any]:
    """Compute verification metrics for the posterior analysis."""
    metrics = {}

    if "ensemble_mae" in results and results["ensemble_mae"].size > 0:
        mae = results["ensemble_mae"]
        metrics["mean_absolute_error"] = {
            var: mae[i].mean() if mae.ndim > 1 else mae.mean()
            for i, var in enumerate(variable_names)
        }

    if "ensemble_spread" in results and results["ensemble_spread"].size > 0:
        spread = results["ensemble_spread"]
        metrics["ensemble_spread"] = {
            var: spread[i].mean() if spread.ndim > 1 else spread.mean()
            for i, var in enumerate(variable_names)
        }

    if "posterior_samples" in results:
        samples = results["posterior_samples"]
        if samples.size > 0:
            n_eff = estimate_effective_sample_size(samples)
            metrics["effective_sample_size"] = {
                var: n_eff[i] if hasattr(n_eff, "__len__") else n_eff
                for i, var in enumerate(variable_names)
            }

    return metrics


def estimate_effective_sample_size(samples: np.ndarray) -> np.ndarray:
    if samples.size == 0:
        return np.array([0])

    n_steps = samples.shape[0]
    if n_steps < 4:
        return np.array([n_steps] * samples.shape[1] if samples.ndim > 1 else [n_steps])

    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)

    n_vars = samples.shape[1]
    n_eff = np.zeros(n_vars)

    for i in range(n_vars):
        var_samples = samples[:, i, 0] if samples.ndim > 2 else samples[:, i]

        var_val = np.var(var_samples)

        if var_val > 1e-8:
            autocorr_1 = np.corrcoef(var_samples[:-1], var_samples[1:])[0, 1]
            autocorr_1 = max(0, min(0.99, autocorr_1))

            n_eff[i] = n_steps * (1 - autocorr_1) / (1 + autocorr_1)
        else:
            n_eff[i] = n_steps

    return n_eff


def generate_diagnostic_plots(
    results: Dict[str, Any],
    variable_names: list,
    output_path: Path,
    logger: logging.Logger,
):
    """Generate additional diagnostic plots for comprehensive analysis."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib

        matplotlib.use("Agg")

        if "posterior_samples" in results and results["posterior_samples"].size > 0:
            samples = results["posterior_samples"]

            fig, axes = plt.subplots(
                len(variable_names), 1, figsize=(10, 2 * len(variable_names))
            )
            if len(variable_names) == 1:
                axes = [axes]

            for i, (var, ax) in enumerate(zip(variable_names, axes)):
                var_samples = samples[:, i, 0] if samples.ndim > 2 else samples[:, i]
                steps = np.arange(len(var_samples))

                ax.plot(steps, var_samples, "b-", alpha=0.7, linewidth=1)

                if len(var_samples) > 10:
                    window = max(3, len(var_samples) // 10)
                    running_mean = np.convolve(
                        var_samples, np.ones(window) / window, mode="same"
                    )
                    ax.plot(
                        steps, running_mean, "r-", linewidth=2, label="Running mean"
                    )
                    ax.legend()

                ax.set_title(f"{var} - Parameter Trace")
                ax.set_xlabel("Step")
                ax.set_ylabel("Alpha Scale")
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(
                output_path / "parameter_traces_detailed.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()

        if "posterior_scores" in results and results["posterior_scores"].size > 0:
            scores = results["posterior_scores"]

            fig, ax = plt.subplots(figsize=(10, 6))

            if scores.ndim > 1:
                steps = np.arange(scores.shape[0])
                for i, var in enumerate(variable_names):
                    ax.plot(steps, scores[:, i], label=f"{var}", alpha=0.7)
                ax.legend()
            else:
                steps = np.arange(len(scores))
                ax.plot(steps, scores, "b-", linewidth=2)

            ax.set_title("Score Evolution with Convergence Analysis")
            ax.set_xlabel("Gibbs Step")
            ax.set_ylabel("Score")
            ax.grid(True, alpha=0.3)

            if scores.ndim > 1:
                mean_scores = scores.mean(axis=1)
            else:
                mean_scores = scores

            if len(mean_scores) > 2:
                z = np.polyfit(steps, mean_scores, 1)
                p = np.poly1d(z)
                ax.plot(
                    steps,
                    p(steps),
                    "r--",
                    linewidth=2,
                    label=f"Trend (slope: {z[0]:.4f})",
                )
                ax.legend()

            plt.tight_layout()
            plt.savefig(
                output_path / "score_convergence_analysis.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()

        logger.info("Diagnostic plots generated successfully")

    except ImportError:
        logger.warning("matplotlib not available, skipping diagnostic plots")
    except Exception as e:
        logger.warning(f"Error generating diagnostic plots: {e}")
