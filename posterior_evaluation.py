import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

DEFAULT_VARIABLE_NAMES = ["z500", "t850", "t2m", "u10", "v10"]
DEFAULT_DPI = 300

def main():
    parser = argparse.ArgumentParser(description="Evaluate ABC posterior samples")
    parser.add_argument("result_directory", type=str, help="Path to inference result directory")
    args = parser.parse_args()

    result_path = Path(args.result_directory)
    if not result_path.exists():
        raise FileNotFoundError(f"Directory not found: {result_path}")

    samples_path = result_path / "posterior_samples.npy"
    scores_path = result_path / "posterior_scores.npy"

    if not samples_path.exists() or not scores_path.exists():
        raise FileNotFoundError("Expected files not found in result directory")

    samples = np.load(samples_path) 
    scores = np.load(scores_path)  
    T, P, _ = samples.shape
    variable_names = DEFAULT_VARIABLE_NAMES

    if len(variable_names) != P:
        raise ValueError(f"Expected {P} variable names, got {len(variable_names)}")

    sns.set(style="whitegrid", font_scale=1.2)

    fig_combined, ax_combined = plt.subplots(figsize=(10, 4))
    for p in range(P):
        trace = samples[:, p, 0]
        sns.lineplot(x=np.arange(T), y=trace, ax=ax_combined, label=variable_names[p], lw=1.2)
    ax_combined.set_title("Trace of α for all variables")
    ax_combined.set_xlabel("Gibbs Step")
    ax_combined.set_ylabel("α")
    ax_combined.legend(loc="upper right", ncol=2)
    fig_combined.tight_layout()
    fig_combined.savefig(result_path / "trace_all.png", dpi=DEFAULT_DPI)
    plt.close(fig_combined)

    for p in range(P):
        name = variable_names[p]
        trace = samples[:, p, 0]

        fig, ax = plt.subplots(figsize=(6, 3))
        sns.histplot(trace, ax=ax, kde=True, bins=30, stat="density", linewidth=0.5, color="steelblue")
        ax.set_title(f"Posterior of α for {name}")
        ax.set_xlabel("α")
        ax.set_ylabel("Density")
        fig.tight_layout()
        fig.savefig(result_path / f"posterior_{name}.png", dpi=DEFAULT_DPI)
        plt.close(fig)

    mean_scores = scores.mean(axis=1)
    fig_score, ax_score = plt.subplots(figsize=(8, 3))
    sns.lineplot(x=np.arange(T), y=mean_scores, ax=ax_score, lw=1.2)
    ax_score.set_title("Mean Score per Gibbs Step")
    ax_score.set_xlabel("Gibbs Step")
    ax_score.set_ylabel("Score")
    fig_score.tight_layout()
    fig_score.savefig(result_path / "score_trajectory.png", dpi=DEFAULT_DPI)
    plt.close(fig_score)

    print(f"Saved {P + 2} plots to: {result_path}")

if __name__ == "__main__":
    main()
