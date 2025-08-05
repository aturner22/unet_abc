import argparse
import datetime as dt
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np

from core.evaluation import (
    generate_comprehensive_analysis,
    generate_evaluation_metrics,
    generate_rank_histograms,
    generate_score_evolution,
    generate_trace_plots,
    print_posterior_summary,
)


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

    print_posterior_summary(results, config, result_path, logger)
    generate_trace_plots(results, config, result_path, logger)
    generate_rank_histograms(results, config, result_path, logger)
    generate_score_evolution(results, config, result_path, logger)
    generate_evaluation_metrics(results, config, result_path, logger)
    generate_comprehensive_analysis(results, config, result_path, logger)

    logger.info("Evaluation completed; outputs written to %s", result_path)


if __name__ == "__main__":
    main()
