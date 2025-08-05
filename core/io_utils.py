import datetime
import json
import logging
import multiprocessing
import os
import platform
import shutil
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .datasets import ERA5Dataset
from .models import DetPrecond

logger = logging.getLogger(__name__)


def load_model_and_test_data(
    config: Any,
    device: torch.device,
    random_subset_seed: int,
) -> Tuple[DataLoader, torch.nn.Module, np.ndarray, np.ndarray, Path]:
    result_path: Path = config.result_directory
    result_path.mkdir(parents=True, exist_ok=True)
    shutil.copy(config.config_path, result_path / "config.json")

    with open(config.data_directory / "norm_factors.json", "r", encoding="utf-8") as f:
        stats = json.load(f)

    mean = torch.tensor(
        [stats[k]["mean"] for k in config.variable_names],
        dtype=torch.float32,
        device=device,
    )
    std = torch.tensor(
        [stats[k]["std"] for k in config.variable_names],
        dtype=torch.float32,
        device=device,
    )
    norm_factors = np.stack([mean.cpu().numpy(), std.cpu().numpy()], axis=0)

    ti = pd.date_range(
        datetime.datetime(1979, 1, 1, 0), datetime.datetime(2018, 12, 31, 23), freq="1h"
    )
    n_samples = len(ti)

    lat, lon = np.load(config.data_directory / "latlon_1979-2018_5.625deg.npz").values()

    dataset_kwargs = {
        "dataset_path": str(
            config.data_directory / "z500_t850_t2m_u10_v10_1979-2018_5.625deg.npy"
        ),
        "sample_counts": (n_samples, 0, 0),
        "dimensions": (config.num_variables, len(lat), len(lon)),
        "max_horizon": config.max_horizon,
        "norm_factors": norm_factors,
        "device": device,
        "spacing": config.spacing,
        "dtype": "float32",
        "conditioning_times": config.conditioning_times,
        "lead_time_range": [config.t_direct, config.t_direct, config.t_direct],
        "static_data_path": str(
            config.data_directory / "orog_lsm_1979-2018_5.625deg.npy"
        ),
        "random_lead_time": 0,
    }

    input_channels = (
        len(config.conditioning_times) * config.num_variables + config.num_static_fields
    )
    model = DetPrecond(
        filters=32,
        img_channels=input_channels,
        out_channels=config.num_variables,
        img_resolution=64,
    )
    model_ckpt = "./models/deterministic-iterative-6h/best_model.pth"
    model.load_state_dict(torch.load(model_ckpt, map_location=device))
    model.to(device).eval()

    full_dataset = ERA5Dataset(
        lead_time=[config.t_direct], dataset_mode="test", **dataset_kwargs
    )

    if config.sample_size is not None:
        np.random.seed(random_subset_seed)
        subset_indices = np.random.choice(
            len(full_dataset), size=config.sample_size, replace=False
        )
        dataset = torch.utils.data.Subset(full_dataset, subset_indices)
    else:
        dataset = full_dataset

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    return loader, model, lat, lon, result_path


def save_posterior_statistics(results: dict, output_directory: Path) -> None:
    np.save(output_directory / "posterior_samples.npy", results["posterior_samples"])
    np.save(output_directory / "posterior_scores.npy", results["posterior_scores"])
    np.save(output_directory / "posterior_mean.npy", results["posterior_mean"])
    np.save(output_directory / "posterior_variance.npy", results["posterior_variance"])
    np.save(output_directory / "ensemble_mae.npy", results["ensemble_mae"])
    np.save(output_directory / "ensemble_spread.npy", results["ensemble_spread"])
    np.savez(output_directory / "results.npz", **results)


def log_computing_configuration() -> None:
    logger.info("--- Computing Configuration ---")
    logger.info("Platform: %s %s", platform.system(), platform.release())
    logger.info("Python version: %s", platform.python_version())
    logger.info("CUDA available: %s", torch.cuda.is_available())

    if torch.cuda.is_available():
        logger.info("CUDA device count: %d", torch.cuda.device_count())
        logger.info("CUDA device name: %s", torch.cuda.get_device_name(0))
        logger.info("CUDA device capability: %s", torch.cuda.get_device_capability(0))
        logger.info(
            "CUDA memory allocated: %.2f GB", torch.cuda.memory_allocated(0) / 1e9
        )
        logger.info(
            "CUDA memory reserved: %.2f GB", torch.cuda.memory_reserved(0) / 1e9
        )

    logger.info("torch version: %s", torch.__version__)
    logger.info("Number of CPUs: %d", os.cpu_count())
    logger.info("Physical CPU cores: %d", multiprocessing.cpu_count())
    logger.info("TORCH_NUM_THREADS: %s", torch.get_num_threads())
    logger.info("N_WORKERS: %s", os.getenv("N_WORKERS"))
    logger.info("PARALLEL_BACKEND: %s", os.getenv("PARALLEL_BACKEND"))
    logger.info("--------------------------------")


def materialise_batches(
    loader: DataLoader,
    device: torch.device,
    num_variables: int,
    max_horizon: int,
    latitude: np.ndarray,
    longitude: np.ndarray,
):
    batches = []
    for previous_fields, current_fields, valid_time in loader:
        previous_fields = previous_fields.to(device)
        current_fields = current_fields.view(
            -1, num_variables, len(latitude), len(longitude)
        ).to(device)
        time_normalised = (
            torch.tensor([valid_time[0]], dtype=torch.float32, device=device)
            / max_horizon
        )
        batches.append((previous_fields, current_fields, time_normalised))
    return batches
