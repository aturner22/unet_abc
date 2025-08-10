import torch
import numpy as np
import json
import os
import gc
import logging
import argparse
from pathlib import Path
from tqdm import tqdm

from core.config import Config
from core.io_utils import (
    load_model_and_test_data,
    save_posterior_statistics,
    materialise_batches,
    log_computing_configuration,
)
from core.algorithm import run_gibbs_abc_rfp


def setup_logging(result_path):
    level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(result_path / "abc_run.log"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def load_or_generate_standardised_reference(
    config, latitude, longitude, logger
) -> np.memmap:
    standardised_path = config.data_directory / "z500_t850_t2m_u10_v10_standardized.npy"
    raw_path = config.data_directory / "z500_t850_t2m_u10_v10_1979-2018_5.625deg.npy"
    norm_path = config.data_directory / "norm_factors.json"

    with open(norm_path, "r") as f:
        norm_stats = json.load(f)

    mean_data = torch.tensor(
        [norm_stats[v]["mean"] for v in config.variable_names], dtype=torch.float32
    )
    std_data = torch.tensor(
        [norm_stats[v]["std"] for v in config.variable_names], dtype=torch.float32
    )

    if standardised_path.exists():
        logger.info("Loading precomputed standardised ERA5 tensor")
        return np.load(standardised_path, mmap_mode="r")

    logger.info("Standardising raw dataset - this may take several minutes")
    temporal_len = 350640  # Should probably change this from being hard coded
    shape = (temporal_len, config.num_variables, len(latitude), len(longitude))
    raw_array = np.memmap(raw_path, dtype=np.float32, mode="r", shape=shape)

    full_tensor = torch.tensor(raw_array, dtype=torch.float32)
    del raw_array
    gc.collect()

    full_tensor.sub_(mean_data[:, None, None]).div_(std_data[:, None, None])
    np.save(standardised_path, full_tensor.cpu().numpy())

    del full_tensor, mean_data, std_data
    gc.collect()
    return np.load(standardised_path, mmap_mode="r")


def main():
    parser = argparse.ArgumentParser(description="Run ABC-RFP procedure")
    parser.add_argument(
        "--config", type=str, default="config.json", help="Path to configuration file"
    )
    parser.add_argument(
        "--resume-from", type=str, help="Path to existing result directory to resume from"
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.resume_from:
        resume_path = Path(args.resume_from)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume directory not found: {resume_path}")
        
        checkpoint_file = resume_path / "gibbs_checkpoint_step.npz"
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"No checkpoint found in: {resume_path}")
        
        dir_name = resume_path.name
        if "_" in dir_name:
            extracted_timestamp = dir_name.split("_", 1)[-1] 
            timestamp = extracted_timestamp
        else:
            timestamp = os.environ.get("CONFIG_TIMESTAMP")
            
        config = Config(args.config, timestamp=timestamp)
        
        config.result_directory = resume_path
        print(f"Resuming from existing directory: {resume_path}")
        
    else:
        timestamp = os.environ.get(
            "CONFIG_TIMESTAMP"
        ) 
        config = Config(args.config, timestamp=timestamp)

    logger = setup_logging(config.result_directory)
    log_computing_configuration()

    logger.info(f"Starting ABC-RFP procedure with device: {device}")
    logger.info(
        f"Configuration: {config.score_function} scoring, {config.n_gibbs_steps} steps, {config.ensemble_size} ensemble size"
    )

    loader, model, latitude, longitude, result_path = load_model_and_test_data(
        config, device, config.SEED
    )

    use_temporal_resampling = getattr(config, "temporal_resampling", False)

    if use_temporal_resampling:
        logger.info(
            f"Using temporal resampling: {config.sample_size} samples per Gibbs step"
        )
        full_dataset = (
            loader.dataset.dataset
            if hasattr(loader.dataset, "dataset")
            else loader.dataset
        )
        cached_batches = None
    else:
        logger.info("Materialising input batches")
        cached_batches = list(
            tqdm(
                materialise_batches(
                    loader,
                    device,
                    config.num_variables,
                    config.max_horizon,
                    latitude,
                    longitude,
                ),
                total=config.sample_size,
                desc="Loading batches",
            )
        )
        full_dataset = None

    reference_mmap = load_or_generate_standardised_reference(
        config, latitude, longitude, logger
    )

    try:
        logger.info("Beginning ABC-Gibbs inference with RFP perturbations")
        results = run_gibbs_abc_rfp(
            model=model,
            config=config,
            batches=cached_batches,
            full_dataset=full_dataset,
            sample_size=config.sample_size,
            ensemble_size=config.ensemble_size,
            n_steps=config.n_gibbs_steps,
            n_proposals=config.n_proposals_per_variable,
            num_variables=config.num_variables,
            variable_names=config.variable_names,
            reference_mmap=reference_mmap,
            result_directory=result_path,
            log_diagnostics=True,
            resample_temporal=use_temporal_resampling,
            score_function=config.score_function,
            logger=logger,
        )

        logger.info("Saving posterior results")
        save_posterior_statistics(results, result_path)

        logger.info("ABC-Gibbs procedure completed successfully")
        logger.info(f"Results saved to: {result_path}")

    except Exception as e:
        logger.error(f"ABC procedure failed: {str(e)}")
        raise
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        logger.info("Memory cleanup completed")


if __name__ == "__main__":
    main()
