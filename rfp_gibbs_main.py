import torch
import numpy as np
import json
import os
import gc
from tqdm import tqdm
import psutil

from config import Config
from core.io_utils import (
    load_model_and_test_data,
    save_posterior_statistics,
    materialise_batches,
    print_computing_configuration,
)
from core.plotting import (
    produce_trace_and_histogram_plots,
    produce_rank_histograms,
    plot_crps_trace,
)
from core.evaluation import print_posterior_summary
from core.gibbs_abc_threaded_rfp import run_gibbs_abc_rfp


def load_or_generate_standardized_reference(config, latitude, longitude) -> np.memmap:
    standardized_path = config.data_directory / "z500_t850_t2m_u10_v10_standardized.npy"
    raw_path = config.data_directory / "z500_t850_t2m_u10_v10_1979-2018_5.625deg.npy"
    norm_path = config.data_directory / "norm_factors.json"

    with open(norm_path, "r") as f:
        norm_stats = json.load(f)

    mean_data = torch.tensor([norm_stats[v]["mean"] for v in config.variable_names], dtype=torch.float32)
    std_data = torch.tensor([norm_stats[v]["std"] for v in config.variable_names], dtype=torch.float32)

    if standardized_path.exists():
        print("Loading precomputed standardized ERA5 tensor...")
        return np.load(standardized_path, mmap_mode='r')

    print("Standardized tensor not found. Standardizing raw dataset...")
    temporal_len = 350640
    shape = (temporal_len, config.num_variables, len(latitude), len(longitude))
    raw_array = np.memmap(raw_path, dtype=np.float32, mode='r', shape=shape)

    full_tensor = torch.tensor(raw_array, dtype=torch.float32)
    del raw_array
    gc.collect()

    full_tensor.sub_(mean_data[:, None, None]).div_(std_data[:, None, None])
    np.save(standardized_path, full_tensor.cpu().numpy())

    del full_tensor, mean_data, std_data
    gc.collect()
    return np.load(standardized_path, mmap_mode='r')

def main():
    print("Initializing device...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_computing_configuration()
    print(f"Using device: {device}")

    timestamp = os.environ.get("CONFIG_TIMESTAMP")
    config = Config("config.json", timestamp=timestamp)

    print("Preparing model and data loader...")
    loader, model, latitude, longitude, result_path = load_model_and_test_data(config, device, config.SEED)

    # Check if temporal resampling is enabled
    use_temporal_resampling = getattr(config, 'temporal_resampling', False)
    
    if use_temporal_resampling:
        print(f"Temporal resampling enabled: {config.sample_size} samples per Gibbs step")
        # Get full dataset without subset sampling for resampling
        full_dataset = loader.dataset.dataset if hasattr(loader.dataset, 'dataset') else loader.dataset
        cached_batches = None
    else:
        print("Materializing input batches...")
        cached_batches = list(tqdm(
            materialise_batches(loader, device, config.num_variables, config.max_horizon, latitude, longitude),
            total=config.sample_size,
            desc="Loading batches"
        ))
        full_dataset = None

    print("Preparing standardized reference tensor...")
    reference_mmap = load_or_generate_standardized_reference(config, latitude, longitude)

    # Get example dimensions (not used in current implementation but kept for future compatibility)
    if cached_batches:
        example_input, example_output = cached_batches[0][0], cached_batches[0][1]
    else:
        # For temporal resampling, get dimensions from a sample
        sample_data = next(iter(torch.utils.data.DataLoader(full_dataset, batch_size=1)))
        example_input, example_output = sample_data[0].squeeze(0), sample_data[1].squeeze(0)

    print("Dynamic batch management will be handled automatically during inference...")

    try:
        print("Commencing ABC-Gibbs inference with RFP perturbations...")
        results = run_gibbs_abc_rfp(
            model=model,
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
            resample_temporal=use_temporal_resampling
        )


        print("Saving posterior results...")
        save_posterior_statistics(results, result_path)

        print("Releasing memory before plotting...")
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        torch.cuda.empty_cache()
        gc.collect()

        results_to_keep = {
            "posterior_samples": results["posterior_samples"],
            "rank_histograms": results["rank_histograms"],
            "step_mean_crps": results["step_mean_crps"],
            "posterior_mean": results["posterior_mean"],
            "posterior_variance": results["posterior_variance"],
        }
        del results
        gc.collect()

        print("Generating posterior plots...")
        produce_trace_and_histogram_plots(
            results_to_keep["posterior_samples"],
            result_path,
            config.variable_names,
            ["alpha_scale"]
        )

        produce_rank_histograms(
            results_to_keep["rank_histograms"],
            result_path,
            config.variable_names,
            config.ensemble_size
        )

        plot_crps_trace(
            results_to_keep["step_mean_crps"],
            result_path
        )

        print("Final posterior parameter summary:")
        print_posterior_summary(
            results_to_keep["posterior_mean"],
            results_to_keep["posterior_variance"],
            config.variable_names,
            ["alpha_scale"]
        )

        print("ABC-Gibbs with RFP complete.")

    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()