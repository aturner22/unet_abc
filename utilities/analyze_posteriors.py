#!/usr/bin/env python3
"""
Analyze posterior samples from ABC runs to compute parameter statistics.
"""

import numpy as np
import json
from pathlib import Path
import pandas as pd

def load_posterior_samples(run_dir):
    """Load posterior samples from a run directory."""
    samples_path = Path(run_dir) / "posterior_samples.npy"
    if samples_path.exists():
        samples = np.load(samples_path)
        # Shape should be (n_samples, n_params, 1) -> squeeze to (n_samples, n_params)
        if samples.ndim == 3 and samples.shape[2] == 1:
            samples = samples.squeeze(axis=2)
        return samples
    return None

def compute_statistics(samples):
    """Compute mean and 90% credible intervals for each parameter."""
    stats = {}
    param_names = ['z500', 't850', 't2m', 'u10', 'v10']
    
    for i, param in enumerate(param_names):
        param_samples = samples[:, i]
        stats[param] = {
            'mean': float(np.mean(param_samples)),
            'q05': float(np.percentile(param_samples, 5)),
            'q95': float(np.percentile(param_samples, 95)),
            'std': float(np.std(param_samples))
        }
    
    return stats

def load_performance_stats(run_dir):
    """Load performance statistics from statistics.json."""
    stats_path = Path(run_dir) / "statistics.json"
    if stats_path.exists():
        with open(stats_path, 'r') as f:
            return json.load(f)
    return None

def main():
    # Production run directories
    run_dirs = [
        "results/2nd_run_conditional_energy_gibbs_abc_2025-08-11T11:11:57Z",
        "results/2nd_run_conditional_crps_gibbs_abc_2025-08-16T07:49:39Z", 
        "results/2nd_run_smc_energy_gibbs_abc_2025-08-13T06:33:17Z",
        "results/2nd_run_smc_crps_gibbs_abc_2025-08-14T22:48:53Z",
        "results/2nd_run_greedy_energy_gibbs_abc_2025-08-17T18:14:01Z",
        "results/2nd_run_greedy_crps_gibbs_abc_2025-08-17T17:33:01Z"
    ]
    
    # Method names for display
    method_names = {
        "2nd_run_conditional_energy_gibbs_abc_2025-08-11T11:11:57Z": "Conditional (Energy)",
        "2nd_run_conditional_crps_gibbs_abc_2025-08-16T07:49:39Z": "Conditional (CRPS)",
        "2nd_run_smc_energy_gibbs_abc_2025-08-13T06:33:17Z": "SMC (Energy)", 
        "2nd_run_smc_crps_gibbs_abc_2025-08-14T22:48:53Z": "SMC (CRPS)",
        "2nd_run_greedy_energy_gibbs_abc_2025-08-17T18:14:01Z": "Greedy (Energy)",
        "2nd_run_greedy_crps_gibbs_abc_2025-08-17T17:33:01Z": "Greedy (CRPS)"
    }
    
    results = {}
    
    for run_dir in run_dirs:
        if not Path(run_dir).exists():
            print(f"Warning: {run_dir} does not exist")
            continue
            
        run_name = Path(run_dir).name
        method_name = method_names.get(run_name, run_name)
        
        # Load posterior samples
        samples = load_posterior_samples(run_dir)
        if samples is None:
            print(f"Warning: No posterior samples found in {run_dir}")
            continue
            
        # Load performance statistics
        perf_stats = load_performance_stats(run_dir)
        if perf_stats is None:
            print(f"Warning: No performance statistics found in {run_dir}")
            continue
        
        # Compute parameter statistics
        param_stats = compute_statistics(samples)
        
        results[method_name] = {
            'performance': perf_stats,
            'parameters': param_stats,
            'n_samples': samples.shape[0]
        }
        
        print(f"Processed {method_name}: {samples.shape[0]} samples")
    
    # Save results
    with open('posterior_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to posterior_analysis.json")
    print(f"Analyzed {len(results)} ABC runs")
    
    return results

if __name__ == "__main__":
    results = main()