#!/usr/bin/env python3
"""
Analysis of benchmark results across different ABC setups.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

def load_all_results():
    """Load all statistics.json files and organize by setup type."""
    results_dir = Path("results")
    all_results = {}
    
    for stats_file in results_dir.glob("*/statistics.json"):
        setup_name = stats_file.parent.name
        with open(stats_file, 'r') as f:
            results = json.load(f)
        all_results[setup_name] = results
    
    return all_results

def create_comparison_table(all_results):
    """Create a comparison table for analysis."""
    
    # Extract setup types from directory names
    setup_mapping = {
        'gibbs_abc_crps': 'Gibbs+CRPS',
        'gibbs_abc_energy': 'Gibbs+Energy', 
        'smc_gibbs_abc_energy': 'SMC-Gibbs+Energy',
        'greedy_abc_energy': 'Greedy+Energy'
    }
    
    methods = ['rfp_posterior_mean', 'rfp_posterior_sample', 'persistence', 'climatology', 'ar1', 'deterministic_gaussian']
    metrics = ['crps', 'energy_score', 'mae', 'spread']
    
    # Create DataFrame
    data = []
    for setup_dir, results in all_results.items():
        setup_type = None
        for key, name in setup_mapping.items():
            if key in setup_dir:
                setup_type = name
                break
        
        if setup_type is None:
            continue
            
        for method in methods:
            if method in results:
                row = {'Setup': setup_type, 'Method': method}
                row.update(results[method])
                data.append(row)
    
    df = pd.DataFrame(data)
    return df

def analyze_results(df):
    """Analyze the benchmark results and provide insights."""
    
    print("=== COMPREHENSIVE BENCHMARK ANALYSIS ===\n")
    
    # 1. Method ranking by CRPS (lower is better)
    print("1. CRPS RANKING (Lower = Better)")
    print("-" * 50)
    crps_ranking = df.groupby('Method')['crps'].agg(['mean', 'std', 'min', 'max']).round(4)
    crps_ranking = crps_ranking.sort_values('mean')
    print(crps_ranking)
    print()
    
    # 2. Method ranking by Energy Score (lower is better) 
    print("2. ENERGY SCORE RANKING (Lower = Better)")
    print("-" * 50)
    energy_ranking = df.groupby('Method')['energy_score'].agg(['mean', 'std', 'min', 'max']).round(4)
    energy_ranking = energy_ranking.sort_values('mean')
    print(energy_ranking)
    print()
    
    # 3. Method ranking by MAE (lower is better)
    print("3. MAE RANKING (Lower = Better)")
    print("-" * 50)
    mae_ranking = df.groupby('Method')['mae'].agg(['mean', 'std', 'min', 'max']).round(4)
    mae_ranking = mae_ranking.sort_values('mean')
    print(mae_ranking)
    print()
    
    # 4. Spread analysis (measure of uncertainty)
    print("4. SPREAD ANALYSIS (Ensemble Uncertainty)")
    print("-" * 50)
    spread_ranking = df.groupby('Method')['spread'].agg(['mean', 'std', 'min', 'max']).round(4)
    spread_ranking = spread_ranking.sort_values('mean', ascending=False)
    print(spread_ranking)
    print()
    
    # 5. ABC Setup comparison
    print("5. ABC SETUP PERFORMANCE (RFP Methods Only)")
    print("-" * 50)
    rfp_methods = df[df['Method'].str.contains('rfp')]
    setup_comparison = rfp_methods.groupby(['Setup', 'Method'])[['crps', 'energy_score', 'mae']].mean().round(4)
    print(setup_comparison)
    print()
    
    # 6. Key insights
    print("6. KEY INSIGHTS")
    print("-" * 50)
    
    # Best performing method overall
    best_crps = df.loc[df['crps'].idxmin()]
    best_energy = df.loc[df['energy_score'].idxmin()]
    best_mae = df.loc[df['mae'].idxmin()]
    
    print(f"Best CRPS: {best_crps['Method']} ({best_crps['crps']:.4f})")
    print(f"Best Energy Score: {best_energy['Method']} ({best_energy['energy_score']:.4f})")
    print(f"Best MAE: {best_mae['Method']} ({best_mae['mae']:.4f})")
    print()
    
    # ABC vs baselines
    abc_methods = df[df['Method'].str.contains('rfp')]
    baseline_methods = df[~df['Method'].str.contains('rfp')]
    
    print("ABC-RFP vs Baselines:")
    print(f"  ABC-RFP mean CRPS: {abc_methods['crps'].mean():.4f}")
    print(f"  Baselines mean CRPS: {baseline_methods['crps'].mean():.4f}")
    print()
    
    # Deterministic + Gaussian analysis
    det_gauss = df[df['Method'] == 'deterministic_gaussian']
    print("Deterministic + Gaussian Performance:")
    print(f"  Mean CRPS: {det_gauss['crps'].mean():.4f} (std: {det_gauss['crps'].std():.4f})")
    print(f"  Mean Energy: {det_gauss['energy_score'].mean():.4f} (std: {det_gauss['energy_score'].std():.4f})")
    print(f"  Mean MAE: {det_gauss['mae'].mean():.4f} (std: {det_gauss['mae'].std():.4f})")
    print(f"  Mean Spread: {det_gauss['spread'].mean():.4f} (std: {det_gauss['spread'].std():.4f})")
    print()
    
    # Performance ratios
    print("7. PERFORMANCE ANALYSIS")
    print("-" * 50)
    
    # Compare ABC to deterministic gaussian
    abc_mean_crps = abc_methods['crps'].mean()
    det_mean_crps = det_gauss['crps'].mean()
    
    print(f"ABC-RFP vs Deterministic+Gaussian:")
    print(f"  CRPS ratio: {abc_mean_crps/det_mean_crps:.2f}x worse")
    print(f"  ABC-RFP CRPS: {abc_mean_crps:.4f}")
    print(f"  Det+Gauss CRPS: {det_mean_crps:.4f}")
    print()
    
    # AR(1) performance
    ar1_results = df[df['Method'] == 'ar1']
    print(f"AR(1) Performance:")
    print(f"  Mean CRPS: {ar1_results['crps'].mean():.4f}")
    print(f"  Competitive with ABC-RFP: {ar1_results['crps'].mean() < abc_mean_crps}")
    print()

def main():
    all_results = load_all_results()
    df = create_comparison_table(all_results)
    analyze_results(df)

if __name__ == "__main__":
    main()