#!/usr/bin/env python3
"""
Script to analyze filtered hyperparameter sweep results from sweep_1
and generate a LaTeX table based on specific hyperparameter ranges.
"""

import os
import re
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze_top import (
    parse_results_file,
    parse_config_from_dirname,
    compute_composite_score,
    compute_sweep_ranges,
    generate_latex_table
)

def config_matches_ranges(config: Dict[str, float], filter_ranges: Dict[str, List[float]]) -> bool:
    """
    Check if a configuration matches the specified filter ranges.
    
    Args:
        config: Configuration dictionary with hyperparameter values
        filter_ranges: Dictionary mapping parameter names to allowed values
    
    Returns:
        True if config matches all filter criteria, False otherwise
    """
    for param_name, allowed_values in filter_ranges.items():
        if param_name not in config:
            return False
        
        config_value = config[param_name]
        
        # Check if config value is in allowed values (with tolerance for floating point)
        matches = False
        for allowed_val in allowed_values:
            if abs(config_value - allowed_val) < 1e-6:
                matches = True
                break
        
        if not matches:
            return False
    
    return True

def analyze_filtered_sweep(sweep_dir: str = 'sweep_1', 
                          filter_ranges: Dict[str, List[float]] = None,
                          w_snr: float = 1.0, 
                          w_mse: float = 0.3, 
                          w_coherence: float = 0.3) -> Tuple[List[Tuple[Dict, Dict, float, str]], Dict[str, List[float]]]:
    """
    Analyze sweep results filtered by specific hyperparameter ranges.
    
    Args:
        sweep_dir: Directory containing sweep results
        filter_ranges: Dictionary mapping parameter names to allowed values
        w_snr: Weight for SNR in composite score
        w_mse: Weight for MSE in composite score
        w_coherence: Weight for Coherence in composite score
    
    Returns:
        Tuple of (filtered_results_list, sweep_ranges_dict)
    """
    if filter_ranges is None:
        filter_ranges = {}
    
    sweep_path = Path(sweep_dir)
    if not sweep_path.exists():
        raise ValueError(f"Sweep directory {sweep_dir} does not exist")
    
    all_results = []
    
    # Find all results.txt files
    results_files = glob.glob(str(sweep_path / '*/results.txt'))
    
    print(f"Found {len(results_files)} result files")
    print(f"\nFiltering by hyperparameter ranges:")
    for param, values in filter_ranges.items():
        print(f"  {param}: {values}")
    
    # First pass: collect all metrics to compute actual ranges (only from filtered results)
    all_snr_improvement_values = []
    all_mse_values = []
    all_coherence_values = []
    
    filtered_count = 0
    for results_file in results_files:
        # Get directory name
        dir_path = Path(results_file).parent
        dirname = dir_path.name
        
        # Parse metrics
        metrics = parse_results_file(results_file)
        if metrics is None:
            continue
        
        # Parse config from directory name
        config = parse_config_from_dirname(dirname)
        
        # Filter by ranges
        if not config_matches_ranges(config, filter_ranges):
            continue
        
        filtered_count += 1
        
        # Collect values for range computation
        all_snr_improvement_values.append(metrics['snr_improvement'])
        all_mse_values.append(metrics['mse'])
        all_coherence_values.append(metrics['median_coherence'])
        
        all_results.append((config, metrics, None, dirname))  # Score will be computed later
    
    print(f"\nFound {filtered_count} configurations matching filter criteria")
    
    if not all_results:
        print("No results match the filter criteria!")
        return [], {}
    
    # Compute actual ranges from the filtered data
    snr_range = (min(all_snr_improvement_values), max(all_snr_improvement_values)) if all_snr_improvement_values else (0, 0)
    mse_range = (min(all_mse_values), max(all_mse_values)) if all_mse_values else (0, 0)
    coherence_range = (min(all_coherence_values), max(all_coherence_values)) if all_coherence_values else (0, 0)
    
    print(f"\nActual metric ranges (filtered):")
    print(f"  SNR Improvement: [{snr_range[0]:.2f}, {snr_range[1]:.2f}] dB")
    print(f"  MSE: [{mse_range[0]:.2f}, {mse_range[1]:.2f}]")
    print(f"  Coherence: [{coherence_range[0]:.4f}, {coherence_range[1]:.4f}]")
    
    # Second pass: compute composite scores using actual ranges
    for i, (config, metrics, _, dirname) in enumerate(all_results):
        score = compute_composite_score(
            metrics['snr_improvement'],
            metrics['mse'],
            metrics['median_coherence'],
            snr_range,
            mse_range,
            coherence_range,
            w_snr=w_snr,
            w_mse=w_mse,
            w_coherence=w_coherence
        )
        all_results[i] = (config, metrics, score, dirname)
    
    # Compute sweep ranges (only for filtered results)
    sweep_ranges = compute_sweep_ranges(all_results)
    
    # Sort by composite score (descending)
    all_results.sort(key=lambda x: x[2], reverse=True)
    
    return all_results, sweep_ranges

def main():
    import argparse
    
    # Define the specific filter ranges from hyperparameter_sweep.py
    DEFAULT_FILTER_RANGES = {
    'f_cutoff': [10,25],
    'w_cosine': [1, 2],
    'w_rank_s': [0, 0.2],
    'w_spectral': [2,3],
    'w_spectral_slope': [0,0.2],
    'w_rank_a': [0.2, 1],
}
    
    parser = argparse.ArgumentParser(description='Analyze filtered hyperparameter sweep and generate LaTeX table')
    parser.add_argument('--sweep-dir', type=str, default='sweep_1',
                        help='Directory containing sweep results (default: sweep_1)')
    parser.add_argument('--output', type=str, default='sweep_filtered_table.tex',
                        help='Output LaTeX file (default: sweep_filtered_table.tex)')
    parser.add_argument('--w-snr', type=float, default=1.0,
                        help='Weight for SNR Improvement in composite score (default: 1.0)')
    parser.add_argument('--w-mse', type=float, default=0.3,
                        help='Weight for MSE in composite score (default: 0.3)')
    parser.add_argument('--w-coherence', type=float, default=0.3,
                        help='Weight for Coherence in composite score (default: 0.3)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("ANALYZING FILTERED SWEEP RESULTS")
    print("="*80)
    print(f"Score weights: SNR Improvement={args.w_snr}, MSE={args.w_mse}, Coherence={args.w_coherence}")
    
    filtered_results, sweep_ranges = analyze_filtered_sweep(
        args.sweep_dir,
        filter_ranges=DEFAULT_FILTER_RANGES,
        w_snr=args.w_snr,
        w_mse=args.w_mse,
        w_coherence=args.w_coherence
    )
    
    if not filtered_results:
        print("No valid results found matching the filter criteria!")
        return
    
    print(f"\nFound {len(filtered_results)} configurations matching filter criteria")
    print(f"\nGenerating LaTeX table for all {len(filtered_results)} configurations...")
    
    generate_latex_table(
        filtered_results, 
        sweep_ranges, 
        args.output,
        w_snr=args.w_snr, 
        w_mse=args.w_mse, 
        w_coherence=args.w_coherence
    )
    
    # Print top 10 summary
    print("\n" + "="*80)
    print("TOP 10 CONFIGURATIONS (from filtered results)")
    print("="*80)
    print(f"{'Rank':<6} {'SNR Imp. (dB)':<15} {'MSE':<12} {'Coherence':<12} {'Score':<10}")
    print("-"*80)
    for i, (config, metrics, score, dirname) in enumerate(filtered_results[:10], 1):
        print(f"{i:<6} {metrics['snr_improvement']:<15.2f} {metrics['mse']:<12.2f} {metrics['median_coherence']:<12.4f} {score:<10.4f}")
    
    print("\nDone!")

if __name__ == '__main__':
    main()

