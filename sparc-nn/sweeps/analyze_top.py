#!/usr/bin/env python3
"""
Script to analyze hyperparameter sweep results and generate a LaTeX table
of the top 10 configurations based on SNR, MSE, and Median Spectral Coherence.
"""

import re
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional

def parse_results_file(results_path: str) -> Optional[Dict[str, float]]:
    """
    Parse a results.txt file and extract key metrics.
    
    Returns:
        Dictionary with metrics or None if parsing fails
    """
    try:
        with open(results_path, 'r') as f:
            content = f.read()
        
        metrics = {}
        
        # Extract SNR Improvement (required for scoring)
        snr_imp_match = re.search(r'SNR Improvement: ([\d\.-]+) dB', content)
        if snr_imp_match:
            metrics['snr_improvement'] = float(snr_imp_match.group(1))
        else:
            return None
        
        # Extract SNR After (Cleaned) - kept for display
        snr_match = re.search(r'SNR After \(Cleaned\): ([\d\.-]+) dB', content)
        if snr_match:
            metrics['snr_after'] = float(snr_match.group(1))
        
        # Extract MSE
        mse_match = re.search(r'MSE: ([\d\.]+)', content)
        if mse_match:
            metrics['mse'] = float(mse_match.group(1))
        else:
            return None
        
        # Extract Median Spectral Coherence
        coherence_match = re.search(r'Median Spectral Coherence: ([\d\.]+)', content)
        if coherence_match:
            metrics['median_coherence'] = float(coherence_match.group(1))
        else:
            return None
        
        # Extract PSD MSE median (bonus metric)
        psd_mse_match = re.search(r'PSD MSE \(median across channels\): ([\d\.]+)', content)
        if psd_mse_match:
            metrics['psd_mse_median'] = float(psd_mse_match.group(1))
        
        return metrics
    except Exception as e:
        print(f"Error parsing {results_path}: {e}")
        return None

def parse_config_from_dirname(dirname: str) -> Dict[str, float]:
    """
    Parse hyperparameters from directory name.
    
    Example: f_cutoff_25_w_clean_0_w_cosine_1_w_rank_a_0.50_w_rank_s_0.50_w_spectral_0.50_w_spectral_slope_0.50
    """
    config = {}
    
    # Extract f_cutoff
    match = re.search(r'f_cutoff_([\d\.]+)', dirname)
    if match:
        config['f_cutoff'] = float(match.group(1))
    
    # Extract w_cosine
    match = re.search(r'w_cosine_([\d\.]+)', dirname)
    if match:
        config['w_cosine'] = float(match.group(1))
    
    # Extract w_rank_a
    match = re.search(r'w_rank_a_([\d\.]+)', dirname)
    if match:
        config['w_rank_a'] = float(match.group(1))
    
    # Extract w_rank_s
    match = re.search(r'w_rank_s_([\d\.]+)', dirname)
    if match:
        config['w_rank_s'] = float(match.group(1))
    
    # Extract w_spectral
    match = re.search(r'w_spectral_([\d\.]+)', dirname)
    if match:
        config['w_spectral'] = float(match.group(1))
    
    # Extract w_spectral_slope
    match = re.search(r'w_spectral_slope_([\d\.]+)', dirname)
    if match:
        config['w_spectral_slope'] = float(match.group(1))
    
    return config

def compute_composite_score(snr_improvement: float, mse: float, median_coherence: float, 
                           snr_range: Tuple[float, float], mse_range: Tuple[float, float], 
                           coherence_range: Tuple[float, float],
                           w_snr: float = 1.0, w_mse: float = 0.3, w_coherence: float = 0.3) -> float:
    """
    Compute a composite score for ranking.
    Higher is better.
    
    Normalize each metric using actual ranges from the sweep:
    - SNR Improvement: higher is better
    - MSE: lower is better (invert: use 1 - normalized_mse)
    - Coherence: higher is better
    
    Weighted combination:
    score = w_snr * normalized_snr + w_mse * normalized_mse + w_coherence * normalized_coherence
    """
    snr_min, snr_max = snr_range
    mse_min, mse_max = mse_range
    coherence_min, coherence_max = coherence_range
    
    # Normalize SNR Improvement (higher is better)
    if snr_max > snr_min:
        normalized_snr = (snr_improvement - snr_min) / (snr_max - snr_min)
    else:
        normalized_snr = 0.5  # If all values are the same
    normalized_snr = max(0, min(1, normalized_snr))  # Clamp to [0, 1]
    
    # Normalize MSE (lower is better, so invert)
    if mse_max > mse_min:
        normalized_mse_raw = (mse - mse_min) / (mse_max - mse_min)
        normalized_mse = 1.0 - normalized_mse_raw  # Invert: lower MSE -> higher score
    else:
        normalized_mse = 0.5  # If all values are the same
    normalized_mse = max(0, min(1, normalized_mse))  # Clamp to [0, 1]
    
    # Normalize Coherence (higher is better)
    if coherence_max > coherence_min:
        normalized_coherence = (median_coherence - coherence_min) / (coherence_max - coherence_min)
    else:
        normalized_coherence = 0.5  # If all values are the same
    normalized_coherence = max(0, min(1, normalized_coherence))  # Clamp to [0, 1]
    
    # Weighted combination
    score = w_snr * normalized_snr + w_mse * normalized_mse + w_coherence * normalized_coherence
    
    return score

def compute_sweep_ranges(all_results: List[Tuple[Dict, Dict, float, str]]) -> Dict[str, List[float]]:
    """
    Compute the unique discrete values of hyperparameters covered in the sweep.
    
    Returns:
        Dictionary mapping parameter names to sorted lists of unique values
    """
    unique_values = {}
    
    # Collect all values for each parameter
    param_values = {
        'f_cutoff': [],
        'w_cosine': [],
        'w_rank_a': [],
        'w_rank_s': [],
        'w_spectral': [],
        'w_spectral_slope': []
    }
    
    for config, _, _, _ in all_results:
        for param in param_values.keys():
            if param in config:
                param_values[param].append(config[param])
    
    # Get unique sorted values for each parameter
    for param, values in param_values.items():
        if values:
            unique_values[param] = sorted(set(values))
    
    return unique_values

def analyze_sweep(sweep_dir: str = 'sweep_1', top_n: int = 30, 
                  w_snr: float = 1.0, w_mse: float = 0.3, w_coherence: float = 0.3) -> Tuple[List[Tuple[Dict, Dict, float, str]], Dict[str, List[float]]]:
    """
    Analyze all results in the sweep directory and return top N configurations.
    
    Returns:
        Tuple of (top_results_list, sweep_ranges_dict)
    """
    sweep_path = Path(sweep_dir)
    if not sweep_path.exists():
        raise ValueError(f"Sweep directory {sweep_dir} does not exist")
    
    all_results = []
    
    # Find all results.txt files
    results_files = glob.glob(str(sweep_path / '*/results.txt'))
    
    print(f"Found {len(results_files)} result files")
    
    # First pass: collect all metrics to compute actual ranges
    all_snr_improvement_values = []
    all_mse_values = []
    all_coherence_values = []
    
    for results_file in results_files:
        # Get directory name
        dir_path = Path(results_file).parent
        dirname = dir_path.name
        
        # Parse metrics
        metrics = parse_results_file(results_file)
        if metrics is None:
            print(f"Warning: Could not parse {results_file}")
            continue
        
        # Collect values for range computation
        all_snr_improvement_values.append(metrics['snr_improvement'])
        all_mse_values.append(metrics['mse'])
        all_coherence_values.append(metrics['median_coherence'])
        
        # Parse config from directory name
        config = parse_config_from_dirname(dirname)
        
        all_results.append((config, metrics, None, dirname))  # Score will be computed later
    
    # Compute actual ranges from the data
    snr_range = (min(all_snr_improvement_values), max(all_snr_improvement_values)) if all_snr_improvement_values else (0, 0)
    mse_range = (min(all_mse_values), max(all_mse_values)) if all_mse_values else (0, 0)
    coherence_range = (min(all_coherence_values), max(all_coherence_values)) if all_coherence_values else (0, 0)
    
    print(f"\nActual metric ranges:")
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
    
    # Compute sweep ranges
    sweep_ranges = compute_sweep_ranges(all_results)
    
    # Sort by composite score (descending)
    all_results.sort(key=lambda x: x[2], reverse=True)
    
    # Return top N and ranges
    return all_results[:top_n], sweep_ranges

def generate_latex_table(top_results: List[Tuple[Dict, Dict, float, str]], sweep_ranges: Dict[str, List[float]], 
                         output_file: str = 'sweep_top10_table.tex',
                         w_snr: float = 1.0, w_mse: float = 0.3, w_coherence: float = 0.3):
    """
    Generate a LaTeX table from the top results.
    """
    latex_lines = []
    
    # Table header
    latex_lines.append("\\begin{table}[h]")
    latex_lines.append("\\centering")
    num_results = len(top_results)
    caption = f"Top {num_results} Hyperparameter Configurations Based on Composite Score (weights: SNR Improvement={w_snr}, MSE={w_mse}, Coherence={w_coherence})"
    latex_lines.append(f"\\caption{{{caption}}}")
    
    # Add sweep ranges as a note
    if sweep_ranges:
        ranges_text = "Hyperparameter values tested: "
        range_parts = []
        if 'f_cutoff' in sweep_ranges:
            values = sweep_ranges['f_cutoff']
            values_str = ", ".join([f"{v:.0f}" for v in values])
            range_parts.append(f"$f_c \\in \\{{{values_str}\\}}$ Hz")
        if 'w_cosine' in sweep_ranges:
            values = sweep_ranges['w_cosine']
            values_str = ", ".join([f"{v:.2f}" for v in values])
            range_parts.append(f"$w_{{\\text{{cosine}}}} \\in \\{{{values_str}\\}}$")
        if 'w_rank_a' in sweep_ranges:
            values = sweep_ranges['w_rank_a']
            values_str = ", ".join([f"{v:.2f}" for v in values])
            range_parts.append(f"$w_{{\\text{{artifact rank}}}} \\in \\{{{values_str}\\}}$")
        if 'w_rank_s' in sweep_ranges:
            values = sweep_ranges['w_rank_s']
            values_str = ", ".join([f"{v:.2f}" for v in values])
            range_parts.append(f"$w_{{\\text{{neural rank}}}} \\in \\{{{values_str}\\}}$")
        if 'w_spectral' in sweep_ranges:
            values = sweep_ranges['w_spectral']
            values_str = ", ".join([f"{v:.2f}" for v in values])
            range_parts.append(f"$w_{{\\text{{spectral}}}} \\in \\{{{values_str}\\}}$")
        if 'w_spectral_slope' in sweep_ranges:
            values = sweep_ranges['w_spectral_slope']
            values_str = ", ".join([f"{v:.2f}" for v in values])
            range_parts.append(f"$w_{{\\text{{slope}}}} \\in \\{{{values_str}\\}}$")
        
        ranges_text += ", ".join(range_parts) + "."
        latex_lines.append(f"\\footnotesize{{{ranges_text}}}")
    
    latex_lines.append(f"\\label{{tab:sweep_top{num_results}}}")
    latex_lines.append("\\resizebox{\\textwidth}{!}{%")
    latex_lines.append("\\begin{tabular}{cccccc|cccc}")
    latex_lines.append("\\toprule")
    latex_lines.append("\\textbf{$f_c$ (Hz)} & \\textbf{$w_{\\text{cosine}}$} & \\textbf{$w_{\\text{artifact rank}}$} & \\textbf{$w_{\\text{neural rank}}$} & \\textbf{$w_{\\text{spectral}}$} & \\textbf{$w_{\\text{slope}}$} & \\textbf{SNR Imp.} & \\textbf{MSE} & \\textbf{Coherence} & \\textbf{Score} \\\\")
    latex_lines.append("\\midrule")
    
    # Table rows
    for config, metrics, score, dirname in top_results:
        f_cutoff = config.get('f_cutoff', 0)
        w_cosine = config.get('w_cosine', 0)
        w_rank_a = config.get('w_rank_a', 0)
        w_rank_s = config.get('w_rank_s', 0)
        w_spectral = config.get('w_spectral', 0)
        w_spectral_slope = config.get('w_spectral_slope', 0)
        
        snr_improvement = metrics['snr_improvement']
        mse = metrics['mse']
        median_coherence = metrics['median_coherence']
        
        # Format numbers
        row = f"{f_cutoff:.0f} & {w_cosine:.2f} & {w_rank_a:.2f} & {w_rank_s:.2f} & {w_spectral:.2f} & {w_spectral_slope:.2f} & {snr_improvement:.2f} & {mse:.2f} & {median_coherence:.4f} & {score:.4f} \\\\"
        latex_lines.append(row)
    
    # Table footer
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("}%")
    latex_lines.append("\\end{table}")
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(latex_lines))
    
    print(f"\nLaTeX table saved to: {output_file}")
    
    # Print sweep ranges
    print("\n" + "="*80)
    print("SWEEP HYPERPARAMETER VALUES TESTED")
    print("="*80)
    for param, values in sorted(sweep_ranges.items()):
        param_name = param.replace('_', ' ').title()
        values_str = ", ".join([f"{v:.2f}" if v < 1 or v != int(v) else f"{int(v)}" for v in values])
        print(f"{param_name:<25} {{{values_str}}}")
    
    # Also print a summary
    print("\n" + "="*80)
    print(f"TOP {len(top_results)} CONFIGURATIONS SUMMARY")
    print("="*80)
    print(f"{'SNR Imp. (dB)':<15} {'MSE':<12} {'Coherence':<12} {'Score':<10}")
    print("-"*80)
    for config, metrics, score, dirname in top_results:
        print(f"{metrics['snr_improvement']:<15.2f} {metrics['mse']:<12.2f} {metrics['median_coherence']:<12.4f} {score:<10.4f}")
    
    return output_file

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze hyperparameter sweep and generate LaTeX table')
    parser.add_argument('--sweep-dir', type=str, default='sweep_1',
                        help='Directory containing sweep results (default: sweep_1)')
    parser.add_argument('--top-n', type=int, default=30,
                        help='Number of top configurations to include (default: 30)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output LaTeX file (default: sweep_top{top_n}_table.tex based on --top-n)')
    parser.add_argument('--w-snr', type=float, default=1.0,
                        help='Weight for SNR Improvement in composite score (default: 1.0)')
    parser.add_argument('--w-mse', type=float, default=0.3,
                        help='Weight for MSE in composite score (default: 0.3)')
    parser.add_argument('--w-coherence', type=float, default=0.3,
                        help='Weight for Coherence in composite score (default: 0.3)')
    
    args = parser.parse_args()
    
    # Set default output filename based on top_n if not provided
    if args.output is None:
        args.output = f'sweep_top{args.top_n}_table.tex'
    
    print("Analyzing sweep results...")
    print(f"Score weights: SNR Improvement={args.w_snr}, MSE={args.w_mse}, Coherence={args.w_coherence}")
    top_results, sweep_ranges = analyze_sweep(
        args.sweep_dir, 
        args.top_n,
        w_snr=args.w_snr,
        w_mse=args.w_mse,
        w_coherence=args.w_coherence
    )
    
    if not top_results:
        print("No valid results found!")
        return
    
    print(f"\nGenerating LaTeX table for top {args.top_n} configurations...")
    generate_latex_table(top_results, sweep_ranges, args.output,
                        w_snr=args.w_snr, w_mse=args.w_mse, w_coherence=args.w_coherence)
    
    print("\nDone!")

if __name__ == '__main__':
    main()

