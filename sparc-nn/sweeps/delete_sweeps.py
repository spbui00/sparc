#!/usr/bin/env python3
"""
Script to delete sweep results matching specific hyperparameter ranges.
"""

import os
import re
import shutil
import glob
from pathlib import Path
from typing import Dict, List

def parse_config_from_dirname(dirname: str) -> Dict[str, float]:
    """
    Parse hyperparameters from directory name.
    
    Example: f_cutoff_25_w_cosine_1_w_rank_a_0.50_w_rank_s_0.50_w_spectral_0.50_w_spectral_slope_0.50
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

def find_matching_directories(sweep_dir: str, filter_ranges: Dict[str, List[float]]) -> List[str]:
    """
    Find all directories in sweep_dir that match the filter ranges.
    
    Returns:
        List of directory paths that match
    """
    sweep_path = Path(sweep_dir)
    if not sweep_path.exists():
        raise ValueError(f"Sweep directory {sweep_dir} does not exist")
    
    matching_dirs = []
    
    # Find all subdirectories
    for item in sweep_path.iterdir():
        if item.is_dir():
            dirname = item.name
            config = parse_config_from_dirname(dirname)
            
            # Check if config matches filter ranges
            if config_matches_ranges(config, filter_ranges):
                matching_dirs.append(str(item))
    
    return matching_dirs

def delete_sweeps(sweep_dir: str = 'sweep_1', 
                 filter_ranges: Dict[str, List[float]] = None,
                 dry_run: bool = True,
                 confirm: bool = True):
    """
    Delete sweep directories matching the filter ranges.
    
    Args:
        sweep_dir: Directory containing sweep results
        filter_ranges: Dictionary mapping parameter names to allowed values
        dry_run: If True, only print what would be deleted without actually deleting
        confirm: If True, ask for confirmation before deleting
    """
    if filter_ranges is None:
        filter_ranges = {}
    
    print("="*80)
    print("DELETE SWEEP RESULTS")
    print("="*80)
    print(f"Sweep directory: {sweep_dir}")
    print(f"Filter ranges:")
    for param, values in filter_ranges.items():
        print(f"  {param}: {values}")
    print(f"Dry run: {dry_run}")
    print("="*80)
    
    matching_dirs = find_matching_directories(sweep_dir, filter_ranges)
    
    if not matching_dirs:
        print("\nNo matching directories found.")
        return
    
    print(f"\nFound {len(matching_dirs)} matching directories:")
    for i, dir_path in enumerate(matching_dirs, 1):
        dirname = Path(dir_path).name
        print(f"  {i}. {dirname}")
    
    if dry_run:
        print(f"\n[DRY RUN] Would delete {len(matching_dirs)} directories.")
        print("Run with --no-dry-run to actually delete them.")
        return
    
    if confirm:
        response = input(f"\nAre you sure you want to delete {len(matching_dirs)} directories? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Cancelled.")
            return
    
    # Delete directories
    deleted_count = 0
    failed_count = 0
    
    for dir_path in matching_dirs:
        try:
            shutil.rmtree(dir_path)
            deleted_count += 1
            print(f"Deleted: {Path(dir_path).name}")
        except Exception as e:
            print(f"Failed to delete {Path(dir_path).name}: {e}")
            failed_count += 1
    
    print(f"\n{'='*80}")
    print(f"Deletion complete!")
    print(f"Deleted: {deleted_count}")
    print(f"Failed: {failed_count}")
    print(f"Total: {len(matching_dirs)}")
    print(f"{'='*80}")

def main():
    import argparse
    
    # Default filter ranges from hyperparameter_sweep.py
    DEFAULT_FILTER_RANGES = {
        'f_cutoff': [10, 25],
        'w_cosine': [1, 2],
        'w_rank_s': [0],
        'w_spectral': [2, 3],
        'w_spectral_slope': [0, 0.2],
        'w_rank_a': [0.2, 1],
    }
    
    parser = argparse.ArgumentParser(description='Delete sweep results matching specific hyperparameter ranges')
    parser.add_argument('--sweep-dir', type=str, default='sweep_1',
                        help='Directory containing sweep results (default: sweep_1)')
    parser.add_argument('--no-dry-run', action='store_true',
                        help='Actually delete directories (default: dry run)')
    parser.add_argument('--no-confirm', action='store_true',
                        help='Skip confirmation prompt (default: ask for confirmation)')
    parser.add_argument('--f-cutoff', type=float, nargs='+', default=None,
                        help='Filter by f_cutoff values (e.g., --f-cutoff 10 25)')
    parser.add_argument('--w-cosine', type=float, nargs='+', default=None,
                        help='Filter by w_cosine values (e.g., --w-cosine 1 2)')
    parser.add_argument('--w-rank-s', type=float, nargs='+', default=None,
                        help='Filter by w_rank_s values (e.g., --w-rank-s 0)')
    parser.add_argument('--w-spectral', type=float, nargs='+', default=None,
                        help='Filter by w_spectral values (e.g., --w-spectral 2 3)')
    parser.add_argument('--w-spectral-slope', type=float, nargs='+', default=None,
                        help='Filter by w_spectral_slope values (e.g., --w-spectral-slope 0 0.2)')
    parser.add_argument('--w-rank-a', type=float, nargs='+', default=None,
                        help='Filter by w_rank_a values (e.g., --w-rank-a 0.2 1)')
    parser.add_argument('--use-defaults', action='store_true',
                        help='Use default filter ranges from hyperparameter_sweep.py')
    
    args = parser.parse_args()
    
    # Build filter ranges
    if args.use_defaults:
        filter_ranges = DEFAULT_FILTER_RANGES
    else:
        filter_ranges = {}
        if args.f_cutoff is not None:
            filter_ranges['f_cutoff'] = args.f_cutoff
        if args.w_cosine is not None:
            filter_ranges['w_cosine'] = args.w_cosine
        if args.w_rank_s is not None:
            filter_ranges['w_rank_s'] = args.w_rank_s
        if args.w_spectral is not None:
            filter_ranges['w_spectral'] = args.w_spectral
        if args.w_spectral_slope is not None:
            filter_ranges['w_spectral_slope'] = args.w_spectral_slope
        if args.w_rank_a is not None:
            filter_ranges['w_rank_a'] = args.w_rank_a
        
        # If no filters specified, use defaults
        if not filter_ranges:
            filter_ranges = DEFAULT_FILTER_RANGES
    
    delete_sweeps(
        sweep_dir=args.sweep_dir,
        filter_ranges=filter_ranges,
        dry_run=not args.no_dry_run,
        confirm=not args.no_confirm
    )

if __name__ == '__main__':
    main()

