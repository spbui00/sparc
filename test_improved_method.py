#!/usr/bin/env python3
"""Test script to compare original and improved template subtraction methods."""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the improved methods to path
sys.path.insert(0, '/Users/bui/code/sparc')

from sparc import AverageTemplateSubtraction
from sparc.methods.template_subtraction.improved_average_template_subtraction import ImprovedAverageTemplateSubtraction
from examples.swec_ethz_template_subtraction import load_swec_ethz_data

def test_method_comparison():
    """Compare original vs improved template subtraction methods."""
    print("=== Comparing Original vs Improved Methods ===")
    
    # Load data
    mixed_data, ground_truth, artifacts, sampling_rate = load_swec_ethz_data('research/datasets/SWEC-ETHZ/selected_clips_pat123.pkl')
    if mixed_data is None:
        print("Could not load data")
        return
    
    # Focus on first trial for detailed comparison
    trial_idx = 0
    
    print(f"Data loaded: {mixed_data.shape}")
    print(f"Sampling rate: {sampling_rate} Hz")
    
    # Test configurations
    methods = {
        'Original (2ms)': {
            'class': AverageTemplateSubtraction,
            'params': {
                'sampling_rate': sampling_rate,
                'template_length_ms': 2,
                'onset_threshold': 1.0,
                'detection_method': 'amplitude'
            }
        },
        'Original (20ms)': {
            'class': AverageTemplateSubtraction,
            'params': {
                'sampling_rate': sampling_rate,
                'template_length_ms': 20,
                'onset_threshold': 1.0,
                'detection_method': 'amplitude'
            }
        },
        'Improved (default)': {
            'class': ImprovedAverageTemplateSubtraction,
            'params': {
                'sampling_rate': sampling_rate,
                # Uses defaults: template_length_ms=50, clean_based detection
            }
        },
        'Improved (30ms)': {
            'class': ImprovedAverageTemplateSubtraction,
            'params': {
                'sampling_rate': sampling_rate,
                'template_length_ms': 30,
                'onset_threshold': 3.0,
                'detection_method': 'clean_based'
            }
        }
    }
    
    results = {}
    
    # Test each method
    for method_name, config in methods.items():
        print(f"\n--- Testing {method_name} ---")
        
        try:
            # Initialize method
            method_class = config['class']
            method = method_class(**config['params'])
            
            # Fit with ground truth for improved methods
            if isinstance(method, ImprovedAverageTemplateSubtraction):
                method.fit(mixed_data[trial_idx:trial_idx+1], ground_truth=ground_truth[trial_idx:trial_idx+1])
            else:
                method.fit(mixed_data[trial_idx:trial_idx+1])
            
            # Analyze detection
            if isinstance(method.template_indices_, list):
                artifact_mask = method.template_indices_[0]
            else:
                artifact_mask = method.template_indices_
            
            n_artifact_points = np.sum(artifact_mask)
            
            # Analyze templates for first few active channels
            active_channels = []
            for ch in range(min(5, mixed_data.shape[2])):
                if np.any(mixed_data[trial_idx, :, ch] != 0):
                    active_channels.append(ch)
            
            print(f"  Total artifact points detected: {n_artifact_points}")
            print(f"  Templates learned: {len(method.templates_)}")
            
            # Check template quality for first active channel
            if active_channels and active_channels[0] in method.templates_:
                ch = active_channels[0]
                template = method.templates_[ch]
                template_norm = np.linalg.norm(template)
                template_std = np.std(template)
                artifact_regions = method._find_artifact_regions(artifact_mask[:, ch])
                
                print(f"  Channel {ch} template: length={len(template)}, norm={template_norm:.2f}, std={template_std:.2f}")
                print(f"  Channel {ch} regions: {len(artifact_regions)}")
                if artifact_regions:
                    region_lengths = [end - start for start, end in artifact_regions]
                    print(f"  Channel {ch} region lengths: mean={np.mean(region_lengths):.1f}, max={np.max(region_lengths)}")
            
            # Test cleaning performance
            cleaned = method.transform(mixed_data[trial_idx:trial_idx+1])
            cleaned_signal = cleaned[0]
            mixed_signal = mixed_data[trial_idx]
            clean_signal = ground_truth[trial_idx]
            
            # Compute performance metrics
            performance = compute_performance_metrics(mixed_signal, cleaned_signal, clean_signal, active_channels)
            
            print(f"  MSE improvement: {performance['mse_improvement']:.2f} ({performance['mse_improvement_pct']:.1f}%)")
            print(f"  SNR improvement: {performance['snr_improvement_db']:.2f} dB")
            print(f"  Artifact removal: {performance['artifact_removal_ratio']:.3f}")
            
            results[method_name] = {
                'method': method,
                'cleaned': cleaned_signal,
                'performance': performance,
                'n_artifact_points': n_artifact_points,
                'active_channels': active_channels
            }
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Plot comparison
    if results:
        plot_method_comparison(mixed_data[trial_idx], ground_truth[trial_idx], artifacts[trial_idx], 
                             results, sampling_rate)
    
    return results

def compute_performance_metrics(mixed_signal, cleaned_signal, clean_signal, active_channels):
    """Compute performance metrics for comparison."""
    metrics = {}
    
    # Focus on active channels only
    if active_channels:
        mixed_active = mixed_signal[:, active_channels]
        cleaned_active = cleaned_signal[:, active_channels]
        clean_active = clean_signal[:, active_channels]
    else:
        mixed_active = mixed_signal
        cleaned_active = cleaned_signal
        clean_active = clean_signal
    
    # MSE
    mse_original = np.mean((mixed_active - clean_active)**2)
    mse_cleaned = np.mean((cleaned_active - clean_active)**2)
    mse_improvement = mse_original - mse_cleaned
    mse_improvement_pct = (mse_improvement / mse_original) * 100 if mse_original > 0 else 0
    
    # SNR
    signal_power = np.mean(clean_active**2)
    noise_power_original = np.mean((mixed_active - clean_active)**2)
    noise_power_cleaned = np.mean((cleaned_active - clean_active)**2)
    
    snr_original = 10 * np.log10(signal_power / noise_power_original) if noise_power_original > 0 else float('inf')
    snr_cleaned = 10 * np.log10(signal_power / noise_power_cleaned) if noise_power_cleaned > 0 else float('inf')
    snr_improvement_db = snr_cleaned - snr_original
    
    # Artifact removal ratio
    artifact_power_original = np.mean((mixed_active - clean_active)**2)
    artifact_power_remaining = np.mean((cleaned_active - clean_active)**2)
    artifact_removal_ratio = 1 - (artifact_power_remaining / artifact_power_original) if artifact_power_original > 0 else 0
    
    metrics.update({
        'mse_improvement': mse_improvement,
        'mse_improvement_pct': mse_improvement_pct,
        'snr_improvement_db': snr_improvement_db,
        'artifact_removal_ratio': artifact_removal_ratio
    })
    
    return metrics

def plot_method_comparison(mixed_data, ground_truth, artifacts, results, sampling_rate):
    """Plot comparison of different methods."""
    
    # Find first active channel
    channel = 0
    for ch in range(mixed_data.shape[1]):
        if np.any(mixed_data[:, ch] != 0):
            channel = ch
            break
    
    time_s = np.arange(mixed_data.shape[0]) / sampling_rate
    
    # Create subplot for each method + original
    n_plots = len(results) + 1
    fig, axes = plt.subplots(n_plots, 1, figsize=(15, 4 * n_plots))
    if n_plots == 1:
        axes = [axes]
    
    # Plot 1: Original signals
    axes[0].plot(time_s, ground_truth[:, channel], 'b-', label='Clean', alpha=0.7)
    axes[0].plot(time_s, mixed_data[:, channel], 'r-', label='Mixed', alpha=0.7)
    axes[0].plot(time_s, artifacts[:, channel], 'g-', label='True Artifact', alpha=0.5)
    axes[0].set_title('Original Signals')
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_xlim(0, 0.5)  # First 500ms
    
    # Plot results for each method
    colors = ['purple', 'orange', 'brown', 'pink']
    sorted_methods = sorted(results.keys(), key=lambda k: results[k]['performance']['mse_improvement_pct'], reverse=True)
    
    for i, method_name in enumerate(sorted_methods):
        result = results[method_name]
        cleaned_signal = result['cleaned'][:, channel]
        perf = result['performance']
        
        ax = axes[i + 1]
        ax.plot(time_s, ground_truth[:, channel], 'b--', label='Ground Truth', alpha=0.5)
        ax.plot(time_s, cleaned_signal, color=colors[i % len(colors)], 
               label=f'Cleaned ({perf["mse_improvement_pct"]:.1f}% improvement)', linewidth=1.5)
        ax.set_title(f'{method_name} - SNR: {perf["snr_improvement_db"]:.1f}dB, Removal: {perf["artifact_removal_ratio"]:.3f}')
        ax.legend()
        ax.grid(True)
        ax.set_xlim(0, 0.5)  # First 500ms
    
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"\n=== Performance Summary ===")
    for method_name in sorted_methods:
        perf = results[method_name]['performance']
        print(f"{method_name:20s}: {perf['mse_improvement_pct']:6.1f}% MSE, {perf['snr_improvement_db']:6.1f}dB SNR, {perf['artifact_removal_ratio']:.3f} removal")

if __name__ == "__main__":
    results = test_method_comparison()
