#!/usr/bin/env python3
"""Test the fixed improved template subtraction method."""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/Users/bui/code/sparc')

from sparc import AverageTemplateSubtraction
from sparc.methods.template_subtraction.improved_average_template_subtraction import ImprovedAverageTemplateSubtraction
from examples.swec_ethz_template_subtraction import load_swec_ethz_data

def test_fixed_method():
    """Test the fixed improved method with adjusted parameters."""
    print("=== Testing Fixed Improved Method ===")
    
    # Load data
    mixed_data, ground_truth, artifacts, sampling_rate = load_swec_ethz_data('research/datasets/SWEC-ETHZ/selected_clips_pat123.pkl')
    if mixed_data is None:
        return
    
    trial_idx = 0
    channel = 0
    
    print(f"Data loaded: {mixed_data.shape}")
    
    # Test different configurations of the improved method
    methods = {
        'Original (best)': {
            'class': AverageTemplateSubtraction,
            'params': {
                'sampling_rate': sampling_rate,
                'template_length_ms': 20,
                'onset_threshold': 1.0,
                'detection_method': 'amplitude'
            }
        },
        'Fixed Improved (8x)': {
            'class': ImprovedAverageTemplateSubtraction,
            'params': {
                'sampling_rate': sampling_rate,
                'onset_threshold': 8.0,  # Default now
            }
        },
        'Fixed Improved (6x)': {
            'class': ImprovedAverageTemplateSubtraction,
            'params': {
                'sampling_rate': sampling_rate,
                'onset_threshold': 6.0,
            }
        },
        'Fixed Improved (10x)': {
            'class': ImprovedAverageTemplateSubtraction,
            'params': {
                'sampling_rate': sampling_rate,
                'onset_threshold': 10.0,
            }
        },
        'Fixed Improved (conservative)': {
            'class': ImprovedAverageTemplateSubtraction,
            'params': {
                'sampling_rate': sampling_rate,
                'onset_threshold': 12.0,
                'morphology_kernel_size': 2,  # Even smaller
                'min_region_length': 3
            }
        }
    }
    
    results = {}
    
    # Test each method
    for method_name, config in methods.items():
        print(f"\n--- Testing {method_name} ---")
        
        try:
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
            n_channels_detected = np.sum(np.any(artifact_mask, axis=0))
            
            print(f"  Total artifact points: {n_artifact_points}")
            print(f"  Channels with artifacts: {n_channels_detected}/88")
            
            # Analyze channel 0 in detail
            ch0_mask = artifact_mask[:, channel]
            ch0_regions = method._find_artifact_regions(ch0_mask)
            ch0_points = np.sum(ch0_mask)
            
            print(f"  Channel 0: {ch0_points} points, {len(ch0_regions)} regions")
            if ch0_regions:
                region_lengths = [end - start for start, end in ch0_regions]
                print(f"  Region lengths: mean={np.mean(region_lengths):.1f}, range={np.min(region_lengths)}-{np.max(region_lengths)}")
            
            # Check template quality
            if channel in method.templates_:
                template = method.templates_[channel]
                template_norm = np.linalg.norm(template)
                template_std = np.std(template)
                template_energy = np.sum(template**2)
                
                print(f"  Template: length={len(template)}, norm={template_norm:.1f}, std={template_std:.1f}, energy={template_energy:.1f}")
            
            # Test performance
            cleaned = method.transform(mixed_data[trial_idx:trial_idx+1])
            cleaned_signal = cleaned[0, :, channel]
            mixed_signal = mixed_data[trial_idx, :, channel]
            clean_signal = ground_truth[trial_idx, :, channel]
            
            # Compute metrics
            mse_original = np.mean((mixed_signal - clean_signal)**2)
            mse_cleaned = np.mean((cleaned_signal - clean_signal)**2)
            mse_improvement = mse_original - mse_cleaned
            mse_improvement_pct = (mse_improvement / mse_original) * 100
            
            # Signal energy comparison
            total_change = np.sum(np.abs(mixed_signal - cleaned_signal))
            signal_energy = np.sum(np.abs(clean_signal))
            change_ratio = total_change / signal_energy if signal_energy > 0 else 0
            
            print(f"  MSE improvement: {mse_improvement:.1f} ({mse_improvement_pct:.1f}%)")
            print(f"  Total signal change: {total_change:.1f}")
            print(f"  Change/signal ratio: {change_ratio:.3f}")
            
            results[method_name] = {
                'method': method,
                'cleaned': cleaned[0],
                'n_artifact_points': n_artifact_points,
                'n_channels_detected': n_channels_detected,
                'mse_improvement_pct': mse_improvement_pct,
                'total_change': total_change,
                'change_ratio': change_ratio,
                'ch0_regions': len(ch0_regions),
                'ch0_points': ch0_points
            }
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Compare with ground truth
    print(f"\n=== Ground Truth Reference ===")
    true_artifact = artifacts[trial_idx, :, channel]
    true_threshold = 3 * np.std(true_artifact[true_artifact != 0])
    true_mask = np.abs(true_artifact) > true_threshold
    true_points = np.sum(true_mask)
    print(f"True artifact points: {true_points} ({true_points/len(true_mask)*100:.1f}%)")
    
    # Performance summary
    if results:
        print(f"\n=== Performance Summary ===")
        sorted_methods = sorted(results.keys(), key=lambda k: results[k]['mse_improvement_pct'], reverse=True)
        
        for method_name in sorted_methods:
            r = results[method_name]
            print(f"{method_name:25s}: {r['mse_improvement_pct']:5.1f}% MSE, {r['n_artifact_points']:6d} points, {r['ch0_regions']:3d} regions, ratio {r['change_ratio']:.3f}")
        
        # Plot comparison
        plot_fixed_comparison(mixed_data[trial_idx], ground_truth[trial_idx], 
                            artifacts[trial_idx], results, sampling_rate, channel)
    
    return results

def plot_fixed_comparison(mixed_data, ground_truth, artifacts, results, sampling_rate, channel):
    """Plot comparison of the fixed methods."""
    
    time_s = np.arange(mixed_data.shape[0]) / sampling_rate
    
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
    axes[0].set_xlim(0, 0.5)
    
    # Plot results
    colors = ['purple', 'orange', 'brown', 'pink', 'cyan']
    sorted_methods = sorted(results.keys(), key=lambda k: results[k]['mse_improvement_pct'], reverse=True)
    
    for i, method_name in enumerate(sorted_methods):
        result = results[method_name]
        cleaned_signal = result['cleaned'][:, channel]
        perf = result['mse_improvement_pct']
        
        ax = axes[i + 1]
        ax.plot(time_s, ground_truth[:, channel], 'b--', label='Ground Truth', alpha=0.5)
        ax.plot(time_s, cleaned_signal, color=colors[i % len(colors)], 
               label=f'Cleaned ({perf:.1f}% improvement)', linewidth=1.5)
        ax.set_title(f'{method_name} - {result["n_artifact_points"]} points, {result["ch0_regions"]} regions')
        ax.legend()
        ax.grid(True)
        ax.set_xlim(0, 0.5)
    
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    results = test_fixed_method()
