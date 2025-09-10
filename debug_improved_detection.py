#!/usr/bin/env python3
"""Debug the improved detection method to understand why it's over-detecting."""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/Users/bui/code/sparc')

from sparc.methods.template_subtraction.improved_average_template_subtraction import ImprovedAverageTemplateSubtraction
from examples.swec_ethz_template_subtraction import load_swec_ethz_data

def debug_improved_detection():
    """Debug why improved detection is over-detecting artifacts."""
    print("=== Debugging Improved Detection ===")
    
    # Load data
    mixed_data, ground_truth, artifacts, sampling_rate = load_swec_ethz_data('research/datasets/SWEC-ETHZ/selected_clips_pat123.pkl')
    if mixed_data is None:
        return
    
    # Focus on first trial, first channel
    trial_idx = 0
    channel = 0
    
    mixed_signal = mixed_data[trial_idx, :, channel]
    clean_signal = ground_truth[trial_idx, :, channel]
    true_artifact = artifacts[trial_idx, :, channel]
    
    print(f"Signal stats:")
    print(f"  Mixed: mean={np.mean(mixed_signal):.2f}, std={np.std(mixed_signal):.2f}")
    print(f"  Clean: mean={np.mean(clean_signal):.2f}, std={np.std(clean_signal):.2f}")
    print(f"  True artifact: mean={np.mean(true_artifact):.2f}, std={np.std(true_artifact):.2f}")
    
    # Test clean signal estimation
    method = ImprovedAverageTemplateSubtraction(
        sampling_rate=sampling_rate,
        template_length_ms=30,
        onset_threshold=3.0,
        detection_method='clean_based'
    )
    
    print(f"\n=== Clean Signal Estimation ===")
    
    # Test direct estimation from ground truth
    estimated_std_direct = method._estimate_clean_stats(ground_truth[trial_idx:trial_idx+1])
    print(f"Direct estimation from ground truth: {estimated_std_direct:.2f}")
    
    # Test estimation from mixed data
    estimated_std_mixed = method._estimate_clean_stats_from_mixed(mixed_data[trial_idx:trial_idx+1])
    print(f"Estimation from mixed data (MAD): {estimated_std_mixed:.2f}")
    
    # Actual clean std for comparison
    actual_clean_std = np.std(clean_signal[clean_signal != 0])
    print(f"Actual clean std: {actual_clean_std:.2f}")
    
    # Test thresholds
    threshold_direct = 3.0 * estimated_std_direct
    threshold_mixed = 3.0 * estimated_std_mixed
    threshold_actual = 3.0 * actual_clean_std
    
    print(f"\n=== Thresholds ===")
    print(f"Threshold (direct): {threshold_direct:.2f}")
    print(f"Threshold (mixed MAD): {threshold_mixed:.2f}")
    print(f"Threshold (actual clean): {threshold_actual:.2f}")
    
    # True artifact threshold for comparison
    true_artifact_threshold = 3.0 * np.std(true_artifact[true_artifact != 0])
    print(f"True artifact threshold: {true_artifact_threshold:.2f}")
    
    # Test detection with different thresholds
    thresholds_to_test = {
        'Direct estimation': threshold_direct,
        'Mixed MAD estimation': threshold_mixed,
        'Actual clean': threshold_actual,
        'Manual 300': 300.0,  # From our previous analysis
        'Manual 500': 500.0,
        'Manual 1000': 1000.0
    }
    
    print(f"\n=== Detection Results ===")
    detection_results = {}
    
    for name, threshold in thresholds_to_test.items():
        mask = np.abs(mixed_signal) > threshold
        n_points = np.sum(mask)
        pct = n_points / len(mask) * 100
        
        print(f"{name:20s}: {threshold:8.2f} -> {n_points:5d} points ({pct:4.1f}%)")
        detection_results[name] = {
            'threshold': threshold,
            'mask': mask,
            'n_points': n_points,
            'pct': pct
        }
    
    # Compare with ground truth
    true_mask = np.abs(true_artifact) > 3 * np.std(true_artifact[true_artifact != 0])
    true_points = np.sum(true_mask)
    true_pct = true_points / len(true_mask) * 100
    print(f"{'Ground truth':20s}: {'N/A':8s} -> {true_points:5d} points ({true_pct:4.1f}%)")
    
    # Test morphological operations
    print(f"\n=== Morphological Operations Effect ===")
    from scipy.ndimage import binary_closing, binary_dilation
    
    # Use manual threshold of 300 (which seemed good from previous analysis)
    base_mask = np.abs(mixed_signal) > 300.0
    print(f"Base mask (300 threshold): {np.sum(base_mask)} points ({np.sum(base_mask)/len(base_mask)*100:.1f}%)")
    
    # Apply morphological operations with different kernel sizes
    for kernel_size in [5, 10, 20]:
        morpho_mask = binary_closing(base_mask, structure=np.ones(kernel_size))
        morpho_points = np.sum(morpho_mask)
        morpho_pct = morpho_points / len(morpho_mask) * 100
        print(f"  After closing (k={kernel_size}): {morpho_points} points ({morpho_pct:.1f}%)")
    
    # Test with optimal parameters
    print(f"\n=== Testing Optimal Parameters ===")
    
    optimal_method = ImprovedAverageTemplateSubtraction(
        sampling_rate=sampling_rate,
        template_length_ms=30,
        onset_threshold=3.0,
        detection_method='clean_based',
        morphology_kernel_size=5,  # Reduced
        min_region_length=3
    )
    
    # Manually set clean std to actual value
    optimal_method.clean_signal_std_ = actual_clean_std
    
    # Test detection
    artifact_mask = optimal_method._detect_artifacts_single_trial(mixed_data[trial_idx])
    optimal_points = np.sum(artifact_mask)
    optimal_pct = optimal_points / artifact_mask.size * 100
    print(f"Optimal method detection: {optimal_points} points ({optimal_pct:.1f}%)")
    
    # Check regions
    regions = optimal_method._find_artifact_regions(artifact_mask[:, channel])
    print(f"Optimal method regions: {len(regions)}")
    if regions:
        region_lengths = [end - start for start, end in regions]
        print(f"Region lengths: mean={np.mean(region_lengths):.1f}, max={np.max(region_lengths)}, min={np.min(region_lengths)}")
    
    # Plot comparison
    plot_detection_debug(mixed_signal, clean_signal, true_artifact, detection_results, 
                        artifact_mask[:, channel], sampling_rate)
    
    return detection_results

def plot_detection_debug(mixed_signal, clean_signal, true_artifact, detection_results, 
                        optimal_mask, sampling_rate):
    """Plot debug information about detection methods."""
    
    time_s = np.arange(len(mixed_signal)) / sampling_rate
    
    fig, axes = plt.subplots(6, 1, figsize=(15, 12))
    
    # Plot 1: Original signals
    axes[0].plot(time_s, clean_signal, 'b-', label='Clean', alpha=0.7)
    axes[0].plot(time_s, mixed_signal, 'r-', label='Mixed', alpha=0.7)
    axes[0].plot(time_s, true_artifact, 'g-', label='True Artifact', alpha=0.5)
    axes[0].set_title('Original Signals')
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_xlim(0, 0.5)
    
    # Plot 2: Ground truth mask
    true_mask = np.abs(true_artifact) > 3 * np.std(true_artifact[true_artifact != 0])
    axes[1].plot(time_s, true_mask.astype(int), 'g-', linewidth=2, label='Ground Truth')
    axes[1].set_title('Ground Truth Artifact Regions')
    axes[1].legend()
    axes[1].grid(True)
    axes[1].set_xlim(0, 0.5)
    axes[1].set_ylim(-0.1, 1.1)
    
    # Plot 3-5: Different threshold results
    test_names = ['Direct estimation', 'Actual clean', 'Manual 300']
    colors = ['orange', 'purple', 'red']
    
    for i, name in enumerate(test_names[:3]):
        if name in detection_results:
            mask = detection_results[name]['mask']
            threshold = detection_results[name]['threshold']
            pct = detection_results[name]['pct']
            
            axes[i + 2].plot(time_s, mask.astype(int), color=colors[i], linewidth=2, 
                           label=f'{name} (thresh={threshold:.0f}, {pct:.1f}%)')
            axes[i + 2].set_title(f'{name} Detection')
            axes[i + 2].legend()
            axes[i + 2].grid(True)
            axes[i + 2].set_xlim(0, 0.5)
            axes[i + 2].set_ylim(-0.1, 1.1)
    
    # Plot 6: Optimal method result
    axes[5].plot(time_s, optimal_mask.astype(int), 'brown', linewidth=2, label='Optimal Method')
    axes[5].set_title('Optimal Method Detection')
    axes[5].legend()
    axes[5].grid(True)
    axes[5].set_xlim(0, 0.5)
    axes[5].set_ylim(-0.1, 1.1)
    axes[5].set_xlabel('Time (s)')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    results = debug_improved_detection()
