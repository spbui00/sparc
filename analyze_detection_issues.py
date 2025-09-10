#!/usr/bin/env python3
"""Analyze the specific issues with artifact detection."""

import numpy as np
import matplotlib.pyplot as plt
from sparc import AverageTemplateSubtraction
from examples.swec_ethz_template_subtraction import load_swec_ethz_data
from scipy import signal as sp_signal

def analyze_detection_issues():
    """Analyze why artifact detection is failing."""
    print("=== Analyzing Artifact Detection Issues ===")
    
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
    
    # Test current detection methods
    method = AverageTemplateSubtraction(
        sampling_rate=sampling_rate,
        template_length_ms=20,
        onset_threshold=1.0,
        detection_method='amplitude'
    )
    
    # Manual amplitude detection
    print(f"\n=== Manual Amplitude Detection ===")
    signal_std = np.std(mixed_signal)
    signal_median = np.median(np.abs(mixed_signal))
    
    threshold_std = method.onset_threshold * signal_std
    threshold_median = method.onset_threshold * signal_median * 3
    threshold = min(threshold_std, threshold_median)
    
    print(f"Signal std: {signal_std:.2f}")
    print(f"Signal median: {signal_median:.2f}")
    print(f"Threshold (std-based): {threshold_std:.2f}")
    print(f"Threshold (median-based): {threshold_median:.2f}")
    print(f"Final threshold: {threshold:.2f}")
    
    amplitude_mask = np.abs(mixed_signal) > threshold
    print(f"Points above amplitude threshold: {np.sum(amplitude_mask)}")
    
    # Manual gradient detection
    print(f"\n=== Manual Gradient Detection ===")
    gradient = np.gradient(mixed_signal)
    if len(mixed_signal) > 5:
        gradient_smooth = sp_signal.savgol_filter(gradient, window_length=5, polyorder=2)
    else:
        gradient_smooth = gradient
    gradient_std = np.std(gradient_smooth)
    gradient_threshold = method.onset_threshold * gradient_std
    
    print(f"Gradient std: {gradient_std:.2f}")
    print(f"Gradient threshold: {gradient_threshold:.2f}")
    print(f"Max gradient: {np.max(np.abs(gradient_smooth)):.2f}")
    
    gradient_mask = np.abs(gradient_smooth) > gradient_threshold
    print(f"Points above gradient threshold: {np.sum(gradient_mask)}")
    
    # Compare with ground truth
    print(f"\n=== Ground Truth Comparison ===")
    true_artifact_threshold = 3 * np.std(true_artifact)
    true_mask = np.abs(true_artifact) > true_artifact_threshold
    print(f"True artifact threshold: {true_artifact_threshold:.2f}")
    print(f"True artifact points: {np.sum(true_mask)}")
    
    # Find continuous regions in ground truth
    def find_continuous_regions(mask, min_gap=5):
        regions = []
        in_region = False
        start_idx = 0
        
        for i in range(len(mask)):
            if mask[i] and not in_region:
                start_idx = i
                in_region = True
            elif not mask[i] and in_region:
                # Check if gap is small (part of same artifact)
                gap_start = i
                gap_end = gap_start
                while gap_end < len(mask) and not mask[gap_end] and gap_end - gap_start < min_gap:
                    gap_end += 1
                
                if gap_end < len(mask) and mask[gap_end]:
                    # Small gap, continue region
                    continue
                else:
                    # End of region
                    regions.append((start_idx, i))
                    in_region = False
        
        if in_region:
            regions.append((start_idx, len(mask)))
        
        return regions
    
    true_regions = find_continuous_regions(true_mask)
    amplitude_regions = find_continuous_regions(amplitude_mask)
    gradient_regions = find_continuous_regions(gradient_mask)
    
    print(f"True artifact regions: {len(true_regions)}")
    print(f"Amplitude detection regions: {len(amplitude_regions)}")
    print(f"Gradient detection regions: {len(gradient_regions)}")
    
    if true_regions:
        print(f"True region lengths: {[end-start for start, end in true_regions[:5]]}")
        print(f"True region mean length: {np.mean([end-start for start, end in true_regions]):.1f} samples")
    
    if amplitude_regions:
        print(f"Amplitude region lengths: {[end-start for start, end in amplitude_regions[:5]]}")
        print(f"Amplitude region mean length: {np.mean([end-start for start, end in amplitude_regions]):.1f} samples")
    
    if gradient_regions:
        print(f"Gradient region lengths: {[end-start for start, end in gradient_regions[:5]]}")
        print(f"Gradient region mean length: {np.mean([end-start for start, end in gradient_regions]):.1f} samples")
    
    # Test improved detection method
    print(f"\n=== Improved Detection Method ===")
    
    # Method 1: Use true artifact-based threshold
    artifact_based_threshold = np.std(clean_signal) * 3  # 3 std of clean signal
    artifact_mask_improved = np.abs(mixed_signal) > artifact_based_threshold
    improved_regions = find_continuous_regions(artifact_mask_improved)
    
    print(f"Improved threshold (3*clean_std): {artifact_based_threshold:.2f}")
    print(f"Improved detection regions: {len(improved_regions)}")
    if improved_regions:
        print(f"Improved region mean length: {np.mean([end-start for start, end in improved_regions]):.1f} samples")
    
    # Method 2: Use morphological operations to connect nearby detections
    from scipy.ndimage import binary_dilation, binary_closing
    
    morpho_mask = binary_closing(amplitude_mask, structure=np.ones(10))  # Connect within 10 samples
    morpho_mask = binary_dilation(morpho_mask, structure=np.ones(5))     # Expand by 5 samples
    morpho_regions = find_continuous_regions(morpho_mask)
    
    print(f"Morphological regions: {len(morpho_regions)}")
    if morpho_regions:
        print(f"Morphological region mean length: {np.mean([end-start for start, end in morpho_regions]):.1f} samples")
    
    # Plot comparison
    plot_detection_comparison(mixed_signal, clean_signal, true_artifact, 
                             amplitude_mask, gradient_mask, true_mask, 
                             artifact_mask_improved, morpho_mask, 
                             sampling_rate)
    
    return {
        'amplitude_regions': amplitude_regions,
        'gradient_regions': gradient_regions,
        'true_regions': true_regions,
        'improved_regions': improved_regions,
        'morpho_regions': morpho_regions
    }

def plot_detection_comparison(mixed_signal, clean_signal, true_artifact,
                            amplitude_mask, gradient_mask, true_mask,
                            improved_mask, morpho_mask, sampling_rate):
    """Plot comparison of different detection methods."""
    
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
    
    # Plot 2: True artifact mask
    axes[1].plot(time_s, true_mask.astype(int), 'g-', linewidth=2, label='True Artifact Mask')
    axes[1].set_title('Ground Truth Artifact Regions')
    axes[1].legend()
    axes[1].grid(True)
    axes[1].set_xlim(0, 0.5)
    axes[1].set_ylim(-0.1, 1.1)
    
    # Plot 3: Current amplitude detection
    axes[2].plot(time_s, amplitude_mask.astype(int), 'orange', linewidth=2, label='Amplitude Detection')
    axes[2].set_title('Current Amplitude Detection')
    axes[2].legend()
    axes[2].grid(True)
    axes[2].set_xlim(0, 0.5)
    axes[2].set_ylim(-0.1, 1.1)
    
    # Plot 4: Current gradient detection
    axes[3].plot(time_s, gradient_mask.astype(int), 'purple', linewidth=2, label='Gradient Detection')
    axes[3].set_title('Current Gradient Detection')
    axes[3].legend()
    axes[3].grid(True)
    axes[3].set_xlim(0, 0.5)
    axes[3].set_ylim(-0.1, 1.1)
    
    # Plot 5: Improved detection
    axes[4].plot(time_s, improved_mask.astype(int), 'red', linewidth=2, label='Improved Detection')
    axes[4].set_title('Improved Detection (3*clean_std)')
    axes[4].legend()
    axes[4].grid(True)
    axes[4].set_xlim(0, 0.5)
    axes[4].set_ylim(-0.1, 1.1)
    
    # Plot 6: Morphological detection
    axes[5].plot(time_s, morpho_mask.astype(int), 'brown', linewidth=2, label='Morphological Detection')
    axes[5].set_title('Morphological Detection (Connected)')
    axes[5].legend()
    axes[5].grid(True)
    axes[5].set_xlim(0, 0.5)
    axes[5].set_ylim(-0.1, 1.1)
    axes[5].set_xlabel('Time (s)')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    results = analyze_detection_issues()
