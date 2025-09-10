#!/usr/bin/env python3
"""Analyze the characteristics of synthetic artifacts to design better detection."""

import numpy as np
import matplotlib.pyplot as plt
from examples.swec_ethz_template_subtraction import load_swec_ethz_data

def analyze_artifact_characteristics():
    """Analyze the synthetic artifacts to understand their properties."""
    print("=== Analyzing Synthetic Artifact Characteristics ===")
    
    mixed_data, ground_truth, artifacts, sampling_rate = load_swec_ethz_data('research/datasets/SWEC-ETHZ/selected_clips_pat123.pkl')
    if mixed_data is None:
        return
    
    # Focus on first trial, first active channel
    trial_idx = 0
    channel = 0
    
    clean_signal = ground_truth[trial_idx, :, channel]
    mixed_signal = mixed_data[trial_idx, :, channel]
    artifact_signal = artifacts[trial_idx, :, channel]
    
    print(f"Clean signal stats: min={np.min(clean_signal):.2f}, max={np.max(clean_signal):.2f}, "
          f"std={np.std(clean_signal):.2f}, rms={np.sqrt(np.mean(clean_signal**2)):.2f}")
    print(f"Artifact signal stats: min={np.min(artifact_signal):.2f}, max={np.max(artifact_signal):.2f}, "
          f"std={np.std(artifact_signal):.2f}, rms={np.sqrt(np.mean(artifact_signal**2)):.2f}")
    print(f"Mixed signal stats: min={np.min(mixed_signal):.2f}, max={np.max(mixed_signal):.2f}, "
          f"std={np.std(mixed_signal):.2f}, rms={np.sqrt(np.mean(mixed_signal**2)):.2f}")
    
    # Analyze artifact amplitude relative to clean signal
    artifact_to_clean_ratio = np.std(artifact_signal) / np.std(clean_signal)
    print(f"Artifact-to-clean std ratio: {artifact_to_clean_ratio:.2f}")
    
    # Find artifact regions using amplitude-based detection
    artifact_threshold_factors = [1, 2, 3, 5, 10]
    
    for factor in artifact_threshold_factors:
        threshold = factor * np.std(clean_signal)
        artifact_mask = np.abs(artifact_signal) > threshold
        artifact_points = np.sum(artifact_mask)
        percentage = (artifact_points / len(artifact_signal)) * 100
        print(f"Threshold {factor}x clean std ({threshold:.2f}): {artifact_points} points ({percentage:.1f}%)")
    
    # Try mixed signal thresholding
    print(f"\nMixed signal thresholding:")
    for factor in artifact_threshold_factors:
        threshold = factor * np.std(mixed_signal)
        artifact_mask = np.abs(mixed_signal) > threshold
        artifact_points = np.sum(artifact_mask)
        percentage = (artifact_points / len(mixed_signal)) * 100
        print(f"Threshold {factor}x mixed std ({threshold:.2f}): {artifact_points} points ({percentage:.1f}%)")
    
    # Analyze gradient-based detection (current method)
    gradient = np.gradient(mixed_signal)
    gradient_std = np.std(gradient)
    print(f"\nGradient analysis:")
    print(f"Gradient std: {gradient_std:.2f}")
    
    for factor in [1, 2, 3, 5, 10]:
        threshold = factor * gradient_std
        gradient_mask = np.abs(gradient) > threshold
        gradient_points = np.sum(gradient_mask)
        percentage = (gradient_points / len(gradient)) * 100
        print(f"Gradient threshold {factor}x std ({threshold:.2f}): {gradient_points} points ({percentage:.1f}%)")
    
    # Plot comparison
    time_s = np.arange(len(mixed_signal)) / sampling_rate
    
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    
    # Plot 1: All signals
    axes[0].plot(time_s, clean_signal, 'b-', label='Clean', alpha=0.7)
    axes[0].plot(time_s, artifact_signal, 'r-', label='Artifact', alpha=0.7)
    axes[0].plot(time_s, mixed_signal, 'g-', label='Mixed', alpha=0.7)
    axes[0].set_title('Signal Comparison')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot 2: Artifact detection methods
    axes[1].plot(time_s, np.abs(artifact_signal), 'r-', label='|Artifact|', alpha=0.7)
    axes[1].axhline(y=3*np.std(clean_signal), color='blue', linestyle='--', label='3x Clean STD')
    axes[1].axhline(y=2*np.std(clean_signal), color='cyan', linestyle='--', label='2x Clean STD')
    axes[1].set_title('Amplitude-based Detection')
    axes[1].legend()
    axes[1].grid(True)
    
    # Plot 3: Gradient analysis
    axes[2].plot(time_s, np.abs(gradient), 'purple', label='|Gradient|', alpha=0.7)
    axes[2].axhline(y=5*gradient_std, color='red', linestyle='--', label='5x Gradient STD (current)')
    axes[2].axhline(y=2*gradient_std, color='orange', linestyle='--', label='2x Gradient STD')
    axes[2].set_title('Gradient-based Detection')
    axes[2].legend()
    axes[2].grid(True)
    
    # Plot 4: Detection comparison
    artifact_mask_amp = np.abs(artifact_signal) > 2*np.std(clean_signal)
    artifact_mask_grad = np.abs(gradient) > 2*gradient_std
    
    axes[3].plot(time_s, artifact_mask_amp.astype(int), 'b-', label='Amplitude-based', linewidth=2)
    axes[3].plot(time_s, artifact_mask_grad.astype(int) + 0.1, 'r-', label='Gradient-based', linewidth=2)
    axes[3].set_title('Detection Method Comparison')
    axes[3].legend()
    axes[3].grid(True)
    axes[3].set_xlabel('Time (s)')
    
    plt.tight_layout()
    plt.show()
    
    return mixed_data, ground_truth, artifacts

if __name__ == "__main__":
    analyze_artifact_characteristics()
