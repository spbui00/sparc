#!/usr/bin/env python3
"""Debug script to investigate template subtraction issues."""

import numpy as np
import matplotlib.pyplot as plt
from sparc import AverageTemplateSubtraction
from examples.swec_ethz_template_subtraction import load_swec_ethz_data, generate_synthetic_artifacts

def debug_artifact_detection():
    """Debug the artifact detection process."""
    print("=== Debugging Template Subtraction ===")
    
    # Load data
    mixed_data, ground_truth, artifacts, sampling_rate = load_swec_ethz_data('research/datasets/SWEC-ETHZ/selected_clips_pat123.pkl')
    if mixed_data is None:
        print("Could not load data")
        return
    
    print(f"Data shape: {mixed_data.shape}")
    print(f"Sampling rate: {sampling_rate}")
    
    # Initialize method with debug parameters
    method = AverageTemplateSubtraction(
        sampling_rate=sampling_rate,
        template_length_ms=2,
        onset_threshold=1.0,
        detection_method='amplitude'
    )
    
    # Test on first trial only
    trial_data = mixed_data[0]  # Shape: (timesteps, channels)
    print(f"Trial data shape: {trial_data.shape}")
    
    # Check for active channels
    active_channels = []
    for ch in range(trial_data.shape[1]):
        if np.any(trial_data[:, ch] != 0):
            active_channels.append(ch)
    print(f"Active channels: {len(active_channels)} out of {trial_data.shape[1]}")
    print(f"First 10 active channels: {active_channels[:10]}")
    
    # Debug artifact detection on first active channel
    if active_channels:
        ch = active_channels[0]
        print(f"\n=== Debugging Channel {ch} ===")
        
        signal_ch = trial_data[:, ch]
        print(f"Signal stats: min={np.min(signal_ch):.2f}, max={np.max(signal_ch):.2f}, "
              f"mean={np.mean(signal_ch):.2f}, std={np.std(signal_ch):.2f}")
        
        # Manual artifact detection
        gradient = np.gradient(signal_ch)
        from scipy import signal as sp_signal
        if len(signal_ch) > 5:
            gradient_smooth = sp_signal.savgol_filter(gradient, window_length=5, polyorder=2)
        else:
            gradient_smooth = gradient
        
        gradient_std = np.std(gradient_smooth)
        threshold = method.onset_threshold * gradient_std
        
        print(f"Gradient std: {gradient_std:.6f}")
        print(f"Threshold: {threshold:.6f}")
        print(f"Max gradient: {np.max(np.abs(gradient_smooth)):.6f}")
        
        # Count points above threshold
        above_threshold = np.sum(np.abs(gradient_smooth) > threshold)
        print(f"Points above threshold: {above_threshold} out of {len(gradient_smooth)}")
        
        # Check actual artifacts vs detected
        if ground_truth is not None:
            true_artifact = mixed_data[0, :, ch] - ground_truth[0, :, ch]
            artifact_std = np.std(true_artifact)
            artifact_threshold = 3 * artifact_std
            true_artifact_points = np.sum(np.abs(true_artifact) > artifact_threshold)
            print(f"True artifact points (3*std): {true_artifact_points}")
            print(f"True artifact std: {artifact_std:.6f}")
    
    # Run full artifact detection
    print(f"\n=== Running Full Artifact Detection ===")
    artifact_mask = method._detect_artifacts_single_trial(trial_data)
    print(f"Artifact mask shape: {artifact_mask.shape}")
    
    total_artifacts = np.sum(artifact_mask)
    print(f"Total artifact points detected: {total_artifacts}")
    
    # Check per channel
    for ch in active_channels[:5]:  # First 5 active channels
        ch_artifacts = np.sum(artifact_mask[:, ch])
        print(f"Channel {ch}: {ch_artifacts} artifact points")
    
    # Try fitting the method
    print(f"\n=== Fitting Method ===")
    try:
        method.fit(mixed_data)
        print("✓ Method fitted successfully")
        
        # Check templates
        if method.templates_:
            print(f"Number of templates learned: {len(method.templates_)}")
            for ch in active_channels[:3]:
                if ch in method.templates_:
                    template = method.templates_[ch]
                    template_norm = np.linalg.norm(template)
                    print(f"Channel {ch} template norm: {template_norm:.6f}")
                    print(f"Channel {ch} template stats: min={np.min(template):.6f}, "
                          f"max={np.max(template):.6f}, mean={np.mean(template):.6f}")
        
        # Test transform
        print(f"\n=== Testing Transform ===")
        cleaned = method.transform(mixed_data)
        print(f"Cleaned data shape: {cleaned.shape}")
        
        # Check if anything changed
        diff = np.mean(np.abs(mixed_data - cleaned))
        print(f"Mean absolute difference: {diff:.6f}")
        
        if diff < 1e-10:
            print("⚠️  WARNING: No significant change detected - method may not be working")
        
    except Exception as e:
        print(f"✗ Error during fitting: {e}")
        import traceback
        traceback.print_exc()

def plot_debug_signals():
    """Plot signals for visual debugging."""
    mixed_data, ground_truth, artifacts, sampling_rate = load_swec_ethz_data('research/datasets/SWEC-ETHZ/selected_clips_pat123.pkl')
    if mixed_data is None:
        return
    
    # Find first active channel
    active_ch = 0
    for ch in range(mixed_data.shape[2]):
        if np.any(mixed_data[0, :, ch] != 0):
            active_ch = ch
            break
    
    time_s = np.arange(mixed_data.shape[1]) / sampling_rate
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    
    # Plot 1: Original signals
    axes[0].plot(time_s, ground_truth[0, :, active_ch], 'b-', label='Clean', alpha=0.7)
    axes[0].plot(time_s, mixed_data[0, :, active_ch], 'r-', label='Mixed', alpha=0.7)
    axes[0].set_title(f'Original Signals (Channel {active_ch})')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot 2: Artifact signal
    artifact_signal = mixed_data[0, :, active_ch] - ground_truth[0, :, active_ch]
    axes[1].plot(time_s, artifact_signal, 'g-', label='Artifact')
    axes[1].set_title('True Artifact Signal')
    axes[1].legend()
    axes[1].grid(True)
    
    # Plot 3: Gradient
    signal_ch = mixed_data[0, :, active_ch]
    gradient = np.gradient(signal_ch)
    axes[2].plot(time_s, gradient, 'purple', label='Gradient')
    axes[2].axhline(y=5.0 * np.std(gradient), color='red', linestyle='--', label='Threshold')
    axes[2].axhline(y=-5.0 * np.std(gradient), color='red', linestyle='--')
    axes[2].set_title('Signal Gradient and Threshold')
    axes[2].legend()
    axes[2].grid(True)
    
    # Plot 4: Artifact detection
    method = AverageTemplateSubtraction(sampling_rate=sampling_rate, template_length_ms=2, onset_threshold=1.0, detection_method='amplitude')
    artifact_mask = method._detect_artifacts_single_trial(mixed_data[0])
    axes[3].plot(time_s, artifact_mask[:, active_ch].astype(int), 'orange', label='Detected Artifacts')
    axes[3].set_title('Detected Artifact Regions')
    axes[3].legend()
    axes[3].grid(True)
    
    axes[3].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    debug_artifact_detection()
    plot_debug_signals()
