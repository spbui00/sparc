#!/usr/bin/env python3
"""Debug script to test different template parameters."""

import numpy as np
import matplotlib.pyplot as plt
from sparc import AverageTemplateSubtraction
from examples.swec_ethz_template_subtraction import load_swec_ethz_data

def test_template_parameters():
    """Test different template parameters to understand the issues."""
    print("=== Testing Template Parameters ===")
    
    # Load data
    mixed_data, ground_truth, artifacts, sampling_rate = load_swec_ethz_data('research/datasets/SWEC-ETHZ/selected_clips_pat123.pkl')
    if mixed_data is None:
        print("Could not load data")
        return
    
    # Test different template lengths
    template_lengths_ms = [2, 5, 10, 20, 50]
    onset_thresholds = [1.0, 2.0, 5.0]
    
    # Focus on first trial, first active channel
    trial_idx = 0
    channel = 0
    
    # Find first active channel
    for ch in range(mixed_data.shape[2]):
        if np.any(mixed_data[trial_idx, :, ch] != 0):
            channel = ch
            break
    
    mixed_signal = mixed_data[trial_idx, :, channel]
    clean_signal = ground_truth[trial_idx, :, channel]
    true_artifact = artifacts[trial_idx, :, channel]
    
    print(f"Using trial {trial_idx}, channel {channel}")
    print(f"Mixed signal stats: mean={np.mean(mixed_signal):.2f}, std={np.std(mixed_signal):.2f}")
    print(f"True artifact stats: mean={np.mean(true_artifact):.2f}, std={np.std(true_artifact):.2f}")
    
    # Test each combination
    results = {}
    
    for template_len in template_lengths_ms:
        for threshold in onset_thresholds:
            config_name = f"len_{template_len}ms_thresh_{threshold}"
            print(f"\n--- Testing {config_name} ---")
            
            try:
                method = AverageTemplateSubtraction(
                    sampling_rate=sampling_rate,
                    template_length_ms=template_len,
                    onset_threshold=threshold,
                    detection_method='amplitude'
                )
                
                # Fit on single trial for debugging
                method.fit(mixed_data[trial_idx:trial_idx+1])
                
                # Check artifact detection
                if isinstance(method.template_indices_, list):
                    artifact_mask = method.template_indices_[0][:, channel]
                else:
                    artifact_mask = method.template_indices_[:, channel]
                
                n_artifact_points = np.sum(artifact_mask)
                artifact_regions = method._find_artifact_regions(artifact_mask)
                n_regions = len(artifact_regions)
                
                # Check template quality
                template = method.templates_.get(channel, np.zeros(method.template_length_samples))
                template_norm = np.linalg.norm(template)
                template_std = np.std(template)
                
                print(f"  Template length: {len(template)} samples ({template_len}ms)")
                print(f"  Artifact points detected: {n_artifact_points}")
                print(f"  Artifact regions: {n_regions}")
                print(f"  Template norm: {template_norm:.2f}")
                print(f"  Template std: {template_std:.2f}")
                
                if n_regions > 0:
                    region_lengths = [end - start for start, end in artifact_regions]
                    avg_region_length = np.mean(region_lengths)
                    print(f"  Avg region length: {avg_region_length:.1f} samples")
                
                # Test cleaning
                cleaned = method.transform(mixed_data[trial_idx:trial_idx+1])
                cleaned_signal = cleaned[0, :, channel]
                
                # Compute performance
                mse_original = np.mean((mixed_signal - clean_signal)**2)
                mse_cleaned = np.mean((cleaned_signal - clean_signal)**2)
                improvement = mse_original - mse_cleaned
                improvement_pct = improvement / mse_original * 100
                
                print(f"  MSE improvement: {improvement:.2f} ({improvement_pct:.1f}%)")
                
                results[config_name] = {
                    'template_len': template_len,
                    'threshold': threshold,
                    'n_artifact_points': n_artifact_points,
                    'n_regions': n_regions,
                    'template_norm': template_norm,
                    'template_std': template_std,
                    'mse_improvement': improvement,
                    'improvement_pct': improvement_pct,
                    'cleaned_signal': cleaned_signal.copy()
                }
                
            except Exception as e:
                print(f"  ✗ Failed: {e}")
                results[config_name] = {'error': str(e)}
    
    # Find best configuration
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    if valid_results:
        best_config = max(valid_results.keys(), key=lambda k: valid_results[k]['improvement_pct'])
        print(f"\n=== Best Configuration: {best_config} ===")
        best_result = valid_results[best_config]
        for key, value in best_result.items():
            if key != 'cleaned_signal':
                print(f"  {key}: {value}")
    
    # Plot comparison of different configurations
    if len(valid_results) > 0:
        plot_parameter_comparison(mixed_signal, clean_signal, true_artifact, valid_results, sampling_rate)
    
    return results

def analyze_artifact_structure():
    """Analyze the structure of synthetic artifacts to understand expected templates."""
    print("\n=== Analyzing Artifact Structure ===")
    
    # Load data
    mixed_data, ground_truth, artifacts, sampling_rate = load_swec_ethz_data('research/datasets/SWEC-ETHZ/selected_clips_pat123.pkl')
    if mixed_data is None:
        return
    
    # Focus on first active channel
    channel = 0
    for ch in range(mixed_data.shape[2]):
        if np.any(artifacts[0, :, ch] != 0):
            channel = ch
            break
    
    artifact_signal = artifacts[0, :, channel]
    
    # Find artifact events
    threshold = 3 * np.std(artifact_signal)
    artifact_points = np.where(np.abs(artifact_signal) > threshold)[0]
    
    print(f"Channel {channel} artifact analysis:")
    print(f"  Artifact threshold: {threshold:.2f}")
    print(f"  Artifact points: {len(artifact_points)} ({len(artifact_points)/len(artifact_signal)*100:.1f}%)")
    
    if len(artifact_points) > 1:
        # Analyze spacing
        gaps = np.diff(artifact_points)
        large_gaps = gaps[gaps > 5]  # More than 5 samples apart
        
        if len(large_gaps) > 0:
            print(f"  Mean spacing: {np.mean(large_gaps):.1f} samples ({np.mean(large_gaps)/sampling_rate*1000:.1f} ms)")
            print(f"  Expected 130 Hz period: {sampling_rate/130:.1f} samples ({1000/130:.1f} ms)")
        
        # Extract individual artifact events
        artifact_events = []
        event_starts = []
        
        i = 0
        while i < len(artifact_points):
            start_idx = artifact_points[i]
            
            # Find end of this event (consecutive points)
            end_idx = start_idx
            while i + 1 < len(artifact_points) and artifact_points[i + 1] - artifact_points[i] <= 2:
                i += 1
                end_idx = artifact_points[i]
            
            # Extract the event with some padding
            event_start = max(0, start_idx - 5)
            event_end = min(len(artifact_signal), end_idx + 10)
            event = artifact_signal[event_start:event_end]
            
            if len(event) > 10:  # Only keep events with reasonable length
                artifact_events.append(event)
                event_starts.append(event_start)
            
            i += 1
        
        print(f"  Found {len(artifact_events)} individual artifact events")
        
        if artifact_events:
            # Analyze event characteristics
            event_lengths = [len(event) for event in artifact_events]
            print(f"  Event lengths: {np.min(event_lengths)} - {np.max(event_lengths)} samples")
            print(f"  Mean event length: {np.mean(event_lengths):.1f} samples ({np.mean(event_lengths)/sampling_rate*1000:.1f} ms)")
            
            # Show first few events
            print(f"  First 5 events:")
            for i, event in enumerate(artifact_events[:5]):
                print(f"    Event {i+1}: length={len(event)}, max_amp={np.max(np.abs(event)):.1f}, energy={np.sum(event**2):.1f}")
            
            # Recommend template length
            recommended_length_samples = int(np.mean(event_lengths) * 1.5)  # 1.5x average event length
            recommended_length_ms = recommended_length_samples / sampling_rate * 1000
            print(f"  Recommended template length: {recommended_length_samples} samples ({recommended_length_ms:.1f} ms)")
            
            return artifact_events, event_starts, recommended_length_ms
    
    return None, None, None

def plot_parameter_comparison(mixed_signal, clean_signal, true_artifact, results, sampling_rate):
    """Plot comparison of different parameter configurations."""
    time_s = np.arange(len(mixed_signal)) / sampling_rate
    
    # Select best few configurations to plot
    sorted_configs = sorted(results.keys(), key=lambda k: results[k]['improvement_pct'], reverse=True)
    top_configs = sorted_configs[:4]  # Top 4 configurations
    
    fig, axes = plt.subplots(len(top_configs) + 1, 1, figsize=(12, 3 * (len(top_configs) + 1)))
    
    # Plot 1: Original signals
    axes[0].plot(time_s, clean_signal, 'b-', label='Clean', alpha=0.7)
    axes[0].plot(time_s, mixed_signal, 'r-', label='Mixed', alpha=0.7)
    axes[0].plot(time_s, true_artifact, 'g-', label='True Artifact', alpha=0.5)
    axes[0].set_title('Original Signals')
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_xlim(0, 0.5)  # First 500ms
    
    # Plot cleaned signals
    colors = ['purple', 'orange', 'brown', 'pink']
    for i, config_name in enumerate(top_configs):
        result = results[config_name]
        cleaned_signal = result['cleaned_signal']
        improvement = result['improvement_pct']
        
        axes[i + 1].plot(time_s, clean_signal, 'b--', label='Ground Truth', alpha=0.5)
        axes[i + 1].plot(time_s, cleaned_signal, color=colors[i], 
                        label=f'Cleaned ({improvement:.1f}% improvement)', linewidth=1.5)
        axes[i + 1].set_title(f'{config_name}')
        axes[i + 1].legend()
        axes[i + 1].grid(True)
        axes[i + 1].set_xlim(0, 0.5)  # First 500ms
    
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # First analyze the artifact structure to understand what we're dealing with
    events, starts, recommended_ms = analyze_artifact_structure()
    
    # Then test different parameters
    results = test_template_parameters()
