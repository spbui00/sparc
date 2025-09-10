import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from sparc import AverageTemplateSubtraction
from tqdm import tqdm


def load_eraasr_data(data_path='research/datasets/eraasr-1.0.0/exampleDataTensor.mat'):
    try:
        data = scipy.io.loadmat(data_path)
        
        # Extract the main data array: (trials, timesteps, channels)
        mixed_data = data['data_trials_by_time_by_channels']
        n_trials, n_timesteps, n_channels = mixed_data.shape
        
        sampling_rate = 30000  # 30 kHz
        stim_onset_sample = 1500  # ~50ms in (1500 samples)
        stim_duration_samples = int(60 * sampling_rate / 1000)  # 60ms duration
        pulse_spacing_samples = int(3 * sampling_rate / 1000)  # 3ms between pulses
        n_pulses = 20
        
        # Calculate stimulation window
        stim_start = stim_onset_sample
        stim_end = stim_start + stim_duration_samples
        
        print(f"Loaded ERAASR data:")
        print(f"  Shape: {mixed_data.shape}")
        print(f"  Sampling rate: {sampling_rate} Hz")
        print(f"  Duration per trial: {n_timesteps / sampling_rate * 1000:.1f} ms")
        print(f"  Trials: {n_trials}, Channels: {n_channels}")
        print(f"  Stim onset: sample {stim_start} ({stim_start/sampling_rate*1000:.1f} ms)")
        print(f"  Stim end: sample {stim_end} ({stim_end/sampling_rate*1000:.1f} ms)")
        print(f"  Stim duration: {stim_duration_samples} samples ({stim_duration_samples/sampling_rate*1000:.1f} ms)")
        print(f"  Time range: -50ms to +200ms relative to stim onset")
        
        return mixed_data, sampling_rate, stim_start, stim_end
        
    except FileNotFoundError:
        print(f"ERAASR data file not found at {data_path}")
        return None, None, None, None
    except Exception as e:
        print(f"Error loading ERAASR data: {str(e)}")
        return None, None, None, None


def create_artifact_indices(data_shape, stim_start, stim_end):
    n_trials, n_timesteps, n_channels = data_shape
    artifact_indices = []
    
    for trial in range(n_trials):
        trial_mask = np.zeros((n_timesteps, n_channels), dtype=bool)
        trial_mask[stim_start:stim_end, :] = True
        artifact_indices.append(trial_mask)
    
    return artifact_indices


def demonstrate_eraasr_template_subtraction():
    """Demonstrate template subtraction on ERAASR dataset"""
    
    # Load ERAASR data
    mixed_data, sampling_rate, stim_start, stim_end = load_eraasr_data()
    
    if mixed_data is None:
        print("Could not load ERAASR data. Please ensure the dataset is available.")
        return None, None
    
    # Create artifact indices for known stimulation periods
    artifact_indices = create_artifact_indices(mixed_data.shape, stim_start, stim_end)
    
    print(f"\nArtifact regions:")
    print(f"  Stimulation period: samples {stim_start}-{stim_end}")
    print(f"  Artifact percentage: {np.mean([np.mean(mask) for mask in artifact_indices]) * 100:.1f}%")
    
    methods = {
        '2ms': AverageTemplateSubtraction(
            sampling_rate=sampling_rate,
            template_length_ms=2,
            num_templates_for_avg=5
        )
    }
    
    cleaned_signals = {}
    
    print(f"\n=== Testing Template Subtraction Methods on ERAASR ===")
    
    for method_name, method in methods.items():
        print(f"\n--- Testing {method_name} ---")
        
        try:
            # Fit the method
            print("  Fitting method...")
            method.fit(mixed_data)
            
            # Transform the data
            print("  Applying artifact removal...")
            cleaned = method.transform(mixed_data)
            cleaned_signals[method_name] = cleaned
            
            # Calculate some basic statistics
            original_power = np.mean(mixed_data ** 2)
            cleaned_power = np.mean(cleaned ** 2)
            power_reduction = (original_power - cleaned_power) / original_power * 100
            
            # Calculate artifact region statistics
            stim_region_original = mixed_data[:, stim_start:stim_end, :]
            stim_region_cleaned = cleaned[:, stim_start:stim_end, :]
            
            stim_power_original = np.mean(stim_region_original ** 2)
            stim_power_cleaned = np.mean(stim_region_cleaned ** 2)
            stim_power_reduction = (stim_power_original - stim_power_cleaned) / stim_power_original * 100
            
            print(f"✓ {method_name} completed successfully")
            print(f"  Overall power reduction: {power_reduction:.1f}%")
            print(f"  Stimulation region power reduction: {stim_power_reduction:.1f}%")
            print(f"  Max amplitude reduction: {(np.max(np.abs(mixed_data)) - np.max(np.abs(cleaned))) / np.max(np.abs(mixed_data)) * 100:.1f}%")
            
        except Exception as e:
            print(f"✗ {method_name} failed: {str(e)}")
            continue
    
    return cleaned_signals, mixed_data, sampling_rate, stim_start, stim_end


def plot_eraasr_results(mixed_data, cleaned_signals, sampling_rate, stim_start, stim_end,
                       trial_idx=0, channel=0, zoom_ms=50):
    """Plot ERAASR template subtraction results"""
    
    if mixed_data is None or not cleaned_signals:
        print("No data or results to plot")
        return
    
    # Create time vector
    n_timesteps = mixed_data.shape[1]
    time_ms = np.arange(n_timesteps) / sampling_rate * 1000 - 50  # Relative to stim onset
    
    # Find an active channel if the specified one is inactive
    if not np.any(mixed_data[trial_idx, :, channel]):
        print(f"Warning: Channel {channel} appears to be inactive. Finding active channel...")
        for ch in range(mixed_data.shape[2]):
            if np.any(mixed_data[trial_idx, :, ch]):
                channel = ch
                print(f"Using channel {channel}")
                break
    
    # Determine zoom window
    stim_onset_ms = stim_start / sampling_rate * 1000 - 50  # Relative to stim onset (should be ~0)
    zoom_start_ms = stim_onset_ms - zoom_ms/2
    zoom_end_ms = stim_onset_ms + zoom_ms/2
    
    zoom_mask = (time_ms >= zoom_start_ms) & (time_ms <= zoom_end_ms)
    
    # Create plots
    n_methods = len(cleaned_signals)
    fig, axes = plt.subplots(n_methods + 1, 2, figsize=(15, 4 * (n_methods + 1)))
    
    if n_methods == 0:
        return
    
    fig.suptitle(f'ERAASR Template Subtraction Results\n(Trial {trial_idx}, Channel {channel})', fontsize=16)
    
    # Plot 1: Original signal (full and zoomed)
    ax_full, ax_zoom = axes[0]
    
    # Full view
    ax_full.plot(time_ms, mixed_data[trial_idx, :, channel], 'b-', alpha=0.7, linewidth=0.5)
    ax_full.axvspan(stim_start/sampling_rate*1000-50, stim_end/sampling_rate*1000-50, 
                    alpha=0.2, color='red', label='Stimulation period')
    ax_full.set_title('Original Signal (Full View)')
    ax_full.set_ylabel('Amplitude (μV)')
    ax_full.legend()
    ax_full.grid(True, alpha=0.3)
    
    # Zoomed view
    ax_zoom.plot(time_ms[zoom_mask], mixed_data[trial_idx, zoom_mask, channel], 'b-', alpha=0.7)
    ax_zoom.axvspan(stim_start/sampling_rate*1000-50, stim_end/sampling_rate*1000-50, 
                    alpha=0.2, color='red', label='Stimulation period')
    ax_zoom.set_title(f'Original Signal (Zoomed: ±{zoom_ms/2:.0f}ms around stim)')
    ax_zoom.set_ylabel('Amplitude (μV)')
    ax_zoom.legend()
    ax_zoom.grid(True, alpha=0.3)
    
    # Plot cleaned signals
    colors = ['green', 'purple', 'orange', 'brown', 'pink']
    
    for i, (method_name, cleaned) in enumerate(cleaned_signals.items()):
        ax_full, ax_zoom = axes[i + 1]
        color = colors[i % len(colors)]
        
        # Full view
        ax_full.plot(time_ms, mixed_data[trial_idx, :, channel], 'b-', 
                    alpha=0.3, linewidth=0.5, label='Original')
        ax_full.plot(time_ms, cleaned[trial_idx, :, channel], 
                    color=color, alpha=0.8, linewidth=0.5, label=f'Cleaned ({method_name})')
        ax_full.axvspan(stim_start/sampling_rate*1000-50, stim_end/sampling_rate*1000-50, 
                        alpha=0.2, color='red')
        ax_full.set_title(f'After {method_name} (Full View)')
        ax_full.set_ylabel('Amplitude (μV)')
        ax_full.legend()
        ax_full.grid(True, alpha=0.3)
        
        # Zoomed view
        ax_zoom.plot(time_ms[zoom_mask], mixed_data[trial_idx, zoom_mask, channel], 
                    'b-', alpha=0.3, label='Original')
        ax_zoom.plot(time_ms[zoom_mask], cleaned[trial_idx, zoom_mask, channel], 
                    color=color, alpha=0.8, label=f'Cleaned ({method_name})')
        ax_zoom.axvspan(stim_start/sampling_rate*1000-50, stim_end/sampling_rate*1000-50, 
                        alpha=0.2, color='red')
        ax_zoom.set_title(f'After {method_name} (Zoomed)')
        ax_zoom.set_ylabel('Amplitude (μV)')
        ax_zoom.legend()
        ax_zoom.grid(True, alpha=0.3)
    
    # Set x-axis labels
    for ax in axes[-1]:
        ax.set_xlabel('Time relative to stim onset (ms)')
    
    plt.tight_layout()
    plt.show()


def analyze_artifact_removal(mixed_data, cleaned_signals, stim_start, stim_end):
    """Analyze the effectiveness of artifact removal"""
    
    print(f"\n=== Artifact Removal Analysis ===")
    
    # Pre-stim, stim, and post-stim periods
    pre_stim_end = stim_start
    post_stim_start = stim_end
    
    pre_stim_data = mixed_data[:, :pre_stim_end, :]
    stim_data = mixed_data[:, stim_start:stim_end, :]
    post_stim_data = mixed_data[:, post_stim_start:, :]
    
    print(f"Analysis periods:")
    print(f"  Pre-stim: samples 0-{pre_stim_end} ({pre_stim_end/30:.1f} ms)")
    print(f"  Stim: samples {stim_start}-{stim_end} ({(stim_end-stim_start)/30:.1f} ms)")
    print(f"  Post-stim: samples {post_stim_start}-end ({(mixed_data.shape[1]-post_stim_start)/30:.1f} ms)")
    
    for method_name, cleaned in cleaned_signals.items():
        print(f"\n--- {method_name} ---")
        
        # Calculate RMS in each period
        pre_stim_rms_orig = np.sqrt(np.mean(pre_stim_data ** 2))
        pre_stim_rms_clean = np.sqrt(np.mean(cleaned[:, :pre_stim_end, :] ** 2))
        
        stim_rms_orig = np.sqrt(np.mean(stim_data ** 2))
        stim_rms_clean = np.sqrt(np.mean(cleaned[:, stim_start:stim_end, :] ** 2))
        
        post_stim_rms_orig = np.sqrt(np.mean(post_stim_data ** 2))
        post_stim_rms_clean = np.sqrt(np.mean(cleaned[:, post_stim_start:, :] ** 2))
        
        print(f"  Pre-stim RMS: {pre_stim_rms_orig:.1f} → {pre_stim_rms_clean:.1f} μV "
              f"({(pre_stim_rms_orig-pre_stim_rms_clean)/pre_stim_rms_orig*100:+.1f}%)")
        print(f"  Stim RMS: {stim_rms_orig:.1f} → {stim_rms_clean:.1f} μV "
              f"({(stim_rms_orig-stim_rms_clean)/stim_rms_orig*100:+.1f}%)")
        print(f"  Post-stim RMS: {post_stim_rms_orig:.1f} → {post_stim_rms_clean:.1f} μV "
              f"({(post_stim_rms_orig-post_stim_rms_clean)/post_stim_rms_orig*100:+.1f}%)")
        
        # Calculate peak-to-peak reduction in stim period
        stim_p2p_orig = np.max(stim_data) - np.min(stim_data)
        stim_p2p_clean = np.max(cleaned[:, stim_start:stim_end, :]) - np.min(cleaned[:, stim_start:stim_end, :])
        
        print(f"  Stim peak-to-peak: {stim_p2p_orig:.1f} → {stim_p2p_clean:.1f} μV "
              f"({(stim_p2p_orig-stim_p2p_clean)/stim_p2p_orig*100:+.1f}%)")


def main():
    """Run ERAASR template subtraction demonstration"""
    
    print("ERAASR Template Subtraction Demonstration")
    print("=" * 50)
    
    # Run the demonstration
    cleaned_signals, mixed_data, sampling_rate, stim_start, stim_end = demonstrate_eraasr_template_subtraction()
    
    if cleaned_signals and mixed_data is not None:
        # Find an active channel for visualization
        active_channel = 0
        for ch in range(min(10, mixed_data.shape[2])):
            if np.any(mixed_data[0, :, ch]):
                active_channel = ch
                break
        
        print(f"\nGenerating plots for channel {active_channel}...")
        
        # Plot results
        plot_eraasr_results(mixed_data, cleaned_signals, sampling_rate, 
                           stim_start, stim_end, channel=active_channel)
        
        # Analyze artifact removal effectiveness
        analyze_artifact_removal(mixed_data, cleaned_signals, stim_start, stim_end)
        
        # Summary
        print(f"\n=== Summary ===")
        print(f"Successfully tested {len(cleaned_signals)} template subtraction methods")
        print(f"Dataset: {mixed_data.shape[0]} trials, {mixed_data.shape[2]} channels")
        print(f"Stimulation period: {(stim_end-stim_start)/sampling_rate*1000:.1f} ms")
        
        if cleaned_signals:
            # Find method with best stimulation period artifact reduction
            best_method = None
            best_reduction = 0
            
            for method_name, cleaned in cleaned_signals.items():
                stim_power_orig = np.mean(mixed_data[:, stim_start:stim_end, :] ** 2)
                stim_power_clean = np.mean(cleaned[:, stim_start:stim_end, :] ** 2)
                reduction = (stim_power_orig - stim_power_clean) / stim_power_orig * 100
                
                if reduction > best_reduction:
                    best_reduction = reduction
                    best_method = method_name
            
            print(f"Best method: {best_method} ({best_reduction:.1f}% power reduction in stim period)")
    
    else:
        print("Failed to load or process ERAASR data")
    
    return cleaned_signals, mixed_data


if __name__ == "__main__":
    cleaned_signals, mixed_data = main()