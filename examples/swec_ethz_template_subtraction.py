import numpy as np
import matplotlib.pyplot as plt
import pickle
from sparc import SPARCEvaluator, AverageTemplateSubtraction, TrialTemplateSubtraction, DictionaryTemplateSubtraction
from tqdm import tqdm
from scipy import signal as sp_signal

def load_swec_ethz_data(data_path='research/datasets/SWEC-ETHZ/selected_clips_pat123.pkl'):
    try:
        with open(data_path, 'rb') as f:
            selected_data = pickle.load(f)
        
        seizure_clips = []
        for patient_id in selected_data:
            for clip in selected_data[patient_id]['seizure_clips']:
                seizure_clips.append(clip.reshape(clip.shape[0], -1))
        
        if not seizure_clips:
            return None, None, None, None
            
        # Stack clips into (trials, timesteps, channels) format
        max_channels = max(clip.shape[0] for clip in seizure_clips)
        n_trials = len(seizure_clips)
        n_timesteps = seizure_clips[0].shape[1]  # 2048 timesteps
        
        # Pre-allocate with zeros
        clean_data = np.zeros((n_trials, n_timesteps, max_channels))
        
        # Fill in the data
        for i, clip in enumerate(seizure_clips):
            n_channels = clip.shape[0]
            # Transpose to (timesteps, channels)
            clean_data[i, :, :n_channels] = clip.T
        
        # Generate synthetic artifacts
        sampling_rate = 512
        artifacts = generate_synthetic_artifacts(clean_data, sampling_rate)
        
        # Create mixed data (neural + artifacts)
        mixed_data = clean_data + artifacts
        
        print(f"Loaded SWEC-ETHZ data:")
        print(f"  Shape: {mixed_data.shape}")
        print(f"  Sampling rate: {sampling_rate} Hz")
        print(f"  Duration per trial: {n_timesteps / sampling_rate:.1f} s")
        print(f"  Trials: {n_trials}, Max channels: {max_channels}")
        print(f"  Active channels per trial: {[np.sum(np.any(clean_data[i], axis=0)) for i in range(min(5, n_trials))]}")
        
        return mixed_data, clean_data, artifacts, sampling_rate
        
    except FileNotFoundError:
        print(f"SWEC-ETHZ data file not found at {data_path}")
        return None, None, None, None
    except Exception as e:
        print(f"Error loading SWEC-ETHZ data: {str(e)}")
        return None, None, None, None

def resample_signal(sig, original_rate, target_rate):
    """Resample signal to a new sampling rate"""
    num_original_samples = sig.shape[-1]
    num_target_samples = int(num_original_samples * target_rate / original_rate)
    resampled_sig = sp_signal.resample(sig, num_target_samples, axis=-1)
    return resampled_sig

def generate_sine_exp_decay_artifact(input_data, sampling_rate_signal, sampling_rate_artifact, stim_rate, stimulation_channel, stim_current_strength, 
                                      strip_distance=1e-1, normal_strip_distance=1e-3, f_pulse=2500):
    num_clips, num_channels, num_timesteps = input_data.shape
    dt = 1 / sampling_rate_signal
    Total_time = num_timesteps * dt
    time_artifact = np.arange(0, Total_time, 1 / sampling_rate_artifact)

    artifact = np.zeros((num_clips, num_channels, len(time_artifact)))
    
    # Identify silent channels (all-zero input across clips)
    silent_channels = np.all(input_data == 0, axis=(0, 2))  # Shape: (num_channels,)

    # Calculate distances for non-silent channels
    k1, k2 = -0.201, 0.102  # Constants from the formula
    strip_id = stimulation_channel // 8  # Identify which strip the stimulation channel belongs to
    channel_distances = np.zeros(num_channels)

    for ch in range(num_channels):
        if silent_channels[ch]:  # Skip silent channels
            continue
        ch_strip = ch // 8  # Identify strip
        if ch_strip == strip_id:
            channel_distances[ch] = abs(ch - stimulation_channel) * normal_strip_distance
        else:
            channel_distances[ch] = abs(ch_strip - strip_id) * strip_distance

    # Compute amplitude for each non-silent channel
    log_amplitudes = k1 * channel_distances + k2 * stim_current_strength - 1.92
    amplitudes = np.zeros(num_channels)  # Default to zero for silent channels
    amplitudes[~silent_channels] = 10 ** log_amplitudes[~silent_channels]  # Convert log10(Amplitude) to linear scale

    # Generate artifact for each clip
    stim_period = 1 / stim_rate
    for clip_idx in tqdm(range(num_clips), desc="Generating Artifacts"):
        clip_artifact = np.zeros((num_channels, len(time_artifact)))
        for start_time in np.arange(0, time_artifact[-1], stim_period):
            indices_sine = np.where((time_artifact >= start_time) & (time_artifact < start_time + 1 / f_pulse))
            rand_delay = np.random.uniform(3/8 * (1/f_pulse), 7/8 * (1/f_pulse)) 
            indices_exp = np.where(time_artifact >= start_time + rand_delay)
            
            for ch in range(num_channels):
                if silent_channels[ch]:  # Skip silent channels
                    continue
                clip_artifact[ch, indices_sine] = -np.sin(2 * np.pi * f_pulse * time_artifact[indices_sine])
                clip_artifact[ch, indices_exp] += -np.exp(-3000 * time_artifact[indices_exp]) - np.exp(-5000 * time_artifact[indices_exp])
        
        # Scale by amplitude for each channel
        artifact[clip_idx] = clip_artifact * amplitudes[:, np.newaxis]

    return artifact

def generate_synthetic_artifacts(clean_data, sampling_rate, f_pulse=2500):
    sampling_rate_artifact = 2000  # Generate artifact at a higher sampling rate
    stimulation_channel = 0
    stim_current_strength = 57

    # stim_rate = np.random.uniform(200, 300)
    stim_rate = 200  # Fixed stimulation rate for consistency

    # Transpose clean_data to (trials, channels, timesteps) for artifact generation function
    clean_data_transposed = clean_data.transpose(0, 2, 1)

    artifacts_high_res = generate_sine_exp_decay_artifact(
        input_data=clean_data_transposed,
        sampling_rate_signal=sampling_rate,
        sampling_rate_artifact=sampling_rate_artifact,
        stim_rate=stim_rate,
        stimulation_channel=stimulation_channel,
        stim_current_strength=stim_current_strength,
        f_pulse=f_pulse
    )

    # resample artisacts down to the signal's sampling rate
    artifacts_resampled = resample_signal(
        artifacts_high_res,
        original_rate=sampling_rate_artifact,
        target_rate=sampling_rate
    )

    # Transpose back to (trials, timesteps, channels)
    artifacts = artifacts_resampled.transpose(0, 2, 1)
    
    # Ensure the length matches clean_data due to resampling
    if artifacts.shape[1] != clean_data.shape[1]:
        # Pad or truncate
        diff = artifacts.shape[1] - clean_data.shape[1]
        if diff > 0:
            artifacts = artifacts[:, :-diff, :]
        else:
            artifacts = np.pad(artifacts, ((0,0), (0, -diff), (0,0)), 'constant')

    return artifacts

def demonstrate_swec_ethz_template_subtraction():
    mixed_data, ground_truth, artifacts, sampling_rate = load_swec_ethz_data()

    if mixed_data is None:
        print("Could not load SWEC-ETHZ data. Please ensure the dataset is available.")
        return None, None

    known_artifact_indices = [np.abs(artifacts[i]) > 1e-9 for i in range(artifacts.shape[0])] 
    
    methods = {
        'Simple Average': AverageTemplateSubtraction(
            sampling_rate=sampling_rate,
            template_length_ms=5,
            num_templates_for_avg=3
        ),
    }
    
    evaluator = SPARCEvaluator(sampling_rate)
    
    results = {}
    cleaned_signals = {}
    
    for method_name, method in methods.items():
        print(f"\n--- Testing {method_name} ---")

        try:
            method.fit(mixed_data, artifact_indices=known_artifact_indices)
            cleaned = method.transform(mixed_data)
            cleaned_signals[method_name] = cleaned
            
            metrics = evaluator.evaluate_method_comprehensive(
                ground_truth[0], mixed_data[0], cleaned[0], method_name
            )
            results[method_name] = metrics
            
            print(f"✓ {method_name} completed successfully")
            print(f"  MSE: {metrics['mse']:.6f}")
            print(f"  SNR Improvement: {metrics['snr_improvement_db']:.2f} dB")
            print(f"  Artifact Removal Ratio: {metrics['artifact_removal_ratio']:.4f}")
            print(f"  Hit Rate: {metrics['hit_rate_mean']:.4f}")
            print(f"  Miss Rate: {metrics['miss_rate_mean']:.4f}")
            print(f"  False Positive Rate: {metrics['false_positive_rate_mean']:.4f}")
            print(f"  LFP PSD Correlation: {metrics['lfp_psd_correlation_mean']:.4f}")
            
        except Exception as e:
            print(f"✗ {method_name} failed: {str(e)}")
            continue
    
    return results, cleaned_signals, mixed_data, ground_truth, artifacts

def plot_swec_ethz_results(mixed_data, ground_truth, artifacts, cleaned_signals, 
                          trial_idx=0, channel=0, zoom_ms=2000, zoom_center_ms=None):
    if mixed_data is None or not cleaned_signals:
        print("No data or results to plot")
        return

    # --- Data and Time Axis Setup ---
    sampling_rate = 512
    time_s = np.arange(mixed_data.shape[1]) / sampling_rate
    
    # Ensure channel is active
    if not np.any(ground_truth[trial_idx, :, channel]):
        print(f"Warning: Channel {channel} appears to be inactive. Trying channel 0.")
        channel = 0

    if zoom_center_ms is None:
        # Auto-find the first artifact to center the zoom
        artifact_signal = mixed_data[trial_idx, :, channel] - ground_truth[trial_idx, :, channel]
        artifact_onsets = np.where(np.abs(artifact_signal) > np.std(artifact_signal) * 3)[0]
        if artifact_onsets.any():
            zoom_center_s = artifact_onsets[0] / sampling_rate
        else:
            zoom_center_s = time_s[len(time_s) // 2] # Default to center if no artifacts found
    else:
        zoom_center_s = zoom_center_ms / 1000

    zoom_half_width_s = (zoom_ms / 1000) / 2
    zoom_start_s = max(0, zoom_center_s - zoom_half_width_s)
    zoom_end_s = min(time_s[-1], zoom_center_s + zoom_half_width_s)

    # --- Plotting ---
    fig, axes = plt.subplots(len(cleaned_signals) + 1, 1, 
                             figsize=(12, 4 * (len(cleaned_signals) + 1)),
                             sharex=True)
    fig.suptitle(f'Zoomed-In Results (Trial {trial_idx}, Channel {channel})', fontsize=16)

    # Plot 1: Original Mixed vs. Ground Truth
    ax = axes[0]
    ax.plot(time_s, ground_truth[trial_idx, :, channel], 'b-', 
            label='Ground Truth (Clean)', alpha=0.8)
    ax.plot(time_s, mixed_data[trial_idx, :, channel], 'r-', 
            label='Mixed (w/ Artifacts)', alpha=0.6)
    ax.set_title('Original Signals')
    ax.set_ylabel('Amplitude')
    ax.legend()
    ax.grid(True, alpha=0.4)

    # Plot 2 onwards: Cleaned signals
    colors = ['green', 'purple', 'brown']
    for i, (method_name, cleaned) in enumerate(cleaned_signals.items()):
        ax = axes[i + 1]
        ax.plot(time_s, ground_truth[trial_idx, :, channel], 'b-', 
                alpha=0.5, label='Ground Truth')
        ax.plot(time_s, cleaned[trial_idx, :, channel], 
                color=colors[i % len(colors)], label=f'Cleaned ({method_name})', linewidth=1.5)
        ax.set_title(f'Result: {method_name}')
        ax.set_ylabel('Amplitude')
        ax.legend()
        ax.grid(True, alpha=0.4)

    # Apply zoom to all subplots
    for ax in axes:
        ax.set_xlim(zoom_start_s, zoom_end_s)

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def main():
    results, cleaned_signals, mixed_data, ground_truth, artifacts = demonstrate_swec_ethz_template_subtraction()
    
    if results:
        sampling_rate = 512
        
        # Find an active channel for analysis
        active_channel = 0
        for ch in range(min(10, ground_truth.shape[2])):
            if np.any(ground_truth[0, :, ch]):
                active_channel = ch
                break
        
        # Plot results
        print(f"\nGenerating comparison plots for channel {active_channel}...")
        plot_swec_ethz_results(mixed_data, ground_truth, artifacts, cleaned_signals,
                              channel=active_channel)
        
        # Summary
        print("\n=== Summary ===")
        if results:
            best_method = min(results.keys(), key=lambda x: results[x]['mse'])
            print(f"Best performing method: {best_method}")
            print(f"Best MSE: {results[best_method]['mse']:.6f}")
            print(f"Best SNR improvement: {results[best_method]['snr_improvement_db']:.2f} dB")
            print(f"Best artifact removal ratio: {results[best_method]['artifact_removal_ratio']:.4f}")
            print(f"Best Hit Rate: {results[best_method]['hit_rate_mean']:.4f}")
            print(f"Best Miss Rate: {results[best_method]['miss_rate_mean']:.4f}")
            print(f"Best LFP PSD Correlation: {results[best_method]['lfp_psd_correlation_mean']:.4f}")
    
    return results, cleaned_signals

if __name__ == "__main__":
    results, cleaned_signals = main()
