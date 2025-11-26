from sparc import DataHandler, SignalDataWithGroundTruth, SignalData, NeuralAnalyzer, NeuralPlotter
from sparc.core.signal_data import ArtifactTriggers
import numpy as np 
from scipy import signal
from typing import cast
import matplotlib.pyplot as plt
from utils import create_stim_trace


SWEC_SAMPLING_RATE = 512
SAMPLING_RATE = 30000

def split_data(selected_data, patient_id:str='01'):
    seizure = np.array(selected_data[patient_id]['seizure_clips'])
    non_seizure = np.array(selected_data[patient_id]['non_seizure_clips'])
    trials, channels, segments, samples = seizure.shape

    seizure_data = seizure.reshape(trials, channels, segments * samples)
    non_seizure_data = non_seizure.reshape(trials, channels, segments * samples)

    seizure_signal = SignalData(
        raw_data=seizure_data,
        sampling_rate=SWEC_SAMPLING_RATE
    )

    non_seizure_signal = SignalData(
        raw_data=non_seizure_data,
        sampling_rate=SWEC_SAMPLING_RATE
    )

    return seizure_signal, non_seizure_signal

def resample_signal(data, original_rate, target_rate) -> np.ndarray:
    data_float64 = data.astype(np.float64)
    
    num_original_samples = data_float64.shape[0]  # samples are along axis 0
    num_target_samples = int(num_original_samples * target_rate / original_rate)
    
    resampled_data = signal.resample(data_float64, num_target_samples, axis=0)
    
    return cast(np.ndarray, resampled_data)

def downsample_artifact_markers(artifact_indices, original_rate, target_rate):
    downsample_ratio = original_rate / target_rate
    downsampled_indices = [int(idx / downsample_ratio) for idx in artifact_indices]
    
    return downsampled_indices

def main():
    data_handler = DataHandler()
    data = data_handler.load_pickle_data('../research/datasets/SWEC-ETHZ/selected_clips_pat123.pkl')
    seizure_signal, non_seizure_signal = split_data(data, patient_id='01')
    trials_original, channels, samples = non_seizure_signal.raw_data.shape
    final_data = non_seizure_signal.raw_data[1:, :, :]  # Exclude first trial (trial 0)
    trials = trials_original - 1  # Update trials count
    print(f"Original data shape: ({trials_original}, {channels}, {samples})")
    print(f"Final data shape (excluding first trial): {final_data.shape}")

    # Load raw MATLAB data directly to avoid incorrect reshaping
    artifacts_data = data_handler.load_matlab_data('../research/generate_dataset/SimulatedData_lower_freq.mat')
    artifacts = np.array(artifacts_data['SimArtifact']) # (samples, channels, arrays)
    artifacts_stim_params_raw = artifacts_data['StimParam']
    print(f"Original artifacts shape: {artifacts.shape}")
    print(f"Artifacts min/max: {artifacts.min():.2f}, {artifacts.max():.2f}")
    
    # Extract stim parameters
    stim_param_struct = artifacts_stim_params_raw.item()
    t_phase_ms = stim_param_struct[2]
    t_inter_ms = stim_param_struct[3]
    amplitudes_vector = stim_param_struct[4].squeeze()
    
    # Create simple artifact markers for downsampling
    raw_indices = artifacts_data['AllStimIdx'].squeeze().astype(int)
    artifact_indices_all = raw_indices - 1
    
    # Downsample all artifact indices to target sampling rate
    artifacts_indices_downsampled_all = downsample_artifact_markers(artifact_indices_all, SAMPLING_RATE, SWEC_SAMPLING_RATE)
    print(f"Total downsampled artifact indices: {len(artifacts_indices_downsampled_all)}")
    
    # Filter to only include artifacts in the last 2 seconds
    last_two_seconds_samples = 2 * SWEC_SAMPLING_RATE  # 2 seconds * 512 Hz = 1024 samples
    # Note: target_length will be set later, but we'll calculate relative to signal end
    
    # We'll filter after we know target_length, but first let's keep track of which original indices
    # correspond to which downsampled indices for stim param filtering

    num_samples_orig = artifacts.shape[0]
    total_channels = artifacts.shape[1] * artifacts.shape[2] # 64 * 2 = 128
    artifacts_reshaped = artifacts.reshape(num_samples_orig, total_channels, order='F')
    print(f"Artifacts reshaped shape: {artifacts_reshaped.shape}")

    artifacts_downsampled = resample_signal(artifacts_reshaped, SAMPLING_RATE, SWEC_SAMPLING_RATE)
    print(f"Artifacts downsampled shape: {artifacts_downsampled.shape}")

    # plot artifact and their indices before downsampling
    plt.figure(figsize=(12, 6))
    # Plot the first channel of the reshaped artifact signal and mark artifact indices
    channel_idx = 0
    signal_to_plot = artifacts_reshaped[:, channel_idx]
    times = np.arange(signal_to_plot.shape[0]) / SAMPLING_RATE * 1000  # ms (original sampling rate)

    plt.plot(times, signal_to_plot, label='Artifacts (trial 0, channel 0)')
    # Show all original artifact indices in the plot (for visualization)
    for idx in artifact_indices_all[:50]:  # Limit to first 50 for readability
        if 0 <= idx < len(times):
            plt.axvline(x=times[idx], color='r', linestyle='--', alpha=0.3)
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    plt.title('Original artifact signal with all artifact indices (showing first 50)')
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.show()

    first_artifact_start_orig = int(raw_indices.flat[0]) - 1
    window_orig = 1000 # number of samples to plot around the start
    pulse_orig = artifacts_reshaped[first_artifact_start_orig:first_artifact_start_orig + window_orig, 0]
    time_orig = np.arange(len(pulse_orig)) / SAMPLING_RATE * 1000 # in ms

    first_artifact_start_down = int(artifacts_indices_downsampled_all[0])
    window_down = int(window_orig * SWEC_SAMPLING_RATE / SAMPLING_RATE) + 2 # equivalent window
    pulse_down = artifacts_downsampled[first_artifact_start_down:first_artifact_start_down + window_down, 0]
    time_down = np.arange(len(pulse_down)) / SWEC_SAMPLING_RATE * 1000 # in ms

    plt.figure(figsize=(12, 6))
    plt.plot(time_orig, pulse_orig, 'o-', label=f'Original @ {SAMPLING_RATE} Hz')
    plt.plot(time_down, pulse_down, 's-', label=f'Downsampled @ {SWEC_SAMPLING_RATE} Hz')
    plt.title('Artifact Pulse Shape Before and After Downsampling')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude (uV)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Create empty artifact array
    target_length = samples
    print(f"Target length: {target_length}")
    
    # Initialize empty artifact array (trials, channels, samples)
    final_artifacts = np.zeros((trials, channels, target_length), dtype=np.float32)
    
    # Extract single artifact pulse
    artifact_duration_ms = 40  # 40ms duration
    artifact_duration_downsampled = int(artifact_duration_ms / 1000 * SWEC_SAMPLING_RATE)
    print(f"Artifact duration: {artifact_duration_downsampled} samples ({artifact_duration_ms}ms at {SWEC_SAMPLING_RATE} Hz)")
    
    # Filter to only include artifacts in the last 2 seconds
    last_two_seconds_start = max(0, target_length - last_two_seconds_samples)
    last_two_seconds_end = target_length - artifact_duration_downsampled  # Ensure artifacts fit
    
    # Filter artifact indices to be within the last 2 seconds
    # Also track which original indices correspond to the filtered ones for stim param filtering
    artifact_indices_filtered = []
    original_indices_filtered = []
    
    for i, idx_downsampled in enumerate(artifacts_indices_downsampled_all):
        if last_two_seconds_start <= idx_downsampled < last_two_seconds_end:
            if idx_downsampled + artifact_duration_downsampled <= target_length:
                artifact_indices_filtered.append(idx_downsampled)
                original_indices_filtered.append(i)  # Track original index position
    
    target_positions = artifact_indices_filtered
    
    print(f"Placing {len(target_positions)} artifacts in last 2 seconds (samples {last_two_seconds_start} to {last_two_seconds_end})")
    print(f"Artifact positions: {target_positions[:10]}..." if len(target_positions) > 10 else f"Artifact positions: {target_positions}")
    
    # Filter stim params to match only the artifacts we're placing
    filtered_amplitudes_vector = amplitudes_vector[original_indices_filtered] if len(original_indices_filtered) > 0 else amplitudes_vector
    print(f"Filtered amplitudes vector shape: {filtered_amplitudes_vector.shape} (matching {len(target_positions)} artifacts)")
    
    # Extract the reference artifact pulse from the first artifact location in the original signal
    first_artifact_location = artifact_indices_all[0]
    first_artifact_downsampled = int(first_artifact_location * SWEC_SAMPLING_RATE / SAMPLING_RATE)
    
    # Add artifacts at all positions within the last 2 seconds
    for trial_idx in range(trials):
        for channel_idx in range(channels):
            # Extract artifact pulse from the first artifact location in artifacts_downsampled
            pulse_start = first_artifact_downsampled
            pulse_end = min(pulse_start + artifact_duration_downsampled, artifacts_downsampled.shape[0])
            pulse = artifacts_downsampled[pulse_start:pulse_end, channel_idx % artifacts_downsampled.shape[1]]
            
            # Place this pulse at each of our target positions
            for target_pos in target_positions:
                if target_pos >= 0 and target_pos < target_length:
                    signal_end = min(target_pos + len(pulse), target_length)
                    actual_pulse_length = signal_end - target_pos
                    final_artifacts[trial_idx, channel_idx, target_pos:signal_end] = pulse[:actual_pulse_length]
    
    print(f"Final artifacts shape: {final_artifacts.shape}")
    print(f"Number of non-zero artifact samples per trial/channel: ~{np.count_nonzero(final_artifacts[0, 0, :])}")
    print(f"Artifact amplitude range: {final_artifacts.min():.2f} to {final_artifacts.max():.2f}")
    
    # Verify artifacts at target positions
    for i, pos in enumerate(target_positions):
        artifact_region = final_artifacts[0, 0, pos:pos+artifact_duration_downsampled]
        if len(artifact_region) > 0:
            print(f"Artifact {i+1} at position {pos}: samples={len(artifact_region)}, amplitude range=[{artifact_region.min():.2f}, {artifact_region.max():.2f}]")
        else:
            print(f"Artifact {i+1} at position {pos}: empty region (no samples)")

    # Create stim trace at 30kHz first (to avoid phase width issues at 512Hz), then downsample
    # We need to find which original 30kHz artifact indices correspond to the filtered ones
    # The filtered artifacts are at positions in original_indices_filtered
    filtered_artifact_indices_30k = artifact_indices_all[original_indices_filtered] if len(original_indices_filtered) > 0 else artifact_indices_all
    
    # Create stim trace at 30kHz with filtered artifacts
    filtered_markers_30k = ArtifactTriggers(starts=filtered_artifact_indices_30k)
    stim_trace_30k = create_stim_trace(
        n_trials=1,
        n_samples=num_samples_orig,
        artifact_markers=filtered_markers_30k,
        amplitudes_array=filtered_amplitudes_vector,
        sampling_rate=SAMPLING_RATE,  # 30kHz
        t_phase_ms=t_phase_ms,
        t_inter_ms=t_inter_ms
    ).numpy()
    
    stim_trace_30k = stim_trace_30k.reshape(num_samples_orig, 1)
    print(f"Stim trace at 30kHz shape: {stim_trace_30k.shape}")
    
    # Downsample stim trace to 512Hz
    print("Downsampling stim trace from 30kHz to 512Hz...")
    stim_trace_downsampled = resample_signal(stim_trace_30k, SAMPLING_RATE, SWEC_SAMPLING_RATE)
    print(f"Stim trace downsampled shape: {stim_trace_downsampled.shape}")
    
    # Extract only the last 2 seconds portion
    stim_trace_length = stim_trace_downsampled.shape[0]
    if stim_trace_length >= target_length:
        # If downsampled trace is longer, take the last portion that matches target_length
        stim_trace_filtered = stim_trace_downsampled[-target_length:, :]
    else:
        # If shorter, pad with zeros or take what we have
        stim_trace_filtered = np.zeros((target_length, 1), dtype=np.float32)
        stim_trace_filtered[-stim_trace_length:, :] = stim_trace_downsampled
    
    print(f"Filtered stim trace shape (last portion): {stim_trace_filtered.shape}")
    
    # Tile stim trace for all trials
    final_stim_trace = np.tile(stim_trace_filtered.T, (trials, 1, 1))
    print(f"Final stim trace shape: {final_stim_trace.shape}")
    
    starts_3d = []
    for trial_idx in range(trials):
        trial_markers = []
        for channel_idx in range(channels):
            # Each channel gets the same artifact indices (our target positions)
            trial_markers.append(np.array(target_positions))
        starts_3d.append(trial_markers)
    
    clean_stim_params = {
        'frequency': stim_param_struct[0],
        'period': stim_param_struct[1],
        'timeperphase': t_phase_ms,
        'timeinterphase': t_inter_ms,
        'amplitudes_vector': filtered_amplitudes_vector 
    }
    
    mixed = SignalDataWithGroundTruth(
        raw_data=final_data + final_artifacts,
        sampling_rate=SWEC_SAMPLING_RATE,
        ground_truth=final_data,
        artifacts=final_artifacts,
        artifact_markers=starts_3d,
        artifacts_stim_params=clean_stim_params
    )

    # Plot artifacts and stim trace together (similar to add_artifacts_to_swec.py)
    print("Generating artifacts and stim trace plot...")
    fig_stim, axs_stim = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    time_stim = np.arange(target_length) / SWEC_SAMPLING_RATE  # Time in seconds
    
    # -- Top: Artifacts (last 2 seconds) --
    axs_stim[0].plot(time_stim, final_artifacts[0, 0, :], 'b-', label='Artifacts', linewidth=1.5)
    axs_stim[0].set_title(f'Artifacts in Last 2 Seconds @ {SWEC_SAMPLING_RATE} Hz (Trial 0, Channel 0)')
    axs_stim[0].set_ylabel('Amplitude (uV)')
    axs_stim[0].legend(loc='upper right')
    axs_stim[0].grid(True, alpha=0.3)
    
    # Mark artifact positions
    for pos in target_positions:
        time_pos = pos / SWEC_SAMPLING_RATE
        axs_stim[0].axvline(x=time_pos, color='orange', linestyle=':', alpha=0.6, linewidth=1)
    
    # -- Bottom: Stim Trace --
    axs_stim[1].plot(time_stim, stim_trace_filtered[:, 0], 'r-', label='Stim Trace (Normalized)', linewidth=1.5)
    axs_stim[1].set_xlabel('Time (s)')
    axs_stim[1].set_ylabel('Normalized Amplitude')
    axs_stim[1].legend(loc='upper right')
    axs_stim[1].grid(True, alpha=0.3)
    
    # Mark artifact positions on stim trace too
    for pos in target_positions:
        time_pos = pos / SWEC_SAMPLING_RATE
        axs_stim[1].axvline(x=time_pos, color='orange', linestyle=':', alpha=0.6, linewidth=1)
    
    plt.suptitle('Artifacts and Stim Trace (Last 2 Seconds)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Plot comparison: Ground Truth, Mixed, and Artifacts overlaid
    trial_idx = 0
    channel_idx = 0
    time_axis = np.arange(target_length) / SWEC_SAMPLING_RATE * 1000  # Time in ms
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot ground truth (neural signal)
    ax.plot(time_axis, final_data[trial_idx, channel_idx, :], 
            label='Ground Truth (Neural)', color='#2ca02c', linewidth=1.5, alpha=0.8)
    
    # Plot mixed signal (neural + artifacts)
    ax.plot(time_axis, mixed.raw_data[trial_idx, channel_idx, :], 
            label='Mixed (Neural + Artifacts)', color='#1f77b4', linewidth=1.5, alpha=0.8)
    
    # Plot artifacts only
    ax.plot(time_axis, final_artifacts[trial_idx, channel_idx, :], 
            label='Artifacts', color='#d62728', linewidth=1.5, alpha=0.7, linestyle='--')
    
    # Mark artifact positions with vertical lines
    for pos in target_positions:
        time_pos = pos / SWEC_SAMPLING_RATE * 1000
        ax.axvline(x=time_pos, color='orange', linestyle=':', alpha=0.6, linewidth=1)
        ax.text(time_pos, ax.get_ylim()[1] * 0.95, f'Artifact\n{pos}', 
                color='orange', ha='center', fontsize=8, alpha=0.8)
    
    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel('Amplitude (µV)', fontsize=12)
    ax.set_title(f'Signal Comparison: Ground Truth vs Mixed vs Artifacts (last 2 seconds)\nTrial {trial_idx}, Channel {channel_idx}', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()

    # save to npz 
    np.savez_compressed(f'../data/added_artifacts_swec_data_{SWEC_SAMPLING_RATE}_lower_freq_last_2s.npz', 
        mixed_data=mixed.raw_data, 
        ground_truth=mixed.ground_truth, 
        artifacts=mixed.artifacts, 
        sampling_rate=mixed.sampling_rate,
        artifact_markers=mixed.artifact_markers,
        artifacts_stim_params=mixed.artifacts_stim_params,
        stim_trace=final_stim_trace
    )

    analyzer = NeuralAnalyzer(mixed.sampling_rate)
    plotter = NeuralPlotter(analyzer)

    plotter.plot_all_channels_trial(
        mixed.raw_data,
        0
    )
    plotter.plot_artifacts_and_mixed_comparison(mixed.artifacts, mixed.raw_data, mixed.artifact_markers, 0, 0)


if __name__ == "__main__":
    main()
