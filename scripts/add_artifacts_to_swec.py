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

    # plot non_seizure_signal of trial 0 channel 56
    plt.figure(figsize=(12, 6))
    time_non_seizure = np.arange(final_data.shape[2]) / SWEC_SAMPLING_RATE * 1000  # ms
    plt.plot(time_non_seizure, final_data[0, 56, :], label='Non-Seizure Signal (trial 0, channel 56)')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    plt.title('Non-Seizure Signal (Trial 0, Channel 56 - Original Trial 1)')
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.show()

    stim_param_struct = artifacts_stim_params_raw.item()
    t_phase_ms = stim_param_struct[2]
    t_inter_ms = stim_param_struct[3]
    amplitudes_vector = stim_param_struct[4].squeeze()
    print(f"Original artifacts shape: {artifacts.shape}")
    print(f"Artifacts min/max: {artifacts.min():.2f}, {artifacts.max():.2f}")
    num_samples_orig = artifacts.shape[0]
    raw_indices = artifacts_data['AllStimIdx'].squeeze().astype(int)
    artifact_indices = raw_indices - 1

    dummy_markers_30k = ArtifactTriggers(starts=artifact_indices) 
    stim_trace_orig_30k = create_stim_trace(
        n_trials=1, # Just create one master trace
        n_samples=num_samples_orig,
        artifact_markers=dummy_markers_30k,
        amplitudes_array=amplitudes_vector,
        sampling_rate=SAMPLING_RATE, # 30000 Hz
        t_phase_ms=t_phase_ms,
        t_inter_ms=t_inter_ms
    ).numpy()

    stim_trace_orig_30k = stim_trace_orig_30k.reshape(num_samples_orig, 1)
    print(f"Original stim trace shape: {stim_trace_orig_30k.shape}")
    print("Downsampling 30kHz stim trace...")
    stim_trace_downsampled = resample_signal(stim_trace_orig_30k, SAMPLING_RATE, SWEC_SAMPLING_RATE)
    print(f"Stim trace downsampled shape: {stim_trace_downsampled.shape}")

    # Create simple artifact markers for downsampling
    artifacts_indices_downsampled = downsample_artifact_markers(artifact_indices, SAMPLING_RATE, SWEC_SAMPLING_RATE)

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
    times = np.arange(signal_to_plot.shape[0]) / SWEC_SAMPLING_RATE * 1000  # ms

    plt.plot(times, signal_to_plot, label='Artifacts (trial 0, channel 0)')
    for idx in artifact_indices:
        if 0 <= idx < len(times):
            plt.axvline(x=times[idx], color='r', linestyle='--', alpha=0.5)
            plt.text(times[idx], signal_to_plot.min(), str(idx), color='red', rotation=90,
                     verticalalignment='bottom', horizontalalignment='right', fontsize=7, alpha=0.7)
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    plt.title('First trial, first channel with artifact indices (indices labeled on time axis)')
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.show()

    first_artifact_start_orig = int(raw_indices.flat[0]) - 1
    window_orig = 1000 # number of samples to plot around the start
    pulse_orig = artifacts_reshaped[first_artifact_start_orig:first_artifact_start_orig + window_orig, 0]
    time_orig = np.arange(len(pulse_orig)) / SAMPLING_RATE * 1000 # in ms

    first_artifact_start_down = int(artifacts_indices_downsampled[0])
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

    print("Generating full trace plot (30kHz)...")
    fig_before_full, axs_before_full = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    time_orig_full = np.arange(num_samples_orig) / SAMPLING_RATE # Time in seconds

    # -- Top: Original Artifact (30kHz) --
    axs_before_full[0].plot(time_orig_full, artifacts_reshaped[:, 0], 'b-', label='Artifact')
    axs_before_full[0].set_title(f'Original Full Trace @ {SAMPLING_RATE} Hz (Channel 0)')
    axs_before_full[0].set_ylabel('Amplitude (uV)')
    axs_before_full[0].legend(loc='upper right')
    axs_before_full[0].grid(True)
    
    # -- Bottom: Original Stim Trace (30kHz) --
    axs_before_full[1].plot(time_orig_full, stim_trace_orig_30k[:, 0], 'r-', label='Stim Trace (Normalized)')
    axs_before_full[1].set_xlabel('Time (s)')
    axs_before_full[1].set_ylabel('Normalized Amplitude')
    axs_before_full[1].legend(loc='upper right')
    axs_before_full[1].grid(True)
    
    plt.suptitle('BEFORE Downsampling (Full Trace, Channel 0)')
    plt.tight_layout()
    plt.show()

    # --- PLOT: FULL TRACE (AFTER DOWNSAMPLING) ---
    print("Generating full trace plot (512Hz)...")
    fig_after_full, axs_after_full = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    time_down_full = np.arange(len(artifacts_downsampled)) / SWEC_SAMPLING_RATE # Time in seconds

    # -- Top: Downsampled Artifact (512Hz) --
    axs_after_full[0].plot(time_down_full, artifacts_downsampled[:, 0], 'b-', label='Artifact')
    axs_after_full[0].set_title(f'Downsampled Full Trace @ {SWEC_SAMPLING_RATE} Hz (Channel 0)')
    axs_after_full[0].set_ylabel('Amplitude (uV)')
    axs_after_full[0].legend(loc='upper right')
    axs_after_full[0].grid(True)
    
    # -- Bottom: Downsampled Stim Trace (512Hz) --
    axs_after_full[1].plot(time_down_full, stim_trace_downsampled[:, 0], 'r-', label='Stim Trace (Aliased/Filtered)')
    axs_after_full[1].set_xlabel('Time (s)')
    axs_after_full[1].set_ylabel('Normalized Amplitude')
    axs_after_full[1].legend(loc='upper right')
    axs_after_full[1].grid(True)

    plt.suptitle('AFTER Downsampling (Full Trace, Channel 0)')
    plt.tight_layout()
    plt.show()
    
    # Adjust length to match target data length
    artifact_length = artifacts_downsampled.shape[0]
    target_length = samples
    print(f"Artifact length: {artifact_length}, Target length: {target_length}")
    
    if artifact_length > target_length:
        artifacts_downsampled = artifacts_downsampled[:target_length, :]
        stim_trace_downsampled = stim_trace_downsampled[:target_length, :]
    elif artifact_length < target_length:
        repeat_factor = int(np.ceil(target_length / artifact_length))
        artifacts_repeated = np.tile(artifacts_downsampled, (repeat_factor, 1))
        artifacts_downsampled = artifacts_repeated[:target_length, :]

        stim_trace_repeated = np.tile(stim_trace_downsampled, (repeat_factor, 1))
        stim_trace_downsampled = stim_trace_repeated[:target_length, :]
    
    print(f"Artifacts after length adjustment: {artifacts_downsampled.shape}")
    print(f"Stim trace after length adjustment: {stim_trace_downsampled.shape}")
    
    final_artifacts = np.tile(artifacts_downsampled.T, (trials, 1, 1))  # (trials, 128, samples)
    final_artifacts = final_artifacts[:, :channels, :]  # (trials, 88, samples)
    print(f"Final artifacts shape: {final_artifacts.shape}")

    final_stim_trace = np.tile(stim_trace_downsampled.T, (trials, 1, 1))
    print(f"Final stim trace shape: {final_stim_trace.shape}")

    starts_3d = []
    for trial_idx in range(trials):
        trial_markers = []
        for channel_idx in range(channels):
            # Each channel gets the same artifact indices as a numpy array
            trial_markers.append(np.array(artifacts_indices_downsampled))
        starts_3d.append(trial_markers)

    clean_stim_params = {
        'frequency': stim_param_struct[0],
        'period': stim_param_struct[1],
        'timeperphase': t_phase_ms,
        'timeinterphase': t_inter_ms,
        'amplitudes_vector': amplitudes_vector 
    }

    mixed = SignalDataWithGroundTruth(
        raw_data=final_data + final_artifacts,
        sampling_rate=SWEC_SAMPLING_RATE,
        ground_truth=final_data,
        artifacts=final_artifacts,
        artifact_markers=starts_3d,
        artifacts_stim_params=clean_stim_params
    )

    # save to npz 
    np.savez_compressed(f'../data/added_artifacts_swec_data_{SWEC_SAMPLING_RATE}_lower_freq_wo0.npz', 
        mixed_data=mixed.raw_data, 
        ground_truth=mixed.ground_truth, 
        artifacts=mixed.artifacts, 
        sampling_rate=mixed.sampling_rate,
        artifact_markers=mixed.artifact_markers,
        artifacts_stim_params=mixed.artifacts_stim_params,
        stim_trace=final_stim_trace
    )

    print(f"saved to ../data/added_artifacts_swec_data_{SWEC_SAMPLING_RATE}_lower_freq_wo0.npz")

    analyzer = NeuralAnalyzer(mixed.sampling_rate)
    plotter = NeuralPlotter(analyzer)

    plotter.plot_all_channels_trial(
        mixed.raw_data,
        0
    )
    plotter.plot_all_channels_trial(
        mixed.raw_data,
        1
    )
    plotter.plot_all_channels_trial(
        mixed.raw_data,
        2
    )
    plotter.plot_all_channels_trial(
        mixed.raw_data,
        3
    )
    plotter.plot_all_channels_trial(
        mixed.raw_data,
        4
    )
    plotter.plot_artifacts_and_mixed_comparison(mixed.artifacts, mixed.raw_data, mixed.artifact_markers, 1, 86)
    plotter.plot_artifacts_and_mixed_comparison(mixed.artifacts, mixed.raw_data, mixed.artifact_markers, 1, 87)


if __name__ == "__main__":
    main()
