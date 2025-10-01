from sparc import DataHandler, SignalDataWithGroundTruth, SignalData, NeuralAnalyzer, NeuralPlotter
from sparc.core.signal_data import ArtifactTriggers
import numpy as np 
from scipy import signal
from typing import cast
import matplotlib.pyplot as plt


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
    
    # Simple downsampling of indices
    downsampled_indices = [int(idx / downsample_ratio) for idx in artifact_indices]
    
    return downsampled_indices

def main():
    data_handler = DataHandler()
    data = data_handler.load_pickle_data('../research/datasets/SWEC-ETHZ/selected_clips_pat123.pkl')
    seizure_signal, non_seizure_signal = split_data(data, patient_id='01')
    trials, channels, samples = non_seizure_signal.raw_data.shape
    final_data = non_seizure_signal.raw_data
    print(f"Final data shape: {final_data.shape}")

    # Load raw MATLAB data directly to avoid incorrect reshaping
    artifacts_data = data_handler.load_matlab_data('../research/generate_dataset/SimulatedData.mat')
    artifacts = np.array(artifacts_data['SimArtifact']) # (samples, channels, arrays)
    print(f"Original artifacts shape: {artifacts.shape}")
    print(f"Artifacts min/max: {artifacts.min():.2f}, {artifacts.max():.2f}")
    
    # Create simple artifact markers for downsampling
    raw_indices = artifacts_data['AllStimIdx'].squeeze().astype(int)
    artifact_indices = raw_indices - 1
    artifacts_indices_downsampled = downsample_artifact_markers(artifact_indices, SAMPLING_RATE, SWEC_SAMPLING_RATE)

    num_samples_orig = artifacts.shape[0]
    total_channels = artifacts.shape[1] * artifacts.shape[2] # 64 * 2 = 128
    artifacts_reshaped = artifacts.reshape(num_samples_orig, total_channels, order='F')
    print(f"Artifacts reshaped shape: {artifacts_reshaped.shape}")

    artifacts_downsampled = resample_signal(artifacts_reshaped, SAMPLING_RATE, SWEC_SAMPLING_RATE)
    print(f"Artifacts downsampled shape: {artifacts_downsampled.shape}")

    first_artifact_start_orig = int(raw_indices.flat[1]) - 1
    window_orig = 200 # number of samples to plot around the start
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
    
    # Adjust length to match target data length
    artifact_length = artifacts_downsampled.shape[0]
    target_length = samples
    print(f"Artifact length: {artifact_length}, Target length: {target_length}")
    
    if artifact_length > target_length:
        artifacts_downsampled = artifacts_downsampled[:target_length, :]
    elif artifact_length < target_length:
        repeat_factor = int(np.ceil(target_length / artifact_length))
        artifacts_repeated = np.tile(artifacts_downsampled, (repeat_factor, 1))
        artifacts_downsampled = artifacts_repeated[:target_length, :]
    
    print(f"Artifacts after length adjustment: {artifacts_downsampled.shape}")
    
    final_artifacts = np.tile(artifacts_downsampled.T, (trials, 1, 1))  # (trials, 128, samples)
    final_artifacts = final_artifacts[:, :channels, :]  # (trials, 88, samples)
    print(f"Final artifacts shape: {final_artifacts.shape}")

    starts_3d = []
    for trial_idx in range(trials):
        trial_markers = []
        for channel_idx in range(channels):
            # Each channel gets the same artifact indices as a numpy array
            trial_markers.append(np.array(artifacts_indices_downsampled))
        starts_3d.append(trial_markers)
    
    mixed = SignalDataWithGroundTruth(
        raw_data=final_data + final_artifacts,
        sampling_rate=SWEC_SAMPLING_RATE,
        ground_truth=final_data,
        artifacts=final_artifacts,
        artifact_markers=starts_3d
    )

    # save to npz 
    np.savez_compressed(f'../data/added_artifacts_swec_data_{SWEC_SAMPLING_RATE}.npz', 
        mixed_data=mixed.raw_data, 
        ground_truth=mixed.ground_truth, 
        artifacts=mixed.artifacts, 
        sampling_rate=mixed.sampling_rate,
        artifact_markers=mixed.artifact_markers
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
