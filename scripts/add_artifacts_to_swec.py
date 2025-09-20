from sparc import DataHandler, SignalDataWithGroundTruth, SignalData, NeuralAnalyzer, NeuralPlotter
import numpy as np 
from scipy import signal
from typing import cast, final
import matplotlib.pyplot as plt


SWEC_SAMPLING_RATE = 512 

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
    
    num_original_samples = data_float64.shape[-1]
    num_target_samples = int(num_original_samples * target_rate / original_rate)
    
    resampled_data = signal.resample(data_float64, num_target_samples, axis=-1)
    
    return cast(np.ndarray, resampled_data)

def plot_raw_artifact(raw_artifact, original_sampling_rate):
    num_samples = raw_artifact.shape[0]
    time_vec = np.arange(num_samples) / original_sampling_rate

    plt.figure(figsize=(14, 7))
    
    # Plot channel 0 from the first array (index 0)
    plt.plot(time_vec, raw_artifact[:, 0, 0], label='Channel 1 from Array 1')
    
    # Plot channel 0 from the second array (index 1) if it exists
    if raw_artifact.shape[2] > 1:
        plt.plot(time_vec, raw_artifact[:, 0, 1], label='Channel 1 from Array 2', alpha=0.8)

    plt.title(f'Raw Artifact Signal at {original_sampling_rate} Hz (Zoomed In)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Zoom in to the first 100ms to see the detailed spike structure
    plt.xlim(0, 0.1)
    
    plt.show()

def main():
    data_handler = DataHandler()
    data = data_handler.load_pickle_data('../research/datasets/SWEC-ETHZ/selected_clips_pat123.pkl')
    _, non_seizure_signal = split_data(data, patient_id='01')
    trials, channels, samples = non_seizure_signal.raw_data.shape

    artifacts_data = data_handler.load_matlab_data('../research/generate_dataset/SimulatedArtifact.mat')
    artifacts = np.array(artifacts_data['SimArtifact'])
    print(max(artifacts.flatten()), min(artifacts.flatten()))
    artifacts_indices = np.array(artifacts_data['AllStimIdx'])

    num_samples_orig = artifacts.shape[0]
    total_channels = artifacts.shape[1] * artifacts.shape[2] # 64 * 2 = 128
    artifacts_reshaped = artifacts.reshape(num_samples_orig, total_channels, order='F')
    final_artifacts = np.tile(artifacts_reshaped.T, (trials, 1, 1))
    final_artifacts = final_artifacts[:, :channels, :]

    upsampled_data = signal.resample(non_seizure_signal.raw_data, int(samples * 30000 / SWEC_SAMPLING_RATE), axis=-1)

    print(f"Final artifact shape: {final_artifacts.shape}")

    final_data = upsampled_data
    sampling_rate = 30000 

    mixed = SignalDataWithGroundTruth(
        raw_data=final_data + final_artifacts,
        sampling_rate=sampling_rate,
        ground_truth=final_data,
        artifacts=final_artifacts
    )

    # save to npz 
    np.savez_compressed('../data/added_artifacts_swec_data_30000.npz', 
        mixed_data=mixed.raw_data, 
        ground_truth=mixed.ground_truth, 
        artifacts=mixed.artifacts, 
        sampling_rate=mixed.sampling_rate
    )

    analyzer = NeuralAnalyzer(mixed.sampling_rate)
    plotter = NeuralPlotter(analyzer)

    plotter.plot_trial_channels(mixed.artifacts, 0, [0], title='Mixed Signal with Artifacts')
    plotter.plot_trace_comparison(
        ground_truth=mixed.ground_truth,
        mixed_data=mixed.raw_data,
        trial_idx=0,
        channel_idx=0,
    )

if __name__ == "__main__":
    main()
