from sparc import DataHandler, NeuralAnalyzer, NeuralPlotter
from sparc.core.signal_data import SignalDataWithGroundTruth, SimulatedData, ArtifactTriggers
import numpy as np
from scipy.stats import shapiro, probplot, jarque_bera
import matplotlib.pyplot as plt
from utils import extract_windows


def non_gaussianity(neural_windows, artifact_windows, dataset_name):
    clean_flat = np.ravel(neural_windows)
    artifact_flat = np.ravel(artifact_windows)

    jb_clean_stat, jb_clean_p = jarque_bera(clean_flat)
    jb_artifact_stat, jb_artifact_p = jarque_bera(artifact_flat)

    print(f"[{dataset_name}] Jarque-Bera test for clean neural signal: JB={jb_clean_stat:.4f} (p={jb_clean_p:.4e})")
    print(f"[{dataset_name}] Jarque-Bera test for artifact signal:    JB={jb_artifact_stat:.4f} (p={jb_artifact_p:.4e})")
    print(f"[{dataset_name}] Number of windows analyzed: {neural_windows.shape[0]}")

if __name__ == "__main__":
    data_handler = DataHandler()
    data_obj = data_handler.load_npz_data('../../data/simulated_data_2x64_1000.npz')
    data_obj = SimulatedData(
        raw_data=data_obj['raw_data'],
        sampling_rate=data_obj['sampling_rate'],
        ground_truth=data_obj['ground_truth'],
        artifacts=data_obj['artifacts'],
        artifact_markers=ArtifactTriggers(starts=data_obj['artifact_markers']),  
        firing_rate=data_obj['firing_rate'],
        spike_train=data_obj['spike_train'],
        lfp=data_obj['lfp'],
        stim_params=None,
        snr=data_obj['snr'],
    )
    window_sizes = [10, 100, 200, 500, int(1000 * data_obj.raw_data.shape[2] / data_obj.sampling_rate)]
    for window_size in window_sizes:
        clean_neural_windows, artifact_windows = extract_windows(data_obj, window_size)
        non_gaussianity(clean_neural_windows, artifact_windows, f'simulated data 1000Hz (window size {window_size} ms)')
        print("-" * 50)

    data_obj = data_handler.load_npz_data('../../data/simulated_data_2x64_30000.npz')
    data_obj = SimulatedData(
        raw_data=data_obj['raw_data'],
        sampling_rate=data_obj['sampling_rate'],
        ground_truth=data_obj['ground_truth'],
        artifacts=data_obj['artifacts'],
        artifact_markers=ArtifactTriggers(starts=data_obj['artifact_markers']),  
        firing_rate=data_obj['firing_rate'],
        spike_train=data_obj['spike_train'],
        lfp=data_obj['lfp'],
        stim_params=None,
        snr=data_obj['snr'],
    )
    window_sizes = [10, 100, 200, 500, int(1000 * data_obj.raw_data.shape[2] / data_obj.sampling_rate)]
    for window_size in window_sizes:
        clean_neural_windows, artifact_windows = extract_windows(data_obj, window_size)
        non_gaussianity(clean_neural_windows, artifact_windows, f'simulated data 30000Hz (window size {window_size} ms)')
        print("-" * 50)

    data_obj = data_handler.load_npz_data('../../data/added_artifacts_swec_data_512.npz')
    artifact_markers_data = data_obj['artifact_markers']
    data_obj = SignalDataWithGroundTruth(
        raw_data=data_obj['mixed_data'],
        sampling_rate=data_obj['sampling_rate'],
        ground_truth=data_obj['ground_truth'],
        artifacts=data_obj['artifacts'],
        artifact_markers=ArtifactTriggers(starts=artifact_markers_data)
    )
    window_sizes = [10, 100, 200, 500, int(1000 * data_obj.raw_data.shape[2] / data_obj.sampling_rate)]
    for window_size in window_sizes:
        clean_neural_windows, artifact_windows = extract_windows(data_obj, window_size)
        non_gaussianity(clean_neural_windows, artifact_windows, f'swec data (window size {window_size} ms)')
        print("-" * 50)

    data_obj = data_handler.load_npz_data('../../data/added_artifacts_swec_data_seizure_512.npz')
    artifact_markers_data = data_obj['artifact_markers']
    data_obj = SignalDataWithGroundTruth(
        raw_data=data_obj['mixed_data'],
        sampling_rate=data_obj['sampling_rate'],
        ground_truth=data_obj['ground_truth'],
        artifacts=data_obj['artifacts'],
        artifact_markers=ArtifactTriggers(starts=artifact_markers_data)
    )
    window_sizes = [10, 100, 200, 500, int(1000 * data_obj.raw_data.shape[2] / data_obj.sampling_rate)]
    for window_size in window_sizes:
        clean_neural_windows, artifact_windows = extract_windows(data_obj, window_size)
        non_gaussianity(clean_neural_windows, artifact_windows, f'swec seizure data (window size {window_size} ms)')
        print("-" * 50)