from sparc import DataHandler, NeuralAnalyzer, NeuralPlotter
from sparc.core.signal_data import SignalDataWithGroundTruth, SimulatedData, ArtifactTriggers
import numpy as np
from utils import extract_windows


def orthogonality(neural_windows, artifact_windows, dataset_name):
    dot_products = []
    
    for i in range(neural_windows.shape[0]):
        clean_window = neural_windows[i, :]
        artifact_window = artifact_windows[i, :]
        
        # Normalize each window
        clean_norm = clean_window / np.linalg.norm(clean_window) if np.linalg.norm(clean_window) > 0 else clean_window
        artifact_norm = artifact_window / np.linalg.norm(artifact_window) if np.linalg.norm(artifact_window) > 0 else artifact_window
        
        # Compute dot product
        dot_product = np.dot(clean_norm, artifact_norm)
        dot_products.append(dot_product)
    
    # Compute average dot product across all windows
    avg_dot_product = np.mean(dot_products)
    
    print(f"[{dataset_name}] Average dot product: {avg_dot_product:.6f}")
    print(f"[{dataset_name}] Number of windows analyzed: {len(dot_products)}")
    
    if abs(avg_dot_product) < 0.01:
        print(f"[{dataset_name}] Result: Orthogonal")
    elif abs(avg_dot_product) < 0.1:
        print(f"[{dataset_name}] Result: Nearly orthogonal")
    else:
        print(f"[{dataset_name}] Result: Not orthogonal")
    
    return avg_dot_product

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
    window_sizes = [10, 100, 200, 500, int(1000 * data_obj.raw_data.shape[2] / data_obj.sampling_rate)] # ms
    for window_size in window_sizes:
        clean_neural_windows, artifact_windows = extract_windows(data_obj, window_size)
        orthogonality(clean_neural_windows, artifact_windows, f'simulated data 1000Hz (window size {window_size} ms)')

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
    window_sizes = [10, 100, 200, 500, int(1000 * data_obj.raw_data.shape[2] / data_obj.sampling_rate)] # ms
    for window_size in window_sizes:
        clean_neural_windows, artifact_windows = extract_windows(data_obj, window_size)
        orthogonality(clean_neural_windows, artifact_windows, f'simulated data 30000Hz (window size {window_size} ms)')

    data_obj = data_handler.load_npz_data('../../data/added_artifacts_swec_data_512.npz')
    artifact_markers_data = data_obj['artifact_markers']
    data_obj = SignalDataWithGroundTruth(
        raw_data=data_obj['mixed_data'],
        sampling_rate=data_obj['sampling_rate'],
        ground_truth=data_obj['ground_truth'],
        artifacts=data_obj['artifacts'],
        artifact_markers=ArtifactTriggers(starts=artifact_markers_data)
    )
    window_sizes = [10, 100, 200, 500, int(1000 * data_obj.raw_data.shape[2] / data_obj.sampling_rate)] # ms
    for window_size in window_sizes:
        clean_neural_windows, artifact_windows = extract_windows(data_obj, window_size)
        orthogonality(clean_neural_windows, artifact_windows, f'swec data (window size {window_size} ms)')

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
        orthogonality(clean_neural_windows, artifact_windows, f'swec seizure data (window size {window_size} ms)')