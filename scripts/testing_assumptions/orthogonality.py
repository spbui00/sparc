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
    
    datasets = [
        # ('../../data/simulated_data_2x64_1000.npz', 'simulated data 1000Hz'),
        # ('../../data/simulated_data_2x64_30000.npz', 'simulated data 30000Hz'),
        ('../../data/added_artifacts_swec_data_512.npz', 'swec data 512Hz'),
        ('../../data/added_artifacts_swec_data_seizure_512.npz', 'swec seizure data 512Hz'),
        ('../../data/added_artifacts_swec_data_seizure_512_lower_freq.npz', 'swec seizure data 512Hz lower freq'),
    ]

    for data_path, dataset_name in datasets:
        print(f"\n{'='*60}\nProcessing: {dataset_name}\n{'='*60}")
        data_obj_dict = data_handler.load_npz_data(data_path)
        
        if 'simulated' in dataset_name:
            data_obj = SimulatedData(
                raw_data=data_obj_dict['raw_data'],
                sampling_rate=data_obj_dict['sampling_rate'],
                ground_truth=data_obj_dict['ground_truth'],
                artifacts=data_obj_dict['artifacts'],
                artifact_markers=ArtifactTriggers(starts=data_obj_dict['artifact_markers']),  
                firing_rate=data_obj_dict.get('firing_rate'),
                spike_train=data_obj_dict.get('spike_train'),
                lfp=data_obj_dict.get('lfp'),
                stim_params=None,
                snr=data_obj_dict.get('snr'),
            )
        else:
            data_obj = SignalDataWithGroundTruth(
                raw_data=data_obj_dict['mixed_data'],
                sampling_rate=data_obj_dict['sampling_rate'],
                ground_truth=data_obj_dict['ground_truth'],
                artifacts=data_obj_dict['artifacts'],
                artifact_markers=ArtifactTriggers(starts=data_obj_dict['artifact_markers']),
            )
        
        max_time_samples = data_obj.raw_data.shape[-1]
        window_sizes = [10, 100, 200, 500, int(1000 * max_time_samples / data_obj.sampling_rate)]
        
        for window_size in window_sizes:
            clean_neural_windows, artifact_windows = extract_windows(data_obj, window_size)
            print(f"clean_neural_windows: {clean_neural_windows.shape}")
            orthogonality(clean_neural_windows, artifact_windows, f'{dataset_name} (window size {window_size} ms)')