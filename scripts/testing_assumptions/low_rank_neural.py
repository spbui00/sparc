import numpy as np
from sparc import DataHandler
from sparc.core.signal_data import SignalDataWithGroundTruth, SimulatedData, ArtifactTriggers


def analyze_rank(matrix, dimension, dimension_name, dataset_name, window_info):
    if matrix.shape[0] < 2 or matrix.shape[1] < 2:
        print(f"[{dataset_name} @ {window_info}] Cannot analyze rank for matrix with shape {matrix.shape}")
        return
        
    try:
        _, s, _ = np.linalg.svd(matrix, full_matrices=False)
    except np.linalg.LinAlgError as e:
        print(f"[{dataset_name} @ {window_info}] SVD failed: {e}")
        return

    explained_variance = s**2 / np.sum(s**2)
    cumulative_variance = np.cumsum(explained_variance)
    
    threshold = 0.95
    min_rank = np.searchsorted(cumulative_variance, threshold) + 1
    
    max_rank = min(matrix.shape)
    
    is_low_rank = min_rank <= max(1, dimension // 10)
    
    print(f"[{dataset_name} @ {window_info}] {dimension_name}: Min rank for 95% variance: {min_rank} (out of {max_rank})")
    
    if is_low_rank: 
        print(f"[{dataset_name} @ {window_info}] {dimension_name}: Neural data are LOW rank.")
    else:
        print(f"[{dataset_name} @ {window_info}] {dimension_name}: Neural data are NOT low rank (relative to {dimension_name}).")


def perform_rank_analysis(artifact_windows_3d, dataset_name, window_info):
    if artifact_windows_3d is None or artifact_windows_3d.ndim != 3:
        print(f"[{dataset_name} @ {window_info}] Invalid 3D data for analysis.")
        return
        
    w, c, n = artifact_windows_3d.shape # (trials, channels, samples)

    temporal_matrix = artifact_windows_3d.reshape(w * c, n)
    analyze_rank(temporal_matrix, n, "Temporal Rank", dataset_name, window_info)
    
    spatial_matrix = artifact_windows_3d.transpose(0, 2, 1).reshape(w * n, c)
    analyze_rank(spatial_matrix, c, "Spatial Rank", dataset_name, window_info)
    
    print()

if __name__ == "__main__":
    data_handler = DataHandler()
    
    datasets = [
        ('../../data/simulated_data_2x64_1000.npz', 'simulated data 1000Hz'),
        ('../../data/simulated_data_2x64_30000.npz', 'simulated data 30000Hz'),
        ('../../data/added_artifacts_swec_data_512.npz', 'swec data 512Hz'),
        ('../../data/added_artifacts_swec_data_seizure_512.npz', 'swec seizure data 512Hz')
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
                    
        original_neural_tensor = data_obj.ground_truth
        fs = data_obj.sampling_rate
            
        w, c, n_total_in_trial = original_neural_tensor.shape
        max_time_samples = data_obj.raw_data.shape[-1]
        window_sizes_ms = [10, 100, 200, 500, int(1000 * max_time_samples / fs)] # ms

        for window_size_ms in window_sizes_ms:            
            n_samples_window = int(window_size_ms * fs / 1000)
            
            if n_samples_window == 0:
                print(f"[{dataset_name}] Window size {window_size_ms}ms is too small.")
                continue
            
            truncated_3d_tensor = original_neural_tensor[:, :, :n_samples_window]
            window_info = f"{window_size_ms} ms ({n_samples_window} samples)"
            perform_rank_analysis(truncated_3d_tensor, dataset_name, window_info)