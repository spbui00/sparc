import numpy as np
from sparc import DataHandler
from sparc.core.signal_data import SignalDataWithGroundTruth, SimulatedData, ArtifactTriggers


def _compute_covariance(M):
    T, C, N = M.shape
    M_mean = np.mean(M, axis=-1, keepdims=True)
    M_centered = M - M_mean
    Cov = (M_centered @ M_centered.transpose(0, 2, 1)) / (N - 1)
    return Cov


def compute_soft_rank_from_covariance(data_3d, data_type, dataset_name, window_info):
    if data_3d is None or data_3d.ndim != 3:
        print(f"[{dataset_name} @ {window_info}] Invalid 3D {data_type} data for covariance analysis.")
        return None
    
    T, C, N = data_3d.shape
    
    if N < 2:
        print(f"[{dataset_name} @ {window_info}] Not enough samples for covariance computation.")
        return None
    
    cov = _compute_covariance(data_3d)
    
    nuc_norms = []
    fro_norms = []
    soft_ranks = []
    
    for t in range(T):
        cov_t = cov[t]
        _, s, _ = np.linalg.svd(cov_t, full_matrices=False)
        nuc = np.sum(s)
        fro = np.linalg.norm(cov_t, 'fro')
        soft_rank = nuc / (fro + 1e-6)
        nuc_norms.append(nuc)
        fro_norms.append(fro)
        soft_ranks.append(soft_rank)
    
    mean_nuc = np.mean(nuc_norms)
    mean_fro = np.mean(fro_norms)
    mean_soft_rank = np.mean(soft_ranks)
    
    print(f"[{dataset_name} @ {window_info}] {data_type} Covariance-based Soft Rank:")
    print(f"  Mean nuclear norm of covariance: {mean_nuc:.6e}")
    print(f"  Mean frobenius norm of covariance: {mean_fro:.6e}")
    print(f"  Mean soft rank (nuc/fro): {mean_soft_rank:.6e}")
    
    return mean_soft_rank


def perform_nuclear_norm_analysis(neural_windows_3d, artifact_windows_3d, dataset_name, window_info):
    if neural_windows_3d is None or neural_windows_3d.ndim != 3:
        print(f"[{dataset_name} @ {window_info}] Invalid 3D neural data for analysis.")
        return
    
    if artifact_windows_3d is None or artifact_windows_3d.ndim != 3:
        print(f"[{dataset_name} @ {window_info}] Invalid 3D artifact data for analysis.")
        return
        
    w, c, n = neural_windows_3d.shape
    
    if artifact_windows_3d.shape != (w, c, n):
        print(f"[{dataset_name} @ {window_info}] Shape mismatch: neural {neural_windows_3d.shape} vs artifact {artifact_windows_3d.shape}")
        return
    
    print(f"\n[{dataset_name} @ {window_info}] Analyzing covariance-based soft rank...")
    print(f"Data shape: {neural_windows_3d.shape} (trials, channels, samples)")
    
    neural_soft_rank = compute_soft_rank_from_covariance(neural_windows_3d, "Neural", dataset_name, window_info)
    artifact_soft_rank = compute_soft_rank_from_covariance(artifact_windows_3d, "Artifact", dataset_name, window_info)
    
    print("\n--- Comparison ---")
    if neural_soft_rank is not None and artifact_soft_rank is not None:
        ratio = artifact_soft_rank / neural_soft_rank if neural_soft_rank > 0 else float('inf')
        print(f"Soft rank ratio (Artifact/Neural): {ratio:.4f}")
    
    print()


if __name__ == "__main__":
    data_handler = DataHandler()
    
    datasets = [
        # ('../../data/simulated_data_2x64_1000.npz', 'simulated data 1000Hz'),
        # ('../../data/simulated_data_2x64_30000.npz', 'simulated data 30000Hz'),
        ('../../data/added_artifacts_swec_data_512.npz', 'swec data 512Hz'),
        ('../../data/added_artifacts_swec_data_seizure_512.npz', 'swec seizure data 512Hz'),
        ('../../data/added_artifacts_swec_data_seizure_512_lower_freq.npz', 'swec seizure data 512Hz lower freq')
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
                    
        neural_tensor = data_obj.ground_truth
        artifact_tensor = data_obj.artifacts
        fs = data_obj.sampling_rate
            
        w, c, n_total_in_trial = neural_tensor.shape
        max_time_samples = data_obj.raw_data.shape[-1]
        window_sizes_ms = [10, 100, 200, 500, int(1000 * max_time_samples / fs)]

        for window_size_ms in window_sizes_ms:            
            n_samples_window = int(window_size_ms * fs / 1000)
            
            if n_samples_window == 0:
                print(f"[{dataset_name}] Window size {window_size_ms}ms is too small.")
                continue
            
            truncated_neural = neural_tensor[:, :, :n_samples_window]
            truncated_artifacts = artifact_tensor[:, :, :n_samples_window]
            window_info = f"{window_size_ms} ms ({n_samples_window} samples)"
            perform_nuclear_norm_analysis(truncated_neural, truncated_artifacts, dataset_name, window_info)

