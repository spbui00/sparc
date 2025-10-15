from sparc import DataHandler, NeuralAnalyzer, NeuralPlotter
from sparc.core.signal_data import SignalDataWithGroundTruth, SimulatedData, ArtifactTriggers
import numpy as np


def low_rank_artifacts_spatial(data_obj, dataset_name):
    artifact = data_obj.artifacts
    n_trials, n_channels, n_time = artifact.shape
    artifact_matrix = artifact.transpose(1, 0, 2).reshape(n_channels, -1)  # (channels, trials * time)
    _, s, _ = np.linalg.svd(artifact_matrix, full_matrices=False)

    explained_variance = s**2 / np.sum(s**2) # squared singular values are the eigenvalues of the covariance matrix
    cumulative_variance = np.cumsum(explained_variance)

    # Heuristic: artifact is 'low rank' if most variance explained by few components
    threshold = 0.95
    min_rank = np.searchsorted(cumulative_variance, threshold) + 1
    print(f"[{dataset_name}] Minimum rank to explain {threshold*100:.0f}% variance: {min_rank} out of {n_channels}")
    if min_rank <= max(1, n_channels // 10):
        print(f"[{dataset_name}] Spatial: Artifacts are low rank (most variance explained by {min_rank} components).")
    else:
        print(f"[{dataset_name}] Spatial: Artifacts are not strongly low rank (need {min_rank} components).")

    direct_rank = np.linalg.matrix_rank(artifact_matrix)
    print(f"[{dataset_name}] Direct Mathematical Spatial Rank: {direct_rank} out of {n_channels}")
    if direct_rank < n_channels:
         print(f"    - Result: Confirmed to be spatially low-rank (rank < num_channels).")
    else:
         print(f"    - Result: Confirmed to be spatially full-rank.")

def low_rank_artifacts_temporal(data_obj, dataset_name):
    artifact = data_obj.artifacts
    n_trials, n_channels, n_time = artifact.shape
    artifact_matrix = artifact.transpose(2, 0, 1).reshape(n_time, -1)  # (time, trials * channels)
    _, s, _ = np.linalg.svd(artifact_matrix, full_matrices=False)
    explained_variance = s**2 / np.sum(s**2) # squared singular values are the eigenvalues of the covariance matrix
    cumulative_variance = np.cumsum(explained_variance)
    threshold = 0.95
    min_rank = np.searchsorted(cumulative_variance, threshold) + 1
    print(f"[{dataset_name}] Minimum rank to explain {threshold*100:.0f}% variance: {min_rank} out of {n_time}")

    if min_rank <= max(1, n_time // 10):
        print(f"[{dataset_name}] Temporal: Artifacts are low rank (most variance explained by {min_rank} components).")
    else:
        print(f"[{dataset_name}] Temporal: Artifacts are not strongly low rank (need {min_rank} components).")

    direct_rank = np.linalg.matrix_rank(artifact_matrix)
    print(f"[{dataset_name}] Direct Mathematical Temporal Rank: {direct_rank} out of {n_time}")
    if direct_rank < n_time:
         print(f"    - Result: Confirmed to be temporally low-rank (rank < num_time).")
    else:
         print(f"    - Result: Confirmed to be temporally full-rank.")

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
    low_rank_artifacts_spatial(data_obj, 'simulated data 1000Hz')
    low_rank_artifacts_temporal(data_obj, 'simulated data 1000Hz')
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
    low_rank_artifacts_spatial(data_obj, 'simulated data 30000Hz')
    low_rank_artifacts_temporal(data_obj, 'simulated data 30000Hz') 
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
    low_rank_artifacts_spatial(data_obj, 'swec data')
    low_rank_artifacts_temporal(data_obj, 'swec data')
    print("-" * 50)