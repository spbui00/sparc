from sparc import DataHandler, NeuralAnalyzer, NeuralPlotter
from sparc.core.signal_data import SignalDataWithGroundTruth, SimulatedData, ArtifactTriggers
import numpy as np


def orthogonality(data_obj, dataset_name):
    clean_neural = data_obj.ground_truth
    artifact = data_obj.artifacts
    
    clean_flat = np.ravel(clean_neural)
    artifact_flat = np.ravel(artifact)
    
    neural_norm = clean_flat / np.linalg.norm(clean_flat)
    artifact_norm = artifact_flat / np.linalg.norm(artifact_flat)
    
    dot_product = np.dot(neural_norm, artifact_norm)
    
    print(f"[{dataset_name}] Dot product: {dot_product:.6f}")
    if abs(dot_product) < 0.01:
        print(f"[{dataset_name}] Result: Orthogonal")
    elif abs(dot_product) < 0.1:
        print(f"[{dataset_name}] Result: Nearly orthogonal")
    else:
        print(f"[{dataset_name}] Result: Not orthogonal")
    
    return dot_product

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
    orthogonality(data_obj, 'simulated data 1000Hz')


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
    orthogonality(data_obj, 'simulated data 30000Hz')


    data_obj = data_handler.load_npz_data('../../data/added_artifacts_swec_data_512.npz')
    artifact_markers_data = data_obj['artifact_markers']
    data_obj = SignalDataWithGroundTruth(
        raw_data=data_obj['mixed_data'],
        sampling_rate=data_obj['sampling_rate'],
        ground_truth=data_obj['ground_truth'],
        artifacts=data_obj['artifacts'],
        artifact_markers=ArtifactTriggers(starts=artifact_markers_data)
    )
    orthogonality(data_obj, 'swec data')