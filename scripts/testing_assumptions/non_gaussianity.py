from sparc import DataHandler, NeuralAnalyzer, NeuralPlotter
from sparc.core.signal_data import SignalDataWithGroundTruth, SimulatedData, ArtifactTriggers
import numpy as np
from scipy.stats import shapiro, probplot, jarque_bera
import matplotlib.pyplot as plt


def non_gaussianity(data_obj, dataset_name):
    clean_neural = data_obj.ground_truth
    artifact = data_obj.artifacts

    clean_flat = np.ravel(clean_neural)
    artifact_flat = np.ravel(artifact)

    # plt.figure(figsize=(12, 5))
    # plt.subplot(1, 2, 1)
    # plt.hist(clean_flat, bins=100, alpha=0.7, color='green', edgecolor='black')
    # plt.title(f'{dataset_name}: Clean Neural Signal Histogram')
    # plt.xlabel('Value')
    # plt.ylabel('Count')

    # plt.subplot(1, 2, 2)
    # plt.hist(artifact_flat, bins=100, alpha=0.7, color='red', edgecolor='black')
    # plt.title(f'{dataset_name}: Artifact Signal Histogram')
    # plt.xlabel('Value')
    # plt.ylabel('Count')

    # plt.tight_layout()
    # plt.show()

    jb_clean_stat, jb_clean_p = jarque_bera(clean_flat)
    jb_artifact_stat, jb_artifact_p = jarque_bera(artifact_flat)

    print(f"[{dataset_name}] Jarque-Bera test for clean neural signal: JB={jb_clean_stat:.4f} (p={jb_clean_p:.4e})")
    print(f"[{dataset_name}] Jarque-Bera test for artifact signal:    JB={jb_artifact_stat:.4f} (p={jb_artifact_p:.4e})")

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
    non_gaussianity(data_obj, 'simulated data 1000Hz')


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
    non_gaussianity(data_obj, 'simulated data 30000Hz')


    data_obj = data_handler.load_npz_data('../../data/added_artifacts_swec_data_512.npz')
    artifact_markers_data = data_obj['artifact_markers']
    data_obj = SignalDataWithGroundTruth(
        raw_data=data_obj['mixed_data'],
        sampling_rate=data_obj['sampling_rate'],
        ground_truth=data_obj['ground_truth'],
        artifacts=data_obj['artifacts'],
        artifact_markers=ArtifactTriggers(starts=artifact_markers_data)
    )
    non_gaussianity(data_obj, 'swec data')