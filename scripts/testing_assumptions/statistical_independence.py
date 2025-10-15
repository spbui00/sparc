import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from sparc import DataHandler, NeuralAnalyzer, NeuralPlotter
from sparc.core.signal_data import SimulatedData, ArtifactTriggers, SignalDataWithGroundTruth
import numpy as np
from scipy.stats import pearsonr, entropy
from matplotlib import pyplot as plt


def statistical_independence(data_obj, dataset_name):
    analyzer = NeuralAnalyzer(sampling_rate=data_obj.sampling_rate)
    plotter = NeuralPlotter(analyzer)

    clean_neural = data_obj.ground_truth
    artifact = data_obj.artifacts

    clean_flat = np.ravel(clean_neural)
    artifact_flat = np.ravel(artifact)

    corr_coef, pval = pearsonr(clean_flat, artifact_flat)
    print(f"[{dataset_name}] Pearson correlation coefficient: {corr_coef:.10f} (p={pval})")

    def mutual_information(x, y, bins=50):
        c_xy = np.histogram2d(x, y, bins)[0]
        pxy = c_xy / np.sum(c_xy)
        px = np.sum(pxy, axis=1)
        py = np.sum(pxy, axis=0)
        Hxy = entropy(pxy.flatten(), base=2)
        Hx = entropy(px, base=2)
        Hy = entropy(py, base=2)
        MI = Hx + Hy - Hxy
        norm_MI = MI / Hxy if Hxy > 0 else 0.0
        return MI, norm_MI

    mi, norm_mi = mutual_information(clean_flat, artifact_flat)
    print(f"[{dataset_name}] Mutual information: {mi:.4f} bits (normalized MI: {norm_mi:.4f})")

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
    statistical_independence(data_obj, 'simulated data 1000Hz')


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
    statistical_independence(data_obj, 'simulated data 30000Hz')


    data_obj = data_handler.load_npz_data('../../data/added_artifacts_swec_data_512.npz')
    artifact_markers_data = data_obj['artifact_markers']
    data_obj = SignalDataWithGroundTruth(
        raw_data=data_obj['mixed_data'],
        sampling_rate=data_obj['sampling_rate'],
        ground_truth=data_obj['ground_truth'],
        artifacts=data_obj['artifacts'],
        artifact_markers=ArtifactTriggers(starts=artifact_markers_data)
    )
    statistical_independence(data_obj, 'swec data')