import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from sparc import DataHandler, NeuralAnalyzer, NeuralPlotter
from sparc.core.signal_data import SimulatedData, ArtifactTriggers, SignalDataWithGroundTruth
import numpy as np
from scipy.stats import pearsonr, entropy
from matplotlib import pyplot as plt
from utils import extract_windows


def statistical_independence(clean_neural, artifact, dataset_name):
    correlations = []
    p_values = []
    mutual_infos = []
    normalized_mis = []
    
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
    
    for i in range(clean_neural.shape[0]):
        clean_window = clean_neural[i, :]
        artifact_window = artifact[i, :]
        
        try:
            corr_coef, pval = pearsonr(clean_window, artifact_window)
            correlations.append(corr_coef)
            p_values.append(pval)
            
            mi, norm_mi = mutual_information(clean_window, artifact_window)
            mutual_infos.append(mi)
            normalized_mis.append(norm_mi)
        except Exception as e:
            print(f"[{dataset_name}] Error: {e}")
            continue
    
    avg_corr = np.mean(correlations) if correlations else np.nan
    avg_pval = np.mean(p_values) if p_values else np.nan
    avg_mi = np.mean(mutual_infos) if mutual_infos else np.nan
    avg_norm_mi = np.mean(normalized_mis) if normalized_mis else np.nan
    
    print(f"[{dataset_name}] Pearson correlation coefficient: {avg_corr:.10f} (avg p={avg_pval:.10f})")
    print(f"[{dataset_name}] Mutual information: {avg_mi:.4f} bits (normalized MI: {avg_norm_mi:.4f})")
    print(f"[{dataset_name}] Number of windows analyzed: {len(correlations)}")

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
        statistical_independence(clean_neural_windows, artifact_windows, f'simulated data 1000Hz (window size {window_size} ms)')

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
        statistical_independence(clean_neural_windows, artifact_windows, f'simulated data 30000Hz (window size {window_size} ms)')


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
        statistical_independence(clean_neural_windows, artifact_windows, f'swec data (window size {window_size} ms)')

    data_obj = data_handler.load_npz_data('../../data/added_artifacts_swec_data_seizure_512.npz')
    artifact_markers_data = data_obj['artifact_markers']
    data_obj = SignalDataWithGroundTruth(
        raw_data=data_obj['mixed_data'], sampling_rate=data_obj['sampling_rate'],
        ground_truth=data_obj['ground_truth'], artifacts=data_obj['artifacts'],
        artifact_markers=ArtifactTriggers(starts=artifact_markers_data)
    )
    window_sizes = [10, 100, 200, 500, int(1000 * data_obj.raw_data.shape[2] / data_obj.sampling_rate)]
    for window_size in window_sizes:
        clean_neural_windows, artifact_windows = extract_windows(data_obj, window_size)
        statistical_independence(clean_neural_windows, artifact_windows, f'swec seizure data (window size {window_size} ms)')