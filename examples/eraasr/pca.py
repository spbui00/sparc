import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from sparc.core.data_handler import DataHandler
from sparc.core.signal_data import SignalData, ArtifactTriggers
from sparc.methods.decomposition.pca import PCA
from sparc.method_tester import MethodTester
from sparc.core.plotting import NeuralAnalyzer, NeuralPlotter


def pca():
    data_handler = DataHandler()
    data_obj = data_handler.load_npz_data('../../data/eraasr_1000.npz')
    
    pre_artifact_train_samples = 200
    post_artifact_train_samples = 500

    orig_data = data_obj['arr_0']  # (trials, channels, samples)
    artifact_markers = ArtifactTriggers(
        starts=[[np.array([pre_artifact_train_samples])]]
    )
    
    data_obj = SignalData(
        raw_data=orig_data,
        sampling_rate=1000,
        artifact_markers=artifact_markers
    )
    
    print(f"Data shape: {data_obj.raw_data.shape}")
    print(f"Sampling rate: {data_obj.sampling_rate} Hz")
    
    pca = PCA(
        n_components=4,
        features_axis=1,
        noise_identify_method='variance',
        mode='global',
        pre_ms=0,
        post_ms=(post_artifact_train_samples - pre_artifact_train_samples) * 1000 / data_obj.sampling_rate,
        highpass_cutoff=1.0,
        variance_threshold=0.05
    )
    
    pca.set_sampling_rate(data_obj.sampling_rate)
    
    print("Fitting PCA...")
    pca.fit(data_obj.raw_data, data_obj.artifact_markers)
    
    print("Transforming data...")
    cleaned_data = pca.transform(data_obj.raw_data)
    
    print(f"Explained variance ratios: {pca.get_explained_variance_ratio()}")
    print(f"Noise components: {pca.get_noise_components()}")
    print(f"Reconstruction error: {pca.get_reconstruction_error(data_obj.raw_data):.6f}")

    analyzer = NeuralAnalyzer(sampling_rate=data_obj.sampling_rate)
    plotter = NeuralPlotter(analyzer)
    
    plotter.plot_trace_comparison(cleaned_data, data_obj.raw_data, 0, 0, "PCA Noise Removal")
    pca.plot_components(data_obj.raw_data, trial_idx=0, channel_idx=0)


if __name__ == "__main__":
    pca()
