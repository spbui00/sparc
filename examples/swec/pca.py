import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from sparc.core.data_handler import DataHandler
from sparc.core.signal_data import SignalDataWithGroundTruth, ArtifactTriggers
from sparc.methods.decomposition.pca import PCA
from sparc.method_tester import MethodTester
from sparc.core.plotting import NeuralAnalyzer
from sparc.core.plotting import NeuralPlotter


def pca():    
    data_handler = DataHandler()
    data_obj = data_handler.load_npz_data('../../data/added_artifacts_swec_data_512.npz')
    
    artifact_markers_data = data_obj['artifact_markers']
    if hasattr(artifact_markers_data, 'starts'):
        artifact_markers = artifact_markers_data
    else:
        artifact_markers = ArtifactTriggers(starts=artifact_markers_data)

    data_obj = SignalDataWithGroundTruth(
        raw_data=data_obj['mixed_data'],
        sampling_rate=data_obj['sampling_rate'],
        ground_truth=data_obj['ground_truth'],
        artifacts=data_obj['artifacts'],
        artifact_markers=artifact_markers
    )
    
    print(f"Data shape: {data_obj.raw_data.shape}")
    print(f"Sampling rate: {data_obj.sampling_rate} Hz")
    
    pca = PCA(
        n_components=4,
        features_axis=1,
        noise_identify_method='variance',
        mode='global',
        pre_ms=0.5,
        post_ms=4.0,
    )

    tester = MethodTester(
        data=data_obj,
        methods={'pca': pca},
    )
    tester.run()
    tester.print_results()
    tester.plot_results()
    tester.compare()
    
    # pca.set_sampling_rate(data_obj.sampling_rate)
    
    # print("Fitting PCA...")
    # pca.fit(data_obj.raw_data, data_obj.artifact_markers)
    
    # print("Transforming data...")
    # cleaned_data = pca.transform(data_obj.raw_data)
    
    # plotter = NeuralPlotter(NeuralAnalyzer(sampling_rate=data_obj.sampling_rate))
    
    # plotter.plot_cleaned_comparison(data_obj.ground_truth, data_obj.raw_data, cleaned_data, 0, 0)


if __name__ == "__main__":
    pca()
