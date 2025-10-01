import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from sparc.core.data_handler import DataHandler
from sparc.core.signal_data import SignalData, ArtifactTriggers
from sparc.methods.decomposition.ica import ICA
from sparc.methods.decomposition.pca import PCA
from sparc.core.plotting import NeuralAnalyzer, NeuralPlotter
from sparc.method_tester import MethodTester


def run():
    data_handler = DataHandler()
    data_obj = data_handler.load_npz_data('../../data/eraasr_1000.npz')
    
    pre_artifact_train_samples = 200
    post_artifact_train_samples = 500

    orig_data = data_obj['arr_0']
    artifact_markers = ArtifactTriggers(
        starts=[[np.array([pre_artifact_train_samples])]]
    )
    
    data_obj = SignalData(
        raw_data=orig_data,
        sampling_rate=1000,
        artifact_markers=artifact_markers
    )
    
    methods = {
        'ica_kurtosis_max_targeted': ICA(
            n_components=2, features_axis=1, artifact_identify_method='kurtosis_max', 
            mode='global', pre_ms=0, 
            post_ms=(post_artifact_train_samples - pre_artifact_train_samples) * 1000 / data_obj.sampling_rate
        ),
        
        'pca_variance_global': PCA(
            n_components=4, features_axis=1, noise_identify_method='variance',
            mode='global'
        ),
    }
    
    tester = MethodTester(
        data=data_obj,
        methods=methods,
    )
    
    print(f"Testing {len(methods)} methods...")
    tester.run()
    tester.print_results()
    tester.plot_results()
    tester.compare()


if __name__ == "__main__":
    run()
