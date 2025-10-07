import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from sparc.core.data_handler import DataHandler
from sparc.core.signal_data import SignalDataWithGroundTruth, SimulatedData, ArtifactTriggers
from sparc.methods.decomposition.pca import PCA
from sparc.method_tester import MethodTester
from sparc.core.plotting import NeuralAnalyzer, NeuralPlotter


def pca():
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
    
    print(f"Data shape: {data_obj.raw_data.shape}")
    print(f"Sampling rate: {data_obj.sampling_rate} Hz")
    
    pca = PCA(
        n_components=32,
        features_axis=1,
        noise_identify_method='variance',
        mode='global',
        highpass_cutoff=1.0,
        variance_threshold=0.03
    )
    
    pca.set_sampling_rate(data_obj.sampling_rate)
    
    print("Fitting PCA...")
    pca.fit(data_obj.raw_data, data_obj.artifact_markers)
    
    print("Transforming data...")
    cleaned_data = pca.transform(data_obj.raw_data)
    
    # Plot results
    analyzer = NeuralAnalyzer(sampling_rate=data_obj.sampling_rate)
    plotter = NeuralPlotter(analyzer)
    
    plotter.plot_cleaned_comparison(
        data_obj.ground_truth,
        data_obj.raw_data,
        cleaned_data,
        0,
        1
    )
    
    plotter.plot_cleaned_comparison(
        data_obj.ground_truth,
        data_obj.raw_data,
        cleaned_data,
        0,
        20
    )
    
    pca.plot_components(data_obj.raw_data, trial_idx=0, channel_idx=1)


def multiple_pcas():
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
    
    print(f"Data shape: {data_obj.raw_data.shape}")
    
    # Test different PCA configurations
    noise_methods = ['variance']
    modes = ['global']
    components = [28, 32, 36, 40]
    highpass_cutoffs = [0.5, 1.0, 2.0]
    
    methods = {}
    for noise_method in noise_methods:
        for mode in modes:
            for cp in components:
                for hpc in highpass_cutoffs:
                    hpc_str = "no_filter" if hpc is None else str(hpc)
                    method_name = f"pca_{noise_method}_{mode}_{cp}_{hpc_str}"
                    
                    methods[method_name] = PCA(
                        n_components=cp,
                        features_axis=1,
                        noise_identify_method=noise_method,
                        mode=mode,
                        highpass_cutoff=hpc,
                    )
    
    # Test methods
    tester = MethodTester(
        data=data_obj,
        methods=methods,
    )
    
    print(f"Testing {len(methods)} PCA methods...")
    tester.run()
    tester.print_results()
    tester.plot_results()
    tester.compare()


if __name__ == "__main__":
    multiple_pcas()
