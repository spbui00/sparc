from sparc.methods.decomposition import ICA, LocalICA, SparseLocalProjection
from sparc import DataHandler, NeuralAnalyzer, NeuralPlotter
from sparc.core.signal_data import ArtifactTriggers
from sparc import MethodTester
from sparc.core.signal_data import SignalData, SignalDataWithGroundTruth, SimulatedData
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import resample


def ica():
    data_handler = DataHandler()
    # data_obj = data_handler.load_concatenated_simulated_data('../data/SimulatedData_2x64_30000_10trials.npz', 30000)
    data_obj = data_handler.load_npz_data('../../data/eraasr_1000.npz')
    
    pre_artifact_train_samples = 200
    post_artifact_train_samples = 600

    orig_data = data_obj['arr_0'] # (trials, channels, samples)
    raw_data = orig_data[:, :, pre_artifact_train_samples:pre_artifact_train_samples + post_artifact_train_samples]
    
    data_obj = SignalData(
        raw_data=raw_data,
        sampling_rate=1000
    )
    data = data_obj.raw_data
    print(data.shape)
    analyzer = NeuralAnalyzer(sampling_rate=data_obj.sampling_rate)
    plotter = NeuralPlotter(analyzer)
    plotter.plot_all_channels_trial(data_obj.raw_data, 0)

    ica = ICA(
        n_components=2,
        features_axis=1, 
        artifact_identify_method='kurtosis_max',
        mode='global'
    )
    ica.set_sampling_rate(data_obj.sampling_rate)
    ica.fit(data, artifact_markers=data_obj.artifact_markers)
    ica.plot_components()
    cleaned_data = ica.transform(data)

    plotter.plot_trace_comparison(cleaned_data, data_obj.raw_data, 0, 0, "Just artifact part")
    cleaned_data = np.concatenate([orig_data[:, :, :pre_artifact_train_samples], cleaned_data, orig_data[:, :, pre_artifact_train_samples + post_artifact_train_samples:]], axis=2)
    plotter.plot_trace_comparison(cleaned_data, orig_data, 0, 0, "All data")

def multiple_icas():
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
    data = data_obj.raw_data
    print(data.shape)
    analyzer = NeuralAnalyzer(sampling_rate=data_obj.sampling_rate)
    plotter = NeuralPlotter(analyzer)

    artifact_identify_methods = ['kurtosis_min', 'variance']
    modes = ['global', 'targeted']
    pre_ms = [1.0, 2.0, 3.0, 4.0]
    post_ms = [1.0, 2.0, 3.0, 4.0]
    
    methods = {}
    for artifact_identify_method in artifact_identify_methods:
        for mode in modes:
            for pr in pre_ms:
                for po in post_ms:
                    method_name = f"ica_{artifact_identify_method}_{mode}_{pr}_{po}"
                    methods[method_name] = ICA(n_components=2, features_axis=1,artifact_identify_method=artifact_identify_method, mode=mode, pre_ms=pr, post_ms=po)

    tester = MethodTester(
        data=data_obj,
        methods=methods,
    )
    tester.run()
    tester.print_results()
    tester.plot_results()
    tester.compare()


if __name__ == "__main__":
    ica()
