from sparc.methods.decomposition import ICA, LocalICA, SparseLocalProjection
from sparc import DataHandler, NeuralAnalyzer, NeuralPlotter
from sparc.core.signal_data import ArtifactTriggers
from sparc import MethodTester
from sparc.core.signal_data import SignalData, SignalDataWithGroundTruth, SimulatedData
import matplotlib.pyplot as plt
import numpy as np


def ica():
    data_handler = DataHandler()
    data_obj = data_handler.load_npz_data('../../data/eraasr_1000.npz')
    
    pre_artifact_train_samples = 200
    post_artifact_train_samples = 500

    orig_data = data_obj['arr_0'] # (trials, channels, samples)
    artifact_markers = ArtifactTriggers(
        starts=[[np.array([pre_artifact_train_samples])]]
    )
    
    data_obj = SignalData(
        raw_data=orig_data,
        sampling_rate=1000,
        artifact_markers=artifact_markers
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
        mode='targeted',
        pre_ms=0,
        post_ms=(post_artifact_train_samples - pre_artifact_train_samples) * 1000 / data_obj.sampling_rate,
    )
    ica.set_sampling_rate(data_obj.sampling_rate)
    ica.fit(data, artifact_markers=data_obj.artifact_markers)
    ica.plot_components()
    cleaned_data = ica.transform(data)

    plotter.plot_trace_comparison(cleaned_data, data_obj.raw_data, 0, 0, "Just artifact part")
    # cleaned_data = np.concatenate([orig_data[:, :, :pre_artifact_train_samples], cleaned_data, orig_data[:, :, pre_artifact_train_samples + post_artifact_train_samples:]], axis=2)
    # plotter.plot_trace_comparison(cleaned_data, orig_data, 0, 0, "All data")
    # plotter.plot_trace_comparison(cleaned_data, orig_data, 0, 20, "All data")

def multiple_icas():
    data_handler = DataHandler()
    data_obj = data_handler.load_npz_data('../../data/eraasr_1000.npz')
    pre_artifact_train_samples = 200
    post_artifact_train_samples = 500

    orig_data = data_obj['arr_0'] # (trials, channels, samples)
    artifact_markers = ArtifactTriggers(
        starts=[[np.array([pre_artifact_train_samples])]]
    )
    
    data_obj = SignalData(
        raw_data=orig_data,
        sampling_rate=1000,
        artifact_markers=artifact_markers
    )
    data = data_obj.raw_data
    print(data.shape)

    artifact_identify_methods = ['kurtosis_max']
    modes = ['global', 'targeted']
    components = [2, 4, 8, 16, 24]
    
    methods = {}
    for artifact_identify_method in artifact_identify_methods:
        for mode in modes:
            for cp in components:
                method_name = f"ica_{artifact_identify_method}_{mode}_{cp}"
                methods[method_name] = ICA(
                    n_components=cp, features_axis=1,artifact_identify_method=artifact_identify_method, mode=mode,
                    pre_ms=0,
                    post_ms=(post_artifact_train_samples - pre_artifact_train_samples) * 1000 / data_obj.sampling_rate,
                )

    tester = MethodTester(
        data=data_obj,
        methods=methods,
    )
    tester.run()
    tester.print_results()
    tester.plot_results()
    tester.compare()

def ica_one_trial_one_channel():
    data_handler = DataHandler()
    data_obj = data_handler.load_npz_data('../../data/eraasr_1000.npz')
    pre_artifact_train_samples = 200
    post_artifact_train_samples = 500
    
    orig_data = data_obj['arr_0'] # (trials, channels, samples)
    artifact_markers = ArtifactTriggers(
        starts=[[np.array([pre_artifact_train_samples])]]
    )

    orig_data = orig_data[0, 0:4, :][np.newaxis, :, :]
    # Only use first channel's artifact markers to avoid IndexError
    artifact_markers_starts_ch0 = artifact_markers.starts[0][0]
    artifact_markers = ArtifactTriggers(
        starts=[[np.array(artifact_markers_starts_ch0)]]
    )
    
    data_obj = SignalData(
        raw_data=orig_data,
        sampling_rate=1000,
        artifact_markers=artifact_markers
    )
    data = data_obj.raw_data
    print(data.shape)
    
    analyzer = NeuralAnalyzer(sampling_rate=data_obj.sampling_rate)
    plotter = NeuralPlotter(analyzer)
    plotter.plot_trace_comparison(data, data, 0, 0, "Just artifact part")
    ica = ICA(
        n_components=3,
        features_axis=1,
        artifact_identify_method='kurtosis_max',
        mode='targeted',
        pre_ms=0,
        post_ms=(post_artifact_train_samples - pre_artifact_train_samples) * 1000 / data_obj.sampling_rate,
    )
    ica.set_sampling_rate(data_obj.sampling_rate)
    ica.fit(data, artifact_markers=data_obj.artifact_markers)
    ica.plot_components()
    cleaned_data = ica.transform(data)
    plotter.plot_trace_comparison(cleaned_data, data, 0, 0)


if __name__ == "__main__":
    # ica()
    ica_one_trial_one_channel()
