from sparc.methods.decomposition import ICA, LocalICA, SparseLocalProjection
from sparc import DataHandler, NeuralAnalyzer, NeuralPlotter
from sparc.core.signal_data import ArtifactTriggers
from sparc import MethodTester
from sparc.core.signal_data import SignalData, SignalDataWithGroundTruth, SimulatedData
import matplotlib.pyplot as plt
import numpy as np


def ica():
    data_handler = DataHandler()
    data_obj = data_handler.load_npz_data('../../data/added_artifacts_swec_data_512.npz')
    
    artifact_markers_data = data_obj['artifact_markers']
    
    data_obj = SignalDataWithGroundTruth(
        raw_data=data_obj['mixed_data'],
        sampling_rate=data_obj['sampling_rate'],
        ground_truth=data_obj['ground_truth'],
        artifacts=data_obj['artifacts'],
        artifact_markers=ArtifactTriggers(starts=artifact_markers_data)
    )
    data = data_obj.raw_data
    print(data.shape)
    analyzer = NeuralAnalyzer(sampling_rate=data_obj.sampling_rate)
    plotter = NeuralPlotter(analyzer)
    # plotter.plot_all_channels_trial(data_obj.raw_data, 0)
    # plotter.plot_trial_channel(
    #     data_obj.raw_data,
    #     0,
    #     0
    # )

    ica = ICA(
        n_components=2,
        features_axis=1,
        artifact_identify_method='kurtosis_min',
        mode='targeted',
        pre_ms=0,
        post_ms=4.0,
        highpass_cutoff=0.5,
    )

    tester = MethodTester(
        data=data_obj,
        methods={'ica': ica},
    )
    tester.run()
    tester.print_results()
    tester.plot_results()
    tester.compare()
    # ica.set_sampling_rate(data_obj.sampling_rate)
    # ica.fit(data, artifact_markers=data_obj.artifact_markers)
    # ica.plot_components()
    # cleaned_data = ica.transform(data)

    # plotter.plot_trace_comparison(cleaned_data, data_obj.raw_data, 0, 0)
    # plotter.plot_cleaned_comparison(data_obj.ground_truth, data_obj.raw_data, cleaned_data, 0, 0)
    # plotter.plot_cleaned_comparison(data_obj.ground_truth, data_obj.raw_data, cleaned_data, 0, 30)
    # plotter.plot_cleaned_comparison(data_obj.ground_truth, data_obj.raw_data, cleaned_data, 0, 80)


def multiple_icas():
    data_handler = DataHandler()
    data_obj = data_handler.load_npz_data('../../data/added_artifacts_swec_data_512.npz')
    
    artifact_markers_data = data_obj['artifact_markers']
    
    data_obj = SignalDataWithGroundTruth(
        raw_data=data_obj['mixed_data'],
        sampling_rate=data_obj['sampling_rate'],
        ground_truth=data_obj['ground_truth'],
        artifacts=data_obj['artifacts'],
        artifact_markers=ArtifactTriggers(starts=artifact_markers_data)
    )
    data = data_obj.raw_data
    print(data.shape)

    artifact_identify_methods = ['kurtosis_min']
    modes = ['global', 'targeted']
    components = [2]
    highpass_cutoffs = [None, 0.1, 0.5, 1.0, 2.0]
    
    methods = {}
    for artifact_identify_method in artifact_identify_methods:
        for mode in modes:
            for cp in components:
                for hpc in highpass_cutoffs:
                    hpc_str = "no_filter" if hpc is None else str(hpc)
                    method_name = f"ica_{artifact_identify_method}_{mode}_{cp}_{hpc_str}"
                    methods[method_name] = ICA(
                        n_components=cp, features_axis=1,
                        artifact_identify_method=artifact_identify_method, mode=mode,
                        pre_ms=0,
                        post_ms=4.0,
                        highpass_cutoff=hpc,
                    )

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
