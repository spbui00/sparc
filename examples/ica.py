from sparc.methods.decomposition import ICA, LocalICA, SparseLocalProjection
from sparc import DataHandler, NeuralAnalyzer, NeuralPlotter
from sparc.core.signal_data import SignalData
import matplotlib.pyplot as plt


def ica():
    data_handler = DataHandler()
    # data_obj = data_handler.load_concatenated_simulated_data('../data/SimulatedData_2x64_30000_10trials.npz', 30000)
    data_obj = data_handler.load_simulated_data('../data/simulated_data_2x64_30000.npz', 30000)
    data = data_obj.raw_data
    print(data.shape)
    analyzer = NeuralAnalyzer(sampling_rate=data_obj.sampling_rate)
    plotter = NeuralPlotter(analyzer)
    plotter.plot_all_channels_trial(data_obj.raw_data, 0)

    ica = ICA(n_components=5, features_axis=1)
    ica.fit(data)
    ica.plot_components()
    cleaned_data = ica.transform(data)
    plotter.plot_cleaned_comparison(
        data_obj.ground_truth,
        data_obj.raw_data,
        cleaned_data,
        0,
        0
    )
    plotter.plot_cleaned_comparison(
        data_obj.ground_truth,
        data_obj.raw_data,
        cleaned_data,
        0,
        1
    )

def local_ica():
    data_handler = DataHandler()
    # data_obj = data_handler.load_simulated_data('../data/simulated_data_2x64_30000.npz', 30000)
    data_obj = SignalData(
        raw_data = data_handler.load_npz_data('../data/eraasr_30000.npz')['arr_0'],
        sampling_rate = 30000
    )
    data = data_obj.raw_data
    print(data.shape)
    analyzer = NeuralAnalyzer(sampling_rate=data_obj.sampling_rate)
    plotter = NeuralPlotter(analyzer)
    # plotter.plot_all_channels_trial(data_obj.raw_data, 0)

    ica = LocalICA(
        n_components=2,
        features_axis=1,
        stim_channel=0,
        local_radius=24,
    )
    ica.fit(data)
    ica.plot_components()
    cleaned_data = ica.transform(data)
    plotter.plot_trace_comparison(
        cleaned_data,
        data, 
        0, 0
    )
    plotter.plot_trace_comparison(
        cleaned_data,
        data, 
        0, 23
    )
    # plotter.plot_cleaned_comparison(
    #     data_obj.ground_truth,
    #     data_obj.raw_data,
    #     cleaned_data,
    #     0,
    #     0
    # )
    # plotter.plot_cleaned_comparison(
    #     data_obj.ground_truth,
    #     data_obj.raw_data,
    #     cleaned_data,
    #     0,
    #     1
    # )
    # difference between raw data and cleaned data
    plt.plot(data_obj.raw_data[0, 0, :] - cleaned_data[0, 0, :])
    plt.show()

def sparse_local_projection():
    data_handler = DataHandler()
    data_obj = data_handler.load_concatenated_simulated_data('../data/SimulatedData_2x64_30000_10trials.npz', 30000)
    data = data_obj.raw_data
    print(data.shape)
    analyzer = NeuralAnalyzer(sampling_rate=data_obj.sampling_rate)
    plotter = NeuralPlotter(analyzer)
    # plotter.plot_all_channels_trial(data_obj.raw_data, 0)

    ica = SparseLocalProjection(
        stim_channel=0,
        local_radius=5,
        epoch_pre=50,
        epoch_post=150,
        l1_alpha=0.01,
        features_axis=-1,
    )
    ica.fit(data, stim_times=data_obj.artifact_markers.starts)
    ica.plot_components()
    cleaned_data = ica.transform(data)
    plotter.plot_cleaned_comparison(
        data_obj.ground_truth,
        data_obj.raw_data,
        cleaned_data,
        0,
        0
    )
    plotter.plot_cleaned_comparison(
        data_obj.ground_truth,
        data_obj.raw_data,
        cleaned_data,
        0,
        1
    )

if __name__ == "__main__":
    local_ica()
