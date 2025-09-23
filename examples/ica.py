from sparc.methods import ICA
from sparc import DataHandler, NeuralAnalyzer, NeuralPlotter


def main():
    data_handler = DataHandler()
    data_obj = data_handler.load_concatenated_simulated_data('../data/SimulatedData_2x64_30000_10trials.npz', 30000)
    data = data_obj.raw_data
    print(data.shape)
    analyzer = NeuralAnalyzer(sampling_rate=data_obj.sampling_rate)
    plotter = NeuralPlotter(analyzer)
    plotter.plot_all_channels_trial(data_obj.raw_data, 0)

    ica = ICA(n_components=10, features_axis=1)
    ica.fit(data)
    ica.plot_components()
    cleaned_data = ica.transform(data)
    plotter.plot_cleaned_comparison(
        data_obj.ground_truth,
        data_obj.raw_data,
        cleaned_data,
        0,
        1
    )

if __name__ == "__main__":
    main()