from sparc import DataHandler, NeuralAnalyzer, NeuralPlotter
import numpy as np


def main():
    eraasr_mat = '../research/datasets/eraasr-1.0.0/exampleDataTensor.mat'
    data = DataHandler().load_eraasr_data(eraasr_mat, sampling_rate=30000)
    print(data.raw_data.shape)
    analyzer = NeuralAnalyzer(sampling_rate=data.sampling_rate)
    plotter = NeuralPlotter(analyzer)
    plotter.plot_all_channels_trial(data.raw_data, 0)

    SAVE_PATH = '../data/eraasr_30000.npz'
    np.savez_compressed(SAVE_PATH, data.raw_data)


if __name__ == "__main__":
    main()
