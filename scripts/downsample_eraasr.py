from sparc.core.data_handler import DataHandler
from sparc import NeuralAnalyzer, NeuralPlotter
import numpy as np
from scipy.signal import resample

INPUT_FILE = '../data/eraasr_30000.npz'
OUTPUT_FILE = '../data/eraasr_1000.npz'
DOWNSAMPLE_RATE = 1000

def main():
    data_handler = DataHandler()
    data_obj = data_handler.load_npz_data(INPUT_FILE)
    data = data_obj['arr_0']
    data = resample(data, DOWNSAMPLE_RATE, axis=-1)
    np.savez_compressed(OUTPUT_FILE, data)

    neural_analyzer = NeuralAnalyzer(sampling_rate=DOWNSAMPLE_RATE)
    neural_plotter = NeuralPlotter(neural_analyzer)
    neural_plotter.plot_all_channels_trial(data, 0)

if __name__ == "__main__":
    main()