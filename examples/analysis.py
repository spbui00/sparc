from sparc import DataHandler, NeuralAnalyzer, NeuralPlotter
from sparc.core.signal_data import SignalDataWithGroundTruth
import numpy as np


def load_npz(filepath: str) -> SignalDataWithGroundTruth:
    data = np.load(filepath)
    mixed_data = data['mixed_data']
    ground_truth = data['ground_truth']
    artifacts = data['artifacts']
    sampling_rate = data['sampling_rate'].item()

    return SignalDataWithGroundTruth(
        raw_data=mixed_data,
        sampling_rate=sampling_rate,
        ground_truth=ground_truth,
        artifacts=artifacts
    )

def main():
    data_handler = DataHandler()

    data = data_handler.load_simulated_data('../data/simulated_data_2x64_30000.npz', sampling_rate=30000)
    # data = load_npz('../data/simulated_swec_data_20000_1000.npz')
    # data = load_npz('../data/added_artifacts_swec_data_30000.npz')
    print(f"Data shape: {data.raw_data.shape}, Sampling rate: {data.sampling_rate} Hz")

    analyzer = NeuralAnalyzer(sampling_rate=data.sampling_rate)
    
    # spikes = analyzer.extract_spikes(data.raw_data)
    # lfp = analyzer.extract_lfp(data.raw_data)

    # print(f"Extracted {len(spikes[0])} spikes from the raw mixed signal from channel 0")
    # print(f"LFP shape: {lfp.shape}")

    # analyzer.plot_spectral_analysis(data.raw_data, title="Raw Mixed Signal Spectral Analysis")
    plotter = NeuralPlotter(analyzer)
    plotter.plot_trace_comparison(
        data.ground_truth, 
        data.raw_data,
        0,0
    )
    # plotter.plot_psd(data.raw_data, title="PSD of Raw Mixed Signal - Channel 0")


if __name__ == "__main__":
    main()
