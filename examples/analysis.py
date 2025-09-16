from sparc import DataHandler, NeuralAnalyzer

def main():
    data_handler = DataHandler()
    data = data_handler.load_simulated_data('research/generate_dataset/SimulatedData_2.mat')
    
    analyzer = NeuralAnalyzer(sampling_rate=data.sampling_rate)
    
    spikes = analyzer.extract_spikes(data.raw_data)
    lfp = analyzer.extract_lfp(data.raw_data)

    print(f"Extracted {len(spikes[0])} spikes from the raw mixed signal from channel 0")
    print(f"LFP shape: {lfp.shape}")

    analyzer.plot_spectral_analysis(data.raw_data, title="Raw Mixed Signal Spectral Analysis")


if __name__ == "__main__":
    main()
