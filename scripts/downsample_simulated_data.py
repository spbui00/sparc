from sparc.core.data_handler import DataHandler
from sparc import NeuralAnalyzer, NeuralPlotter
import numpy as np
from scipy.signal import resample

INPUT_FILE = '../research/generate_dataset/SimulatedData.mat'
OUTPUT_FILE = '../data/simulated_data_2x64_1000.npz'
DOWNSAMPLE_RATE = 1000

def rebin_spike_train(spike_train, original_rate, target_rate):
    downsample_factor = int(original_rate / target_rate)
    trimmed_length = (spike_train.shape[-1] // downsample_factor) * downsample_factor
    trimmed_train = spike_train[..., :trimmed_length]
    
    new_shape = trimmed_train.shape[:-1] + (-1, downsample_factor)
    return np.sum(trimmed_train.reshape(new_shape), axis=-1)

def downsample_artifact_markers(artifact_markers, original_rate, target_rate):
    downsample_ratio = original_rate / target_rate
    
    original_starts = artifact_markers.starts
    
    downsampled_starts = []
    for trial_markers in original_starts:
        trial_downsampled = []
        for channel_markers in trial_markers:
            downsampled_indices = [int(idx / downsample_ratio) for idx in channel_markers]
            trial_downsampled.append(downsampled_indices)
        downsampled_starts.append(trial_downsampled)
    
    return downsampled_starts

if __name__ == "__main__":
    handler = DataHandler()
    data_obj = handler.load_simulated_data(INPUT_FILE, 30000)

    original_rate = data_obj.sampling_rate

    analyzer = NeuralAnalyzer(sampling_rate=original_rate)
    plotter = NeuralPlotter(analyzer)
    # plotter.plot_all_channels_trial(data_obj.raw_data, 0)
    plotter.plot_fft(data_obj.raw_data, 0, 0)
    freqs, yf_mag = analyzer.compute_fft(data_obj.raw_data, 0, 0)
    plotter.plot_fft(data_obj.ground_truth, 0, 0)
    freqs, yf_mag = analyzer.compute_fft(data_obj.ground_truth, 0, 0)

    plotter.plot_trace_comparison(data_obj.ground_truth, data_obj.raw_data, 0, 0)   

    exit()
    data_to_save = {}

    continuous_attrs = ['raw_data', 'ground_truth', 'artifacts', 'lfp', 'firing_rate']
    discrete_attrs = ['spike_train', 'artifact_markers']

    print(f"Downsampling data from {original_rate} Hz to {DOWNSAMPLE_RATE} Hz...")

    num_samples_downsampled = int(data_obj.raw_data.shape[-1] * DOWNSAMPLE_RATE / original_rate)
    for attr in continuous_attrs:
        if hasattr(data_obj, attr) and getattr(data_obj, attr) is not None:
            print(f"Resampling {attr}...")
            data = getattr(data_obj, attr)
            resampled_data = resample(data, num_samples_downsampled, axis=-1)
            data_to_save[attr] = resampled_data

    for attr in discrete_attrs:
        if hasattr(data_obj, attr) and getattr(data_obj, attr) is not None:
            if attr == 'spike_train':
                print(f"Re-binning {attr}...")
                data_to_save[attr] = rebin_spike_train(getattr(data_obj, attr), original_rate, DOWNSAMPLE_RATE)
            elif attr == 'artifact_markers':
                print(f"Downsampling {attr}...")
                data_to_save[attr] = downsample_artifact_markers(getattr(data_obj, attr), original_rate, DOWNSAMPLE_RATE)


    data_to_save['stim_params'] = data_obj.stim_params
    data_to_save['snr'] = data_obj.snr

    data_to_save['sampling_rate'] = DOWNSAMPLE_RATE
    np.savez_compressed(OUTPUT_FILE, **data_to_save)

    plotter = NeuralPlotter(NeuralAnalyzer(sampling_rate=DOWNSAMPLE_RATE))
    # plotter.plot_all_channels_trial(data_to_save['raw_data'], 0)
    plotter.plot_trace_comparison(data_to_save['ground_truth'], data_to_save['raw_data'], 0, 0)
    # plotter.plot_trial_channels(data_to_save['raw_data'], 0, [0, 1, 2, 8, 24])

    print(f"\nSuccessfully processed and saved data to '{OUTPUT_FILE}'")
    print(f"New sampling rate: {DOWNSAMPLE_RATE} Hz")
