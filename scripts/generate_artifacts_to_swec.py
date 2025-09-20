import numpy as np
from sparc import DataHandler, SignalData, NeuralPlotter, NeuralAnalyzer, SignalDataWithGroundTruth
import numpy as np
import os
from tqdm import tqdm
from scipy import signal
from typing import cast


SWEC_SAMPLING_RATE = 512  # Original sampling rate of the SWEC dataset

def split_data(selected_data, patient_id:str='01'):
    seizure = np.array(selected_data[patient_id]['seizure_clips'])
    non_seizure = np.array(selected_data[patient_id]['non_seizure_clips'])
    trials, channels, segments, samples = seizure.shape

    seizure_data = seizure.reshape(trials, channels, segments * samples)
    non_seizure_data = non_seizure.reshape(trials, channels, segments * samples)

    seizure_signal = SignalData(
        raw_data=seizure_data,
        sampling_rate=SWEC_SAMPLING_RATE
    )

    non_seizure_signal = SignalData(
        raw_data=non_seizure_data,
        sampling_rate=SWEC_SAMPLING_RATE
    )

    return seizure_signal, non_seizure_signal

def resample_signal(data, original_rate, target_rate) -> np.ndarray:
    data_float64 = data.astype(np.float64)
    
    num_original_samples = data_float64.shape[-1]
    num_target_samples = int(num_original_samples * target_rate / original_rate)
    
    resampled_data = signal.resample(data_float64, num_target_samples, axis=-1)
    
    return cast(np.ndarray, resampled_data)

def generate_sine_exp_decay_artifact(input_data, sampling_rate_signal, sampling_rate_artifact, stim_rate, stimulation_channel, stim_current_strength, 
                                     strip_distance=1e-1, normal_strip_distance=1e-3, f_pulse=1000):
    num_clips, num_channels, num_timesteps = input_data.shape
    dt = 1 / sampling_rate_signal
    Total_time = num_timesteps * dt
    time_artifact = np.arange(0, Total_time, 1 / sampling_rate_artifact)

    artifact = np.zeros((num_clips, num_channels, len(time_artifact)))
    
    # Identify silent channels (all-zero input across clips)
    silent_channels = np.all(input_data == 0, axis=(0, 2))  # Shape: (num_channels,)

    # Calculate distances for non-silent channels
    k1, k2 = -0.201, 0.102  # Constants from the formula
    strip_id = stimulation_channel // 8  # Identify which strip the stimulation channel belongs to
    channel_distances = np.zeros(num_channels)

    for ch in range(num_channels):
        if silent_channels[ch]:  # Skip silent channels
            continue
        ch_strip = ch // 8  # Identify strip
        if ch_strip == strip_id:
            channel_distances[ch] = abs(ch - stimulation_channel) * (normal_strip_distance * 1000)
        else:
            channel_distances[ch] = abs(ch_strip - strip_id) * (strip_distance * 1000)

    # Compute amplitude for each non-silent channel
    log_amplitudes = k1 * channel_distances + k2 * stim_current_strength - 1.92
    amplitudes = np.zeros(num_channels)  # Default to zero for silent channels
    amplitudes[~silent_channels] = 10 ** log_amplitudes[~silent_channels]  # Convert log10(Amplitude) to linear scale

    # Generate artifact for each clip
    stim_period = 1 / stim_rate
    for clip_idx in tqdm(range(num_clips), desc="Generating Artifacts"):
        clip_artifact = np.zeros((num_channels, len(time_artifact)))
        for start_time in np.arange(0, time_artifact[-1], stim_period):
            indices_sine = np.where((time_artifact >= start_time) & (time_artifact < start_time + 1 / f_pulse))
            rand_delay = np.random.uniform(3/8 * (1/f_pulse), 7/8 * (1/f_pulse)) 
            indices_exp = np.where(time_artifact >= start_time + rand_delay)
            
            for ch in range(num_channels):
                if silent_channels[ch]:  # Skip silent channels
                    continue
                clip_artifact[ch, indices_sine] = -np.sin(2 * np.pi * f_pulse * time_artifact[indices_sine]) 
                clip_artifact[ch, indices_exp] += -np.exp(-3000 * (time_artifact[indices_exp]-start_time)) - np.exp(-5000 * (time_artifact[indices_exp]-start_time))
        
        # Scale by amplitude for each channel
        artifact[clip_idx] = clip_artifact * amplitudes[:, np.newaxis]

    return artifact

def save_data(mixed, folder, filename):
    os.makedirs(folder, exist_ok=True)
    full_path = os.path.join(folder, filename)
    np.savez_compressed(
        full_path,
        mixed_data=mixed.raw_data,
        ground_truth=mixed.ground_truth,
        artifacts=mixed.artifacts,
        sampling_rate=mixed.sampling_rate
    )
    
    print(f"Data successfully saved to {filename}")

def load_data(filepath):
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

def _generate_and_save(artifact_sampling_rate=30000, f_pulse=2500, target_sampling_rate=2000):
    data_handler = DataHandler()
    data = data_handler.load_pickle_data('../research/datasets/SWEC-ETHZ/selected_clips_pat123.pkl')

    _, non_seizure = split_data(data)
    print(f"Non-seizure data shape: {non_seizure.raw_data.shape}")

    artifacts = generate_sine_exp_decay_artifact(
        input_data=non_seizure.raw_data,
        sampling_rate_signal=non_seizure.sampling_rate,
        sampling_rate_artifact=artifact_sampling_rate,
        stim_rate=200,
        stimulation_channel=0,
        stim_current_strength=50,
        f_pulse=f_pulse
    )
    data_interp = resample_signal(non_seizure.raw_data, non_seizure.sampling_rate, artifact_sampling_rate)
    mixed = data_interp + artifacts

    analyzer = NeuralAnalyzer(sampling_rate=artifact_sampling_rate)
    plotter = NeuralPlotter(analyzer)

    # plotter.plot_trial_channels(mixed, 0, [0], title=f'Mixed Signal with Artifacts at {artifact_sampling_rate} Hz')

    # resample to target sampling rate 
    final = SignalDataWithGroundTruth(
        raw_data=resample_signal(mixed, artifact_sampling_rate, target_sampling_rate),
        sampling_rate=target_sampling_rate,
        ground_truth=resample_signal(data_interp, artifact_sampling_rate, target_sampling_rate),
        artifacts=resample_signal(artifacts, artifact_sampling_rate, target_sampling_rate)
    )

    save_data(final, folder='../data/', filename=f'simulated_swec_data_{target_sampling_rate}_{f_pulse}.npz')

def main():
    artifact_sampling_rate = 20000
    f_pulse = 1000 # affect the amplitude of the artifact
    target_sampling_rate = 20000
    _generate_and_save(artifact_sampling_rate=artifact_sampling_rate, f_pulse=f_pulse, target_sampling_rate=target_sampling_rate)
    
    mixed = load_data(f'../data/simulated_swec_data_{target_sampling_rate}_{f_pulse}.npz')

    analyzer = NeuralAnalyzer(sampling_rate=target_sampling_rate)
    plotter = NeuralPlotter(analyzer)

    # print max val 
    trial_idx = 0
    channel_idx = 0
    print(max(mixed.raw_data[trial_idx,channel_idx,:]))
    print(max(mixed.ground_truth[trial_idx,channel_idx,:]))

    # plotter.plot_trial_channels(mixed.ground_truth, 0, [0])

    plotter.plot_trace_comparison(
        ground_truth=mixed.ground_truth,
        mixed_data=mixed.raw_data,
        trial_idx=trial_idx,
        channel_idx=channel_idx,
    )

    channel_idx = 1
    plotter.plot_trace_comparison(
        ground_truth=mixed.ground_truth,
        mixed_data=mixed.raw_data,
        trial_idx=trial_idx,
        channel_idx=channel_idx,
    )
    

if __name__ == "__main__":
    main()
