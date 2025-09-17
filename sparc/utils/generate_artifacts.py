import numpy as np
from tqdm import tqdm
from scipy import signal as sp_signal


def resample_signal(sig, original_rate, target_rate):
    num_original_samples = sig.shape[-1]
    num_target_samples = int(num_original_samples * target_rate / original_rate)
    resampled_sig = sp_signal.resample(sig, num_target_samples, axis=-1)
    return resampled_sig

def generate_sine_exp_decay_artifact(input_data, sampling_rate_signal, sampling_rate_artifact, stim_rate, stimulation_channel, stim_current_strength, 
                                      strip_distance=1e-1, normal_strip_distance=1e-3, f_pulse=2500):
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
            channel_distances[ch] = abs(ch - stimulation_channel) * normal_strip_distance
        else:
            channel_distances[ch] = abs(ch_strip - strip_id) * strip_distance

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


def generate_synthetic_artifacts(clean_data, sampling_rate, f_pulse=2500):
    sampling_rate_artifact = 6000  # Generate artifact at a higher sampling rate
    stimulation_channel = 0
    stim_current_strength = 57 # tune too small (40) -> amplitude similar to neural signal

    stim_rate = 200 

    # Transpose clean_data to (trials, channels, timesteps) for artifact generation function
    clean_data_transposed = clean_data.transpose(0, 2, 1)

    artifacts_high_res = generate_sine_exp_decay_artifact(
        input_data=clean_data_transposed,
        sampling_rate_signal=sampling_rate,
        sampling_rate_artifact=sampling_rate_artifact,
        stim_rate=stim_rate,
        stimulation_channel=stimulation_channel,
        stim_current_strength=stim_current_strength,
        f_pulse=f_pulse
    )

    # resample artisacts down to the signal's sampling rate
    artifacts_resampled = resample_signal(
        artifacts_high_res,
        original_rate=sampling_rate_artifact,
        target_rate=sampling_rate
    )

    # Transpose back to (trials, timesteps, channels)
    artifacts = artifacts_resampled.transpose(0, 2, 1)
    
    # Ensure the length matches clean_data due to resampling
    if artifacts.shape[1] != clean_data.shape[1]:
        # Pad or truncate
        diff = artifacts.shape[1] - clean_data.shape[1]
        if diff > 0:
            artifacts = artifacts[:, :-diff, :]
        else:
            artifacts = np.pad(artifacts, ((0,0), (0, -diff), (0,0)), 'constant')

    return artifacts
