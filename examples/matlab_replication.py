import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

def load_swec_ethz_data(data_path='research/datasets/SWEC-ETHZ/selected_clips_pat123.pkl'):
    """Load SWEC-ETHZ data."""
    try:
        with open(data_path, 'rb') as f:
            selected_data = pickle.load(f)
        
        seizure_clips = []
        for patient_id in selected_data:
            for clip in selected_data[patient_id]['seizure_clips']:
                seizure_clips.append(clip.reshape(clip.shape[0], -1))
        
        if not seizure_clips:
            return None, None
            
        max_channels = max(clip.shape[0] for clip in seizure_clips)
        n_trials = len(seizure_clips)
        n_timesteps = seizure_clips[0].shape[1]
        
        clean_data = np.zeros((n_trials, n_timesteps, max_channels))
        
        for i, clip in enumerate(seizure_clips):
            n_channels = clip.shape[0]
            clean_data[i, :, :n_channels] = clip.T
        
        return clean_data, 512
        
    except FileNotFoundError:
        print(f"SWEC-ETHZ data file not found at {data_path}")
        return None, None

def generate_sine_exp_decay_artifact(input_data, sampling_rate_signal, sampling_rate_artifact, stim_rate, stimulation_channel, stim_current_strength,
                                      strip_distance=1e-1, normal_strip_distance=1e-3, f_pulse=2500):
    """Generate sine wave + exponential decay artifacts for all clips, ensuring silent channels remain zero."""
    num_clips, num_channels, num_timesteps = input_data.shape
    dt = 1 / sampling_rate_signal
    Total_time = num_timesteps * dt
    time_artifact = np.arange(0, Total_time, 1 / sampling_rate_artifact)

    artifact = np.zeros((num_clips, num_channels, len(time_artifact)))
    
    silent_channels = np.all(input_data == 0, axis=(0, 2))

    k1, k2 = -0.201, 0.102
    strip_id = stimulation_channel // 8
    channel_distances = np.zeros(num_channels)

    for ch in range(num_channels):
        if silent_channels[ch]:
            continue
        ch_strip = ch // 8
        if ch_strip == strip_id:
            channel_distances[ch] = abs(ch - stimulation_channel) * normal_strip_distance
        else:
            channel_distances[ch] = abs(ch_strip - strip_id) * strip_distance

    log_amplitudes = k1 * channel_distances + k2 * stim_current_strength - 1.92
    amplitudes = np.zeros(num_channels)
    amplitudes[~silent_channels] = 10 ** log_amplitudes[~silent_channels]

    stim_period = 1 / stim_rate
    for clip_idx in tqdm(range(num_clips), desc="Generating Artifacts"):
        clip_artifact = np.zeros((num_channels, len(time_artifact)))
        for start_time in np.arange(0, time_artifact[-1], stim_period):
            indices_sine = np.where((time_artifact >= start_time) & (time_artifact < start_time + 1 / f_pulse))
            rand_delay = np.random.uniform(3/8 * (1/f_pulse), 7/8 * (1/f_pulse)) 
            indices_exp = np.where(time_artifact >= start_time + rand_delay)
            
            for ch in range(num_channels):
                if silent_channels[ch]:
                    continue
                clip_artifact[ch, indices_sine] = -np.sin(2 * np.pi * f_pulse * time_artifact[indices_sine])
                clip_artifact[ch, indices_exp] += -np.exp(-3000 * time_artifact[indices_exp]) - np.exp(-5000 * time_artifact[indices_exp])
        
        artifact[clip_idx] = clip_artifact * amplitudes[:, np.newaxis]

    return artifact

def main():
    """Main function to replicate MATLAB script."""
    # --- Data Loading ---
    ground_truth, sampling_rate = load_swec_ethz_data()
    if ground_truth is None:
        return

    # --- Artifact Generation ---
    template_length_timesteps = 200
    stim_rate = sampling_rate / template_length_timesteps # 2.56 Hz
    stimulation_channel = 0
    stim_current_strength = 57

    artifacts_transposed = generate_sine_exp_decay_artifact(
        input_data=ground_truth.transpose(0, 2, 1),
        sampling_rate_signal=sampling_rate,
        sampling_rate_artifact=sampling_rate,
        stim_rate=stim_rate,
        stimulation_channel=stimulation_channel,
        stim_current_strength=stim_current_strength
    )
    artifacts = artifacts_transposed.transpose(0, 2, 1)
    
    mixed_data = ground_truth + artifacts

    print(f"Data loaded and artifacts generated. Shape: {mixed_data.shape}")

    # --- Template Subtraction ---
    trial_data = mixed_data[0]
    num_timesteps, num_channels = trial_data.shape
    
    num_of_templates_for_avg = 3
    n_cycle = num_timesteps // template_length_timesteps

    cleaned_data = trial_data.copy()

    for ch in range(num_channels):
        signal_ch = trial_data[:, ch]
        last_avg_template = np.zeros(template_length_timesteps)

        if n_cycle > num_of_templates_for_avg:
            for i in range(n_cycle - num_of_templates_for_avg):
                start_idx = i * template_length_timesteps
                
                templates = []
                for k in range(num_of_templates_for_avg):
                    template_start = start_idx + k * template_length_timesteps
                    template_end = template_start + template_length_timesteps
                    templates.append(signal_ch[template_start:template_end])
                
                avg_template = np.mean(np.array(templates), axis=0)
                last_avg_template = avg_template

                cleaned_data[start_idx:start_idx + template_length_timesteps, ch] -= avg_template

            for i in range(n_cycle - num_of_templates_for_avg, n_cycle):
                start_idx = i * template_length_timesteps
                cleaned_data[start_idx:start_idx + template_length_timesteps, ch] -= last_avg_template

    # --- Evaluation ---
    mse = np.mean((cleaned_data - ground_truth[0])**2)
    mse_orig = np.mean((trial_data - ground_truth[0])**2)
    
    print(f"\n--- Evaluation Results ---")
    print(f"Original MSE: {mse_orig:.2f}")
    print(f"Cleaned MSE: {mse:.2f}")
    print(f"MSE Improvement: {mse_orig - mse:.2f}")

    # --- Plotting ---
    plt.figure(figsize=(15, 5))
    plt.plot(ground_truth[0, :, 0], label='Ground Truth')
    plt.plot(mixed_data[0, :, 0], label='Mixed', alpha=0.7)
    plt.plot(cleaned_data[:, 0], label='Cleaned')
    plt.title('MATLAB Replication Results (Channel 0, Trial 0)')
    plt.xlabel('Timesteps')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()