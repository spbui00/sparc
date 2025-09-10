'''
Generate synthetic artifacts using "Review 2024" paper equations.

'''
import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from tqdm import tqdm

def load_selected_clips(file_path):
    """Load the selected clips from the pkl file."""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def split_data(selected_data):
    # Count total number of clips for pre-allocation
    num_seizure_clips = sum(len(selected_data[patient_id]['seizure_clips']) for patient_id in selected_data)
    num_non_seizure_clips = sum(len(selected_data[patient_id]['non_seizure_clips']) for patient_id in selected_data)

    # Find the largest channel number
    max_channels = max(
        max(clip.shape[0] for patient_id in selected_data for clip in selected_data[patient_id]['seizure_clips']),
        max(clip.shape[0] for patient_id in selected_data for clip in selected_data[patient_id]['non_seizure_clips'])
    )

    timesteps = 4 * 512  # Flattening (4, 512) into one dimension

    # Pre-allocate arrays with max_channels
    seizure_data = np.zeros((num_seizure_clips, max_channels, timesteps))
    non_seizure_data = np.zeros((num_non_seizure_clips, max_channels, timesteps))

    # Fill in the arrays
    seizure_idx, non_seizure_idx = 0, 0
    for patient_id in selected_data:
        for clip in selected_data[patient_id]['seizure_clips']:
            channel_num = clip.shape[0]
            seizure_data[seizure_idx, :channel_num, :] = clip.reshape(channel_num, -1)  # Fill from 0 to channel_num
            seizure_idx += 1
        for clip in selected_data[patient_id]['non_seizure_clips']:
            channel_num = clip.shape[0]
            non_seizure_data[non_seizure_idx, :channel_num, :] = clip.reshape(channel_num, -1)  # Fill from 0 to channel_num
            non_seizure_idx += 1

    return seizure_data, non_seizure_data


def plot_one_clip(clip, sampling_rate, title="Neural Signal Clip", channel=0):
    """Plot a neural signal clip for a selected channel with correct time axis."""
    time = np.arange(clip.shape[1]) / sampling_rate
    plt.figure(figsize=(10, 4))
    plt.plot(time, clip[channel], label=f'Channel {channel+1}', color='b')
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.legend()
    plt.savefig(title + '.png')
    plt.show()



def plot_one_clip_allchannels(data_signal, data_artifact, clip_index=0, title="Clip with Artifact", 
                              sampling_rate=20000, stimulation_rate=1e3, timesteps_for_plot = 100):
    """Plot all channels of a single clip with signal and artifact for "timesteps_for_plot" stimulation cycles.

    Parameters:
        data_signal: np.ndarray of shape (num_clips, num_channels, num_timesteps)
            The signal data without artifacts.
        data_artifact: np.ndarray of shape (num_clips, num_channels, num_timesteps)
            The artifact data.
        clip_index: int, index of the clip to plot.
        title: str, title of the plot.
        sampling_rate: int, the sampling rate (Hz) for signal and artifact data.
        stimulation_rate: int, the stimulation rate (Hz), i.e., how often the stimulation occurs.
    """
    data_signal = data_signal / 1e3  # Convert to millivolts
    data_artifact = data_artifact / 1e3  # Convert to mill
    
    num_channels, num_timesteps = data_signal.shape[1], data_signal.shape[2]
    time = np.arange(num_timesteps) / sampling_rate  # Convert time to seconds
    stim_period = 1 / stimulation_rate  # Duration of one stimulation cycle in seconds
    
    # Calculate the number of timesteps 
    total_steps = int(timesteps_for_plot* stim_period * sampling_rate)

    # Plot the original signal and the signal + artifact for all channels (limited to 100 stimulation cycles)
    fig, axes = plt.subplots(num_channels, 1, figsize=(10, 2 * num_channels), sharex=True)
    fig.suptitle(title)

    for ch in range(num_channels):
        # Plot signal + artifact in red
        axes[ch].plot(time[:total_steps], data_signal[clip_index, ch, :total_steps] + data_artifact[clip_index, ch, :total_steps], 
                      label=f'Signal + Artifact {ch}', color='red')
        # Plot original signal in blue
        axes[ch].plot(time[:total_steps], data_signal[clip_index, ch, :total_steps], label=f'Signal {ch}', color='blue')
        
        axes[ch].legend(loc="upper right")
        axes[ch].set_ylabel("mV")

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(title + '_signal_signalplusartifact.png')

    # Generate distinct colors for each channel
    colors = plt.cm.get_cmap('tab20', num_channels)  # Use 'tab20' colormap for distinct colors
    
    # Plot the stacked artifact signals for the 100 cycles with different colors for each channel
    fig, ax = plt.subplots(figsize=(10, 5))
    for ch in range(num_channels):
        ax.plot(time[:total_steps], data_artifact[clip_index, ch, :total_steps], color=colors(ch), alpha=0.7, label=f'Channel {ch}')
    
    ax.set_title("Stacked Artifacts for All Channels")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("mV")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(title + '_stacked_artifacts.png')
    print(f"Saved plot for clip {clip_index} with title: {title}")




def generate_sine_exp_decay_artifact(input_data, sampling_rate_signal, sampling_rate_artifact, stim_rate, stimulation_channel, stim_current_strength, 
                                      strip_distance=1e-1, normal_strip_distance=1e-3, f_pulse=2500):
    """Generate sine wave + exponential decay artifacts for all clips, ensuring silent channels remain zero.

    Parameters:
        input_data: np.ndarray of shape (num_clips, num_channels, num_timesteps), the original input data
        sampling_rate_signal: int, sampling rate of the signal
        sampling_rate_artifact: int, sampling rate of the artifact
        stim_rate: float, stimulation frequency (Hz)
        stimulation_channel: int, index of the stimulation channel (0 to max_channels - 1)
        stim_current_strength: float, stimulation current strength       
        strip_distance: float, distance between different strips (tunable)
        normal_strip_distance: float, distance between channels within the same strip (default 1)
        f_pulse: float, frequency of the pulse (default 2500 Hz)

    Returns:
        artifact: np.ndarray of shape (num_clips, num_channels, num_timesteps)
    """
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
                clip_artifact[ch, indices_exp] += -np.exp(-3000 * time_artifact[indices_exp]) - np.exp(-5000 * time_artifact[indices_exp])
        
        # Scale by amplitude for each channel
        artifact[clip_idx] = clip_artifact * amplitudes[:, np.newaxis]

    return artifact




def resample_signal(signal, original_rate, target_rate):
    """Resample signal to a new sampling rate."""
    num_original_samples = signal.shape[-1]
    num_target_samples = int(num_original_samples * target_rate / original_rate)
    
    if target_rate < original_rate:
        # Downsample by simple decimation
        factor = original_rate // target_rate
        return signal[:, :, ::factor]
    
    elif target_rate > original_rate:
        # Create the original and target time indices
        original_time = np.linspace(0, 1, num_original_samples, endpoint=False)
        target_time = np.linspace(0, 1, num_target_samples, endpoint=False)
        # Interpolate along the last axis (timesteps)
        interpolated_signal = np.zeros((signal.shape[0], signal.shape[1], num_target_samples))       
        for i in range(signal.shape[0]):  # Iterate over clips
            for j in range(signal.shape[1]):  # Iterate over channels
                interpolated_signal[i, j, :] = np.interp(target_time, original_time, signal[i, j, :])
        return interpolated_signal
    else: # Same sampling rate
        return signal

def plot_frequency_spectrum(original_signal, artifact_signal, original_sampling_rate, downsampled_sampling_rate, channel_index=0):
    """Plot the frequency spectrum of the original and artifact signals before and after downsampling for a specific channel.

    Parameters:
        original_signal: np.ndarray, the original signal to be plotted.
        artifact_signal: np.ndarray, the artifact signal to be plotted.
        original_sampling_rate: int, the original sampling rate (Hz).
        downsampled_sampling_rate: int, the downsampled sampling rate (Hz).
        channel_index: int, index of the channel to plot (0 to 87).
    """
    # Select the specific clip and channel from both signals
    clip_num = 0  # Select the first clip
    original_signal_channel = original_signal[clip_num, channel_index, :]
    artifact_signal_channel = artifact_signal[clip_num, channel_index, :]

    # Compute the FFT of the original signal
    n_original = original_signal_channel.shape[-1]
    freqs_original = np.fft.fftfreq(n_original, 1 / original_sampling_rate)
    fft_original = np.fft.fft(original_signal_channel)

    # Compute the FFT of the artifact signal
    fft_artifact = np.fft.fft(artifact_signal_channel)

    # Compute the FFT of the downsampled original signal
    n_downsampled = n_original // 4  # Example downsample factor
    original_signal_downsampled = original_signal_channel[::4]  # Downsample the signal by a factor of 4
    freqs_downsampled = np.fft.fftfreq(n_downsampled, 1 / downsampled_sampling_rate)
    fft_original_downsampled = np.fft.fft(original_signal_downsampled)

    # Compute the FFT of the downsampled artifact signal
    artifact_signal_downsampled = artifact_signal_channel[::4]  # Downsample the artifact signal by a factor of 4
    fft_artifact_downsampled = np.fft.fft(artifact_signal_downsampled)

    # Plot the frequency spectra
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Plot original signal spectrum with log scale
    axes[0].plot(freqs_original[:n_original // 2], np.abs(fft_original[:n_original // 2]), label='Original Signal', color='blue')
    axes[0].plot(freqs_original[:n_original // 2], np.abs(fft_artifact[:n_original // 2]), label='Artifact Signal', color='red')
    axes[0].set_yscale('log')  # Set y-axis to log scale
    axes[0].set_title(f"Frequency Spectrum of Signals (Channel {channel_index})")
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Amplitude (log scale)")
    axes[0].legend()

    # Plot downsampled signal spectrum with log scale
    axes[1].plot(freqs_downsampled[:n_downsampled // 2], np.abs(fft_original_downsampled[:n_downsampled // 2]), label='Downsampled Original Signal', color='blue')
    axes[1].plot(freqs_downsampled[:n_downsampled // 2], np.abs(fft_artifact_downsampled[:n_downsampled // 2]), label='Downsampled Artifact Signal', color='red')
    axes[1].set_yscale('log')  # Set y-axis to log scale
    axes[1].set_title(f"Frequency Spectrum of Downsampled Signals (Channel {channel_index})")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Amplitude (log scale)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('frequency_spectrum.png')

    
if __name__ == "__main__":
    data_folder = '/net/inltitan1/scratch2/Xiaoyong/Artifact_cancellation/SWEC_ETHZ_data/'
    amplitude_folder = data_folder + 'low_amp_57/seizure/'
    file_path = data_folder + 'pat123_ALLclips_1s.pkl'
    sampling_rate_signal = 512  # Fixed sampling rate of the dataset
    sampling_rate_artifact = 2000  # Sampling rate of the artifact
    target_sampling_rate = 2000  # Tunable resampling rate
    selected_channel = 0  # Channel to plot
    stimulation_rate = 1000  # Stimulation frequency in Hz (ERASSR 333Hz, Dictionary learning 185Hz) compare above or below sample rate
    stimulation_channel = 0  # Stimulation channel
    stim_current_strength = 57 # Stimulation current strength
    
    # low_freq_scaler = 5000 # uV   
    low_freq_component = False  # Add low frequency component to the artifact
    
    selected_data = load_selected_clips(file_path)
    
    # split data into seizure and non-seizure clips
    seizure_data, non_seizure_data = split_data(selected_data) # Shape: (num_clips, num_channels, num_timesteps)
    
    # # Select one patient and one clip for plot
    # patient_id = list(selected_data.keys())[0]
    # clip = selected_data[patient_id]['seizure_clips'][0]  # Shape: (channel_number, 4, 512)
    # clip = clip.reshape(clip.shape[0], -1)  # Reshape to (channel_number, time_steps)    
    # plot_one_clip(clip, sampling_rate_signal, title=f"Original Signal of Patient {patient_id}", channel=selected_channel)
    
    ## Add synthetic artifact to the seizure data ###################################################################################################################################
    artifact_seizure = generate_sine_exp_decay_artifact(seizure_data, sampling_rate_signal, sampling_rate_artifact, stimulation_rate, stimulation_channel, stim_current_strength)
    seizure_data_interp = resample_signal(seizure_data, sampling_rate_signal, sampling_rate_artifact)
    # seizure_data_interp = np.load(data_folder+f'seizure_data_interp_rate{sampling_rate_artifact}.npy')
    mixed_seizure = seizure_data_interp + artifact_seizure
    # # # # save the data
    np.save(amplitude_folder + f'artifact_seizure_rate{sampling_rate_artifact}.npy', artifact_seizure)
    # np.save(data_folder+f'seizure_data_interp_rate{sampling_rate_artifact}.npy', seizure_data_interp)
    np.save(amplitude_folder+f'mixed_seizure_rate{sampling_rate_artifact}.npy', mixed_seizure)
    # load the data
    artifact_seizure = np.load(amplitude_folder+f'artifact_seizure_rate{sampling_rate_artifact}.npy')
   
    mixed_seizure = np.load(amplitude_folder+f'mixed_seizure_rate{sampling_rate_artifact}.npy')
    
    # add low freq component
    if low_freq_component:
        # Generate the low-frequency signal (1D) for the time axis
        low_freq = np.sin(2 * np.pi * 0.01 * np.arange(seizure_data_interp.shape[2]))  # Shape (4s * sampling_rate_artifact,)
        # Expand dimensions to match the shape of artifact_seizure (15, 88, 4s * sampling_rate_artifact)
        low_freq = low_freq[np.newaxis, np.newaxis, :]  # Shape (1, 1, 4s * sampling_rate_artifact)
        # Add low-frequency signal across all clips and channels
    #     artifact_seizure += low_freq_scaler * low_freq  # Broadcasting to shape (15, 88, 4s * sampling_rate_artifact)
    #     mixed_seizure += low_freq_scaler * low_freq  # Broadcasting to shape (15, 88, 4s * sampling_rate_artifact)
    
    # plot frequency spectrum
    plot_frequency_spectrum(seizure_data_interp, artifact_seizure, sampling_rate_artifact, target_sampling_rate, channel_index=selected_channel)
    
    # # plot one clip all channels after adding artifact
    plot_one_clip_allchannels(seizure_data_interp, artifact_seizure, clip_index=0, title=f"StimRate_{stimulation_rate}_SampleRate_{sampling_rate_artifact}_{100/stimulation_rate}s", 
                              sampling_rate=sampling_rate_artifact, stimulation_rate=1e3, timesteps_for_plot = 100)
    
    
    # resample the data
    artifact_seizure = resample_signal(artifact_seizure, sampling_rate_artifact, target_sampling_rate)
    seizure_data_interp = resample_signal(seizure_data_interp, sampling_rate_artifact, target_sampling_rate)
    mixed_seizure = resample_signal(mixed_seizure, sampling_rate_artifact, target_sampling_rate)
    
    # # plot one clip all channels after adding artifact
    plot_one_clip_allchannels(seizure_data_interp, artifact_seizure, clip_index=0, title=f"StimRate_{stimulation_rate}_SampleRate_{target_sampling_rate}_{1000/stimulation_rate}s", 
                              sampling_rate=target_sampling_rate, stimulation_rate=1e3, timesteps_for_plot = 1000)
    
    # Save to .mat file
    # scipy.io.savemat(amplitude_folder+f'swec-ethz-ieeg-seizure-data-rate{target_sampling_rate}.mat', {"mixed_seizure": mixed_seizure, "artifact_seizure": artifact_seizure, "signal_seizure": seizure_data_interp})
    
    
    ## Add synthetic artifact to the non-seizure data ###################################################################################################################################
    
    # artifact_nonseizure = generate_sine_exp_decay_artifact(non_seizure_data, sampling_rate_signal, sampling_rate_artifact, stimulation_rate, stimulation_channel, stim_current_strength)
    # # nonseizure_data_interp = resample_signal(non_seizure_data, sampling_rate_signal, sampling_rate_artifact)
    # nonseizure_data_interp = np.load(data_folder+f'nonseizure_data_interp_rate{sampling_rate_artifact}.npy')
    # mixed_nonseizure = nonseizure_data_interp + artifact_nonseizure
    # # # save the data
    # np.save(amplitude_folder+f'artifact_nonseizure_rate{sampling_rate_artifact}.npy', artifact_nonseizure)
    # # np.save(data_folder+f'nonseizure_data_interp_rate{sampling_rate_artifact}.npy', nonseizure_data_interp)
    # np.save(amplitude_folder+f'mixed_nonseizure_rate{sampling_rate_artifact}.npy', mixed_nonseizure)
    # # load the data
    # artifact_nonseizure = np.load(amplitude_folder+f'artifact_nonseizure_rate{sampling_rate_artifact}.npy')
    # mixed_nonseizure = np.load(amplitude_folder+f'mixed_nonseizure_rate{sampling_rate_artifact}.npy') 
    
    # # add low freq component
    # if low_freq_component:
    #     # Generate the low-frequency signal (1D) for the time axis
    #     low_freq = np.sin(2 * np.pi * 0.01 * np.arange(nonseizure_data_interp.shape[2]))  # Shape (4s * sampling_rate_artifact,)
    #     # Expand dimensions to match the shape of artifact_seizure (15, 88, 4s * sampling_rate_artifact)
    #     low_freq = low_freq[np.newaxis, np.newaxis, :]  # Shape (1, 1, 4s * sampling_rate_artifact)
    #     # Add low-frequency signal across all clips and channels
    #     artifact_nonseizure += low_freq_scaler * low_freq  # Broadcasting to shape (15, 88, 4s * sampling_rate_artifact)
    #     mixed_nonseizure += low_freq_scaler * low_freq  # Broadcasting to shape (15, 88, 4s * sampling_rate_artifact)
    
    # # plot frequency spectrum
    # plot_frequency_spectrum(nonseizure_data_interp, artifact_nonseizure, sampling_rate_artifact, target_sampling_rate, channel_index=selected_channel)
    
    # # # plot one clip all channels after adding artifact
    # plot_one_clip_allchannels(nonseizure_data_interp, artifact_nonseizure, clip_index=0, title=f"StimRate_{stimulation_rate}_SampleRate_{sampling_rate_artifact}_{100/stimulation_rate}s", 
    #                           sampling_rate=sampling_rate_artifact, stimulation_rate=1e3, timesteps_for_plot = 100)
      
    # # resample the data
    # artifact_seizure = resample_signal(artifact_nonseizure, sampling_rate_artifact, target_sampling_rate)
    # seizure_data_interp = resample_signal(nonseizure_data_interp, sampling_rate_artifact, target_sampling_rate)
    # mixed_seizure = resample_signal(mixed_nonseizure, sampling_rate_artifact, target_sampling_rate)
    
    # # # plot one clip all channels after adding artifact
    # plot_one_clip_allchannels(nonseizure_data_interp, artifact_nonseizure, clip_index=0, title=f"StimRate_{stimulation_rate}_SampleRate_{target_sampling_rate}_{1000/stimulation_rate}s", 
    #                           sampling_rate=target_sampling_rate, stimulation_rate=1e3, timesteps_for_plot = 1000)
    
    # # Save to .mat file
    # scipy.io.savemat(amplitude_folder+f'swec-ethz-ieeg-nonseizure-data-rate{target_sampling_rate}.mat', {"mixed_nonseizure": mixed_nonseizure, "artifact_nonseizure": artifact_nonseizure, "signal_nonseizure": nonseizure_data_interp})
    
    
