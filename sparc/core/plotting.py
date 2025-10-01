import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from .neural_analyzer import NeuralAnalyzer 


class NeuralPlotter:
    def __init__(self, analyzer: NeuralAnalyzer):
        self.analyzer = analyzer
        self.sampling_rate = analyzer.sampling_rate
        
        self.color_theme = {
            'signal': '#1f77b4',  # Muted blue
            'artifact': '#d62728',  # Brick red
            'spike': '#2ca02c',   # Cooked asparagus green
            'mean': '#ff7f0e',    # Safety orange
            'grid': '#cccccc'     # Light grey
        }

    def plot_trial_channels(self, data: np.ndarray, trial_idx: int, channels_idx: List[int], 
                           title: str = "Neural Signal (Trial Channel Comparison)", time_axis: bool = False):
        if data.ndim != 3:
            raise ValueError("Input data must be 3D (trials, channels, timesteps).")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Extract the 2D array for the specific trial
        trial_data = data[trial_idx]
        x_axis = np.arange(trial_data.shape[1]) / self.sampling_rate if time_axis else np.arange(trial_data.shape[1])
        
        for ch_idx in channels_idx:
            channel_data = trial_data[ch_idx, :]
            ax.plot(x_axis, channel_data, label=f'Channel {ch_idx}')
        
        ax.set_title(title)
        ax.set_xlabel("Time (s)" if time_axis else "Samples")
        ax.set_ylabel("Amplitude (µV)")
        ax.grid(True, color=self.color_theme['grid'], linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend()
        
        plt.tight_layout()
        plt.show()

    def plot_trial_channels_separate(self, data: np.ndarray, trial_idx: int, channels_idx: List[int], 
                                    title: str = "Neural Signal (Trial Channel Comparison)", time_axis: bool = False):
        if data.ndim != 3:
            raise ValueError("Input data must be 3D (trials, channels, timesteps).")
        
        num_channels = len(channels_idx)
        fig, axs = plt.subplots(num_channels, 1, figsize=(12, 4 * num_channels), sharex=True)
        
        # Extract the 2D array for the specific trial
        trial_data = data[trial_idx]
        x_axis = np.arange(trial_data.shape[1]) / self.sampling_rate if time_axis else np.arange(trial_data.shape[1])
        
        for i, ch_idx in enumerate(channels_idx):
            ax = axs[i] if num_channels > 1 else axs
            channel_data = trial_data[ch_idx, :]
            ax.plot(x_axis, channel_data, label=f'Channel {ch_idx}', color=self.color_theme['signal'])
            ax.set_title(f'Channel {ch_idx}')
            ax.set_ylabel("Amplitude (µV)")
            ax.grid(True, color=self.color_theme['grid'], linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.legend()
        
        axs[-1].set_xlabel("Time (s)" if time_axis else "Samples")
        plt.suptitle(title)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def plot_channels(self, data: np.ndarray, channels_idx: List[int], 
                     title: str = "Neural Signal (Channel Comparison)", time_axis: bool = False):
        if data.ndim != 3:
            raise ValueError("Input data must be 3D (trials, channels, timesteps).")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x_axis = np.arange(data.shape[2]) / self.sampling_rate if time_axis else np.arange(data.shape[2])
        
        for ch_idx in channels_idx:
            channel_data = data[:, ch_idx, :]
            mean_trace = np.mean(channel_data, axis=0)
            ax.plot(x_axis, mean_trace, label=f'Channel {ch_idx}')
        
        ax.set_title(title)
        ax.set_xlabel("Time (s)" if time_axis else "Samples")
        ax.set_ylabel("Amplitude (µV)")
        ax.grid(True, color=self.color_theme['grid'], linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend()
        
        plt.tight_layout()
        plt.show()

    def plot_channels_separate(self, data: np.ndarray, channels_idx: List[int], 
                              title: str = "Neural Signal (Channel Comparison)", time_axis: bool = False):
        if data.ndim != 3:
            raise ValueError("Input data must be 3D (trials, channels, timesteps).")
        
        num_channels = len(channels_idx)
        fig, axs = plt.subplots(num_channels, 1, figsize=(12, 4 * num_channels), sharex=True)
        
        x_axis = np.arange(data.shape[2]) / self.sampling_rate if time_axis else np.arange(data.shape[2])
        
        for i, ch_idx in enumerate(channels_idx):
            ax = axs[i] if num_channels > 1 else axs
            channel_data = data[:, ch_idx, :]
            mean_trace = np.mean(channel_data, axis=0)
            ax.plot(x_axis, mean_trace, label=f'Channel {ch_idx}', color=self.color_theme['signal'])
            ax.set_title(f'Channel {ch_idx}')
            ax.set_ylabel("Amplitude (µV)")
            ax.grid(True, color=self.color_theme['grid'], linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.legend()
        
        axs[-1].set_xlabel("Time (s)" if time_axis else "Samples")
        plt.suptitle(title)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()


    def plot_trial_channel(self, data: np.ndarray, trial_idx: int = 0, channel_idx: int = 0, 
                          title: Optional[str] = None, time_axis: bool = False):
        if data.ndim != 3:
            raise ValueError("Input data must be 3D (trials, channels, timesteps).")

        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Extract the 1D time series for the specific trial and channel
        trace = data[trial_idx, channel_idx, :]
        x_axis = np.arange(trace.shape[0]) / self.sampling_rate if time_axis else np.arange(trace.shape[0])
        
        ax.plot(x_axis, trace, color=self.color_theme['signal'], linewidth=1)
        
        ax.set_title(title or f"Time Series - Trial {trial_idx}, Channel {channel_idx}")
        ax.set_xlabel("Time (s)" if time_axis else "Samples")
        ax.set_ylabel("Amplitude (µV)")
        ax.grid(True, color=self.color_theme['grid'], linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.show()

    def plot_psd(self, data: np.ndarray, title: str = "Power Spectral Density (Trial Averaged)"):
        freqs, avg_psd = self.analyzer.compute_psd(data)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.semilogy(freqs, avg_psd, color=self.color_theme['signal'], alpha=0.6)
        
        ax.set_title(title)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power / Frequency (dB/Hz)")
        ax.grid(True, which="both", ls="--", color=self.color_theme['grid'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.show()

    def plot_fft(self, data: np.ndarray, trial_idx: int, channel_idx: int, title: Optional[str] = None):
        freqs, yf_mag = self.analyzer.compute_fft(data, trial_idx, channel_idx)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(freqs, yf_mag, color=self.color_theme['signal'])
        
        ax.set_title(title or f"FFT - Trial {trial_idx}, Channel {channel_idx}")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude")
        ax.grid(True, which="both", ls="--", color=self.color_theme['grid'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.show()

    def plot_spike_raster(self, spikes: List[List[List[Dict]]], channel_idx: int = 0, title: Optional[str] = None):
        fig, ax = plt.subplots(figsize=(10, 5))
        
        spike_times_per_trial = []
        for trial_idx, trial_spikes in enumerate(spikes):
            channel_spikes = trial_spikes[channel_idx]
            spike_indices = [s['index'] for s in channel_spikes]
            spike_times = np.array(spike_indices) / self.sampling_rate
            spike_times_per_trial.append(spike_times)
            
        ax.eventplot(spike_times_per_trial, color=self.color_theme['spike'], linelengths=0.8)
        
        ax.set_title(title or f"Spike Raster - Channel {channel_idx}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Trial Number")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.show()

    def plot_psth(self, spikes: List[List[List[Dict]]], channel_idx: int = 0, bin_width_ms: float = 10.0, title: Optional[str] = None):
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Concatenate all spike times from the specified channel across all trials
        all_spike_indices = [
            s['index'] for trial_spikes in spikes 
            for s in trial_spikes[channel_idx]
        ]
        all_spike_times_s = np.array(all_spike_indices) / self.sampling_rate
        
        num_trials = len(spikes)
        if num_trials == 0 or all_spike_times_s.size == 0:
            ax.text(0.5, 0.5, "No spikes found.", ha='center', va='center')
            plt.show()
            return
            
        # Determine the duration of a trial from the data (assuming consistent length)
        # This part might need the original data, but we can estimate if needed.
        # For now, we'll use the latest spike time as the max duration.
        duration_s = np.max(all_spike_times_s) if all_spike_times_s.size > 0 else 1.0

        bin_width_s = bin_width_ms / 1000
        bins = np.arange(0, duration_s + bin_width_s, bin_width_s)
        
        counts, _ = np.histogram(all_spike_times_s, bins=bins)
        
        # Convert counts to firing rate (spikes/sec)
        firing_rate = counts / (num_trials * bin_width_s)
        
        ax.bar(bins[:-1], firing_rate, width=bin_width_s, align='edge', color=self.color_theme['signal'])
        
        ax.set_title(title or f"PSTH - Channel {channel_idx} ({bin_width_ms}ms bins)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Firing Rate (spikes/s)")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.show()

    def plot_trace_comparison(self, cleaned: np.ndarray, mixed_data: np.ndarray, trial_idx: int, channel_idx: int, 
                             title: Optional[str] = None, time_axis: bool = False):
        if cleaned.ndim != 3 or mixed_data.ndim != 3:
            raise ValueError("Input data must be 3D (trials, channels, timesteps).")
        if cleaned.shape != mixed_data.shape:
            raise ValueError("Shapes of ground_truth and mixed_data must be the same.")

        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Extract traces
        gt_trace = cleaned[trial_idx, channel_idx, :]
        mixed_trace = mixed_data[trial_idx, channel_idx, :]
        
        x_axis = np.arange(gt_trace.shape[0]) / self.sampling_rate if time_axis else np.arange(gt_trace.shape[0])
        
        ax.plot(x_axis, mixed_trace, color=self.color_theme['signal'], linewidth=1, label='Mixed Data')
        ax.plot(x_axis, gt_trace, color=self.color_theme['spike'], linewidth=1, label='Comparison')
        
        ax.set_title(title or f"Comparison - Trial {trial_idx}, Channel {channel_idx}")
        ax.set_xlabel("Time (s)" if time_axis else "Samples")
        ax.set_ylabel("Amplitude (µV)")
        ax.grid(True, color=self.color_theme['grid'], linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend()
        
        plt.tight_layout()
        plt.show()

    def plot_cleaned_comparison(self, ground_truth: np.ndarray, mixed_data: np.ndarray, cleaned_data: np.ndarray, 
                               trial_idx: int, channel_idx: int, title: Optional[str] = None, time_axis: bool = False):
        if ground_truth.ndim != 3 or mixed_data.ndim != 3 or cleaned_data.ndim != 3:
            raise ValueError("Input data must be 3D (trials, channels, timesteps).")
        if ground_truth.shape != mixed_data.shape or ground_truth.shape != cleaned_data.shape:
            raise ValueError("Shapes of ground_truth, mixed_data, and cleaned_data must be the same.")

        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Extract traces
        gt_trace = ground_truth[trial_idx, channel_idx, :]
        mixed_trace = mixed_data[trial_idx, channel_idx, :]
        cleaned_trace = cleaned_data[trial_idx, channel_idx, :]
        
        x_axis = np.arange(gt_trace.shape[0]) / self.sampling_rate if time_axis else np.arange(gt_trace.shape[0])
        
        ax.plot(x_axis, mixed_trace, color=self.color_theme['grid'], linewidth=1, label='Mixed Data')
        ax.plot(x_axis, cleaned_trace, color=self.color_theme['mean'], linewidth=1, label='Cleaned Data')
        ax.plot(x_axis, gt_trace, color=self.color_theme['spike'], linewidth=1, label='Ground Truth')
        
        ax.set_title(title or f"Comparison - Trial {trial_idx}, Channel {channel_idx}")
        ax.set_xlabel("Time (s)" if time_axis else "Samples")
        ax.set_ylabel("Amplitude (µV)")
        ax.grid(True, color=self.color_theme['grid'], linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend()
        
        plt.tight_layout()
        plt.show()

    def plot_channel_average_comparison(self, ground_truth: np.ndarray, mixed_data: np.ndarray, channel_idx: int, 
                                       title: Optional[str] = None, time_axis: bool = False):
        if ground_truth.ndim != 3 or mixed_data.ndim != 3:
            raise ValueError("Input data must be 3D (trials, channels, timesteps).")
        if ground_truth.shape != mixed_data.shape:
            raise ValueError("Shapes of ground_truth and mixed_data must be the same.")

        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Calculate trial averages
        gt_avg = np.mean(ground_truth[:, channel_idx, :], axis=0)
        mixed_avg = np.mean(mixed_data[:, channel_idx, :], axis=0)
        
        x_axis = np.arange(gt_avg.shape[0]) / self.sampling_rate if time_axis else np.arange(gt_avg.shape[0])
        
        ax.plot(x_axis, mixed_avg, color=self.color_theme['signal'], linewidth=1.5, label='Mixed Data (Average)')
        ax.plot(x_axis, gt_avg, color=self.color_theme['spike'], linewidth=1.5, label='Ground Truth (Average)')
        
        ax.set_title(title or f"Channel Average Comparison - Channel {channel_idx}")
        ax.set_xlabel("Time (s)" if time_axis else "Samples")
        ax.set_ylabel("Amplitude (µV)")
        ax.grid(True, color=self.color_theme['grid'], linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend()
        
        plt.tight_layout()
        plt.show()

    def plot_trial_average_comparison(self, ground_truth: np.ndarray, mixed_data: np.ndarray, trial_idx: int, 
                                     title: Optional[str] = None, time_axis: bool = False):
        if ground_truth.ndim != 3 or mixed_data.ndim != 3:
            raise ValueError("Input data must be 3D (trials, channels, timesteps).")
        if ground_truth.shape != mixed_data.shape:
            raise ValueError("Shapes of ground_truth and mixed_data must be the same.")

        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Calculate channel averages for the trial
        gt_trial_avg = np.mean(ground_truth[trial_idx, :, :], axis=0)
        mixed_trial_avg = np.mean(mixed_data[trial_idx, :, :], axis=0)
        
        x_axis = np.arange(gt_trial_avg.shape[0]) / self.sampling_rate if time_axis else np.arange(gt_trial_avg.shape[0])
        
        ax.plot(x_axis, mixed_trial_avg, color=self.color_theme['signal'], linewidth=1.5, label='Mixed Data (Channel Average)')
        ax.plot(x_axis, gt_trial_avg, color=self.color_theme['spike'], linewidth=1.5, label='Ground Truth (Channel Average)')
        
        ax.set_title(title or f"Trial Average Comparison - Trial {trial_idx}")
        ax.set_xlabel("Time (s)" if time_axis else "Samples")
        ax.set_ylabel("Amplitude (µV)")
        ax.grid(True, color=self.color_theme['grid'], linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend()
        
        plt.tight_layout()
        plt.show()

    def plot_all_channels_trial(self, data: np.ndarray, trial_idx: int, time_axis: bool = False):
        trial_data = data[trial_idx, :, :]
    
        x_axis = np.arange(trial_data.shape[1]) / self.sampling_rate if time_axis else np.arange(trial_data.shape[1])
        
        plt.figure(figsize=(15, 8))
        
        for ch in range(trial_data.shape[0]):
            offset = ch * 20
            plt.plot(x_axis, 
                    trial_data[ch, :] + offset, 
                    alpha=0.7, linewidth=0.5)
        
        plt.title(f'All Channels - Trial {trial_idx}', fontsize=16)
        plt.xlabel('Time (s)' if time_axis else 'Samples')
        plt.ylabel('Channel (with vertical offset)')
        plt.grid(True, alpha=0.3)
        
        channel_labels = [f'Ch {i}' for i in range(0, trial_data.shape[0], 10)]
        channel_positions = [i * 20 for i in range(0, trial_data.shape[0], 10)]
        plt.yticks(channel_positions, channel_labels)
        
        plt.tight_layout()
        plt.show()

    def plot_artifacts_with_markers(self, artifacts: np.ndarray, artifact_indices: list, trial_idx: int = 0, 
                                   channel_idx: int = 0, title: Optional[str] = None, time_axis: bool = False):
        if artifacts.ndim != 3:
            raise ValueError("Input artifacts must be 3D (trials, channels, timesteps).")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Extract the trace
        trace = artifacts[trial_idx, channel_idx, :]
        x_axis = np.arange(trace.shape[0]) / self.sampling_rate if time_axis else np.arange(trace.shape[0])
        
        # Plot the artifact trace
        ax.plot(x_axis, trace, color=self.color_theme['artifact'], linewidth=1, label='Artifacts')
        
        # Mark artifact indices as vertical lines
        for idx in artifact_indices[trial_idx][channel_idx]:
            if 0 <= idx < len(trace):
                marker_x = idx / self.sampling_rate if time_axis else idx
                ax.axvline(x=marker_x, color='red', linestyle='--', alpha=0.7, linewidth=1)
        
        ax.set_title(title or f"Artifacts with Markers - Trial {trial_idx}, Channel {channel_idx}")
        ax.set_xlabel("Time (s)" if time_axis else "Samples")
        ax.set_ylabel("Amplitude (µV)")
        ax.grid(True, color=self.color_theme['grid'], linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend()
        
        plt.tight_layout()
        plt.show()

    def plot_mixed_data_with_markers(self, mixed_data: np.ndarray, artifact_indices: list, trial_idx: int = 0, 
                                    channel_idx: int = 0, title: Optional[str] = None, time_axis: bool = False):
        if mixed_data.ndim != 3:
            raise ValueError("Input mixed_data must be 3D (trials, channels, timesteps).")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Extract the trace
        trace = mixed_data[trial_idx, channel_idx, :]
        x_axis = np.arange(trace.shape[0]) / self.sampling_rate if time_axis else np.arange(trace.shape[0])
        
        # Plot the mixed data trace
        ax.plot(x_axis, trace, color=self.color_theme['signal'], linewidth=1, label='Mixed Data')
        
        # Mark artifact indices as vertical lines
        for idx in artifact_indices[trial_idx][channel_idx]:
            if 0 <= idx < len(trace):
                marker_x = idx / self.sampling_rate if time_axis else idx
                ax.axvline(x=marker_x, color='red', linestyle='--', alpha=0.7, linewidth=1)
        
        ax.set_title(title or f"Mixed Data with Artifact Markers - Trial {trial_idx}, Channel {channel_idx}")
        ax.set_xlabel("Time (s)" if time_axis else "Samples")
        ax.set_ylabel("Amplitude (µV)")
        ax.grid(True, color=self.color_theme['grid'], linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend()
        
        plt.tight_layout()
        plt.show()

    def plot_artifacts_and_mixed_comparison(self, artifacts: np.ndarray, mixed_data: np.ndarray, 
                                           artifact_indices: list, trial_idx: int = 0, channel_idx: int = 0, 
                                           title: Optional[str] = None, time_axis: bool = False):
        if artifacts.ndim != 3 or mixed_data.ndim != 3:
            raise ValueError("Input data must be 3D (trials, channels, timesteps).")
        if artifacts.shape != mixed_data.shape:
            raise ValueError("Shapes of artifacts and mixed_data must be the same.")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Extract traces
        artifact_trace = artifacts[trial_idx, channel_idx, :]
        mixed_trace = mixed_data[trial_idx, channel_idx, :]
        x_axis = np.arange(artifact_trace.shape[0]) / self.sampling_rate if time_axis else np.arange(artifact_trace.shape[0])
        
        # Plot artifacts
        ax1.plot(x_axis, artifact_trace, color=self.color_theme['artifact'], linewidth=1, label='Artifacts')
        ax1.set_title(f"Artifacts - Trial {trial_idx}, Channel {channel_idx}")
        ax1.set_ylabel("Amplitude (µV)")
        ax1.grid(True, color=self.color_theme['grid'], linestyle='--')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.legend()
        
        # Plot mixed data
        ax2.plot(x_axis, mixed_trace, color=self.color_theme['signal'], linewidth=1, label='Mixed Data')
        ax2.set_title(f"Mixed Data - Trial {trial_idx}, Channel {channel_idx}")
        ax2.set_xlabel("Time (s)" if time_axis else "Samples")
        ax2.set_ylabel("Amplitude (µV)")
        ax2.grid(True, color=self.color_theme['grid'], linestyle='--')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.legend()
        
        # Mark artifact indices as vertical lines on both plots
        for idx in artifact_indices[trial_idx][channel_idx]:
            if 0 <= idx < len(artifact_trace):
                marker_x = idx / self.sampling_rate if time_axis else idx
                ax1.axvline(x=marker_x, color='red', linestyle='--', alpha=0.7, linewidth=1)
                ax2.axvline(x=marker_x, color='red', linestyle='--', alpha=0.7, linewidth=1)
        
        if title:
            fig.suptitle(title, fontsize=14)
        
        plt.tight_layout()
        plt.show()
