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

    def plot_trial_channel(self, data: np.ndarray, trial_idx: int = 0, channel_idx: int = 0, title: Optional[str] = None):
        if data.ndim != 3:
            raise ValueError("Input data must be 3D (trials, timesteps, channels).")

        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Extract the 1D time series for the specific trial and channel
        trace = data[trial_idx, :, channel_idx]
        time_axis = np.arange(trace.shape[0]) / self.sampling_rate
        
        ax.plot(time_axis, trace, color=self.color_theme['signal'], linewidth=1)
        
        ax.set_title(title or f"Time Series - Trial {trial_idx}, Channel {channel_idx}")
        ax.set_xlabel("Time (s)")
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

# --- Example Usage ---
# if __name__ == "__main__":
#     # Assuming you have a 'data' object with shape (trials, timesteps, channels)
#     # and a sampling_rate
#
#     analyzer = NeuralAnalyzer(sampling_rate=data.sampling_rate)
#     plotter = NeuralPlotting(analyzer)
#
#     # 1. Plot a single trial
#     plotter.plot_trial_channel(data, trial_idx=0, channel_idx=5)
#
#     # 2. Plot the trial-averaged PSD
#     plotter.plot_psd(data)
#
#     # 3. Extract spikes and plot raster and PSTH
#     spikes = analyzer.extract_spikes(data)
#     plotter.plot_spike_raster(spikes, channel_idx=5)
#     plotter.plot_psth(spikes, channel_idx=5)
