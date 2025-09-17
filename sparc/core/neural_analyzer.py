import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, stats
import pywt
from typing import Dict, List, Tuple, Optional, Any
import warnings

class NeuralAnalyzer:
    def __init__(self, sampling_rate: float):
        self.sampling_rate = sampling_rate
        
    def compute_psd(self, data: np.ndarray, method: str = 'welch',
                   nperseg: int = 256, noverlap: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        if data.ndim != 3:
            raise ValueError("Input data for compute_psd must be 3D (trials, timesteps, channels).")
        all_psds = []
        for trial_idx in range(data.shape[0]):
            freqs, psd = signal.welch(data[trial_idx, :, :], fs=self.sampling_rate, nperseg=nperseg, axis=0)
            all_psds.append(psd)
        
        # Average the PSDs across trials
        avg_psd = np.mean(all_psds, axis=0)
        return freqs, avg_psd
    
    def compute_spectrogram(self, data: np.ndarray, trial_idx: int = 0, channel: int = 0, nperseg: int = 256) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if data.ndim != 3:
            raise ValueError("Input data must be 3D (trials, timesteps, channels).")
        
        # Select the specific trial to analyze
        trial_data = data[trial_idx, :, channel]
        freqs, times, Sxx = signal.spectrogram(trial_data, fs=self.sampling_rate, nperseg=nperseg)
        return freqs, times, Sxx
    
    def compute_wavelet_transform(self, data: np.ndarray, channel: int = 0,
                                 freqs: Optional[np.ndarray] = None, 
                                 method: str = 'morlet') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if freqs is None:
            freqs = np.logspace(np.log10(1), np.log10(100), 50)
            
        if method == 'morlet':
            wavelet = 'cmor1.5-1.0'
            sampling_period = 1.0 / self.sampling_rate
            scales = pywt.scale2frequency(wavelet, 1) / (freqs * sampling_period)
            cwt, freqs = pywt.cwt(data[:, channel], scales, wavelet, sampling_period=sampling_period)
            power = np.abs(cwt) ** 2
        else:
            raise NotImplementedError(f"Wavelet method {method} not implemented")
            
        times = np.arange(data.shape[0]) / self.sampling_rate
        return freqs, times, power
    
    def detect_artifacts(self, data: np.ndarray, method: str = 'threshold',
                        threshold: float = 5.0, min_duration: int = 10) -> np.ndarray:
        if method == 'threshold':
            artifact_mask = np.abs(data) > threshold
            
        elif method == 'statistical':
            z_scores = np.abs(stats.zscore(data, axis=0))
            artifact_mask = z_scores > threshold
            
        elif method == 'gradient':
            gradient = np.abs(np.diff(data, axis=0))
            gradient = np.vstack([gradient, gradient[-1:]])  # Pad to same length
            artifact_mask = gradient > threshold
            
        else:
            raise ValueError(f"Unknown artifact detection method: {method}")
            
        for ch in range(data.shape[1]):
            ch_mask = artifact_mask[:, ch]
            diff = np.diff(np.concatenate(([False], ch_mask, [False])).astype(int))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            
            for start, end in zip(starts, ends):
                if end - start < min_duration:
                    ch_mask[start:end] = False
                    
        return artifact_mask
    
    def calculate_snr(self, signal: np.ndarray, noise: np.ndarray) -> float:
        """
        Calculate Signal-to-Noise Ratio (SNR) in dB.
        """
        signal_power = np.mean(signal ** 2)
        noise_power = np.mean(noise ** 2)
        
        if noise_power == 0:
            return np.inf
            
        snr_db = 10 * np.log10(signal_power / noise_power)
        return snr_db
    
    
    def extract_neural_features(self, data: np.ndarray, 
                               features: List[str] = None) -> Dict[str, np.ndarray]:
        """
        Extract various neural signal features.
        
        Args:
            data: Neural data array (timesteps, channels)
            features: List of features to extract
            
        Returns:
            Dictionary of extracted features
        """
        if features is None:
            features = ['rms', 'variance', 'skewness', 'kurtosis', 'spectral_centroid']
            
        feature_dict = {}
        
        for feature in features:
            if feature == 'rms':
                feature_dict[feature] = np.sqrt(np.mean(data ** 2, axis=0))
                
            elif feature == 'variance':
                feature_dict[feature] = np.var(data, axis=0)
                
            elif feature == 'skewness':
                feature_dict[feature] = stats.skew(data, axis=0)
                
            elif feature == 'kurtosis':
                feature_dict[feature] = stats.kurtosis(data, axis=0)
                
            elif feature == 'spectral_centroid':
                freqs, psd = self.compute_psd(data)
                feature_dict[feature] = np.sum(freqs[:, np.newaxis] * psd, axis=0) / np.sum(psd, axis=0)
                
            else:
                warnings.warn(f"Unknown feature: {feature}")
                
        return feature_dict

    def calculate_mse(self, ground_truth: np.ndarray, reconstructed: np.ndarray) -> float:
        return np.mean((ground_truth - reconstructed) ** 2)
    
    def calculate_psd_mse(self, ground_truth: np.ndarray, reconstructed: np.ndarray, 
                         nperseg: int = 256) -> np.ndarray:
        """Calculate MSE of Power Spectral Density (PSD) between ground truth and reconstructed signals."""
        psd_mse = np.zeros(ground_truth.shape[1])
        
        for ch in range(ground_truth.shape[1]):
            freqs_gt, psd_gt = self.compute_psd(ground_truth[:, ch], nperseg=nperseg)
            freqs_rec, psd_rec = self.compute_psd(reconstructed[:, ch], nperseg=nperseg)
            psd_mse[ch] = np.mean((psd_gt - psd_rec) ** 2)
            
        return psd_mse
    
    def calculate_spectral_coherence(self, signal1: np.ndarray, signal2: np.ndarray) -> np.ndarray:
        coherence = np.zeros(signal1.shape[1])
        
        for ch in range(signal1.shape[1]):
            freqs, coh = signal.coherence(signal1[:, ch], signal2[:, ch], 
                                        fs=self.sampling_rate)
            coherence[ch] = np.mean(coh)
            
        return coherence

    def extract_lfp(self, data: np.ndarray, cutoff_freq: float = 200.0) -> np.ndarray:
        if data.ndim != 3:
            raise ValueError("Input data must be 3D (trials, timesteps, channels).")
        
        # Apply filter along the time axis (axis=1)
        sos = signal.butter(4, cutoff_freq, btype='low', fs=self.sampling_rate, output='sos')
        lfp_data = signal.sosfiltfilt(sos, data, axis=1)
        return lfp_data

    def extract_mua(self, data: np.ndarray, high_pass_freq: float = 250.0, low_pass_freq: float = 5000.0,
                    lfp_cutoff_freq: float = 200.0, target_fs: int = 1000) -> np.ndarray:
        if data.ndim != 3:
            raise ValueError("Input data must be 3D (trials, timesteps, channels).")
            
        # Band-pass filter along the time axis (axis=1)
        sos_bp = signal.butter(4, [high_pass_freq, low_pass_freq], btype='bandpass', fs=self.sampling_rate, output='sos')
        mua_data = signal.sosfiltfilt(sos_bp, data, axis=1)
        
        mua_data = np.abs(mua_data)
        
        # Low-pass filter along the time axis (axis=1)
        sos_lp = signal.butter(4, lfp_cutoff_freq, btype='low', fs=self.sampling_rate, output='sos')
        mua_data = signal.sosfiltfilt(sos_lp, mua_data, axis=1)
        return mua_data

    def extract_spikes(self, data: np.ndarray, threshold_factor: float = 4.0, pre_ms: float = 0.4, post_ms: float = 1.2) -> List[List[List[Dict[str, Any]]]]:
        if data.ndim != 3:
            raise ValueError("Input data for extract_spikes must be 3D (trials, timesteps, channels).")
            
        pre_samples = int(self.sampling_rate * pre_ms / 1000)
        post_samples = int(self.sampling_rate * post_ms / 1000)
        
        sos = signal.butter(4, [250, 5000], btype='bandpass', fs=self.sampling_rate, output='sos')
        
        all_trials_spikes = []
        for trial_idx in range(data.shape[0]):
            all_channels_spikes = []
            for ch_idx in range(data.shape[2]):
                channel_data = data[trial_idx, :, ch_idx]
                
                filtered_data = signal.sosfiltfilt(sos, channel_data)
                
                rms = np.sqrt(np.mean(filtered_data**2))
                detection_threshold = threshold_factor * rms
                min_distance = max(1, int(self.sampling_rate / 1000))
                
                spike_indices, _ = signal.find_peaks(np.abs(filtered_data), height=detection_threshold, distance=min_distance)
                
                channel_spikes = []
                for idx in spike_indices:
                    if idx - pre_samples >= 0 and idx + post_samples < len(channel_data):
                        waveform = channel_data[idx - pre_samples : idx + post_samples]
                        spike_info = {
                            'index': idx,
                            'amplitude': channel_data[idx],
                            'waveform': waveform
                        }
                        channel_spikes.append(spike_info)
                all_channels_spikes.append(channel_spikes)
            all_trials_spikes.append(all_channels_spikes)
            
        return all_trials_spikes
    
    def plot_spectral_analysis(self, data: np.ndarray, trial_idx: int = 0, channel_idx: int = 0,
                           title: str = "Spectral Analysis", save_path: Optional[str] = None):
        if data.ndim != 3:
            raise ValueError("Input data must be 3D (trials, timesteps, channels).")
        
        # Check if selected trial and channel are valid
        if trial_idx >= data.shape[0]:
            raise IndexError(f"trial_idx {trial_idx} is out of bounds for data with {data.shape[0]} trials.")
        if channel_idx >= data.shape[2]:
            raise IndexError(f"channel_idx {channel_idx} is out of bounds for data with {data.shape[2]} channels.")

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"{title} - Trial {trial_idx}, Channel {channel_idx}")
        
        trial_data_2d = data[trial_idx, :, :]
        channel_data_1d = trial_data_2d[:, channel_idx]

        # Time series
        time_axis = np.arange(trial_data_2d.shape[0]) / self.sampling_rate
        axes[0, 0].plot(time_axis, channel_data_1d)
        axes[0, 0].set_title('Time Series')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].grid(True)
        
        single_trial_3d = data[np.newaxis, trial_idx, :, :]
        freqs, psd = self.compute_psd(single_trial_3d)
        axes[0, 1].semilogy(freqs, psd[:, channel_idx])
        axes[0, 1].set_title('Power Spectral Density')
        axes[0, 1].set_xlabel('Frequency (Hz)')
        axes[0, 1].set_ylabel('Power')
        axes[0, 1].grid(True)
        
        # Spectrogram
        freqs_spec, times_spec, Sxx = self.compute_spectrogram(data, trial_idx=trial_idx, channel=channel_idx)
        im = axes[1, 0].pcolormesh(times_spec, freqs_spec, 10 * np.log10(Sxx), shading='gouraud')
        axes[1, 0].set_title('Spectrogram')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Frequency (Hz)')
        plt.colorbar(im, ax=axes[1, 0], label='Power (dB)')
        
        # ... (rest of the function is the same) ...
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
