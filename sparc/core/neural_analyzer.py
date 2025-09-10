import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, stats
from typing import Dict, List, Tuple, Optional, Any
import warnings

class NeuralAnalyzer:
    def __init__(self, sampling_rate: float):
        self.sampling_rate = sampling_rate
        
    def compute_psd(self, data: np.ndarray, method: str = 'welch',
                   nperseg: int = 256, noverlap: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        # Let scipy handle noverlap automatically to avoid conflicts when nperseg is adjusted
        if noverlap is None:
            noverlap = None  # Let scipy use its default (nperseg // 2)
        else:
            # If noverlap is explicitly provided, ensure it's valid
            # Check if signal length is shorter than nperseg
            signal_length = data.shape[0] if data.ndim == 1 else data.shape[0]
            if signal_length < nperseg:
                # When signal is shorter than nperseg, scipy will adjust nperseg
                # So we should let noverlap be handled automatically
                noverlap = None
            
        if method == 'welch':
            freqs, psd = signal.welch(data, fs=self.sampling_rate, nperseg=nperseg,
                                    noverlap=noverlap, axis=0)
        elif method == 'periodogram':
            freqs, psd = signal.periodogram(data, fs=self.sampling_rate, axis=0)
        elif method == 'multitaper':
            freqs, psd = signal.periodogram(data, fs=self.sampling_rate,
                                          window=('dpss', 4), axis=0)
        else:
            raise ValueError(f"Unknown PSD method: {method}")
            
        return freqs, psd
    
    def compute_spectrogram(self, data: np.ndarray, channel: int = 0,
                           nperseg: int = 256, noverlap: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Let scipy handle noverlap automatically to avoid conflicts when nperseg is adjusted
        if noverlap is None:
            noverlap = None  # Let scipy use its default (nperseg // 2)
        else:
            # If noverlap is explicitly provided, ensure it's valid
            signal_length = data.shape[0]
            if signal_length < nperseg:
                # When signal is shorter than nperseg, scipy will adjust nperseg
                # So we should let noverlap be handled automatically
                noverlap = None
            
        freqs, times, Sxx = signal.spectrogram(data[:, channel], fs=self.sampling_rate,
                                              nperseg=nperseg, noverlap=noverlap)
        return freqs, times, Sxx
    
    def compute_wavelet_transform(self, data: np.ndarray, channel: int = 0,
                                 freqs: Optional[np.ndarray] = None, 
                                 method: str = 'morlet') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if freqs is None:
            freqs = np.logspace(np.log10(1), np.log10(100), 50)
            
        if method == 'morlet':
            widths = self.sampling_rate * 7 / (2 * freqs * np.pi)
            cwt = signal.cwt(data[:, channel], signal.morlet2, widths)
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
    
    def calculate_snr_improvement(self, original: np.ndarray, cleaned: np.ndarray, 
                                 ground_truth: np.ndarray) -> float:
        # Calculate noise before cleaning
        noise_before = original - ground_truth
        snr_before = self.calculate_snr(ground_truth, noise_before)
        
        # Calculate noise after cleaning
        noise_after = cleaned - ground_truth
        snr_after = self.calculate_snr(ground_truth, noise_after)
        
        return snr_after - snr_before
    
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

    def extract_lfp(self, data: np.ndarray, cutoff_freq: float = 200.0, target_fs: int = 500) -> np.ndarray:
        """
        Extract Local Field Potential (LFP) by low-pass filtering and down-sampling.
        
        Args:
            data: Neural data array (timesteps, channels)
            cutoff_freq: Low-pass filter cutoff frequency (Hz)
            target_fs: Target sampling rate for down-sampling (Hz)
            
        Returns:
            LFP signal
        """
        # Low-pass filter
        sos = signal.butter(4, cutoff_freq, btype='low', fs=self.sampling_rate, output='sos')
        lfp_data = signal.sosfilt(sos, data, axis=0)
        
        # Down-sample
        downsample_factor = int(self.sampling_rate / target_fs)
        if downsample_factor > 1:
            lfp_data = signal.decimate(lfp_data, downsample_factor, axis=0)
            
        return lfp_data

    def extract_mua(self, data: np.ndarray, high_pass_freq: float = 250.0, low_pass_freq: float = 5000.0,
                    lfp_cutoff_freq: float = 200.0, target_fs: int = 1000) -> np.ndarray:
        """
        Extract Multi-Unit Activity (MUA).
        
        Args:
            data: Neural data array (timesteps, channels)
            high_pass_freq: High-pass filter cutoff frequency (Hz)
            low_pass_freq: Low-pass filter cutoff frequency (Hz)
            lfp_cutoff_freq: Low-pass filter cutoff for rectified signal (Hz)
            target_fs: Target sampling rate for down-sampling (Hz)
            
        Returns:
            MUA signal
        """
        # Band-pass filter
        sos_bp = signal.butter(4, [high_pass_freq, low_pass_freq], btype='bandpass', fs=self.sampling_rate, output='sos')
        mua_data = signal.sosfilt(sos_bp, data, axis=0)
        
        mua_data = np.abs(mua_data)
        
        # Low-pass filter
        sos_lp = signal.butter(4, lfp_cutoff_freq, btype='low', fs=self.sampling_rate, output='sos')
        mua_data = signal.sosfilt(sos_lp, mua_data, axis=0)
        
        # Down-sample
        downsample_factor = int(self.sampling_rate / target_fs)
        if downsample_factor > 1:
            mua_data = signal.decimate(mua_data, downsample_factor, axis=0)
            
        return mua_data

    def extract_spikes(self, data: np.ndarray, threshold_factor: float = 4.0, 
                       pre_ms: float = 0.4, post_ms: float = 1.2) -> List[List[Dict[str, Any]]]:
        """
        Extract spikes from the data.
        
        Args:
            data: Neural data array (timesteps, channels)
            threshold_factor: Factor to multiply RMS by for thresholding.
            pre_ms: Time before spike peak to include in waveform (ms).
            post_ms: Time after spike peak to include in waveform (ms).
            
        Returns:
            A list of lists of dictionaries, where each inner list corresponds to a channel
            and each dictionary contains information about a single spike ('index', 'amplitude', 'waveform').
        """
        pre_samples = int(self.sampling_rate * pre_ms / 1000)
        post_samples = int(self.sampling_rate * post_ms / 1000)
        
        all_spikes = []
        for ch in range(data.shape[1]):
            channel_data = data[:, ch]
            
            # Band-pass filter to emphasize spikes (adapt to sampling rate)
            nyquist = self.sampling_rate / 2
            low_freq = min(250, nyquist * 0.1)  # At least 10% of Nyquist
            high_freq = min(5000, nyquist * 0.9)  # At most 90% of Nyquist
            
            if high_freq <= low_freq:
                # If we can't create a proper bandpass, use highpass only
                if low_freq < nyquist * 0.9:
                    sos = signal.butter(4, low_freq, btype='highpass', fs=self.sampling_rate, output='sos')
                else:
                    # Very low sampling rate, just use the raw data
                    filtered_data = channel_data
                    sos = None
            else:
                sos = signal.butter(4, [low_freq, high_freq], btype='bandpass', fs=self.sampling_rate, output='sos')
            
            if sos is not None:
                filtered_data = signal.sosfilt(sos, channel_data)
            else:
                filtered_data = channel_data
            
            # Calculate threshold
            rms = np.sqrt(np.mean(filtered_data**2))
            detection_threshold = threshold_factor * rms
            
            # Find peaks exceeding threshold
            # Minimum distance between spikes (at least 1ms, but ensure >= 1 sample)
            min_distance = max(1, int(self.sampling_rate / 1000))  # 1ms minimum
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
            all_spikes.append(channel_spikes)
            
        return all_spikes
    
    def plot_spectral_analysis(self, data: np.ndarray, channel: int = 0,
                              title: str = "Spectral Analysis", 
                              save_path: Optional[str] = None):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"{title} - Channel {channel}")
        
        # Time series
        time_axis = np.arange(data.shape[0]) / self.sampling_rate
        axes[0, 0].plot(time_axis, data[:, channel])
        axes[0, 0].set_title('Time Series')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].grid(True)
        
        # PSD
        freqs, psd = self.compute_psd(data)
        axes[0, 1].semilogy(freqs, psd[:, channel])
        axes[0, 1].set_title('Power Spectral Density')
        axes[0, 1].set_xlabel('Frequency (Hz)')
        axes[0, 1].set_ylabel('Power')
        axes[0, 1].grid(True)
        
        # Spectrogram
        freqs_spec, times_spec, Sxx = self.compute_spectrogram(data, channel)
        im = axes[1, 0].pcolormesh(times_spec, freqs_spec, 10 * np.log10(Sxx), shading='gouraud')
        axes[1, 0].set_title('Spectrogram')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Frequency (Hz)')
        plt.colorbar(im, ax=axes[1, 0], label='Power (dB)')
        
        # Wavelet transform
        freqs_wav, times_wav, power_wav = self.compute_wavelet_transform(data, channel)
        im2 = axes[1, 1].pcolormesh(times_wav, freqs_wav, power_wav, shading='gouraud')
        axes[1, 1].set_title('Wavelet Transform')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Frequency (Hz)')
        axes[1, 1].set_yscale('log')
        plt.colorbar(im2, ax=axes[1, 1], label='Power')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def compare_signals(self, original: np.ndarray, cleaned: np.ndarray, 
                       ground_truth: Optional[np.ndarray] = None,
                       channel: int = 0, time_window: Optional[Tuple[int, int]] = None):
        """
        Compare original, cleaned, and ground truth signals.
        
        Args:
            original: Original signal with artifacts
            cleaned: Cleaned signal
            ground_truth: Ground truth clean signal (optional)
            channel: Channel to compare
            time_window: Time window to plot (start, end) in samples
        """
        if time_window is None:
            time_window = (0, min(original.shape[0], 2000))
            
        start, end = time_window
        time_axis = np.arange(start, end) / self.sampling_rate * 1000  # Convert to ms
        
        plt.figure(figsize=(15, 10))
        
        # Time series comparison
        plt.subplot(2, 2, 1)
        plt.plot(time_axis, original[start:end, channel], 'r-', label='Original', alpha=0.7)
        plt.plot(time_axis, cleaned[start:end, channel], 'g-', label='Cleaned', alpha=0.7)
        if ground_truth is not None:
            plt.plot(time_axis, ground_truth[start:end, channel], 'b-', label='Ground Truth', linewidth=2)
        plt.title(f'Time Series Comparison - Channel {channel}')
        plt.xlabel('Time (ms)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)
        
        # PSD comparison
        plt.subplot(2, 2, 2)
        freqs_orig, psd_orig = self.compute_psd(original[start:end, :])
        freqs_clean, psd_clean = self.compute_psd(cleaned[start:end, :])
        plt.semilogy(freqs_orig, psd_orig[:, channel], 'r-', label='Original', alpha=0.7)
        plt.semilogy(freqs_clean, psd_clean[:, channel], 'g-', label='Cleaned', alpha=0.7)
        if ground_truth is not None:
            freqs_gt, psd_gt = self.compute_psd(ground_truth[start:end, :])
            plt.semilogy(freqs_gt, psd_gt[:, channel], 'b-', label='Ground Truth', linewidth=2)
        plt.title('Power Spectral Density Comparison')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')
        plt.legend()
        plt.grid(True)
        
        # SNR comparison
        plt.subplot(2, 2, 3)
        if ground_truth is not None:
            snr_orig = self.calculate_snr(ground_truth[start:end, channel], 
                                        original[start:end, channel] - ground_truth[start:end, channel])
            snr_clean = self.calculate_snr(ground_truth[start:end, channel], 
                                         cleaned[start:end, channel] - ground_truth[start:end, channel])
            
            snr_improvement = snr_clean - snr_orig
            
            bars = plt.bar(['Original', 'Cleaned'], [snr_orig, snr_clean])
            plt.title(f'SNR Comparison\nImprovement: {snr_improvement:.2f} dB')
            plt.ylabel('SNR (dB)')
            plt.grid(True, axis='y')
            
            # Color bars
            bars[0].set_color('red')
            bars[1].set_color('green')
        
        # Feature comparison
        plt.subplot(2, 2, 4)
        features = ['rms', 'variance', 'skewness', 'kurtosis']
        orig_features = self.extract_neural_features(original[start:end, :], features)
        clean_features = self.extract_neural_features(cleaned[start:end, :], features)
        
        x = np.arange(len(features))
        width = 0.35
        
        plt.bar(x - width/2, [orig_features[f][channel] for f in features], 
                width, label='Original', alpha=0.7, color='red')
        plt.bar(x + width/2, [clean_features[f][channel] for f in features], 
                width, label='Cleaned', alpha=0.7, color='green')
        
        plt.title('Feature Comparison')
        plt.xlabel('Features')
        plt.ylabel('Value')
        plt.xticks(x, features, rotation=45)
        plt.legend()
        plt.grid(True, axis='y')
        
        plt.tight_layout()
        plt.show()
