# SPARC/core/evaluator.py
import numpy as np
from typing import Dict, Any
from .neural_analyzer import NeuralAnalyzer
from scipy import signal


class Evaluator(NeuralAnalyzer):
    def __init__(self, sampling_rate: float):
        super().__init__(sampling_rate)


    def calculate_artifact_suppression(self, original: np.ndarray, cleaned: np.ndarray, ground_truth: np.ndarray) -> float:
        # amplitude
        artifact_amplitude_original = np.max(np.abs(original))
        artifact_amplitude_cleaned = np.max(np.abs(cleaned - ground_truth))
        # power
        artifact_power_original = np.mean(original ** 2)
        artifact_power_cleaned = np.mean((cleaned - ground_truth) ** 2)
        # calculate amplitude and power ratios
        suppression_amplitude_ratio = np.nan
        suppression_power_ratio_db = np.nan
        if artifact_amplitude_cleaned > 0:
            suppression_amplitude_ratio = artifact_amplitude_original / artifact_amplitude_cleaned
            suppression_amplitude_ratio_db = 20 * np.log10(suppression_amplitude_ratio)

        if  artifact_power_cleaned > 0:
            suppression_power_ratio = artifact_power_original / artifact_power_cleaned
            suppression_power_ratio_db = 10 * np.log10(suppression_power_ratio)
            
        return suppression_amplitude_ratio_db, suppression_power_ratio_db

    
    def evaluate_spikes(self, cleaned_signal: np.ndarray, ground_truth_signal: np.ndarray, bin_width_ms: float = 0.1) -> Dict[str, Any]:
        if cleaned_signal.shape != ground_truth_signal.shape:
            raise ValueError("Cleaned and ground truth signals must have the same shape.")
        if cleaned_signal.ndim != 3:
            raise ValueError("Input signals must be 3D (trials, channels, timesteps).")

        # Extract spikes for all trials and channels at once
        spikes_cleaned = self.extract_spikes(cleaned_signal)
        spikes_ground_truth = self.extract_spikes(ground_truth_signal)

        # Initialize counters for metrics across all channels
        total_hits = 0
        total_misses = 0
        total_false_positives = 0
        total_gt_spikes_all_channels = 0
        total_gt_non_spikes_all_channels = 0

        num_trials, num_channels, num_timesteps = cleaned_signal.shape
        duration_s = (num_trials * num_timesteps) / self.sampling_rate
        bin_width_s = bin_width_ms / 1000
        num_bins_per_channel = int( (num_timesteps * num_trials) / self.sampling_rate / bin_width_s)

        # Loop through each channel to calculate and aggregate metrics
        for ch in range(num_channels):
            # Binning logic (applied per channel)
            binned_cleaned = np.zeros(num_bins_per_channel, dtype=bool)
            binned_gt = np.zeros(num_bins_per_channel, dtype=bool)
            
            # Note: Spike indices are relative to the start of their trial.
            # We need to add the trial offset for proper binning.
            for trial_idx, trial_spikes in enumerate(spikes_cleaned):
                for spike in trial_spikes[ch]:
                    global_index = trial_idx * num_timesteps + spike['index']
                    bin_index = int(global_index / self.sampling_rate / bin_width_s)
                    if bin_index < num_bins_per_channel:
                        binned_cleaned[bin_index] = True

            for trial_idx, trial_spikes in enumerate(spikes_ground_truth):
                for spike in trial_spikes[ch]:
                    global_index = trial_idx * num_timesteps + spike['index']
                    bin_index = int(global_index / self.sampling_rate / bin_width_s)
                    if bin_index < num_bins_per_channel:
                        binned_gt[bin_index] = True

            # Calculate metrics for this channel
            hits = np.sum(binned_cleaned & binned_gt)
            misses = np.sum(~binned_cleaned & binned_gt)
            false_positives = np.sum(binned_cleaned & ~binned_gt)
            total_gt_spikes = np.sum(binned_gt)
            
            # Aggregate results
            total_hits += hits
            total_misses += misses
            total_false_positives += false_positives
            total_gt_spikes_all_channels += total_gt_spikes
            total_gt_non_spikes_all_channels += (num_bins_per_channel - total_gt_spikes)

        # Calculate final rates from aggregated counts
        hit_rate = total_hits / total_gt_spikes_all_channels if total_gt_spikes_all_channels > 0 else np.nan
        miss_rate = total_misses / total_gt_spikes_all_channels if total_gt_spikes_all_channels > 0 else np.nan
        fp_rate = total_false_positives / total_gt_non_spikes_all_channels if total_gt_non_spikes_all_channels > 0 else np.nan
        
        return {'hit_rate': hit_rate, 'miss_rate': miss_rate, 'false_positive_rate': fp_rate}


    def evaluate_lfp(self, cleaned_signal: np.ndarray, ground_truth_signal: np.ndarray) -> Dict[str, float]:
        if cleaned_signal.ndim != 3 or ground_truth_signal.ndim != 3:
            raise ValueError("Input signals must be 3D (trials, timesteps, channels).")
        lfp_cleaned = self.extract_lfp(cleaned_signal)
        lfp_ground_truth = self.extract_lfp(ground_truth_signal)

        freqs_gt, psd_gt = self.compute_psd(lfp_ground_truth)
        freqs_cleaned, psd_cleaned = self.compute_psd(lfp_cleaned)

        if np.std(psd_gt) > 1e-9 and np.std(psd_cleaned) > 1e-9:
            correlation = np.corrcoef(psd_gt.flatten(), psd_cleaned.flatten())[0, 1]
        else:
            correlation = np.nan
            
        return {'lfp_psd_correlation': correlation}

    def evaluate_mua(self, cleaned_signal: np.ndarray, ground_truth_signal: np.ndarray) -> Dict[str, float]:
        if cleaned_signal.ndim != 3 or ground_truth_signal.ndim != 3:
            raise ValueError("Input signals must be 3D (trials, timesteps, channels).")
        mua_cleaned_3d = self.extract_mua(cleaned_signal)
        mua_ground_truth_3d = self.extract_mua(ground_truth_signal)

        num_channels = mua_cleaned_3d.shape[2]
        mua_cleaned = mua_cleaned_3d.reshape(-1, num_channels)
        mua_ground_truth = mua_ground_truth_3d.reshape(-1, num_channels)

        correlations = np.zeros(num_channels)
        for ch in range(num_channels):
            if np.std(mua_ground_truth[:, ch]) > 1e-9 and np.std(mua_cleaned[:, ch]) > 1e-9:
                correlations[ch] = np.corrcoef(mua_ground_truth[:, ch], mua_cleaned[:, ch])[0, 1]
            else:
                correlations[ch] = np.nan
        
        correlation = np.nanmean(correlations)
        return {'mua_correlation': correlation}

    def calculate_snr_improvement(self, original: np.ndarray, cleaned: np.ndarray, 
                                 ground_truth: np.ndarray) -> float:
        # Calculate noise before cleaning
        noise_before = original - ground_truth
        snr_before = self.calculate_snr(ground_truth, noise_before)
        
        # Calculate noise after cleaning
        noise_after = cleaned - ground_truth
        snr_after = self.calculate_snr(ground_truth, noise_after)
        
        return snr_after - snr_before

    def calculate_signal_quality_metrics(self, original: np.ndarray, cleaned: np.ndarray) -> Dict[str, float]:
        metrics = {}
        
        rms_original = np.sqrt(np.mean(original ** 2))
        rms_cleaned = np.sqrt(np.mean(cleaned ** 2))
        metrics['rms_reduction_ratio'] = rms_original / rms_cleaned if rms_cleaned > 0 else np.nan
        
        var_original = np.var(original)
        var_cleaned = np.var(cleaned)
        metrics['variance_reduction_ratio'] = var_original / var_cleaned if var_cleaned > 0 else np.nan
        
        return metrics

    def calculate_lfp_quality_metrics(self, lfp_original: np.ndarray, lfp_cleaned: np.ndarray) -> Dict[str, float]:
        metrics = {}
        
        lfp_power_original = np.mean(lfp_original ** 2)
        lfp_power_cleaned = np.mean(lfp_cleaned ** 2)
        metrics['lfp_power_preservation_ratio'] = lfp_power_cleaned / lfp_power_original if lfp_power_original > 0 else np.nan
        
        freqs_orig, psd_orig = self.compute_psd(lfp_original)
        freqs_clean, psd_clean = self.compute_psd(lfp_cleaned)
        
        if np.std(psd_orig) > 1e-9 and np.std(psd_clean) > 1e-9:
            lfp_spectral_correlation = np.corrcoef(psd_orig.flatten(), psd_clean.flatten())[0, 1]
            metrics['lfp_spectral_preservation'] = lfp_spectral_correlation
        
        return metrics

    def estimate_snr_improvement(self, original: np.ndarray, cleaned: np.ndarray) -> float:
        nyquist_freq = self.sampling_rate / 2
        
        # High-pass filter for artifacts (use 1/4 of Nyquist frequency)
        high_pass_freq = min(nyquist_freq * 0.25, 200) 
        # Low-pass filter for neural signal (use 1/8 of Nyquist frequency)  
        low_pass_freq = min(nyquist_freq * 0.125, 100)
        
        if high_pass_freq >= nyquist_freq * 0.9:
            high_pass_freq = nyquist_freq * 0.1
        if low_pass_freq >= nyquist_freq * 0.9:
            low_pass_freq = nyquist_freq * 0.05
            
        try:
            sos_high = signal.butter(4, high_pass_freq, btype='high', fs=self.sampling_rate, output='sos')
            high_freq_original = signal.sosfiltfilt(sos_high, original, axis=-1)
            high_freq_cleaned = signal.sosfiltfilt(sos_high, cleaned, axis=-1)
            
            noise_power_original = np.mean(high_freq_original ** 2)
            noise_power_cleaned = np.mean(high_freq_cleaned ** 2)
            
            if noise_power_original > 0 and noise_power_cleaned > 0:
                sos_low = signal.butter(4, low_pass_freq, btype='low', fs=self.sampling_rate, output='sos')
                signal_original = signal.sosfiltfilt(sos_low, original, axis=-1)
                signal_cleaned = signal.sosfiltfilt(sos_low, cleaned, axis=-1)
                
                signal_power_original = np.mean(signal_original ** 2)
                signal_power_cleaned = np.mean(signal_cleaned ** 2)
                
                if signal_power_original > 0 and signal_power_cleaned > 0:
                    snr_original = 10 * np.log10(signal_power_original / noise_power_original)
                    snr_cleaned = 10 * np.log10(signal_power_cleaned / noise_power_cleaned)
                    return snr_cleaned - snr_original
        except Exception:
            pass
        
        return np.nan