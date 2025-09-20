# SPARC/core/evaluator.py
import numpy as np
from typing import Dict, Any
from .neural_analyzer import NeuralAnalyzer


class Evaluator(NeuralAnalyzer):
    def __init__(self, sampling_rate: float):
        super().__init__(sampling_rate)

    
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
