# SPARC/core/evaluator.py
import numpy as np
from typing import Dict
from .neural_analyzer import NeuralAnalyzer

class Evaluator(NeuralAnalyzer):
    def __init__(self, sampling_rate: float):
        super().__init__(sampling_rate)

    def evaluate_spikes(self, cleaned_signal: np.ndarray, ground_truth_signal: np.ndarray, bin_width_ms: float = 0.1) -> Dict[str, float]:
        """
        Calculates the three key spike detection metrics: hit rate, miss rate,
        and false positive rate.
        """
        spikes_cleaned = self.extract_spikes(cleaned_signal)
        spikes_ground_truth = self.extract_spikes(ground_truth_signal)

        # Handle cases with multiple channels by analyzing the first one
        if cleaned_signal.ndim > 1:
            cleaned_signal = cleaned_signal[:, 0]
            ground_truth_signal = ground_truth_signal[:, 0]
            spikes_cleaned = [spikes_cleaned[0]]
            spikes_ground_truth = [spikes_ground_truth[0]]
        else: # Ensure it's a list for the loop
            spikes_cleaned = [spikes_cleaned]
            spikes_ground_truth = [spikes_ground_truth]
            
        duration_s = cleaned_signal.shape[0] / self.sampling_rate
        bin_width_s = bin_width_ms / 1000
        num_bins = int(duration_s / bin_width_s)

        # Simplified loop for a single channel
        binned_cleaned = np.zeros(num_bins, dtype=bool)
        binned_ground_truth = np.zeros(num_bins, dtype=bool)

        for spike in spikes_cleaned[0]:
            bin_index = int(spike['index'] / self.sampling_rate / bin_width_s)
            if bin_index < num_bins:
                binned_cleaned[bin_index] = True

        for spike in spikes_ground_truth[0]:
            bin_index = int(spike['index'] / self.sampling_rate / bin_width_s)
            if bin_index < num_bins:
                binned_ground_truth[bin_index] = True

        hits = np.sum(binned_cleaned & binned_ground_truth)
        misses = np.sum(~binned_cleaned & binned_ground_truth)
        false_positives = np.sum(binned_cleaned & ~binned_ground_truth)

        total_gt_spikes = np.sum(binned_ground_truth)
        total_gt_non_spikes = num_bins - total_gt_spikes

        hit_rate = hits / total_gt_spikes if total_gt_spikes > 0 else np.nan
        miss_rate = misses / total_gt_spikes if total_gt_spikes > 0 else np.nan
        fp_rate = false_positives / total_gt_non_spikes if total_gt_non_spikes > 0 else np.nan
        
        return {'hit_rate': hit_rate, 'miss_rate': miss_rate, 'false_positive_rate': fp_rate}

    def evaluate_lfp(self, cleaned_signal: np.ndarray, ground_truth_signal: np.ndarray) -> Dict[str, float]:
        """
        Calculates the key LFP metric: Power Spectral Density (PSD) correlation.
        """
        lfp_cleaned = self.extract_lfp(cleaned_signal)
        lfp_ground_truth = self.extract_lfp(ground_truth_signal)

        # Handle multi-channel data by averaging correlations
        if lfp_ground_truth.ndim > 1:
            num_channels = lfp_ground_truth.shape[1]
            psd_correlations = np.zeros(num_channels)
            for ch in range(num_channels):
                _, psd_gt = self.compute_psd(lfp_ground_truth[:, ch])
                _, psd_cleaned = self.compute_psd(lfp_cleaned[:, ch])
                # Flatten in case of multi-dimensional PSD output
                psd_correlations[ch] = np.corrcoef(psd_gt.flatten(), psd_cleaned.flatten())[0, 1]
            correlation = np.nanmean(psd_correlations)
        else:
             _, psd_gt = self.compute_psd(lfp_ground_truth)
             _, psd_cleaned = self.compute_psd(lfp_cleaned)
             correlation = np.corrcoef(psd_gt.flatten(), psd_cleaned.flatten())[0, 1]

        return {'lfp_psd_correlation': correlation}
        
    def evaluate_mua(self, cleaned_signal: np.ndarray, ground_truth_signal: np.ndarray) -> Dict[str, float]:
        """
        Calculates the correlation between the ground truth and cleaned MUA signals.
        MUA is a measure of the overall high-frequency spiking activity.
        """
        mua_cleaned = self.extract_mua(cleaned_signal)
        mua_ground_truth = self.extract_mua(ground_truth_signal)

        # Handle multi-channel data by averaging correlations
        if mua_ground_truth.ndim > 1:
            num_channels = mua_ground_truth.shape[1]
            correlations = np.zeros(num_channels)
            for ch in range(num_channels):
                if np.std(mua_ground_truth[:, ch]) > 1e-9 and np.std(mua_cleaned[:, ch]) > 1e-9:
                    correlations[ch] = np.corrcoef(mua_ground_truth[:, ch], mua_cleaned[:, ch])[0, 1]
                else:
                    correlations[ch] = np.nan
            correlation = np.nanmean(correlations)
        else:
            if np.std(mua_ground_truth) > 1e-9 and np.std(mua_cleaned) > 1e-9:
                correlation = np.corrcoef(mua_ground_truth, mua_cleaned)[0, 1]
            else:
                correlation = np.nan
       
        return {'mua_correlation': correlation}

    def calculate_artifact_removal_ratio(self, original: np.ndarray, cleaned: np.ndarray, ground_truth: np.ndarray) -> float:
        """
        Calculates the proportion of the artifact energy that was removed.
        A value of 1.0 means 100% of the artifact was removed.
        """
        artifacts_original = np.abs(original - ground_truth)
        artifacts_cleaned = np.abs(cleaned - ground_truth)
        
        total_artifacts_energy = np.sum(artifacts_original)
        remaining_artifacts_energy = np.sum(artifacts_cleaned)
        
        if total_artifacts_energy == 0:
            return 1.0 # No artifacts to remove, so 100% were removed.
            
        removal_ratio = 1 - (remaining_artifacts_energy / total_artifacts_energy)
        return removal_ratio

    def calculate_snr_improvement(self, original: np.ndarray, cleaned: np.ndarray, 
                                 ground_truth: np.ndarray) -> float:
        # Calculate noise before cleaning
        noise_before = original - ground_truth
        snr_before = self.calculate_snr(ground_truth, noise_before)
        
        # Calculate noise after cleaning
        noise_after = cleaned - ground_truth
        snr_after = self.calculate_snr(ground_truth, noise_after)
        
        return snr_after - snr_before
