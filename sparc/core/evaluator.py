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
        if cleaned_signal.ndim != 3 or ground_truth_signal.ndim != 3:
            raise ValueError("Input signals must be 3D (trials, timesteps, channels).")

        # --- Concatenate trials to get a single performance score ---
        num_channels = cleaned_signal.shape[2]
        cleaned_2d = cleaned_signal.reshape(-1, num_channels)
        gt_2d = ground_truth_signal.reshape(-1, num_channels)

        # The rest of the logic can now proceed as it did for 2D data
        # by analyzing the first channel of the concatenated signal.
        cleaned_ch0 = cleaned_2d[:, 0]
        gt_ch0 = gt_2d[:, 0]
        
        # Reshape to (samples, 1) for extract_spikes
        spikes_cleaned_list = self.extract_spikes(cleaned_ch0.reshape(1, -1, 1))
        spikes_ground_truth_list = self.extract_spikes(gt_ch0.reshape(1, -1, 1))

        spikes_cleaned = spikes_cleaned_list[0][0]
        spikes_ground_truth = spikes_ground_truth_list[0][0]
            
        duration_s = cleaned_2d.shape[0] / self.sampling_rate
        bin_width_s = bin_width_ms / 1000
        num_bins = int(duration_s / bin_width_s)

        binned_cleaned = np.zeros(num_bins, dtype=bool)
        binned_ground_truth = np.zeros(num_bins, dtype=bool)

        for spike in spikes_cleaned:
            bin_index = int(spike['index'] / self.sampling_rate / bin_width_s)
            if bin_index < num_bins:
                binned_cleaned[bin_index] = True

        for spike in spikes_ground_truth:
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
        if cleaned_signal.ndim != 3 or ground_truth_signal.ndim != 3:
            raise ValueError("Input signals must be 3D (trials, timesteps, channels).")

        num_channels = cleaned_signal.shape[2]
        cleaned_2d = cleaned_signal.reshape(-1, num_channels)
        gt_2d = ground_truth_signal.reshape(-1, num_channels)

        lfp_cleaned = self.extract_lfp(cleaned_2d)
        lfp_ground_truth = self.extract_lfp(gt_2d)

        psd_correlations = np.zeros(num_channels)
        for ch in range(num_channels):
            _, psd_gt = self.compute_psd(lfp_ground_truth[:, ch])
            _, psd_cleaned = self.compute_psd(lfp_cleaned[:, ch])
            psd_correlations[ch] = np.corrcoef(psd_gt.flatten(), psd_cleaned.flatten())[0, 1]
        
        correlation = np.nanmean(psd_correlations)
        return {'lfp_psd_correlation': correlation}

    def evaluate_mua(self, cleaned_signal: np.ndarray, ground_truth_signal: np.ndarray) -> Dict[str, float]:
        if cleaned_signal.ndim != 3 or ground_truth_signal.ndim != 3:
            raise ValueError("Input signals must be 3D (trials, timesteps, channels).")
        num_channels = cleaned_signal.shape[2]
        cleaned_2d = cleaned_signal.reshape(-1, num_channels)
        gt_2d = ground_truth_signal.reshape(-1, num_channels)

        mua_cleaned = self.extract_mua(cleaned_2d)
        mua_ground_truth = self.extract_mua(gt_2d)

        correlations = np.zeros(num_channels)
        for ch in range(num_channels):
            if np.std(mua_ground_truth[:, ch]) > 1e-9 and np.std(mua_cleaned[:, ch]) > 1e-9:
                correlations[ch] = np.corrcoef(mua_ground_truth[:, ch], mua_cleaned[:, ch])[0, 1]
            else:
                correlations[ch] = np.nan
        
        correlation = np.nanmean(correlations)
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
