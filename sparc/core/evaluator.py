# SPARC/core/evaluator.py
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any
from .neural_analyzer import NeuralAnalyzer

class SPARCEvaluator(NeuralAnalyzer):
    
    def __init__(self, sampling_rate: float):
        super().__init__(sampling_rate)

    def evaluate_spike_detection(self, cleaned_signal: np.ndarray, ground_truth_signal: np.ndarray, bin_width_ms: float = 0.1) -> Dict[str, Any]:
        """
        Evaluate spike detection performance by comparing cleaned signal to ground truth.

        Args:
            cleaned_signal: The signal after artifact removal.
            ground_truth_signal: The ground truth signal without artifacts.
            bin_width_ms: The width of the bins in milliseconds for spike train comparison.

        Returns:
            A dictionary containing various spike detection metrics.
        """
        # Extract spikes from both signals
        spikes_cleaned = self.extract_spikes(cleaned_signal)
        spikes_ground_truth = self.extract_spikes(ground_truth_signal)

        num_channels = cleaned_signal.shape[1]
        duration_s = cleaned_signal.shape[0] / self.sampling_rate
        bin_width_s = bin_width_ms / 1000
        num_bins = int(duration_s / bin_width_s)

        results = {
            'firing_rate_ratio': [],
            'hits': [],
            'misses': [],
            'false_positives': [],
            'true_negatives': [],
            'hit_rate': [],
            'miss_rate': [],
            'false_positive_rate': []
        }

        for ch in range(num_channels):
            # Create binned spike trains
            binned_cleaned = np.zeros(num_bins, dtype=bool)
            binned_ground_truth = np.zeros(num_bins, dtype=bool)

            for spike in spikes_cleaned[ch]:
                bin_index = int(spike['index'] / self.sampling_rate / bin_width_s)
                if bin_index < num_bins:
                    binned_cleaned[bin_index] = True

            for spike in spikes_ground_truth[ch]:
                bin_index = int(spike['index'] / self.sampling_rate / bin_width_s)
                if bin_index < num_bins:
                    binned_ground_truth[bin_index] = True

            # Firing rates
            fr_cleaned = len(spikes_cleaned[ch]) / duration_s
            fr_ground_truth = len(spikes_ground_truth[ch]) / duration_s
            results['firing_rate_ratio'].append(fr_cleaned / fr_ground_truth if fr_ground_truth > 0 else np.nan)

            # num_gt_spikes = len(spikes_ground_truth[ch])
            # num_cleaned_spikes = len(spikes_cleaned[ch])
            # print(f"  Channel {ch}: Ground Truth Spikes = {num_gt_spikes}, Cleaned Spikes = {num_cleaned_spikes}")

            # Confusion matrix components
            hits = np.sum(binned_cleaned & binned_ground_truth)
            misses = np.sum(~binned_cleaned & binned_ground_truth)
            false_positives = np.sum(binned_cleaned & ~binned_ground_truth)
            true_negatives = np.sum(~binned_cleaned & ~binned_ground_truth)

            results['hits'].append(hits)
            results['misses'].append(misses)
            results['false_positives'].append(false_positives)
            results['true_negatives'].append(true_negatives)

            # Rates
            total_gt_spikes = np.sum(binned_ground_truth)
            total_gt_non_spikes = num_bins - total_gt_spikes
            
            results['hit_rate'].append(hits / total_gt_spikes if total_gt_spikes > 0 else np.nan)
            results['miss_rate'].append(misses / total_gt_spikes if total_gt_spikes > 0 else np.nan)
            results['false_positive_rate'].append(false_positives / total_gt_non_spikes if total_gt_non_spikes > 0 else np.nan)


        keys_to_process = list(results.keys())
        for key in keys_to_process:
            results[f'{key}_mean'] = np.nanmean(results[key])
            results[f'{key}_std'] = np.nanstd(results[key])

        return results

    def evaluate_lfp(self, cleaned_signal: np.ndarray, ground_truth_signal: np.ndarray) -> Dict[str, float]:
        """
        Evaluate LFP preservation by comparing power spectra.

        Args:
            cleaned_signal: The signal after artifact removal.
            ground_truth_signal: The ground truth signal without artifacts.

        Returns:
            A dictionary containing LFP evaluation metrics.
        """
        # Extract LFP from both signals
        lfp_cleaned = self.extract_lfp(cleaned_signal)
        lfp_ground_truth = self.extract_lfp(ground_truth_signal)

        # Calculate PSD MSE
        psd_mse = self.calculate_psd_mse(lfp_ground_truth, lfp_cleaned)

        # Calculate PSD correlation
        psd_corr = np.zeros(lfp_ground_truth.shape[1])
        for ch in range(lfp_ground_truth.shape[1]):
            _, psd_gt = self.compute_psd(lfp_ground_truth[:, ch:ch+1])
            _, psd_cleaned = self.compute_psd(lfp_cleaned[:, ch:ch+1])
            psd_corr[ch] = np.corrcoef(psd_gt.flatten(), psd_cleaned.flatten())[0, 1]

        results = {
            'lfp_psd_mse_mean': np.mean(psd_mse),
            'lfp_psd_mse_std': np.std(psd_mse),
            'lfp_psd_correlation_mean': np.mean(psd_corr),
            'lfp_psd_correlation_std': np.std(psd_corr)
        }

        return results

    def calculate_artifact_removal_ratio(self, original: np.ndarray, cleaned: np.ndarray, 
                                       ground_truth: np.ndarray) -> float:
        artifacts_original = np.abs(original - ground_truth)
        artifacts_cleaned = np.abs(cleaned - ground_truth)
        
        # Calculate removal ratio
        total_artifacts = np.sum(artifacts_original)
        remaining_artifacts = np.sum(artifacts_cleaned)
        
        if total_artifacts == 0:
            return 1.0
            
        removal_ratio = 1 - (remaining_artifacts / total_artifacts)
        return removal_ratio
        
    def evaluate_method_comprehensive(self, ground_truth: np.ndarray, original: np.ndarray, 
                                    cleaned: np.ndarray, method_name: str = "Method") -> Dict[str, float]:
        """
        Comprehensive evaluation of a SAC method.
        
        Args:
            ground_truth: Ground truth clean signal
            original: Original signal with artifacts
            cleaned: Cleaned signal
            method_name: Name of the method for reporting
            
        Returns:
            Dictionary of comprehensive evaluation metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['mse'] = self.calculate_mse(ground_truth, cleaned)
        metrics['mse_improvement'] = (self.calculate_mse(ground_truth, original) - 
                                    self.calculate_mse(ground_truth, cleaned))
        
        # PSD metrics
        psd_mse = self.calculate_psd_mse(ground_truth, cleaned)
        metrics['psd_mse_mean'] = np.mean(psd_mse)
        metrics['psd_mse_std'] = np.std(psd_mse)
        
        # SNR metrics
        metrics['snr_improvement_db'] = self.calculate_snr_improvement(original, cleaned, ground_truth)
        
        # Spectral coherence
        coherence = self.calculate_spectral_coherence(ground_truth, cleaned)
        metrics['spectral_coherence_mean'] = np.mean(coherence)
        metrics['spectral_coherence_std'] = np.std(coherence)
        
        # Artifact removal
        metrics['artifact_removal_ratio'] = self.calculate_artifact_removal_ratio(original, cleaned, ground_truth)
        
        # Spike detection metrics
        spike_metrics = self.evaluate_spike_detection(cleaned, ground_truth)
        metrics.update(spike_metrics)

        # LFP evaluation
        lfp_metrics = self.evaluate_lfp(cleaned, ground_truth)
        metrics.update(lfp_metrics)
        
        # Print results
        print(f"\n=== {method_name} Evaluation Results ===")
        print(f"MSE: {metrics['mse']:.6f}")
        print(f"MSE Improvement: {metrics['mse_improvement']:.6f}")
        print(f"PSD MSE: {metrics['psd_mse_mean']:.6f} ± {metrics['psd_mse_std']:.6f}")
        print(f"SNR Improvement: {metrics['snr_improvement_db']:.2f} dB")
        print(f"Spectral Coherence: {metrics['spectral_coherence_mean']:.4f} ± {metrics['spectral_coherence_std']:.4f}")
        print(f"Artifact Removal Ratio: {metrics['artifact_removal_ratio']:.4f}")
        print(f"LFP PSD MSE: {metrics['lfp_psd_mse_mean']:.6f} ± {metrics['lfp_psd_mse_std']:.6f}")
        print(f"LFP PSD Correlation: {metrics['lfp_psd_correlation_mean']:.4f} ± {metrics['lfp_psd_correlation_std']:.4f}")
        
        return metrics
    
    def compare_methods(self, ground_truth: np.ndarray, original: np.ndarray, 
                       methods_results: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple SAC methods.
        
        Args:
            ground_truth: Ground truth clean signal
            original: Original signal with artifacts
            methods_results: Dictionary mapping method names to cleaned signals
            
        Returns:
            Dictionary of evaluation results for each method
        """
        comparison_results = {}
        
        for method_name, cleaned_signal in methods_results.items():
            print(f"\nEvaluating {method_name}...")
            comparison_results[method_name] = self.evaluate_method_comprehensive(
                ground_truth, original, cleaned_signal, method_name
            )
        
        # Create comparison plots
        self._plot_method_comparison(comparison_results)
        
        return comparison_results
    
    def _plot_method_comparison(self, results: Dict[str, Dict[str, float]]):
        """Create comparison plots for multiple methods."""
        methods = list(results.keys())
        metrics = ['mse', 'psd_mse_mean', 'snr_improvement_db', 'artifact_removal_ratio']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('SAC Methods Comparison', fontsize=16)
        
        for i, metric in enumerate(metrics):
            ax = axes[i // 2, i % 2]
            values = [results[method][metric] for method in methods]
            
            bars = ax.bar(methods, values, alpha=0.7)
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.set_ylabel('Value')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, axis='y')
            
            # Color bars differently
            colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
        
        plt.tight_layout()
        plt.show()
