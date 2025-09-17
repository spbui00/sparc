from .core.base_method import BaseSACMethod
from .core.evaluator import Evaluator
from .core.signal_data import SignalDataWithGroundTruth
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import numpy as np


class MethodTester:
    def __init__(self, data: SignalDataWithGroundTruth, methods: Dict[str, BaseSACMethod]):
        """
        Args:
            data: A SignalDataWithGroundTruth object containing the data to test.
            methods: A dictionary of artifact removal methods to test.
        """
        self.data = data
        self.methods = methods
        self.results: Dict[str, Any] = {}
        self.evaluator = Evaluator(sampling_rate=self.data.sampling_rate)
        self.cleaned_signals = {}

        # Set sampling rate for methods if not already set
        for method in self.methods.values():
            if method.sampling_rate is None:
                method.set_sampling_rate(self.data.sampling_rate)

    def run(self):
        """
        Runs the test by fitting and transforming the data with each method,
        then evaluates the results using SPARCEvaluator.
        """
        print(f"Running tests with {list(self.methods.keys())} methods.")
        
        total = len(self.methods)
        for name, method in self.methods.items():
            print(f"\n=== Testing method: {name} ({list(self.methods.keys()).index(name)+1}/{total}) ===")
            # Pass artifact_markers if available
            method.fit(self.data.raw_data, artifact_markers=self.data.artifact_markers)
            self.cleaned_signals[name] = method.transform(self.data.raw_data)
            self.results[name] = self.evaluate(
                ground_truth=self.data.ground_truth,
                original_mixed=self.data.raw_data,
                cleaned=self.cleaned_signals[name]
            )

    def evaluate(self, ground_truth: np.ndarray, original_mixed: np.ndarray, cleaned: np.ndarray) -> Dict[str, float]:
        metrics = {}
        
        metrics['mse'] = self.evaluator.calculate_mse(ground_truth, cleaned)
        metrics['snr_improvement_db'] = self.evaluator.calculate_snr_improvement(original_mixed, cleaned, ground_truth)
        mua_metrics = self.evaluator.evaluate_mua(cleaned, ground_truth)
        metrics.update(mua_metrics)
        spike_metrics = self.evaluator.evaluate_spikes(cleaned, ground_truth)
        metrics.update(spike_metrics)
        lfp_metrics = self.evaluator.evaluate_lfp(cleaned, ground_truth)
        metrics.update(lfp_metrics)

        return metrics

    def get_results(self):
        """Returns the results of the tests."""
        return self.results

    def print_results(self):
        """Prints a summary of the results.
        Note: The main summary is already printed by the evaluator during the run.
        This provides a way to access it again.
        """
        if not self.results:
            print("No results to print. Run the tester first.")
            return

        print("\n=== Full Results Summary ===")
        for method_name, metrics in self.results.items():
            print(f"\n--- {method_name} ---")
            for metric_name, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {metric_name}: {value:.4f}")

    def plot_results(self, trial_idx: int = 0, channel_idx: int = 0):
        if not self.cleaned_signals:
            print("No cleaned signals to plot.")
            return
        if self.data.raw_data.ndim != 3:
            raise ValueError(f"Plotting requires 3D data (trials, timesteps, channels), but got {self.data.raw_data.ndim} dimensions.")
        
        if trial_idx >= self.data.raw_data.shape[0]:
            raise IndexError(f"trial_idx {trial_idx} is out of bounds for data with {self.data.raw_data.shape[0]} trials.")
        if channel_idx >= self.data.raw_data.shape[2]:
            raise IndexError(f"channel_idx {channel_idx} is out of bounds for data with {self.data.raw_data.shape[2]} channels.")

        print(f"\nPlotting results for trial {trial_idx}, channel {channel_idx}...")

        raw_trial_data = self.data.raw_data[trial_idx, :, :]
        gt_trial_data = self.data.ground_truth[trial_idx, :, :]

        fig, axes = plt.subplots(len(self.cleaned_signals) + 1, 1, figsize=(15, 10), sharex=True)
        
        axes[0].plot(raw_trial_data[:, channel_idx], label='Original', color='black', alpha=0.5)
        axes[0].plot(gt_trial_data[:, channel_idx], label='Ground Truth', color='blue', alpha=0.7)
        axes[0].set_title(f"Original Data - Trial {trial_idx}")
        axes[0].legend()

        for i, (name, cleaned_data_3d) in enumerate(self.cleaned_signals.items()):
            ax = axes[i+1]
            
            cleaned_trial_data = cleaned_data_3d[trial_idx, :, :]
            
            ax.plot(raw_trial_data[:, channel_idx], label='Original', color='black', alpha=0.2)
            ax.plot(gt_trial_data[:, channel_idx], label='Ground Truth', color='blue', alpha=0.3)
            ax.plot(cleaned_trial_data[:, channel_idx], label=f'Cleaned with {name}', color='green')
            ax.set_title(f"Cleaned with {name}")
            ax.legend()

        plt.tight_layout()
        plt.show()

    def compare(self, weights: Optional[Dict[str, float]] = None):
        if not self.results:
            print("No results to compare. Run the tester first.")
            return

        print("\n=== Best Method for Each Metric ===")
        
        first_method_name = list(self.results.keys())[0]
        all_metrics = list(self.results[first_method_name].keys())
        
        metrics_higher_is_better = {
            'mse': False,
            'snr_improvement_db': True,
            'mua_correlation': True,
            'hit_rate': True,
            'miss_rate': False,
            'false_positive_rate': False,
            'lfp_psd_correlation': True
        }

        for metric in all_metrics:
            # Default to True if a metric is not in the dictionary
            higher_is_better = metrics_higher_is_better.get(metric, True)
            
            best_method = None
            best_score = -np.inf if higher_is_better else np.inf

            for method_name, metrics in self.results.items():
                score = metrics.get(metric)
                if score is None:
                    continue
                
                if higher_is_better:
                    if score > best_score:
                        best_score = score
                        best_method = method_name
                else:  # lower is better
                    if score < best_score:
                        best_score = score
                        best_method = method_name
            
            if best_method:
                print(f"Best for {metric}: {best_method} ({best_score:.4f})")

        print("\n=== Weighted Scoring Summary ===")

        if weights is None:
            weights = {
                'mse': 1.0,
                'snr_improvement_db': 1.0,
                'mua_correlation': 1.0,
                'hit_rate': 1.5,
                'lfp_psd_correlation': 1.0,
            }
        
        print("Using weights:")
        for metric, weight in weights.items():
            print(f"  - {metric}: {weight}")

        method_scores = {name: 0.0 for name in self.results.keys()}
        
        for metric, weight in weights.items():
            if metric not in all_metrics:
                print(f"Warning: metric '{metric}' from weights not in results. Skipping.")
                continue

            scores = [res.get(metric) for res in self.results.values() if res.get(metric) is not None]
            if not scores:
                continue
                
            min_score, max_score = min(scores), max(scores)

            if min_score == max_score:  # All methods performed equally
                continue

            # get() second argument is a default value if key not found
            higher_is_better = metrics_higher_is_better.get(metric, True)

            for method_name in self.results.keys():
                score = self.results[method_name].get(metric)
                if score is None:
                    continue

                if higher_is_better:
                    norm_score = (score - min_score) / (max_score - min_score)
                else:  # lower is better
                    norm_score = (max_score - score) / (max_score - min_score)
                
                method_scores[method_name] += norm_score * weight

        print("\nFinal Scores:")
        best_method_overall = None
        
        # Sort for printing
        sorted_scores = sorted(method_scores.items(), key=lambda item: item[1], reverse=True)
        
        if sorted_scores:
            best_method_overall = sorted_scores[0][0]
            for method_name, score in sorted_scores:
                print(f"  - {method_name}: {score:.4f}")
        
        if best_method_overall:
            print(f"\nOverall Best Method: {best_method_overall}")
