import numpy as np
import json
import os
from typing import Dict, Any, Optional

from .core.base_method import BaseSACMethod
from .core.evaluator import Evaluator
from .core.signal_data import SignalDataWithGroundTruth
from .core.plotting import NeuralPlotter


class MethodTester:
    def __init__(self, data: SignalDataWithGroundTruth, methods: Dict[str, BaseSACMethod], save: bool = False, save_folder: str = "./results"):
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
        self.save = save
        self.save_folder = save_folder

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
            method.fit(self.data.raw_data, artifact_markers=self.data.artifact_markers)
            self.cleaned_signals[name] = method.transform(self.data.raw_data)
            if self.save:
                os.makedirs(self.save_folder, exist_ok=True)
                np.save(os.path.join(self.save_folder, f"{name}_cleaned.npy"), self.cleaned_signals[name])
                with open(os.path.join(self.save_folder, f"{name}_config.json"), 'w') as f:
                    json.dump(method.get_config(), f, indent=4)
            self.results[name] = self.evaluate(
                ground_truth=self.data.ground_truth,
                original_mixed=self.data.raw_data,
                cleaned=self.cleaned_signals[name]
            )

    def evaluate(self, ground_truth: np.ndarray, original_mixed: np.ndarray, cleaned: np.ndarray) -> Dict[str, float]:
        metrics = {}
        
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

        plotter = NeuralPlotter(self.evaluator) 

        for method_name, cleaned in self.cleaned_signals.items():
            plotter.plot_cleaned_comparison(
                self.data.ground_truth,
                self.data.raw_data,
                cleaned,
                trial_idx,
                channel_idx,
                title=f"Method: {method_name}"
            )

    def compare(self, weights: Optional[Dict[str, float]] = None):
        if not self.results:
            print("No results to compare. Run the tester first.")
            return

        if weights is None:
            # --- CHANGE 2: Default weights no longer include 'mse' ---
            weights = {
                'snr_improvement_db': 2.0,  # Highly weighted
                'mua_correlation': 1.5,
                'hit_rate': 1.5,
                'lfp_psd_correlation': 1.0,
                'false_positive_rate': -1.0, # Negative weight penalizes false positives
            }
        
        print("\n=== Weighted Scoring Summary ===")
        print("Scoring based on absolute performance (higher is better).")
        print("Using weights:")
        for metric, weight in weights.items():
            print(f"  - {metric}: {weight}")

        method_scores = {name: 0.0 for name in self.results.keys()}

        for method_name, metrics in self.results.items():
            score = 0.0
            for metric_name, weight in weights.items():
                value = metrics.get(metric_name)
                
                if value is None or np.isnan(value):
                    continue

                metric_score = 0.0
                if metric_name == 'snr_improvement_db':
                    # Only award points for POSITIVE improvement.
                    metric_score = max(0, value)
                elif metric_name == 'false_positive_rate':
                    # Penalize directly based on the rate. The negative weight handles the rest.
                    metric_score = value
                elif 'correlation' in metric_name or 'hit_rate' in metric_name:
                    # Use the value directly, but clamp at 0 to prevent negative correlations from helping.
                    metric_score = max(0, value)
                else:
                    metric_score = value
                
                score += metric_score * weight
            
            method_scores[method_name] = score

        sorted_scores = sorted(method_scores.items(), key=lambda item: item[1], reverse=True)
        
        print("\nFinal Scores:")
        if sorted_scores:
            best_method_overall = sorted_scores[0][0]
            for method_name, score in sorted_scores:
                print(f"  - {method_name}: {score:.4f}")
            print(f"\nOverall Best Method: {best_method_overall}")
