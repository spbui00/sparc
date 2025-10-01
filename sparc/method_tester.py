import numpy as np
import json
import os
import glob
import time
from typing import Dict, Any, Optional
from tqdm import tqdm

from .core.base_method import BaseSACMethod
from .core.evaluator import Evaluator
from .core.signal_data import SignalData, SignalDataWithGroundTruth
from .core.plotting import NeuralPlotter


class MethodTester:
    def __init__(self, data: SignalData | SignalDataWithGroundTruth, methods: Dict[str, BaseSACMethod], save: bool = False, save_folder: str = "./results"):
        """
        Args:
            data: A SignalData or SignalDataWithGroundTruth object containing the data to test.
            methods: A dictionary of artifact removal methods to test.
        """
        self.data = data
        self.methods = methods
        self.results: Dict[str, Any] = {}
        self.evaluator = Evaluator(sampling_rate=self.data.sampling_rate)
        self.cleaned_signals = {}
        self.save = save
        self.save_folder = save_folder
        self.has_ground_truth = isinstance(data, SignalDataWithGroundTruth)

        for method in self.methods.values():
            if method.sampling_rate is None:
                method.set_sampling_rate(self.data.sampling_rate)

    def run(self):
        print(f"Running tests with {list(self.methods.keys())} methods.")
        
        start_time = time.time()
        pbar = tqdm(self.methods.items(), desc="Testing methods", unit="method")
        
        for name, method in pbar:
            pbar.set_postfix_str(f"Processing {name}")
            
            method.fit(self.data.raw_data, artifact_markers=self.data.artifact_markers)
            self.cleaned_signals[name] = method.transform(self.data.raw_data)
            
            if self.save:
                os.makedirs(self.save_folder, exist_ok=True)
                np.save(os.path.join(self.save_folder, f"{name}_cleaned.npy"), self.cleaned_signals[name])
                with open(os.path.join(self.save_folder, f"{name}_config.json"), 'w') as f:
                    json.dump(method.get_config(), f, indent=4)
            
            if self.has_ground_truth:
                self.results[name] = self.evaluate(
                    ground_truth=self.data.ground_truth,
                    original_mixed=self.data.raw_data,
                    cleaned=self.cleaned_signals[name]
                )
            else:
                self.results[name] = self.evaluate_without_ground_truth(
                    original=self.data.raw_data,
                    cleaned=self.cleaned_signals[name]
                )
            
            pbar.set_postfix_str(f"✓ {name}")
        
        pbar.close()
        
        total_time = time.time() - start_time
        print(f"\nTesting completed in {total_time:.2f} seconds")
        if len(self.methods) > 0:
            avg_time_per_method = total_time / len(self.methods)
            print(f"Average time per method: {avg_time_per_method:.2f} seconds")

    def evaluate(self, ground_truth: np.ndarray, original_mixed: np.ndarray, cleaned: np.ndarray) -> Dict[str, float]:
        metrics = {}
        
        metrics['snr_improvement_db'] = self.evaluator.calculate_snr_improvement(original_mixed, cleaned, ground_truth)
        # mua_metrics = self.evaluator.evaluate_mua(cleaned, ground_truth)
        # metrics.update(mua_metrics)
        # spike_metrics = self.evaluator.evaluate_spikes(cleaned, ground_truth)
        # metrics.update(spike_metrics)
        lfp_metrics = self.evaluator.evaluate_lfp(cleaned, ground_truth)
        metrics.update(lfp_metrics)

        return metrics

    def evaluate_without_ground_truth(self, original: np.ndarray, cleaned: np.ndarray) -> Dict[str, float]:
        metrics = {}
    
        lfp_original = self.evaluator.extract_lfp(original)
        lfp_cleaned = self.evaluator.extract_lfp(cleaned)
        metrics.update(self.evaluator.calculate_signal_quality_metrics(original, cleaned))
        metrics.update(self.evaluator.calculate_lfp_quality_metrics(lfp_original, lfp_cleaned))
        
        estimated_snr_improvement = self.evaluator.estimate_snr_improvement(original, cleaned)
        if not np.isnan(estimated_snr_improvement):
            metrics['estimated_snr_improvement_db'] = estimated_snr_improvement
        
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
            if self.has_ground_truth:
                plotter.plot_cleaned_comparison(
                    self.data.ground_truth,
                    self.data.raw_data,
                    cleaned,
                    trial_idx,
                    channel_idx,
                    title=f"Method: {method_name}"
                )
            else:
                self._plot_original_vs_cleaned(
                    self.data.raw_data,
                    cleaned,
                    trial_idx,
                    channel_idx,
                    title=f"Method: {method_name}"
                )

    def _plot_original_vs_cleaned(self, original: np.ndarray, cleaned: np.ndarray, 
                                 trial_idx: int, channel_idx: int, title: str = "Original vs Cleaned"):
        import matplotlib.pyplot as plt
        
        # Create sample axis (data shape is trials, channels, time)
        sample_axis = np.arange(original.shape[2])
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle(f"{title} - Trial {trial_idx}, Channel {channel_idx}")
        
        # Original signal
        axes[0].plot(sample_axis, original[trial_idx, channel_idx, :], 'b-', alpha=0.7, label='Original')
        axes[0].set_title('Original Signal')
        axes[0].set_ylabel('Amplitude')
        axes[0].set_xlabel('Sample')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Cleaned signal
        axes[1].plot(sample_axis, cleaned[trial_idx, channel_idx, :], 'g-', alpha=0.7, label='Cleaned')
        axes[1].set_title('Cleaned Signal')
        axes[1].set_ylabel('Amplitude')
        axes[1].set_xlabel('Sample')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        # Overlay comparison
        axes[2].plot(sample_axis, original[trial_idx, channel_idx, :], 'b-', alpha=0.5, label='Original')
        axes[2].plot(sample_axis, cleaned[trial_idx, channel_idx, :], 'g-', alpha=0.7, label='Cleaned')
        axes[2].set_title('Overlay Comparison')
        axes[2].set_xlabel('Sample')
        axes[2].set_ylabel('Amplitude')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        
        plt.tight_layout()
        plt.show()

    def compare(self, weights: Optional[Dict[str, float]] = None):
        if not self.results:
            print("No results to compare. Run the tester first.")
            return

        if weights is None:
            if self.has_ground_truth:
                weights = {
                    'snr_improvement_db': 2.0, 
                    'mua_correlation': 1.5,
                    'hit_rate': 1.5,
                    'lfp_psd_correlation': 1.0,
                    'false_positive_rate': -1.0, # Negative weight penalizes false positives
                }
            else:
                weights = {
                    # 'estimated_snr_improvement_db': 1.0,
                    'lfp_spectral_preservation': 2.0,
                    'lfp_power_preservation_ratio': 2.0,
                    # 'kurtosis_reduction': 1.0,
                    # 'rms_reduction_ratio': 0.5,
                    # 'variance_reduction_ratio': 0.5,
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
                if metric_name in ['snr_improvement_db', 'estimated_snr_improvement_db']:
                    # Only award points for POSITIVE improvement.
                    metric_score = max(0, value)
                elif metric_name == 'false_positive_rate':
                    # Penalize directly based on the rate. The negative weight handles the rest.
                    metric_score = value
                elif 'correlation' in metric_name or 'hit_rate' in metric_name or 'preservation' in metric_name:
                    # Use the value directly, but clamp at 0 to prevent negative correlations from helping.
                    metric_score = max(0, value)
                elif 'reduction' in metric_name:
                    metric_score = max(0, value)
                elif 'ratio' in metric_name:
                    if 'power_preservation' in metric_name:
                        metric_score = 1.0 - abs(1.0 - value) if not np.isnan(value) else 0
                    else:
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
    
    def load_specific_result(self, file_path: str) -> bool:
        if not file_path.endswith('_cleaned.npy'):
            if file_path.endswith('.npy'):
                cleaned_file = file_path
            else:
                possible_paths = [
                    f"{file_path}_cleaned.npy",
                    os.path.join(self.save_folder, f"{file_path}_cleaned.npy") if hasattr(self, 'save_folder') else None
                ]
                cleaned_file = None
                for path in possible_paths:
                    if path and os.path.exists(path):
                        cleaned_file = path
                        break
                if not cleaned_file:
                    print(f"❌ Could not find file for method: {file_path}")
                    return False
        else:
            cleaned_file = file_path
            
        if not os.path.exists(cleaned_file):
            print(f"❌ File not found: {cleaned_file}")
            return False
            
        method_name = os.path.basename(cleaned_file).replace("_cleaned.npy", "")
        
        try:
            print(f"📂 Loading {method_name}...")
            cleaned_signal = np.load(cleaned_file)
            self.cleaned_signals[method_name] = cleaned_signal
            
            # Load config if it exists
            config_file = cleaned_file.replace("_cleaned.npy", "_config.json")
            config = None
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    print(f"📋 Config loaded for {method_name}")
            
            # Evaluate this method and store in results
            if self.has_ground_truth:
                self.results[method_name] = self.evaluate(
                    ground_truth=self.data.ground_truth,
                    original_mixed=self.data.raw_data,
                    cleaned=cleaned_signal
                )
            else:
                self.results[method_name] = self.evaluate_without_ground_truth(
                    original=self.data.raw_data,
                    cleaned=cleaned_signal
                )
            
            print(f"✅ Successfully loaded {method_name}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load {method_name}: {str(e)}")
            return False
    
    @classmethod
    def load_saved_results(cls, data: SignalData | SignalDataWithGroundTruth, results_folder: str):
        cleaned_files = glob.glob(os.path.join(results_folder, "*_cleaned.npy"))
        if not cleaned_files:
            raise ValueError(f"No cleaned signal files found in {results_folder}")
        
        tester = cls(data, {}, save=False)
        tester.save_folder = results_folder 
        
        print(f"Loading {len(cleaned_files)} saved results from {results_folder}")
        
        corrupted_files = []
        loaded_count = 0
        start_time = time.time()
        
        pbar = tqdm(cleaned_files, desc="Loading results", unit="file")
        
        for cleaned_file in pbar:
            method_name = os.path.basename(cleaned_file).replace("_cleaned.npy", "")
            pbar.set_postfix_str(f"Processing {method_name}")
            
            success = tester.load_specific_result(cleaned_file)
            
            if success:
                loaded_count += 1
                pbar.set_postfix_str(f"✓ {method_name}")
            else:
                pbar.set_postfix_str(f"✗ {method_name} - CORRUPTED")
                corrupted_files.append((method_name, "Failed to load"))
        
        pbar.close()
        
        total_time = time.time() - start_time
        print(f"\nLoading completed in {total_time:.2f} seconds")
        print(f"Successfully loaded {loaded_count}/{len(cleaned_files)} method results")
        
        if loaded_count > 0:
            avg_time_per_file = total_time / loaded_count
            print(f"Average time per file: {avg_time_per_file:.2f} seconds")
        
        if corrupted_files:
            print(f"\n⚠️  Skipped {len(corrupted_files)} corrupted files:")
            for method_name, error in corrupted_files:
                print(f"    - {method_name}: {error}")
            
        return tester
    
    @classmethod
    def load_specific_method(cls, data: SignalData | SignalDataWithGroundTruth, file_path: str, results_folder: str = None):
        tester = cls(data, {}, save=False)
        if results_folder:
            tester.save_folder = results_folder
            
        success = tester.load_specific_result(file_path)
        
        if not success:
            raise ValueError(f"Failed to load result from: {file_path}")
            
        return tester
    