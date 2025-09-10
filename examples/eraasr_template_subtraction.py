import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sparc import TemplateSubtractionMethod, SPARCEvaluator

def load_eraasr_data(data_path='../research/datasets/eraasr-1.0.0/exampleDataTensor.mat'):
    try:
        mat_data = sio.loadmat(data_path)
        data = mat_data['data_trials_by_time_by_channels']
        
        sampling_rate = 30000
        
        print(f"Loaded ERAASR data:")
        print(f"  Shape: {data.shape}")
        print(f"  Sampling rate: {sampling_rate} Hz")
        print(f"  Duration per trial: {data.shape[1] / sampling_rate * 1000:.1f} ms")
        print(f"  Trials: {data.shape[0]}, Channels: {data.shape[2]}")
        
        return data, sampling_rate
        
    except FileNotFoundError:
        print(f"Data file not found at {data_path}")
        print("Please ensure the ERAASR data is available")
        return None, None
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None, None

def create_synthetic_ground_truth(data_shape, sampling_rate):
    n_trials, n_samples, n_channels = data_shape
    t = np.linspace(0, n_samples / sampling_rate, n_samples)
    
    ground_truth = np.zeros(data_shape)
    
    for trial in range(n_trials):
        for ch in range(n_channels):
            freq = 10 + ch * 5  # Different frequencies per channel
            ground_truth[trial, :, ch] = (
                np.sin(2 * np.pi * freq * t) +  # Main oscillation
                0.5 * np.sin(2 * np.pi * freq * 2 * t) +  # Harmonic
                0.3 * np.random.randn(n_samples)  # Noise
            )
    
    return ground_truth

def demonstrate_eraasr_template_subtraction():
    data, sampling_rate = load_eraasr_data()    
    if data is None:
        raise ValueError("Could not load ERAASR data")
    
    ground_truth = create_synthetic_ground_truth(data.shape, sampling_rate)
    
    assert sampling_rate is not None
    methods = {
        'Simple Average': TemplateSubtractionMethod(
            sampling_rate=sampling_rate,
            method='simple',
            template_length_ms=90,  # 3ms at 30kHz
        ),
        'Channel Average': TemplateSubtractionMethod(
            sampling_rate=sampling_rate,
            method='average',
            template_length_ms=90
        ),
        'Trial Specific': TemplateSubtractionMethod(
            sampling_rate=sampling_rate,
            method='trial',
            template_length_ms=90
        ),
        'Dictionary Learning': TemplateSubtractionMethod(
            sampling_rate=sampling_rate,
            method='dictionary',
            template_length_ms=90,
            distance_metric='correlation',
            min_cluster_size=2
        )
    }
    
    evaluator = SPARCEvaluator(sampling_rate)
    
    results = {}
    cleaned_signals = {}
    
    for method_name, method in methods.items():
        print(f"\n--- Testing {method_name} ---")
        
        try:
            cleaned = method.fit_transform(data)
            cleaned_signals[method_name] = cleaned
            
            metrics = evaluator.evaluate_method_comprehensive(
                ground_truth[0], data[0], cleaned[0], method_name
            )
            results[method_name] = metrics
            
            print(f"✓ {method_name} completed successfully")
            print(f"  MSE: {metrics['mse']:.6f}")
            print(f"  SNR Improvement: {metrics['snr_improvement_db']:.2f} dB")
            print(f"  Artifact Removal Ratio: {metrics['artifact_removal_ratio']:.4f}")
            print(f"  Hit Rate: {metrics['hit_rate_mean']:.4f}")
            print(f"  Miss Rate: {metrics['miss_rate_mean']:.4f}")
            print(f"  False Positive Rate: {metrics['false_positive_rate_mean']:.4f}")
            
        except Exception as e:
            print(f"✗ {method_name} failed: {str(e)}")
            continue
    
    return results, cleaned_signals, data, ground_truth

def plot_eraasr_results(data, ground_truth, cleaned_signals, trial_idx=0, channel=0):
    """Plot results for ERAASR data."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'ERAASR Template Subtraction Results (Trial {trial_idx}, Channel {channel})', fontsize=16)
    
    # Time axis
    sampling_rate = 30000
    time_ms = np.arange(data.shape[1]) / sampling_rate * 1000
    
    # Plot original and ground truth
    axes[0, 0].plot(time_ms, data[trial_idx, :, channel], 'r-', alpha=0.7, label='Original (with artifacts)')
    axes[0, 0].plot(time_ms, ground_truth[trial_idx, :, channel], 'b-', label='Synthetic Ground Truth')
    axes[0, 0].set_title('Original Signal')
    axes[0, 0].set_xlabel('Time (ms)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot different methods
    method_names = list(cleaned_signals.keys())
    colors = ['green', 'orange', 'purple']
    
    for i, (method_name, cleaned) in enumerate(cleaned_signals.items()):
        if i < 3:  # Plot up to 3 methods
            row, col = (i + 1) // 2, (i + 1) % 2
            axes[row, col].plot(time_ms, data[trial_idx, :, channel], 'r-', alpha=0.3, label='Original')
            axes[row, col].plot(time_ms, ground_truth[trial_idx, :, channel], 'b-', alpha=0.7, label='Ground Truth')
            axes[row, col].plot(time_ms, cleaned[trial_idx, :, channel], color=colors[i], label=method_name)
            axes[row, col].set_title(f'{method_name}')
            axes[row, col].set_xlabel('Time (ms)')
            axes[row, col].set_ylabel('Amplitude')
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def analyze_artifact_regions(data, sampling_rate, trial_idx=0, channel=0):
    print(f"\n=== Artifact Analysis (Trial {trial_idx}, Channel {channel}) ===")
    
    signal = data[trial_idx, :, channel]
    
    # Find high-amplitude regions (potential artifacts)
    threshold = np.std(signal) * 3
    artifact_mask = np.abs(signal) > threshold
    
    # Find artifact regions
    artifact_starts = []
    artifact_ends = []
    in_artifact = False
    
    for i, is_artifact in enumerate(artifact_mask):
        if is_artifact and not in_artifact:
            artifact_starts.append(i)
            in_artifact = True
        elif not is_artifact and in_artifact:
            artifact_ends.append(i)
            in_artifact = False
    
    if in_artifact:
        artifact_ends.append(len(artifact_mask))
    
    print(f"Found {len(artifact_starts)} artifact regions")
    
    for i, (start, end) in enumerate(zip(artifact_starts, artifact_ends)):
        start_ms = start / sampling_rate * 1000
        end_ms = end / sampling_rate * 1000
        duration_ms = end_ms - start_ms
        print(f"  Artifact {i+1}: {start_ms:.1f}ms - {end_ms:.1f}ms (duration: {duration_ms:.1f}ms)")
    
    return artifact_starts, artifact_ends

def main():
    print("ERAASR Template Subtraction Demonstration")
    print("=" * 50)
    
    results, cleaned_signals, data, ground_truth = demonstrate_eraasr_template_subtraction()
    
    if results:
        sampling_rate = 30000
        analyze_artifact_regions(data, sampling_rate)
        
        # Plot results
        print("\nGenerating comparison plots...")
        plot_eraasr_results(data, ground_truth, cleaned_signals)
        
        # Summary
        print("\n=== Summary ===")
        best_method = min(results.keys(), key=lambda x: results[x]['mse'])
        print(f"Best performing method: {best_method}")
        print(f"Best MSE: {results[best_method]['mse']:.6f}")
        print(f"Best SNR improvement: {results[best_method]['snr_improvement_db']:.2f} dB")
        print(f"Best artifact removal ratio: {results[best_method]['artifact_removal_ratio']:.4f}")
        print(f"Best LFP PSD Correlation: {results[best_method]['lfp_psd_correlation_mean']:.4f}")
        print(f"Best Hit Rate: {results[best_method]['hit_rate_mean']:.4f}")
    
    return results, cleaned_signals

if __name__ == "__main__":
    results, cleaned_signals = main()
