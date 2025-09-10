import numpy as np
import matplotlib.pyplot as plt
from examples.swec_ethz_template_subtraction import load_swec_ethz_data

def analyze_artifact_structure():
    """Analyze the periodic structure of synthetic artifacts."""
    print("=== Analyzing Artifact Structure ===")
    
    # Load data
    mixed_data, ground_truth, artifacts, sampling_rate = load_swec_ethz_data('research/datasets/SWEC-ETHZ/selected_clips_pat123.pkl')
    if mixed_data is None:
        return
    
    # Focus on first trial, first channel
    trial_idx = 0
    channel = 0
    
    clean_signal = ground_truth[trial_idx, :, channel]
    artifact_signal = artifacts[trial_idx, :, channel]
    
    print(f"Sampling rate: {sampling_rate} Hz")
    print(f"Signal length: {len(artifact_signal)} samples ({len(artifact_signal)/sampling_rate:.1f} s)")
    
    # Find artifact peaks to understand periodicity
    artifact_threshold = 3 * np.std(clean_signal)
    artifact_peaks = np.where(np.abs(artifact_signal) > artifact_threshold)[0]
    
    print(f"Artifact threshold: {artifact_threshold:.2f}")
    print(f"Artifact points: {len(artifact_peaks)} ({len(artifact_peaks)/len(artifact_signal)*100:.1f}%)")
    
    if len(artifact_peaks) > 1:
        # Analyze spacing between artifacts
        peak_diffs = np.diff(artifact_peaks)
        # Filter out consecutive points (same artifact)
        significant_gaps = peak_diffs[peak_diffs > 5]  # More than 5 samples apart
        
        if len(significant_gaps) > 0:
            mean_spacing = np.mean(significant_gaps)
            std_spacing = np.std(significant_gaps)
            print(f"Mean spacing between artifacts: {mean_spacing:.1f} samples ({mean_spacing/sampling_rate*1000:.1f} ms)")
            print(f"Spacing std: {std_spacing:.1f} samples")
            
            # Expected stimulation rate is 130 Hz, so period should be ~3.9 ms = ~2 samples at 512 Hz
            expected_period_samples = sampling_rate / 130
            print(f"Expected period (130 Hz stim): {expected_period_samples:.1f} samples ({1000/130:.1f} ms)")
    
    # Look at the first few artifact events in detail
    print(f"\nFirst 10 artifact peaks:")
    for i, peak_idx in enumerate(artifact_peaks[:10]):
        time_ms = peak_idx / sampling_rate * 1000
        amplitude = artifact_signal[peak_idx]
        print(f"  Peak {i+1}: t={time_ms:.1f}ms, idx={peak_idx}, amp={amplitude:.1f}")
    
    # Extract a few artifact templates manually
    template_length_ms = 20
    template_length_samples = int(template_length_ms * sampling_rate / 1000)
    print(f"\nExtracting templates of length {template_length_samples} samples ({template_length_ms} ms)")
    
    templates = []
    for i, peak_idx in enumerate(artifact_peaks[:10]):  # First 10 peaks
        # Extract template centered on peak
        start_idx = max(0, peak_idx - template_length_samples // 2)
        end_idx = min(len(artifact_signal), start_idx + template_length_samples)
        
        if end_idx - start_idx == template_length_samples:
            template = artifact_signal[start_idx:end_idx]
            templates.append(template)
            print(f"  Template {i+1}: mean={np.mean(template):.2f}, std={np.std(template):.2f}, "
                  f"min={np.min(template):.2f}, max={np.max(template):.2f}")
    
    if templates:
        # Average template
        avg_template = np.mean(templates, axis=0)
        print(f"\nAverage template: mean={np.mean(avg_template):.2f}, std={np.std(avg_template):.2f}")
        print(f"Template norm: {np.linalg.norm(avg_template):.2f}")
        
        # Plot analysis
        time_s = np.arange(len(artifact_signal)) / sampling_rate
        template_time_ms = np.arange(len(avg_template)) / sampling_rate * 1000
        
        fig, axes = plt.subplots(3, 1, figsize=(15, 10))
        
        # Plot 1: Full signal with artifact peaks marked
        axes[0].plot(time_s, clean_signal, 'b-', label='Clean', alpha=0.7)
        axes[0].plot(time_s, artifact_signal, 'r-', label='Artifact', alpha=0.7)
        axes[0].scatter(artifact_peaks[:50]/sampling_rate, artifact_signal[artifact_peaks[:50]], 
                       color='red', s=20, label='Detected Peaks', zorder=5)
        axes[0].set_title('Artifact Signal with Detected Peaks')
        axes[0].legend()
        axes[0].grid(True)
        axes[0].set_xlim(0, 0.5)  # First 500ms
        
        # Plot 2: Individual templates
        for i, template in enumerate(templates[:5]):
            axes[1].plot(template_time_ms, template, alpha=0.6, label=f'Template {i+1}')
        axes[1].set_title('Individual Artifact Templates')
        axes[1].legend()
        axes[1].grid(True)
        axes[1].set_xlabel('Time (ms)')
        
        # Plot 3: Average template
        axes[2].plot(template_time_ms, avg_template, 'purple', linewidth=2, label='Average Template')
        axes[2].set_title('Average Artifact Template')
        axes[2].legend()
        axes[2].grid(True)
        axes[2].set_xlabel('Time (ms)')
        
        plt.tight_layout()
        plt.show()
        
        return avg_template, templates
    
    return None, None

if __name__ == "__main__":
    analyze_artifact_structure()
