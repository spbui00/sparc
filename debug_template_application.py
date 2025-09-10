import numpy as np
import matplotlib.pyplot as plt
from sparc import AverageTemplateSubtraction
from examples.swec_ethz_template_subtraction import load_swec_ethz_data

def debug_template_application():
    """Debug the template application process in detail."""
    print("=== Debugging Template Application ===")
    
    # Load data
    mixed_data, ground_truth, artifacts, sampling_rate = load_swec_ethz_data('research/datasets/SWEC-ETHZ/selected_clips_pat123.pkl')
    if mixed_data is None:
        return
    
    # Initialize method
    method = AverageTemplateSubtraction(
        sampling_rate=sampling_rate,
        template_length_ms=2,
        onset_threshold=1.0,
        detection_method='amplitude'
    )
    
    # Fit on all data
    method.fit(mixed_data)
    
    # Focus on first trial, first channel for detailed analysis
    trial_idx = 0
    channel = 0
    
    print(f"\n=== Analyzing Trial {trial_idx}, Channel {channel} ===")
    
    # Get the signals
    mixed_signal = mixed_data[trial_idx, :, channel]
    clean_signal = ground_truth[trial_idx, :, channel]
    true_artifact = artifacts[trial_idx, :, channel]
    
    print(f"Mixed signal stats: mean={np.mean(mixed_signal):.2f}, std={np.std(mixed_signal):.2f}")
    print(f"Clean signal stats: mean={np.mean(clean_signal):.2f}, std={np.std(clean_signal):.2f}")
    print(f"True artifact stats: mean={np.mean(true_artifact):.2f}, std={np.std(true_artifact):.2f}")
    
    # Get artifact mask for this trial/channel
    if isinstance(method.template_indices_, list):
        artifact_mask = method.template_indices_[trial_idx][:, channel]
    else:
        artifact_mask = method.template_indices_[:, channel]
    
    print(f"Artifact mask: {np.sum(artifact_mask)} points detected")
    
    # Get the learned template
    template = method.templates_.get(channel, np.zeros(method.template_length_samples))
    print(f"Template stats: mean={np.mean(template):.6f}, std={np.std(template):.6f}, norm={np.linalg.norm(template):.6f}")
    print(f"Template length: {len(template)} samples ({len(template)/sampling_rate*1000:.1f} ms)")
    
    # Find artifact regions
    artifact_regions = method._find_artifact_regions(artifact_mask)
    print(f"Found {len(artifact_regions)} artifact regions")
    
    # Analyze each region
    for i, (start, end) in enumerate(artifact_regions[:5]):  # First 5 regions
        region_length = end - start
        segment_length = min(len(template), region_length)
        
        print(f"  Region {i+1}: [{start}:{end}] ({region_length} samples, using {segment_length})")
        
        if segment_length > 0:
            original_segment = mixed_signal[start:start + segment_length]
            template_segment = template[:segment_length]
            
            print(f"    Original segment: mean={np.mean(original_segment):.2f}, std={np.std(original_segment):.2f}")
            print(f"    Template segment: mean={np.mean(template_segment):.6f}, std={np.std(template_segment):.6f}")
            print(f"    Subtraction impact: {np.mean(np.abs(template_segment)):.6f}")
    
    # Apply template subtraction manually to see the effect
    cleaned_manual = mixed_signal.copy()
    total_subtracted = 0
    
    for start, end in artifact_regions:
        segment_length = min(len(template), end - start)
        if segment_length > 0:
            before = cleaned_manual[start:start + segment_length].copy()
            cleaned_manual[start:start + segment_length] -= template[:segment_length]
            after = cleaned_manual[start:start + segment_length]
            
            subtraction = np.sum(np.abs(before - after))
            total_subtracted += subtraction
    
    print(f"Total absolute change from subtraction: {total_subtracted:.6f}")
    
    # Compare with method's transform
    cleaned_method = method.transform(mixed_data[trial_idx:trial_idx+1])
    cleaned_method_signal = cleaned_method[0, :, channel]
    
    method_change = np.sum(np.abs(mixed_signal - cleaned_method_signal))
    print(f"Method's total change: {method_change:.6f}")
    
    # Check if they match
    manual_vs_method = np.sum(np.abs(cleaned_manual - cleaned_method_signal))
    print(f"Difference between manual and method: {manual_vs_method:.6f}")
    
    # Performance comparison
    mse_original = np.mean((mixed_signal - clean_signal)**2)
    mse_cleaned = np.mean((cleaned_method_signal - clean_signal)**2)
    mse_improvement = mse_original - mse_cleaned
    
    print(f"\nPerformance:")
    print(f"Original MSE: {mse_original:.2f}")
    print(f"Cleaned MSE: {mse_cleaned:.2f}")
    print(f"MSE improvement: {mse_improvement:.2f}")
    
    # Plot comparison
    time_s = np.arange(len(mixed_signal)) / sampling_rate
    
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    
    # Plot 1: Original signals
    axes[0].plot(time_s, clean_signal, 'b-', label='Clean', alpha=0.7)
    axes[0].plot(time_s, mixed_signal, 'r-', label='Mixed', alpha=0.7)
    axes[0].plot(time_s, true_artifact, 'g-', label='True Artifact', alpha=0.5)
    axes[0].set_title('Original Signals')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot 2: Template and artifact mask
    axes[1].plot(time_s, artifact_mask.astype(int), 'orange', label='Detected Artifacts', linewidth=2)
    # Show template at first few artifact regions
    for i, (start, end) in enumerate(artifact_regions[:3]):
        if i == 0:
            axes[1].plot(time_s[start:start+len(template)], template, 'purple', label='Template', linewidth=2)
        else:
            axes[1].plot(time_s[start:start+len(template)], template, 'purple', linewidth=2)
    axes[1].set_title('Artifact Detection and Template')
    axes[1].legend()
    axes[1].grid(True)
    
    # Plot 3: Before and after cleaning
    axes[2].plot(time_s, mixed_signal, 'r-', label='Before Cleaning', alpha=0.7)
    axes[2].plot(time_s, cleaned_method_signal, 'g-', label='After Cleaning', alpha=0.7)
    axes[2].plot(time_s, clean_signal, 'b--', label='Ground Truth', alpha=0.5)
    axes[2].set_title('Cleaning Result')
    axes[2].legend()
    axes[2].grid(True)
    
    # Plot 4: Difference signals
    difference = mixed_signal - cleaned_method_signal
    axes[3].plot(time_s, difference, 'purple', label='Subtracted Signal', alpha=0.7)
    axes[3].plot(time_s, true_artifact, 'g--', label='True Artifact', alpha=0.5)
    axes[3].set_title('What Was Subtracted vs True Artifact')
    axes[3].legend()
    axes[3].grid(True)
    axes[3].set_xlabel('Time (s)')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    debug_template_application()
