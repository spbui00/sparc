import numpy as np
import matplotlib.pyplot as plt
from sparc import MethodTester, AverageTemplateSubtraction

def plot_results(tester: MethodTester, trial_idx=0, channel=0, zoom_ms=50):
    """Plot ERAASR template subtraction results"""
    
    results = tester.get_results()
    if not results:
        print("No results to plot")
        return

    mixed_data = tester.data.raw_data
    sampling_rate = tester.data.sampling_rate
    
    # stim info - this should ideally be part of the data object
    stim_start = 1500
    stim_end = stim_start + int(60 * sampling_rate / 1000)

    # Create time vector
    n_timesteps = mixed_data.shape[1]
    time_ms = np.arange(n_timesteps) / sampling_rate * 1000 - 50  # Relative to stim onset
    
    # Determine zoom window
    stim_onset_ms = stim_start / sampling_rate * 1000 - 50
    zoom_start_ms = stim_onset_ms - zoom_ms/2
    zoom_end_ms = stim_onset_ms + zoom_ms/2
    zoom_mask = (time_ms >= zoom_start_ms) & (time_ms <= zoom_end_ms)
    
    # Create plots
    n_methods = len(results)
    fig, axes = plt.subplots(n_methods + 1, 2, figsize=(15, 4 * (n_methods + 1)))
    
    fig.suptitle(f'ERAASR Template Subtraction Results\n(Trial {trial_idx}, Channel {channel})', fontsize=16)
    
    # Plot 1: Original signal (full and zoomed)
    ax_full, ax_zoom = axes[0]
    ax_full.plot(time_ms, mixed_data[trial_idx, :, channel], 'b-', alpha=0.7, linewidth=0.5)
    ax_full.axvspan(stim_start/sampling_rate*1000-50, stim_end/sampling_rate*1000-50, alpha=0.2, color='red', label='Stimulation period')
    ax_full.set_title('Original Signal (Full View)')
    ax_full.set_ylabel('Amplitude (μV)')
    ax_full.legend()
    ax_full.grid(True, alpha=0.3)
    
    ax_zoom.plot(time_ms[zoom_mask], mixed_data[trial_idx, zoom_mask, channel], 'b-', alpha=0.7)
    ax_zoom.axvspan(stim_start/sampling_rate*1000-50, stim_end/sampling_rate*1000-50, alpha=0.2, color='red', label='Stimulation period')
    ax_zoom.set_title(f'Original Signal (Zoomed: ±{zoom_ms/2:.0f}ms around stim)')
    ax_zoom.set_ylabel('Amplitude (μV)')
    ax_zoom.legend()
    ax_zoom.grid(True, alpha=0.3)
    
    # Plot cleaned signals
    colors = ['green', 'purple', 'orange']
    for i, (method_name, result) in enumerate(results.items()):
        cleaned = result['cleaned_data']
        ax_full, ax_zoom = axes[i + 1]
        color = colors[i % len(colors)]
        
        ax_full.plot(time_ms, mixed_data[trial_idx, :, channel], 'b-', alpha=0.3, linewidth=0.5, label='Original')
        ax_full.plot(time_ms, cleaned[trial_idx, :, channel], color=color, alpha=0.8, linewidth=0.5, label=f'Cleaned ({method_name})')
        ax_full.axvspan(stim_start/sampling_rate*1000-50, stim_end/sampling_rate*1000-50, alpha=0.2, color='red')
        ax_full.set_title(f'After {method_name} (Full View)')
        ax_full.set_ylabel('Amplitude (μV)')
        ax_full.legend()
        ax_full.grid(True, alpha=0.3)
        
        ax_zoom.plot(time_ms[zoom_mask], mixed_data[trial_idx, zoom_mask, channel], 'b-', alpha=0.3, label='Original')
        ax_zoom.plot(time_ms[zoom_mask], cleaned[trial_idx, zoom_mask, channel], color=color, alpha=0.8, label=f'Cleaned ({method_name})')
        ax_zoom.axvspan(stim_start/sampling_rate*1000-50, stim_end/sampling_rate*1000-50, alpha=0.2, color='red')
        ax_zoom.set_title(f'After {method_name} (Zoomed)')
        ax_zoom.set_ylabel('Amplitude (μV)')
        ax_zoom.legend()
        ax_zoom.grid(True, alpha=0.3)
    
    for ax in axes[-1]:
        ax.set_xlabel('Time relative to stim onset (ms)')
    
    plt.tight_layout()
    plt.show()

def main():
    """Run ERAASR template subtraction demonstration"""
    
    print("ERAASR Template Subtraction Demonstration")
    print("=" * 50)
    
    # 1. Define methods to test
    methods = {
        '2ms_template': AverageTemplateSubtraction(
            sampling_rate=30000, # this should be retrieved from data, but for now we know it
            template_length_ms=2,
            num_templates_for_avg=5
        ),
        '5ms_template': AverageTemplateSubtraction(
            sampling_rate=30000,
            template_length_ms=5,
            num_templates_for_avg=5
        )
    }
    
    # 2. Initialize the MethodTester
    tester = MethodTester(
        data_path='research/datasets/eraasr-1.0.0/exampleDataTensor.mat',
        methods=methods,
        data_loader="eraasr"
    )
    
    # 3. Run the tests
    tester.run()
    
    # 4. Get and print results
    results = tester.get_results()
    
    print("\n=== Test Results ===")
    for method_name, result in results.items():
        print(f"\n--- {method_name} ---")
        print(f"  Overall Power Reduction: {result['power_reduction']:.2f}%")
        print(f"  LFP Power Reduction: {result['lfp_power_reduction']:.2f}%")
        print(f"  MUA Power Reduction: {result['mua_power_reduction']:.2f}%")
        if "mse" in result:
            print(f"  MSE: {result['mse']:.2f}")
        if "spike_metrics" in result:
            print("  Spike Metrics (Channel 0):")
            metrics = result['spike_metrics']['channel_0']
            print(f"    Hits: {metrics['hits']}")
            print(f"    Misses: {metrics['misses']}")
            print(f"    False Positives: {metrics['false_positives']}")
            print(f"    Precision: {metrics['precision']:.2f}")
            print(f"    Recall: {metrics['recall']:.2f}")

    # 5. Plot the results
    plot_results(tester)

if __name__ == "__main__":
    main()
