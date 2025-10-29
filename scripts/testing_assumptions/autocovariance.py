from sparc import DataHandler, NeuralAnalyzer
from sparc.core.signal_data import SignalDataWithGroundTruth, SimulatedData, ArtifactTriggers
import numpy as np
import matplotlib.pyplot as plt


def analyze_time(neural_windows, artifact_windows, window_info, dataset_name, fs, max_lag):
    print(f"[{dataset_name}] Window: {window_info}")
    
    analyzer = NeuralAnalyzer(sampling_rate=fs)
    channel = 0
    max_lag_samples = min(int(max_lag * fs / 1000), neural_windows.shape[2])
 
    lags_samples, avg_acf_n = analyzer.compute_avg_acf(neural_windows[:,channel,:], max_lag_samples)
    _, avg_acf_a = analyzer.compute_avg_acf(artifact_windows[:,channel,:], max_lag_samples)
    _, avg_ccf_na = analyzer.compute_avg_ccf(neural_windows[:,channel,:], artifact_windows[:,channel,:], max_lag_samples)

    lags_ms = lags_samples * 1000 / fs

    lag_0_idx = max_lag_samples # index of lag 0 is in center of the array
    lag_1_idx = max_lag_samples + 1 # index of lag 1 is after lag 0

    if lag_1_idx < len(avg_acf_n):
        print(f"[{dataset_name}] ACF Neural (lag 1 sample): {avg_acf_n[lag_1_idx]:.4f}")
        print(f"[{dataset_name}] ACF Artifact (lag 1 sample): {avg_acf_a[lag_1_idx]:.4f}")
    else:
        print(f"[{dataset_name}] ACF Neural (lag 1 sample): N/A (max_lag too small)")
        print(f"[{dataset_name}] ACF Artifact (lag 1 sample): N/A (max_lag too small)")
    
    print(f"[{dataset_name}] CCF Neural-Artifact (lag 0): {avg_ccf_na[lag_0_idx]:.4f}")

    non_zero_lag_indices = np.where(lags_samples != 0)[0]
    max_abs_ccf_non_zero_lag = np.nan
    if non_zero_lag_indices.size > 0:
         valid_indices = non_zero_lag_indices[ (non_zero_lag_indices >= 0) & (non_zero_lag_indices < len(avg_ccf_na)) ]
         if valid_indices.size > 0:
              max_abs_ccf_non_zero_lag = np.nanmax(np.abs(avg_ccf_na[valid_indices]))
              print(f"[{dataset_name}] Max abs CCF (lags != 0): {max_abs_ccf_non_zero_lag:.4f}")
         else:
              print(f"[{dataset_name}] Max abs CCF (lags != 0): N/A (no valid non-zero lag indices)")
    else:
         print(f"[{dataset_name}] Max abs CCF (lags != 0): N/A (max_lag is 0)")


    acf_diff_lag1 = np.abs(avg_acf_n[lag_1_idx] - avg_acf_a[lag_1_idx])
    if acf_diff_lag1 > 0.1:
            print(f"[{dataset_name}] -> ACFs differ noticeably!")
    else:
            print(f"[{dataset_name}] -> ACFs are similar.")
   
    if (np.isnan(max_abs_ccf_non_zero_lag) or max_abs_ccf_non_zero_lag < 0.1) and np.abs(avg_ccf_na[lag_0_idx]) < 0.1: # Thresholds are arbitrary
        print(f"[{dataset_name}] -> CCF is low across lags")
    else:
        print(f"[{dataset_name}] -> CCF shows significant correlation")

    fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    fig.suptitle(f'{dataset_name} - Correlations for {window_info}')

# Plot ACFs
    axs[0].stem(lags_ms, avg_acf_n, linefmt='g-', markerfmt='go', basefmt='k-', label=f'Neural Ch {channel}')
    axs[0].stem(lags_ms, avg_acf_a, linefmt='r:', markerfmt='rx', basefmt='k-', label=f'Artifact Ch {channel}')
    axs[0].set_title('Average Auto-Correlation Functions (ACF)')
    axs[0].set_xlabel('Lag (ms)')
    axs[0].set_ylabel('Correlation')
    axs[0].legend()
    axs[0].grid(True, alpha=0.5)
    axs[0].set_ylim([-1.05, 1.05])

# Plot CCF
    axs[1].stem(lags_ms, avg_ccf_na, linefmt='b-', markerfmt='bo', basefmt='k-')
    axs[1].set_title(f'Avg. Cross-Correlation on channel {channel}')
    axs[1].set_xlabel('Lag (ms)')
    axs[1].grid(True, alpha=0.5)
    axs[1].set_ylim([-1.05, 1.05])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
    safe_name = dataset_name.replace(' ', '_').replace('/', '_').replace(':', '').replace('[','').replace(']','')
    safe_window_info = window_info.replace(' ', '_').replace('(','').replace(')','').replace(',','')
    filename = f"figures/correlation_analysis_{safe_name}_{safe_window_info}.png"
    plt.savefig(filename)
    print(f"[{dataset_name}] Plot saved to {filename}")
    plt.close(fig)


if __name__ == "__main__":
    data_handler = DataHandler()
    
    datasets = [
        ('../../data/simulated_data_2x64_1000.npz', 'simulated data 1000Hz'),
        ('../../data/simulated_data_2x64_30000.npz', 'simulated data 30000Hz'),
        ('../../data/added_artifacts_swec_data_512.npz', 'swec data 512Hz'),
        ('../../data/added_artifacts_swec_data_seizure_512.npz', 'swec seizure data 512Hz')
    ]

    for data_path, dataset_name in datasets:
        print(f"\n{'='*60}\nProcessing: {dataset_name}\n{'='*60}")
        data_obj_dict = data_handler.load_npz_data(data_path)
        
        if 'simulated' in dataset_name:
            data_obj = SimulatedData(
                raw_data=data_obj_dict['raw_data'],
                sampling_rate=data_obj_dict['sampling_rate'],
                ground_truth=data_obj_dict['ground_truth'],
                artifacts=data_obj_dict['artifacts'],
                artifact_markers=ArtifactTriggers(starts=data_obj_dict['artifact_markers']),  
                firing_rate=data_obj_dict.get('firing_rate'),
                spike_train=data_obj_dict.get('spike_train'),
                lfp=data_obj_dict.get('lfp'),
                stim_params=None,
                snr=data_obj_dict.get('snr'),
            )
        else:
            data_obj = SignalDataWithGroundTruth(
                raw_data=data_obj_dict['mixed_data'],
                sampling_rate=data_obj_dict['sampling_rate'],
                ground_truth=data_obj_dict['ground_truth'],
                artifacts=data_obj_dict['artifacts'],
                artifact_markers=ArtifactTriggers(starts=data_obj_dict['artifact_markers']),
            )
                    
        original_neural_tensor = data_obj.ground_truth
        original_artifact_tensor = data_obj.artifacts
        fs = data_obj.sampling_rate
            
        w, c, n_total_in_trial = original_artifact_tensor.shape
        max_time_samples = data_obj.raw_data.shape[-1]
        window_sizes_ms = [10, 100, 200, 500, int(1000 * max_time_samples / fs)] # ms

        for window_size_ms in window_sizes_ms:            
            n_samples_window = int(window_size_ms * fs / 1000)
            
            if n_samples_window == 0:
                print(f"[{dataset_name}] Window size {window_size_ms}ms is too small.")
                continue
            
            truncated_neural = original_neural_tensor[:, :, :n_samples_window]
            truncated_artifact = original_artifact_tensor[:, :, :n_samples_window]
            window_info = f"{window_size_ms} ms ({n_samples_window} samples)"
            analyze_time(truncated_neural, truncated_artifact, window_info, dataset_name, fs=data_obj.sampling_rate, max_lag=window_size_ms)
