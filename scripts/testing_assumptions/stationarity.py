from sparc import DataHandler, NeuralAnalyzer
from sparc.core.signal_data import SignalDataWithGroundTruth, SimulatedData, ArtifactTriggers
import numpy as np
from statsmodels.tsa.stattools import adfuller


def analyze_stationarity(neural_windows, artifact_windows, window_info, dataset_name, fs, significance_level=0.05):
    print(f"--- Stationarity Analysis for [{dataset_name}] Window: {window_info} ---")
    num_trials, num_channels, num_samples = neural_windows.shape
    channel_to_analyze = 0

    if num_samples < 5: # ADF test needs a minimum number of samples
        print(f"Skipping ADF test: Window too short ({num_samples} samples).")
        return

    print(f"Analyzing Channel: {channel_to_analyze}")

    print("\nNeural Signal Stationarity (ADF Test):")
    stationary_neural_trials = 0
    for trial in range(num_trials):
        time_series = neural_windows[trial, channel_to_analyze, :]
        try:
            # Run ADF test
            # autolag='AIC' lets the test automatically select the optimal lag length
            result = adfuller(time_series, autolag='AIC')
            p_value = result[1]

            print(f"  Trial {trial + 1}: ADF Statistic={result[0]:.4f}, p-value={p_value:.4f}", end="")
            if p_value <= significance_level:
                print(" -> Likely Stationary (Reject H0)")
                stationary_neural_trials += 1
            else:
                print(" -> Likely Non-Stationary (Fail to Reject H0)")
        except Exception as e:
            print(f"  Trial {trial + 1}: Error running ADF test - {e}")
            if np.all(time_series == time_series[0]):
                 print("     Note: Series is constant. ADF may not be appropriate, technically stationary.")
            elif len(np.unique(time_series)) < 2:
                 print("     Note: Series has very few unique values. ADF may not be appropriate.")


    print("\nArtifact Signal Stationarity (ADF Test):")
    stationary_artifact_trials = 0
    for trial in range(num_trials):
        time_series = artifact_windows[trial, channel_to_analyze, :]
        try:
            result = adfuller(time_series, autolag='AIC')
            p_value = result[1]

            print(f"  Trial {trial + 1}: ADF Statistic={result[0]:.4f}, p-value={p_value:.4f}", end="")
            if p_value <= significance_level:
                print(" -> Likely Stationary (Reject H0)")
                stationary_artifact_trials += 1
            else:
                print(" -> Likely Non-Stationary (Fail to Reject H0)")
        except Exception as e:
            print(f"  Trial {trial + 1}: Error running ADF test - {e}")
            if np.all(time_series == time_series[0]):
                 print("     Note: Series is constant. ADF may not be appropriate, technically stationary.")
            elif len(np.unique(time_series)) < 2:
                 print("     Note: Series has very few unique values. ADF may not be appropriate.")

    print("\nSummary:")
    print(f"  Neural: {stationary_neural_trials}/{num_trials} trials likely stationary (p <= {significance_level}).")
    print(f"  Artifact: {stationary_artifact_trials}/{num_trials} trials likely stationary (p <= {significance_level}).")
    print("-" * (len(window_info) + len(dataset_name) + 33))

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
            analyze_stationarity(truncated_neural, truncated_artifact, window_info, dataset_name, fs=fs)
