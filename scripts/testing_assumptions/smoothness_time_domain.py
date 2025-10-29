from sparc import DataHandler, NeuralAnalyzer
from sparc.core.signal_data import SignalDataWithGroundTruth, SimulatedData, ArtifactTriggers
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from utils import extract_windows


def smoothness_time_domain(neural_windows, artifact_windows, dataset_name, freq_cutoff_hz=200.0, fs=1000):
    def get_energy_concentration_windowed(signal_windows, fs, freq_cutoff):
        if signal_windows.shape[0] == 0:
            return 0, np.array([]), np.array([])
            
        nperseg = min(256, signal_windows.shape[1])
        freqs = None
        psd_per_window = []
        
        for i in range(signal_windows.shape[0]):
            window_data = signal_windows[i, :]
            f, psd = signal.welch(window_data, fs=fs, nperseg=nperseg)
            if freqs is None:
                freqs = f
            psd_per_window.append(psd)
        
        psd_per_window = np.array(psd_per_window)
        low_freq_indices = freqs < freq_cutoff
        
        total_power_per_window = np.sum(psd_per_window, axis=1)
        low_freq_power_per_window = np.sum(psd_per_window[:, low_freq_indices], axis=1)
        ratios = np.divide(low_freq_power_per_window, total_power_per_window, 
                           out=np.ones_like(total_power_per_window), 
                           where=total_power_per_window != 0)
        
        avg_psd = np.mean(psd_per_window, axis=0)
        
        return np.mean(ratios), freqs, avg_psd

    neural_concentration, neural_freqs, neural_psd = get_energy_concentration_windowed(neural_windows, fs, freq_cutoff_hz)
    artifact_concentration, artifact_freqs, artifact_psd = get_energy_concentration_windowed(artifact_windows, fs, freq_cutoff_hz)

    print(f"[{dataset_name}] Neural Signal: {neural_concentration:.2%} of energy is below {freq_cutoff_hz} Hz.")
    print(f"[{dataset_name}] Artifact Signal: {artifact_concentration:.2%} of energy is below {freq_cutoff_hz} Hz.")

    if neural_concentration > 0.90:
        print("    - Result: Neural signal is smooth.")
    else:
        print("    - Result: Neural signal is not smooth.")
        
    if artifact_concentration < neural_concentration:
        print("    - Result: Artifact is less smooth than the neural signal.")
    else:
         print("    - Result: Artifact is smoother than the neural signal.")


def smoothness_tv_norm(neural_windows, artifact_windows, dataset_name):
    def calculate_normalized_tv_windowed(signal_windows):
        if signal_windows.shape[0] == 0 or signal_windows.shape[1] < 2:
            return 0
        
        tv_per_window = np.sum(np.abs(np.diff(signal_windows, axis=1)), axis=1)
        normalized_tv = np.mean(tv_per_window) / (signal_windows.shape[1] - 1)
        return normalized_tv
    
    neural_tv = calculate_normalized_tv_windowed(neural_windows)
    artifact_tv = calculate_normalized_tv_windowed(artifact_windows)

    print(f"[{dataset_name}] Normalized TV Norm of Neural Signal: {neural_tv:.6f}")
    print(f"[{dataset_name}] Normalized TV Norm of Artifact Signal: {artifact_tv:.6f}")

    if neural_tv < artifact_tv:
        ratio = artifact_tv / neural_tv if neural_tv > 0 else float('inf')
        print(f"    - Result: Neural signal is smoother than the artifact (TV norm is {ratio:.2f}x lower).")
        print("      This strongly supports the TV-based separation assumption.")
    else:
        ratio = neural_tv / artifact_tv if artifact_tv > 0 else float('inf')
        print(f"    - Result: WARNING: Artifact is smoother than the neural signal (TV norm is {ratio:.2f}x lower).")
        print("      This may violate the TV-based separation assumption.")


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

        window_sizes = [10, 100, 200, 500, int(1000 * data_obj.raw_data.shape[2] / data_obj.sampling_rate)] # ms
        for window_size in window_sizes:
            clean_neural_windows, artifact_windows = extract_windows(data_obj, window_size)
            smoothness_time_domain(clean_neural_windows, artifact_windows, f'[{dataset_name}] (window size {window_size} ms)', fs=data_obj.sampling_rate)
            smoothness_tv_norm(clean_neural_windows, artifact_windows, f'[{dataset_name}] (window size {window_size} ms)')
