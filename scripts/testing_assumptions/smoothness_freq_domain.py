from sparc import DataHandler
from sparc.core.signal_data import SignalDataWithGroundTruth, SimulatedData, ArtifactTriggers
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from utils import extract_windows

def smoothness_freq_tv_norm(neural_windows, artifact_windows, dataset_name, fs=1000, plot=False):
    def get_avg_psd(signal_windows, fs):
        if signal_windows.shape[0] == 0 or signal_windows.shape[1] < 2:
             nperseg_dummy = min(256, 100)
             freqs_dummy = np.fft.rfftfreq(nperseg_dummy, 1/fs)
             return freqs_dummy, np.zeros_like(freqs_dummy)

        nperseg = min(256, signal_windows.shape[1])
        freqs, psd_per_window = signal.welch(signal_windows, fs=fs, nperseg=nperseg, axis=1)
        avg_psd = np.mean(psd_per_window, axis=0)
        return freqs, avg_psd

    def calculate_normalized_tv_freq(freqs, avg_psd):
        if len(avg_psd) < 2:
            return 0.0

        max_psd = np.max(avg_psd)
        if max_psd > 1e-12:
             avg_psd_norm = avg_psd / max_psd
        else:
             avg_psd_norm = avg_psd

        tv_psd = np.sum(np.abs(np.diff(avg_psd_norm)))

        normalized_tv_psd = tv_psd / (len(freqs) - 1)
        return normalized_tv_psd

    neural_freqs, neural_avg_psd = get_avg_psd(neural_windows, fs)
    artifact_freqs, artifact_avg_psd = get_avg_psd(artifact_windows, fs)

    neural_tv_freq = calculate_normalized_tv_freq(neural_freqs, neural_avg_psd)
    artifact_tv_freq = calculate_normalized_tv_freq(artifact_freqs, artifact_avg_psd)

    print(f"[{dataset_name}] Spectral Smoothness (TV Norm of avg PSD):")
    print(f"    Normalized TV Norm of Neural Spectrum: {neural_tv_freq:.6f}")
    print(f"    Normalized TV Norm of Artifact Spectrum: {artifact_tv_freq:.6f}")

    if neural_tv_freq < artifact_tv_freq:
        ratio = artifact_tv_freq / neural_tv_freq if neural_tv_freq > 1e-9 else float('inf') # Avoid division by zero
        print(f"    - Interpretation: Neural spectrum graph is smoother (lower TV) (Ratio Art/Neu: {ratio:.2f}x).")
    else:
        ratio = neural_tv_freq / artifact_tv_freq if artifact_tv_freq > 1e-9 else float('inf')
        print(f"    - Interpretation: Artifact spectrum graph is smoother or equal (lower TV) (Ratio Neu/Art: {ratio:.2f}x).")

        
    if plot:
        plt.figure(figsize=(10, 6))
        if len(neural_freqs) > 0 and len(neural_avg_psd) > 0 :
            plt.semilogy(neural_freqs, neural_avg_psd, label=f'Neural (TV={neural_tv_freq:.4f})', color='green')
        if len(artifact_freqs) > 0 and len(artifact_avg_psd) > 0:
            plt.semilogy(artifact_freqs, artifact_avg_psd, label=f'Artifact (TV={artifact_tv_freq:.4f})', color='red', linestyle='--')

        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Average Power Spectral Density (Energy per Hz)')
        plt.title(f'Spectral Smoothness Comparison: {dataset_name}')
        plt.legend()
        plt.grid(True, which="both", ls="--")
        plt.tight_layout()
        # plt.show() # Show plot immediately, or save it later
        # Construct a filename based on dataset_name and fs, replacing invalid chars
        safe_dataset_name = dataset_name.replace(' ', '_').replace('/', '_').replace(':', '')
        filename = f"psd_smoothness_{safe_dataset_name}_fs{int(fs)}.png"
        plt.savefig(filename)
        print(f"    Plot saved to {filename}")
        plt.close() # Close the figure to free memory


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
            smoothness_freq_tv_norm(clean_neural_windows, artifact_windows, f'[{dataset_name}] (window size {window_size} ms)', fs=data_obj.sampling_rate, plot=window_size > 500)