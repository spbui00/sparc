from sparc import DataHandler, NeuralAnalyzer
from sparc.core.signal_data import SignalDataWithGroundTruth, SimulatedData, ArtifactTriggers
import numpy as np
import matplotlib.pyplot as plt


def smoothness_freq_domain(data_obj, dataset_name, freq_cutoff_hz=200.0):
    neural_signal = data_obj.ground_truth
    artifact_signal = data_obj.artifacts
    fs = data_obj.sampling_rate

    analyzer = NeuralAnalyzer(sampling_rate=fs)

    def get_energy_concentration(signal_3d, analyzer, freq_cutoff):
        if signal_3d.shape[-1] == 0:
            return 0, np.array([]), np.array([])
            
        nperseg = min(256, signal_3d.shape[-1])
        freqs, psd_per_channel = analyzer.compute_psd(signal_3d, nperseg=nperseg)
        low_freq_indices = freqs < freq_cutoff
        
        total_power_per_channel = np.sum(psd_per_channel, axis=1)
        low_freq_power_per_channel = np.sum(psd_per_channel[:, low_freq_indices], axis=1)
        ratios = np.divide(low_freq_power_per_channel, total_power_per_channel, 
                           out=np.ones_like(total_power_per_channel), 
                           where=total_power_per_channel != 0)
        
        avg_psd = np.mean(psd_per_channel, axis=0)
        
        return np.mean(ratios), freqs, avg_psd

    neural_concentration, neural_freqs, neural_psd = get_energy_concentration(neural_signal, analyzer, freq_cutoff_hz)
    artifact_concentration, artifact_freqs, artifact_psd = get_energy_concentration(artifact_signal, analyzer, freq_cutoff_hz)

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

    plt.figure(figsize=(10, 6))
    if len(neural_freqs) > 0:
        plt.semilogy(neural_freqs, neural_psd, label='Neural Signal', color='green', alpha=0.8)
    if len(artifact_freqs) > 0:
        plt.semilogy(artifact_freqs, artifact_psd, label='Artifact Signal', color='red', alpha=0.8)
    plt.axvline(x=freq_cutoff_hz, color='k', linestyle='--', label=f'{freq_cutoff_hz} Hz Cutoff')
    plt.title(f'Average Power Spectral Density (Welch\'s Method) - {dataset_name}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density (log scale)')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.xlim(0, fs / 2)
    plt.show()


def smoothness_tv_norm(data_obj, dataset_name):
    neural_signal = data_obj.ground_truth
    artifact_signal = data_obj.artifacts

    def calculate_normalized_tv(signal_3d):
        if signal_3d.shape[-1] < 2:
            return 0        
        tv_per_trace = np.sum(np.abs(np.diff(signal_3d, axis=2)), axis=2)
        normalized_tv = np.mean(tv_per_trace) / (signal_3d.shape[2] - 1)
        return normalized_tv
    
    neural_tv = calculate_normalized_tv(neural_signal)
    artifact_tv = calculate_normalized_tv(artifact_signal)

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
    data_obj = data_handler.load_npz_data('../../data/simulated_data_2x64_1000.npz')
    data_obj = SimulatedData(
        raw_data=data_obj['raw_data'],
        sampling_rate=data_obj['sampling_rate'],
        ground_truth=data_obj['ground_truth'],
        artifacts=data_obj['artifacts'],
        artifact_markers=ArtifactTriggers(starts=data_obj['artifact_markers']),  
        firing_rate=data_obj['firing_rate'],
        spike_train=data_obj['spike_train'],
        lfp=data_obj['lfp'],
        stim_params=None,
        snr=data_obj['snr'],
    )
    smoothness_freq_domain(data_obj, 'simulated data 1000Hz')
    smoothness_tv_norm(data_obj, 'simulated data 1000Hz')
    print("-" * 50)

    data_obj = data_handler.load_npz_data('../../data/simulated_data_2x64_30000.npz')
    data_obj = SimulatedData(
        raw_data=data_obj['raw_data'],
        sampling_rate=data_obj['sampling_rate'],
        ground_truth=data_obj['ground_truth'],
        artifacts=data_obj['artifacts'],
        artifact_markers=ArtifactTriggers(starts=data_obj['artifact_markers']),  
        firing_rate=data_obj['firing_rate'],
        spike_train=data_obj['spike_train'],
        lfp=data_obj['lfp'],
        stim_params=None,
        snr=data_obj['snr'],
    )
    smoothness_freq_domain(data_obj, 'simulated data 30000Hz')
    smoothness_tv_norm(data_obj, 'simulated data 30000Hz')
    print("-" * 50)

    data_obj = data_handler.load_npz_data('../../data/added_artifacts_swec_data_512.npz')
    artifact_markers_data = data_obj['artifact_markers']
    data_obj = SignalDataWithGroundTruth(
        raw_data=data_obj['mixed_data'],
        sampling_rate=data_obj['sampling_rate'],
        ground_truth=data_obj['ground_truth'],
        artifacts=data_obj['artifacts'],
        artifact_markers=ArtifactTriggers(starts=artifact_markers_data)
    )
    smoothness_freq_domain(data_obj, 'swec data (512 Hz)')
    smoothness_tv_norm(data_obj, 'swec data (512 Hz)')
    print("-" * 50)