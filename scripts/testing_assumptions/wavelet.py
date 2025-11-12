from sparc import DataHandler
from sparc.core.signal_data import SignalDataWithGroundTruth, SimulatedData, ArtifactTriggers
import numpy as np
import matplotlib.pyplot as plt
import pywt
from utils import extract_windows


def compute_wavelet_energy_distribution(signal_windows, wavelet='db4', level=None):
    if signal_windows.shape[0] == 0 or signal_windows.shape[1] < 4:
        return None, None
    
    if level is None:
        level = min(pywt.dwt_max_level(signal_windows.shape[1], wavelet), 6)
    
    energy_per_level = []
    
    for window in signal_windows:
        coeffs = pywt.wavedec(window, wavelet, level=level)
        
        energies = [np.sum(c**2) for c in coeffs]
        energy_per_level.append(energies)
    
    energy_per_level = np.array(energy_per_level)
    avg_energy_per_level = np.mean(energy_per_level, axis=0)
    
    total_energy = np.sum(avg_energy_per_level)
    if total_energy > 1e-12:
        energy_distribution = avg_energy_per_level / total_energy
    else:
        energy_distribution = avg_energy_per_level
    
    return energy_distribution, level


def compute_wavelet_sparsity(signal_windows, wavelet='db4', level=None):
    if signal_windows.shape[0] == 0 or signal_windows.shape[1] < 4:
        return 0.0, 0.0
    
    if level is None:
        level = min(pywt.dwt_max_level(signal_windows.shape[1], wavelet), 6)
    
    all_coeffs = []
    
    for window in signal_windows:
        coeffs = pywt.wavedec(window, wavelet, level=level)
        all_coeffs.extend([c for c_array in coeffs for c in c_array])
    
    all_coeffs = np.array(all_coeffs)
    
    threshold = 0.1 * np.std(all_coeffs)
    sparsity = np.sum(np.abs(all_coeffs) < threshold) / len(all_coeffs)
    
    kurtosis = np.mean((all_coeffs - np.mean(all_coeffs))**4) / (np.std(all_coeffs)**4 + 1e-12)
    
    return sparsity, kurtosis


def compute_wavelet_entropy(signal_windows, wavelet='db4', level=None):
    if signal_windows.shape[0] == 0 or signal_windows.shape[1] < 4:
        return 0.0
    
    if level is None:
        level = min(pywt.dwt_max_level(signal_windows.shape[1], wavelet), 6)
    
    entropies = []
    
    for window in signal_windows:
        coeffs = pywt.wavedec(window, wavelet, level=level)
        all_coeffs = np.concatenate([c for c in coeffs])
        
        energy = all_coeffs**2
        total_energy = np.sum(energy)
        
        if total_energy > 1e-12:
            prob = energy / total_energy
            prob = prob[prob > 1e-12]
            entropy = -np.sum(prob * np.log(prob))
        else:
            entropy = 0.0
        
        entropies.append(entropy)
    
    return np.mean(entropies)


def wavelet_analysis(neural_windows, artifact_windows, dataset_name, wavelet='db4', plot=False):
    print(f"\n[{dataset_name}] Wavelet Analysis (wavelet={wavelet}):")
    print(f"    Neural windows: {neural_windows.shape}, Artifact windows: {artifact_windows.shape}")
    
    neural_energy_dist, neural_level = compute_wavelet_energy_distribution(neural_windows, wavelet)
    artifact_energy_dist, artifact_level = compute_wavelet_energy_distribution(artifact_windows, wavelet)
    
    if neural_energy_dist is not None and artifact_energy_dist is not None:
        print(f"    Decomposition levels: {neural_level}")
        print(f"    Neural energy distribution: {neural_energy_dist}")
        print(f"    Artifact energy distribution: {artifact_energy_dist}")
        
        dominant_neural_level = np.argmax(neural_energy_dist)
        dominant_artifact_level = np.argmax(artifact_energy_dist)
        
        print(f"    Dominant scale - Neural: level {dominant_neural_level} ({neural_energy_dist[dominant_neural_level]:.3f})")
        print(f"    Dominant scale - Artifact: level {dominant_artifact_level} ({artifact_energy_dist[dominant_artifact_level]:.3f})")
        
        energy_diff = np.sum(np.abs(neural_energy_dist - artifact_energy_dist))
        print(f"    Energy distribution difference (L1): {energy_diff:.4f}")
    
    neural_sparsity, neural_kurtosis = compute_wavelet_sparsity(neural_windows, wavelet)
    artifact_sparsity, artifact_kurtosis = compute_wavelet_sparsity(artifact_windows, wavelet)
    
    print(f"    Neural sparsity: {neural_sparsity:.4f}, kurtosis: {neural_kurtosis:.4f}")
    print(f"    Artifact sparsity: {artifact_sparsity:.4f}, kurtosis: {artifact_kurtosis:.4f}")
    
    if neural_sparsity > artifact_sparsity:
        print(f"    - Neural coefficients are MORE sparse (ratio: {neural_sparsity/artifact_sparsity:.2f}x)")
    else:
        print(f"    - Artifact coefficients are MORE sparse (ratio: {artifact_sparsity/neural_sparsity:.2f}x)")
    
    neural_entropy = compute_wavelet_entropy(neural_windows, wavelet)
    artifact_entropy = compute_wavelet_entropy(artifact_windows, wavelet)
    
    print(f"    Wavelet entropy - Neural: {neural_entropy:.4f}, Artifact: {artifact_entropy:.4f}")
    
    if neural_entropy < artifact_entropy:
        print(f"    - Neural signal is more ORGANIZED (lower entropy, ratio: {artifact_entropy/neural_entropy:.2f}x)")
    else:
        print(f"    - Artifact signal is more ORGANIZED (lower entropy, ratio: {neural_entropy/artifact_entropy:.2f}x)")
    
    if plot and neural_energy_dist is not None and artifact_energy_dist is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        level_labels = [f'A{neural_level}'] + [f'D{i}' for i in range(neural_level, 0, -1)]
        x = np.arange(len(level_labels))
        width = 0.35
        
        axes[0].bar(x - width/2, neural_energy_dist, width, label='Neural', color='green', alpha=0.7)
        axes[0].bar(x + width/2, artifact_energy_dist, width, label='Artifact', color='red', alpha=0.7)
        axes[0].set_xlabel('Wavelet Decomposition Level')
        axes[0].set_ylabel('Normalized Energy')
        axes[0].set_title(f'Energy Distribution Across Scales\n{dataset_name}')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(level_labels)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        metrics = ['Sparsity', 'Kurtosis', 'Entropy']
        neural_values = [neural_sparsity, neural_kurtosis/10, neural_entropy]
        artifact_values = [artifact_sparsity, artifact_kurtosis/10, artifact_entropy]
        
        x = np.arange(len(metrics))
        axes[1].bar(x - width/2, neural_values, width, label='Neural', color='green', alpha=0.7)
        axes[1].bar(x + width/2, artifact_values, width, label='Artifact', color='red', alpha=0.7)
        axes[1].set_ylabel('Value')
        axes[1].set_title('Wavelet Coefficient Properties')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(metrics)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        safe_dataset_name = dataset_name.replace(' ', '_').replace('/', '_').replace(':', '')
        filename = f"figures/wavelet_analysis_{safe_dataset_name}.png"
        plt.savefig(filename, dpi=150)
        print(f"    Plot saved to {filename}")
        plt.close()


if __name__ == "__main__":
    data_handler = DataHandler()

    datasets = [
        ('../../data/simulated_data_2x64_1000.npz', 'simulated data 1000Hz'),
        ('../../data/simulated_data_2x64_30000.npz', 'simulated data 30000Hz'),
        ('../../data/added_artifacts_swec_data_512.npz', 'swec data 512Hz'),
        ('../../data/added_artifacts_swec_data_seizure_512.npz', 'swec seizure data 512Hz')
    ]

    wavelets = ['db4', 'sym5', 'coif3']

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

        window_sizes = [10, 100, 200, 500, int(1000 * data_obj.raw_data.shape[2] / data_obj.sampling_rate)]
        
        for window_size in window_sizes:
            clean_neural_windows, artifact_windows = extract_windows(data_obj, window_size)
            
            for wavelet in wavelets:
                plot_flag = (window_size > 100 and wavelet == 'db4')
                wavelet_analysis(
                    clean_neural_windows, 
                    artifact_windows, 
                    f'{dataset_name} (window {window_size}ms, {wavelet})',
                    wavelet=wavelet,
                    plot=plot_flag
                )

