import numpy as np
import matplotlib.pyplot as plt
from sparc import DataHandler
from sparc.core.signal_data import SignalDataWithGroundTruth, SimulatedData, ArtifactTriggers
from sparc.core.neural_analyzer import NeuralAnalyzer


def analyze_spectral(neural_windows_3d, fs):
    n_samples = neural_windows_3d.shape[-1]
    
    if n_samples == 0:
        return np.array([]), np.array([])
        
    fft = np.fft.rfft(neural_windows_3d, axis=-1)
    psd = np.abs(fft)**2
    
    freqs = np.fft.rfftfreq(n_samples, d=1.0/fs)
    
    return freqs, psd

def spectral_flatness(log_psd: np.ndarray) -> np.ndarray:
    # Geometric mean / Arithmetic mean in log-space
    # log( (prod(psd))^(1/N) / (sum(psd)/N) ) = mean(log_psd) - log(mean(exp(log_psd)))

    # We use exp(mean(log_psd)) for geometric mean to avoid underflow
    mean_log_psd = np.mean(log_psd, axis=-1)
    geom_mean = np.exp(mean_log_psd)

    # Arithmetic mean of the original PSD
    psd = np.exp(log_psd)
    arith_mean = np.mean(psd, axis=-1)

    # Calculate flatness (and ensure stability)
    flatness = geom_mean / (arith_mean + 1e-8)
    return np.mean(flatness)

def spectral_smoothness_loss(log_psd: np.ndarray) -> np.ndarray:
    diff = log_psd[..., 1:] - log_psd[..., :-1]
    return np.mean(np.abs(diff))

def analyze_log_log_spectrum(freqs: np.ndarray, log_psd: np.ndarray) -> tuple[float, float]:
    if len(freqs) < 2:
        return np.nan, np.nan
        
    X = np.log(freqs[1:])
    Y = log_psd[1:]
    
    if X.size == 0 or Y.size == 0:
        return np.nan, np.nan

    # Fit a 1st-degree polynomial (a line)
    try:
        slope, intercept = np.polyfit(X, Y, 1)
        
        # Calculate the predicted Y values
        Y_pred = slope * X + intercept
        
        # Calculate the Mean Squared Error of the fit
        mse = np.mean((Y - Y_pred)**2)
        
        return slope, mse
    except np.linalg.LinAlgError:
        return np.nan, np.nan

def plot_log_log_psd(freqs, mean_psd, slope, mse, dataset_name, window_info, signal_type, color):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.loglog(freqs[1:], mean_psd[1:], linewidth=1.5, color=color, label='Actual Log-Log PSD')
    
    if not np.isnan(slope) and not np.isnan(mse):
        log_freqs_fit = freqs[1:]
        # Calculate intercept from mean values (more stable)
        intercept = np.mean(np.log(mean_psd[1:])) - slope * np.mean(np.log(log_freqs_fit))
        log_psd_fit = np.exp(slope * np.log(log_freqs_fit) + intercept)
        
        ax.loglog(log_freqs_fit, log_psd_fit, 'r--', 
                  label=f'Best-Fit Line\nSlope: {slope:.2f}\nMSE: {mse:.2f}')
    
    ax.set_xlabel('Frequency (Hz) [Log Scale]')
    ax.set_ylabel('Power Spectral Density [Log Scale]')
    ax.set_title(f'Log-Log PSD Fit - {signal_type} - {dataset_name} ({window_info})')
    ax.grid(True, which="both", alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend()
    plt.tight_layout()

def perform_spectral_analysis(data_windows_3d, dataset_name, window_info, fs, signal_type="Neural", plot=False):
    if data_windows_3d is None or data_windows_3d.ndim != 3:
        print(f"[{dataset_name} @ {window_info}] Invalid 3D data for analysis.")
        return

    analyzer = NeuralAnalyzer(fs)
    w, c, n = data_windows_3d.shape # (trials, channels, samples)
    nperseg = min(n, 256)
    freqs, psd = analyzer.compute_psd(data_windows_3d, nperseg=nperseg)

    mean_psd = np.mean(psd, axis=0)  # shape: (n_freq_bins,)
    cumulative_power = np.cumsum(mean_psd)

    total_power = cumulative_power[-1]
    if total_power == 0:
        print(f"[{dataset_name} @ {window_info}] {signal_type}: Total power is zero.")
        print()
        return

    # --- Define log_mean_psd once for all helpers ---
    log_mean_psd = np.log(mean_psd + 1e-16)

    threshold_95 = 0.95 * total_power
    idx_95 = np.searchsorted(cumulative_power, threshold_95)
    freq_95 = freqs[idx_95] if idx_95 < len(freqs) else freqs[-1]
    print(f"[{dataset_name} @ {window_info}] {signal_type}: 95% of total power is below {freq_95:.2f} Hz")

    idx_200 = np.searchsorted(freqs, 200)
    if idx_200 < len(cumulative_power):
        power_below_200 = cumulative_power[idx_200]
    else:
        power_below_200 = cumulative_power[-1]
    percent_below_200 = 100 * power_below_200 / total_power
    print(f"[{dataset_name} @ {window_info}] {signal_type}: {percent_below_200:.2f}% of total power is below 200 Hz")

    spectral_flatness_value = spectral_flatness(log_mean_psd)
    print(f"[{dataset_name} @ {window_info}] {signal_type}: Spectral flatness: {spectral_flatness_value:.4f}")

    spectral_smoothness_loss_value = spectral_smoothness_loss(log_mean_psd)
    print(f"[{dataset_name} @ {window_info}] {signal_type}: Spectral smoothness loss: {spectral_smoothness_loss_value:.4f}")

    # --- ADDED: Log-Log Spectral Analysis ---
    log_log_slope, log_log_mse = analyze_log_log_spectrum(freqs, log_mean_psd)
    print(f"[{dataset_name} @ {window_info}] {signal_type}: Log-Log Spectral Slope: {log_log_slope:.4f}")
    print(f"[{dataset_name} @ {window_info}] {signal_type}: Log-Log Spectral Fit (MSE): {log_log_mse:.4f}")

    color = '#1f77b4' if signal_type == "Neural" else '#d62728'
    if plot:
        # Original semilogy plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.semilogy(freqs, mean_psd, linewidth=1.5, color=color)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power Spectral Density')
        ax.set_title(f'PSD - {signal_type} - {dataset_name} ({window_info})')
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.show()

        # --- ADDED: Log-Log PSD Plot ---
        plot_log_log_psd(freqs, mean_psd, log_log_slope, log_log_mse,
                         dataset_name, window_info, signal_type, color)
        plt.show()

    print()

if __name__ == "__main__":
    data_handler = DataHandler()

    datasets = [
        # ('../../data/simulated_data_2x64_1000.npz', 'simulated data 1000Hz'),
        # ('../../data/simulated_data_2x64_30000.npz', 'simulated data 30000Hz'),
        ('../../data/added_artifacts_swec_data_512.npz', 'swec data 512Hz'),
        ('../../data/added_artifacts_swec_data_seizure_512.npz', 'swec seizure data 512Hz'),
        ('../../data/added_artifacts_swec_data_seizure_512_lower_freq.npz', 'swec seizure data 512Hz lower freq')
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

        w, c, n_total_in_trial = original_neural_tensor.shape
        max_time_samples = data_obj.raw_data.shape[-1]
        window_sizes_ms = [10, 100, 200, 500, int(1000 * max_time_samples / fs)] # ms

        for idx, window_size_ms in enumerate(window_sizes_ms):            
            n_samples_window = int(window_size_ms * fs / 1000)

            if n_samples_window == 0:
                print(f"[{dataset_name}] Window size {window_size_ms}ms is too small.")
                continue

            is_last = (idx == len(window_sizes_ms) - 1)
            window_info = f"{window_size_ms} ms ({n_samples_window} samples)"

            truncated_neural_tensor = original_neural_tensor[:, :, :n_samples_window]
            perform_spectral_analysis(truncated_neural_tensor, dataset_name, window_info, fs, signal_type="Neural", plot=is_last)

            truncated_artifact_tensor = original_artifact_tensor[:, :, :n_samples_window]
            perform_spectral_analysis(truncated_artifact_tensor, dataset_name, window_info, fs, signal_type="Artifact", plot=is_last)
