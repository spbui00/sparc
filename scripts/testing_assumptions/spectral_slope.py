import numpy as np
import torch
import matplotlib.pyplot as plt
from sparc import DataHandler
from sparc.core.signal_data import SignalDataWithGroundTruth, SimulatedData, ArtifactTriggers


def compute_spectral_slope_loss_welch(data_windows_3d: np.ndarray, fs: float, nperseg: int = 256, eps: float = 1e-8) -> float:
    if data_windows_3d.ndim != 3:
        raise ValueError("Input data must be 3D (trials, channels, samples)")
    
    x = torch.from_numpy(data_windows_3d).float()
    T, C, N = x.shape
    nperseg = min(nperseg, N)
    noverlap = nperseg // 2
    
    if N < nperseg:
        s_fft = torch.fft.rfft(x, dim=-1)
        s_psd = torch.abs(s_fft)**2
        s_log_psd = torch.log(s_psd + eps)
    else:
        window = torch.hann_window(nperseg, device=x.device, dtype=x.dtype)
        psd_segments = []
        n_segments = max(1, (N - noverlap) // (nperseg - noverlap))
        
        for i in range(n_segments):
            start_idx = i * (nperseg - noverlap)
            end_idx = start_idx + nperseg
            if end_idx > N:
                break
            
            segment = x[..., start_idx:end_idx]
            windowed_segment = segment * window.view(1, 1, -1)
            s_fft_seg = torch.fft.rfft(windowed_segment, dim=-1)
            psd_seg = torch.abs(s_fft_seg)**2
            psd_segments.append(psd_seg)
        
        if len(psd_segments) > 0:
            s_psd = torch.stack(psd_segments, dim=0)
            s_psd = torch.mean(s_psd, dim=0)
            s_log_psd = torch.log(s_psd + eps)
        else:
            s_fft = torch.fft.rfft(x, dim=-1)
            s_psd = torch.abs(s_fft)**2
            s_log_psd = torch.log(s_psd + eps)
    
    diff = s_log_psd[..., 1:] - s_log_psd[..., :-1]
    
    positive_slope_penalty = torch.relu(diff) ** 2
    return torch.mean(positive_slope_penalty).item()


def compute_spectral_slope_loss(x: torch.Tensor, eps: float = 1e-8) -> float:
    s_fft = torch.fft.rfft(x, dim=-1)
    s_log_psd = torch.log(torch.abs(s_fft)**2 + eps)
    
    diff = s_log_psd[..., 1:] - s_log_psd[..., :-1]
    
    positive_slope_penalty = torch.relu(diff) ** 2
    return torch.mean(positive_slope_penalty).item()


def plot_spectral_slope(data_windows_3d, dataset_name, window_info, signal_type, fs, spectral_slope_value):
    x = torch.from_numpy(data_windows_3d).float()
    T, C, N = x.shape
    nperseg = min(256, N)
    noverlap = nperseg // 2
    eps = 1e-8
    
    if N < nperseg:
        s_fft = torch.fft.rfft(x, dim=-1)
        s_psd = torch.abs(s_fft)**2
        mean_psd = torch.mean(s_psd, dim=(0, 1)).numpy()
        freqs = np.fft.rfftfreq(N, d=1.0/fs)
    else:
        window = torch.hann_window(nperseg, device=x.device, dtype=x.dtype)
        psd_segments = []
        n_segments = max(1, (N - noverlap) // (nperseg - noverlap))
        
        for i in range(n_segments):
            start_idx = i * (nperseg - noverlap)
            end_idx = start_idx + nperseg
            if end_idx > N:
                break
            
            segment = x[..., start_idx:end_idx]
            windowed_segment = segment * window.view(1, 1, -1)
            s_fft_seg = torch.fft.rfft(windowed_segment, dim=-1)
            psd_seg = torch.abs(s_fft_seg)**2
            psd_segments.append(psd_seg)
        
        if len(psd_segments) > 0:
            s_psd = torch.stack(psd_segments, dim=0)
            s_psd = torch.mean(s_psd, dim=0)
            mean_psd = torch.mean(s_psd, dim=(0, 1)).numpy()
            freqs = np.fft.rfftfreq(nperseg, d=1.0/fs)
        else:
            s_fft = torch.fft.rfft(x, dim=-1)
            s_psd = torch.abs(s_fft)**2
            mean_psd = torch.mean(s_psd, dim=(0, 1)).numpy()
            freqs = np.fft.rfftfreq(N, d=1.0/fs)
    
    color = '#1f77b4' if signal_type == "Neural (s)" else '#d62728'
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.semilogy(freqs, mean_psd, linewidth=1.5, color=color)
    
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power Spectral Density')
    ax.set_title(f'PSD (Welch) - {signal_type} - {dataset_name} ({window_info})\nSpectral Slope Loss (Welch): {spectral_slope_value:.6f}')
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()


def perform_spectral_slope_analysis(data_windows_3d, dataset_name, window_info, signal_type="Neural", fs=None, plot=False):
    if data_windows_3d is None or data_windows_3d.ndim != 3:
        print(f"[{dataset_name} @ {window_info}] Invalid 3D data for analysis.")
        return
    
    w, c, n = data_windows_3d.shape
    
    if n == 0:
        print(f"[{dataset_name} @ {window_info}] {signal_type}: No samples to analyze.")
        return
    
    spectral_slope_value = compute_spectral_slope_loss_welch(data_windows_3d, fs)
    
    print(f"[{dataset_name} @ {window_info}] {signal_type}: Spectral slope loss (Welch): {spectral_slope_value:.6f}")
    
    if plot and fs is not None:
        plot_spectral_slope(data_windows_3d, dataset_name, window_info, signal_type, fs, spectral_slope_value)


if __name__ == "__main__":
    data_handler = DataHandler()
    
    datasets = [
        # ('../../data/simulated_data_2x64_1000.npz', 'simulated data 1000Hz'),
        # ('../../data/simulated_data_2x64_30000.npz', 'simulated data 30000Hz'),
        # ('../../data/added_artifacts_swec_data_512.npz', 'swec data 512Hz'),
        ('../../data/added_artifacts_swec_data_seizure_512_lower_freq_wo0.npz', 'swec seizure data 512Hz'),
        ('../../data/added_artifacts_swec_data_512_lower_freq_wo0.npz', 'swec data 512Hz')
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
        window_sizes_ms = [10, 100, 200, 500, int(1000 * max_time_samples / fs)]
        
        for idx, window_size_ms in enumerate(window_sizes_ms):
            n_samples_window = int(window_size_ms * fs / 1000)
            
            if n_samples_window == 0:
                print(f"[{dataset_name}] Window size {window_size_ms}ms is too small.")
                continue
            
            window_info = f"{window_size_ms} ms ({n_samples_window} samples)"
            is_last = (idx == len(window_sizes_ms) - 1)
            
            truncated_neural_tensor = original_neural_tensor[:, :, :n_samples_window]
            perform_spectral_slope_analysis(truncated_neural_tensor, dataset_name, window_info, 
                                         signal_type="Neural (s)", fs=fs, plot=is_last)
            
            truncated_artifact_tensor = original_artifact_tensor[:, :, :n_samples_window]
            perform_spectral_slope_analysis(truncated_artifact_tensor, dataset_name, window_info, 
                                          signal_type="Artifact", fs=fs, plot=is_last)

