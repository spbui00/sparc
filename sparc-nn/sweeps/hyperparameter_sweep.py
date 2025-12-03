import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import json
import torch.nn.functional as F
from datetime import datetime
from itertools import product
from sparc import DataHandler, SignalDataWithGroundTruth
from sparc.core.signal_data import ArtifactTriggers
from sparc.core.plotting import NeuralPlotter
from sparc.core.evaluator import Evaluator
from sparc.core.neural_analyzer import NeuralAnalyzer
from models import UNet1D
from loss import PhysicsLoss
from uncertainty_loss import UncertaintyWeightedLoss
from data_utils import prepare_dataset
from sweeps.generate_sweep_name import config_to_string

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

LEARNING_RATE = 1e-3
BATCH_SIZE = 1
NUM_EPOCHS = 1000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

SWEEP_DIR = "sweep_2"
os.makedirs(SWEEP_DIR, exist_ok=True)

data_handler = DataHandler()
data_obj_dict = data_handler.load_npz_data('../../data/added_artifacts_swec_data_512_lower_freq_wo0.npz')

data_obj = SignalDataWithGroundTruth(
    raw_data=data_obj_dict['mixed_data'],
    sampling_rate=data_obj_dict['sampling_rate'],
    ground_truth=data_obj_dict['ground_truth'],
    artifacts=data_obj_dict['artifacts'],
    artifact_markers=ArtifactTriggers(starts=data_obj_dict['artifact_markers']),
)

print(f"Data loaded: {data_obj.raw_data.shape}")

class ArtifactDataset(Dataset):
    def __init__(self, data, stim_trace, artifact_masks, data_mean, data_std, eps=1e-8):
        self.data = torch.from_numpy(data)
        self.stim_trace = stim_trace.float()
        self.artifact_masks = artifact_masks
        self.data_mean = data_mean
        self.data_std = data_std
        self.eps = eps
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        norm_data = (self.data[idx] - self.data_mean) / (self.data_std + self.eps)
        norm_data_2d = norm_data.squeeze(0)
        return norm_data_2d, self.stim_trace[idx], self.artifact_masks[idx]

def create_soft_artifact_mask(indices, durations, seq_length, blur_radius=5):
    mask = torch.zeros(1, seq_length)
    for start, duration in zip(indices, durations):
        end = min(start + duration, seq_length)        
        mask[0, start:end] = 1.0
        for j in range(1, blur_radius + 1):
            if start - j >= 0:
                mask[0, start - j] = max(mask[0, start - j], 1 - j/blur_radius)
            if end + j - 1 < seq_length:
                mask[0, end + j - 1] = max(mask[0, end + j - 1], 1 - j/blur_radius)
    return mask

mixed_data = data_obj.raw_data.astype(np.float32)
ARTIFACT_DURATION = int(40 / 1000 * data_obj.sampling_rate)

N_SAMPLES = mixed_data.shape[2]
N_DATA_CHANNELS = mixed_data.shape[1]
IN_CHANNELS = N_DATA_CHANNELS + 1
OUT_CHANNELS = N_DATA_CHANNELS
N_TRIALS = mixed_data.shape[0]

# Use prepare_dataset() for robust normalization (like train.py)
print("Preparing dataset with robust normalization...")
dataset, data_loader, artifact_masks, data_mean, data_std = prepare_dataset(
    data_obj=data_obj,
    data_obj_dict=data_obj_dict,
    batch_size=BATCH_SIZE,
    use_normalization=True,
    shuffle=True,
    artifact_duration_ms=40,
    trial_indices=None  # Use all trials
)

# prepare_dataset handles everything - artifact_masks are already in the dataset
# We keep a reference to data_mean and data_std for denormalization later

def get_model_architecture_info(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_type': model.__class__.__name__,
        'in_channels': model.in_channels,
        'out_channels': model.out_channels
    }


def config_exists(config_str):
    config_dir = os.path.join(SWEEP_DIR, config_str)
    results_file = os.path.join(config_dir, "results.txt")
    return os.path.exists(results_file)

def run_training(config, use_uncertainty_loss=False):
    config_str = config_to_string(config, use_uncertainty_loss=use_uncertainty_loss)
    config_dir = os.path.join(SWEEP_DIR, config_str)
    os.makedirs(config_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Running config: {config_str}")
    print(f"Config: {config}")
    print(f"Use uncertainty loss: {use_uncertainty_loss}")
    print(f"{'='*80}\n")
    
    model = UNet1D(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS).to(DEVICE)
    
    model_info = get_model_architecture_info(model)
    with open(os.path.join(config_dir, "model_architecture.json"), 'w') as f:
        json.dump(model_info, f, indent=2)
    
    criterion = PhysicsLoss(
        f_cutoff=config['f_cutoff'],
        sampling_rate=data_obj.sampling_rate,
    ).to(DEVICE)
    
    uncertainty_loss = None
    if use_uncertainty_loss:
        uncertainty_loss = UncertaintyWeightedLoss(num_losses=5).to(DEVICE)
        optimizer = optim.Adam([
            {'params': model.parameters()},
            {'params': uncertainty_loss.parameters(), 'lr': 1e-3}
        ], lr=LEARNING_RATE)
    else:
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    loss_history = []
    results_output = []
    
    def log_and_save(msg):
        print(msg)
        results_output.append(msg)
    
    log_and_save(f"Config: {config}")
    log_and_save(f"Use uncertainty loss: {use_uncertainty_loss}")
    log_and_save(f"Model architecture: {model_info}")
    log_and_save(f"Total parameters: {model_info['total_parameters']}")
    log_and_save("")
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        
        for batch in data_loader:
            x_mixed_batch = batch[0].to(DEVICE)
            x_stim_batch = batch[1].to(DEVICE)
            artifact_mask_batch = batch[2].to(DEVICE)
            
            a_pred_batch = model(x_mixed_batch, x_stim_batch)
            s_pred_batch = x_mixed_batch - a_pred_batch
            
            raw_loss_dict = criterion(s_pred_batch, a_pred_batch)
            
            if use_uncertainty_loss:
                final_loss = uncertainty_loss(raw_loss_dict)
            else:
                final_loss = (
                    config['w_cosine'] * raw_loss_dict['cosine'] +
                    config['w_rank_s'] * raw_loss_dict['rank_s_penalty'] +
                    config['w_spectral'] * raw_loss_dict['spectral'] +
                    config['w_spectral_slope'] * raw_loss_dict['spectral_slope_s'] +
                    config['w_rank_a'] * raw_loss_dict['rank_a']
                )
            
            optimizer.zero_grad()
            final_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += final_loss.item()
        
        avg_loss = total_loss / len(data_loader)
        loss_history.append(avg_loss)
        
        if (epoch + 1) % 50 == 0:
            log_and_save(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.6f}")
    
    log_and_save(f"\nTraining Complete. Final Loss: {loss_history[-1]:.6f}")
    
    model.eval()
    # Use prepare_dataset for evaluation too (consistent normalization)
    # This will use the same robust stats computed during training
    eval_dataset, eval_loader, eval_artifact_masks, eval_data_mean, eval_data_std = prepare_dataset(
        data_obj=data_obj,
        data_obj_dict=data_obj_dict,
        batch_size=N_TRIALS,
        use_normalization=True,
        shuffle=False,
        artifact_duration_ms=40,
        trial_indices=None  # Use all trials
    )
    
    # Use the same normalization stats from training (should be the same, but use training ones for consistency)
    with torch.no_grad():
        new_mixed_signal_norm, new_stim_trace, artifact_mask_batch = next(iter(eval_loader))
        new_mixed_signal_norm_dev = new_mixed_signal_norm.to(DEVICE)
        new_stim_trace_dev = new_stim_trace.to(DEVICE)
        artifact_mask_batch_dev = artifact_mask_batch.to(DEVICE)
        data_mean_dev = data_mean.to(DEVICE)  # Use training stats
        data_std_dev = data_std.to(DEVICE)    # Use training stats
        
        predicted_artifact_norm = model(new_mixed_signal_norm_dev, new_stim_trace_dev)
        predicted_neural_signal_norm = new_mixed_signal_norm_dev - predicted_artifact_norm
        
        predicted_artifact = (predicted_artifact_norm * data_std_dev) + data_mean_dev
        predicted_neural_signal_unblended = (predicted_neural_signal_norm * data_std_dev) + data_mean_dev
        new_mixed_signal = (new_mixed_signal_norm_dev * data_std_dev) + data_mean_dev
        
        artifact_mask_expanded = artifact_mask_batch_dev.expand_as(new_mixed_signal)
        predicted_neural_signal = (1 - artifact_mask_expanded) * new_mixed_signal + \
                                  artifact_mask_expanded * predicted_neural_signal_unblended
        
        if use_uncertainty_loss:
            weights = torch.exp(-uncertainty_loss.log_vars)
            log_and_save(f"Epoch {epoch}: Loss {loss_history[-1]:.4f} | Weights: {weights.cpu().numpy()}")
    
    def _cov(M: torch.Tensor) -> torch.Tensor:
        T, C, N = M.shape
        M_mean = torch.mean(M, dim=-1, keepdim=True)
        M_centered = M - M_mean
        Cov = (M_centered @ M_centered.transpose(-1, -2)) / (N - 1)
        return Cov
    
    cov_a = _cov(predicted_artifact)
    nuc_a = torch.linalg.norm(cov_a, ord='nuc', dim=(-2, -1))
    fro_a = torch.linalg.norm(cov_a, ord='fro', dim=(-2, -1))
    soft_rank_a = nuc_a / (fro_a + 1e-6)
    log_and_save(f"Soft rank for artifact: {torch.mean(soft_rank_a).item():.4f}")
    
    cov_s = _cov(predicted_neural_signal)
    nuc_s = torch.linalg.norm(cov_s, ord='nuc', dim=(-2, -1))
    fro_s = torch.linalg.norm(cov_s, ord='fro', dim=(-2, -1))
    soft_rank_s = nuc_s / (fro_s + 1e-6)
    log_and_save(f"Soft rank for neural: {torch.mean(soft_rank_s).item():.4f}")
    
    cos_sim_per_channel = F.cosine_similarity(predicted_neural_signal, predicted_artifact, dim=2, eps=1e-8)
    loss_cosine = torch.mean(cos_sim_per_channel**2)
    log_and_save(f"Cosine similarity: {loss_cosine.item():.4f}")
    
    artifact_mask_np = artifact_mask_batch.cpu().numpy()
    evaluator = Evaluator(sampling_rate=data_obj.sampling_rate)
    analyzer = NeuralAnalyzer(sampling_rate=data_obj.sampling_rate)
    plotter = NeuralPlotter(evaluator)
    
    ground_truth_neural = data_obj.ground_truth.astype(np.float32)
    ground_truth_artifacts = data_obj.artifacts.astype(np.float32)
    predicted_neural_np = predicted_neural_signal.cpu().numpy()
    mixed_data_np = new_mixed_signal.cpu().numpy()
    predicted_artifact_np = predicted_artifact.cpu().numpy()
    
    trial_idx = 0
    channel_idx = 0
    
    # Calculate SNR per trial and average (instead of pooling all trials)
    snr_before_per_trial = []
    snr_after_per_trial = []
    snr_improvement_per_trial = []
    
    for trial_idx in range(N_TRIALS):
        trial_gt = ground_truth_neural[trial_idx:trial_idx+1]
        trial_mixed = mixed_data_np[trial_idx:trial_idx+1]
        trial_predicted = predicted_neural_np[trial_idx:trial_idx+1]
        
        trial_noise_before = trial_mixed - trial_gt
        trial_noise_after = trial_predicted - trial_gt
        
        trial_snr_before = evaluator.calculate_snr(trial_gt, trial_noise_before)
        trial_snr_after = evaluator.calculate_snr(trial_gt, trial_noise_after)
        trial_snr_improvement = evaluator.calculate_snr_improvement(trial_mixed, trial_predicted, trial_gt)
        
        snr_before_per_trial.append(trial_snr_before)
        snr_after_per_trial.append(trial_snr_after)
        snr_improvement_per_trial.append(trial_snr_improvement)
    
    snr_before = np.mean(snr_before_per_trial)
    snr_after = np.mean(snr_after_per_trial)
    snr_improvement = np.mean(snr_improvement_per_trial)
    
    log_and_save(f"\nSNR Before (Mixed): {snr_before:.2f} dB (averaged over {N_TRIALS} trials)")
    log_and_save(f"SNR After (Cleaned): {snr_after:.2f} dB (averaged over {N_TRIALS} trials)")
    log_and_save(f"SNR Improvement: {snr_improvement:.2f} dB (averaged over {N_TRIALS} trials)")
    
    suppression_amplitude_db, suppression_power_db = evaluator.calculate_artifact_suppression(
        mixed_data_np, predicted_neural_np, ground_truth_neural
    )
    log_and_save(f"\nArtifact Suppression (Amplitude): {suppression_amplitude_db:.2f} dB")
    log_and_save(f"Artifact Suppression (Power): {suppression_power_db:.2f} dB")
    
    from loss import PhysicsLoss
    criterion = PhysicsLoss(sampling_rate=data_obj.sampling_rate)
    
    predicted_neural_tensor_slope = torch.from_numpy(predicted_neural_np).float()
    ground_truth_neural_tensor_slope = torch.from_numpy(ground_truth_neural).float()
    
    spectral_slope_pred_neural = criterion._spectral_slope_loss(predicted_neural_tensor_slope).item()
    spectral_slope_gt_neural = criterion._spectral_slope_loss(ground_truth_neural_tensor_slope).item()
    
    log_and_save(f"\n--- Spectral Slope Loss (Welch) ---")
    log_and_save(f"GT Neural: {spectral_slope_gt_neural:.6f}")
    log_and_save(f"Predicted Neural: {spectral_slope_pred_neural:.6f}")
    
    MSE = np.mean((predicted_neural_np - ground_truth_neural) ** 2)
    log_and_save(f"\nMSE: {MSE:.4f}")
    
    psd_mse = analyzer.calculate_psd_mse(ground_truth_neural, predicted_neural_np)
    log_and_save(f"\nPSD MSE (max across channels): {np.max(psd_mse):.4f} at channel {np.argmax(psd_mse)}")
    log_and_save(f"PSD MSE (min across channels): {np.min(psd_mse):.4f} at channel {np.argmin(psd_mse)}")
    log_and_save(f"PSD MSE (median across channels): {np.median(psd_mse):.4f}")
    
    coherence_neural = analyzer.calculate_spectral_coherence(ground_truth_neural, predicted_neural_np)
    log_and_save(f"\nMinimum Spectral Coherence: {np.min(coherence_neural):.4f} at channel {np.argmin(coherence_neural)}")
    log_and_save(f"Maximum Spectral Coherence: {np.max(coherence_neural):.4f} at channel {np.argmax(coherence_neural)}")
    log_and_save(f"Median Spectral Coherence: {np.median(coherence_neural):.4f}")
    
    def perform_spectral_analysis(data_3d, analyzer, signal_type, plot=False, save_path=None):
        def spectral_flatness(log_psd: np.ndarray) -> np.ndarray:
            mean_log_psd = np.mean(log_psd, axis=-1)
            geom_mean = np.exp(mean_log_psd)
            psd = np.exp(log_psd)
            arith_mean = np.mean(psd, axis=-1)
            flatness = geom_mean / (arith_mean + 1e-8)
            return np.mean(flatness)
        
        if data_3d is None or data_3d.ndim != 3:
            log_and_save(f"{signal_type}: Invalid 3D data for analysis.")
            return None, None
            
        w, c, n = data_3d.shape
        nperseg = min(n, 256)
        freqs, psd = analyzer.compute_psd(data_3d, nperseg=nperseg)
        
        mean_psd = np.mean(psd, axis=0)
        cumulative_power = np.cumsum(mean_psd)
        
        total_power = cumulative_power[-1]
        if total_power == 0:
            log_and_save(f"{signal_type}: Total power is zero.")
            return None, None
        
        threshold_95 = 0.95 * total_power
        idx_95 = np.searchsorted(cumulative_power, threshold_95)
        freq_95 = freqs[idx_95] if idx_95 < len(freqs) else freqs[-1]
        log_and_save(f"{signal_type}: 95% of total power is below {freq_95:.2f} Hz")
        
        idx_200 = np.searchsorted(freqs, 200)
        if idx_200 < len(cumulative_power):
            power_below_200 = cumulative_power[idx_200]
        else:
            power_below_200 = cumulative_power[-1]
        percent_below_200 = 100 * power_below_200 / total_power
        log_and_save(f"{signal_type}: {percent_below_200:.2f}% of total power is below 200 Hz")
        
        spectral_flatness_value = spectral_flatness(np.log(mean_psd + 1e-10))
        log_and_save(f"{signal_type}: Spectral flatness: {spectral_flatness_value:.4f}")
        
        if plot:
            color_map = {
                'GT Neural': '#2ca02c',
                'Predicted Neural': '#1f77b4',
                'GT Artifact': '#ff7f0e',
                'Predicted Artifact': '#d62728'
            }
            color = color_map.get(signal_type, '#1f77b4')
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.semilogy(freqs, mean_psd, linewidth=1.5, color=color)
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Power Spectral Density')
            ax.set_title(f'PSD - {signal_type}')
            ax.grid(True, alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=150)
            plt.close()
        
        return freqs, mean_psd
    
    log_and_save("\n--- Spectral Analysis ---")
    perform_spectral_analysis(ground_truth_neural, analyzer, "GT Neural", plot=True, 
                             save_path=os.path.join(config_dir, "psd_gt_neural.png"))
    perform_spectral_analysis(predicted_neural_np, analyzer, "Predicted Neural", plot=True,
                             save_path=os.path.join(config_dir, "psd_predicted_neural.png"))
    perform_spectral_analysis(ground_truth_artifacts, analyzer, "GT Artifact", plot=True,
                             save_path=os.path.join(config_dir, "psd_gt_artifact.png"))
    perform_spectral_analysis(predicted_artifact_np, analyzer, "Predicted Artifact", plot=True,
                             save_path=os.path.join(config_dir, "psd_predicted_artifact.png"))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(loss_history) + 1), loss_history, linewidth=2, color='#1f77b4')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(config_dir, 'training_loss.png'), dpi=150)
    plt.close()
    
    fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    x_axis = np.arange(ground_truth_neural.shape[2]) / data_obj.sampling_rate
    
    mask = artifact_mask_np[trial_idx, 0, :]
    mask_threshold = 0
    artifact_regions = mask > mask_threshold
    
    try:
        from scipy.ndimage import label
        labeled_mask, num_features = label(artifact_regions)
        for i in range(1, num_features + 1):
            region = labeled_mask == i
            start_idx = np.where(region)[0][0]
            end_idx = np.where(region)[0][-1]
            start_time = x_axis[start_idx]
            end_time = x_axis[end_idx]
            for ax in axs:
                ax.axvspan(start_time, end_time, color='red', alpha=0.15)
    except ImportError:
        for ax in axs:
            ax.fill_between(x_axis, ax.get_ylim()[0], ax.get_ylim()[1],
                            where=artifact_regions, color='red', alpha=0.15)
    
    axs[0].plot(x_axis, ground_truth_neural[trial_idx, channel_idx, :], 
                linewidth=1.5, color='#2ca02c', label='GT Neural')
    axs[0].set_ylabel('Amplitude (µV)')
    axs[0].set_title(f'Ground Truth Neural Signal - Trial {trial_idx}, Channel {channel_idx}')
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    
    axs[1].plot(x_axis, predicted_neural_np[trial_idx, channel_idx, :], 
                linewidth=1.5, color='#1f77b4', label='Cleaned Neural')
    axs[1].set_ylabel('Amplitude (µV)')
    axs[1].set_title(f'Cleaned Neural Signal - Trial {trial_idx}, Channel {channel_idx}')
    axs[1].grid(True, alpha=0.3)
    axs[1].legend()
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    
    axs[2].plot(x_axis, ground_truth_artifacts[trial_idx, channel_idx, :], 
                linewidth=1.5, color='#ff7f0e', label='GT Artifact')
    axs[2].set_ylabel('Amplitude (µV)')
    axs[2].set_title(f'Ground Truth Artifact - Trial {trial_idx}, Channel {channel_idx}')
    axs[2].grid(True, alpha=0.3)
    axs[2].legend()
    axs[2].spines['top'].set_visible(False)
    axs[2].spines['right'].set_visible(False)
    
    axs[3].plot(x_axis, predicted_artifact_np[trial_idx, channel_idx, :], 
                linewidth=1.5, color='#d62728', label='Predicted Artifact')
    axs[3].set_ylabel('Amplitude (µV)')
    axs[3].set_xlabel('Time (s)')
    axs[3].set_title(f'Predicted Artifact - Trial {trial_idx}, Channel {channel_idx}')
    axs[3].grid(True, alpha=0.3)
    axs[3].legend()
    axs[3].spines['top'].set_visible(False)
    axs[3].spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config_dir, 'all_signals_with_artifact_overlay.png'), dpi=150)
    plt.close()
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'in_channels': IN_CHANNELS,
            'out_channels': OUT_CHANNELS,
        },
        'training_config': {
            'learning_rate': LEARNING_RATE,
            'batch_size': BATCH_SIZE,
            'num_epochs': NUM_EPOCHS,
            'artifact_duration': ARTIFACT_DURATION,
            'sampling_rate': data_obj.sampling_rate,
        },
        'hyperparameters': config,
        'use_uncertainty_loss': use_uncertainty_loss,
        'loss_history': loss_history,
        'data_mean': data_mean,
        'data_std': data_std,
    }, os.path.join(config_dir, 'model.pth'))
    
    with open(os.path.join(config_dir, "results.txt"), 'w') as f:
        f.write("\n".join(results_output))
    
    with open(os.path.join(config_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nConfig {config_str} completed. Results saved to {config_dir}\n")

hyperparameter_ranges = {
    'f_cutoff': [0, 10,25, 50],
    'w_cosine': [0, 1, 2],
    'w_rank_s': [0, 0.2, 0.5],
    'w_spectral': [0, 1,2,3],
    'w_spectral_slope': [0,0.2,0.5,1],
    'w_rank_a': [0, 1,2,3],
}

configs = []
for f_cutoff in hyperparameter_ranges['f_cutoff']:
    for w_cosine in hyperparameter_ranges['w_cosine']:
        for w_rank_s in hyperparameter_ranges['w_rank_s']:
            for w_spectral in hyperparameter_ranges['w_spectral']:
                for w_spectral_slope in hyperparameter_ranges['w_spectral_slope']:
                    for w_rank_a in hyperparameter_ranges['w_rank_a']:
                        configs.append({
                            'f_cutoff': f_cutoff,
                            'w_cosine': w_cosine,
                            'w_rank_s': w_rank_s,
                            'w_spectral': w_spectral,
                            'w_spectral_slope': w_spectral_slope,
                            'w_rank_a': w_rank_a,
                        })

print(f"\nTotal configurations: {len(configs)}")
print(f"Adding one configuration with uncertainty_loss")
configs.append({
    'f_cutoff': 50,
    'w_cosine': 5,
    'w_rank_s': 1,
    'w_spectral': 1.2,
    'w_spectral_slope': 5,
    'w_rank_a': 1.5,
    'use_uncertainty_loss': True
})

completed = 0
skipped = 0

for i, config in enumerate(configs):
    config_copy = config.copy()
    use_uncertainty = config_copy.pop('use_uncertainty_loss', False)
    config_str = config_to_string(config_copy, use_uncertainty_loss=use_uncertainty)
    
    if config_exists(config_str):
        print(f"Skipping config {i+1}/{len(configs)}: {config_str} (already exists)")
        skipped += 1
        continue
    
    try:
        run_training(config_copy, use_uncertainty_loss=use_uncertainty)
        completed += 1
    except Exception as e:
        print(f"Error running config {config_str}: {e}")
        import traceback
        traceback.print_exc()
        continue

print(f"\n{'='*80}")
print(f"Sweep complete!")
print(f"Completed: {completed}")
print(f"Skipped: {skipped}")
print(f"Total: {len(configs)}")
print(f"{'='*80}")

