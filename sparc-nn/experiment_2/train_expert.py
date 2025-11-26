import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import argparse
from datetime import datetime
from glob import glob

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import UNet1D, NeuralExpertAE
from data_utils import prepare_dataset, prepare_swec_expert_dataset
from sparc import DataHandler
from sparc.core.signal_data import ArtifactTriggers, SignalDataWithGroundTruth
from sparc.core.neural_analyzer import NeuralAnalyzer
from sparc.core.evaluator import Evaluator
from sparc.core.plotting import NeuralPlotter
from loss import PhysicsLoss

parser = argparse.ArgumentParser(description='Train expert-guided U-Net for neural signal separation')
parser.add_argument('--train-expert', action='store_true', 
                    help='Train the expert model (default: False, loads from saved_models)')
parser.add_argument('--num-epochs', type=int, default=400,
                    help='Number of epochs for main model training (default: 400)')
parser.add_argument('--swec-mat-file', type=str, default=None,
                    help='Path to SWEC .mat file for expert training (required if --train-expert)')
parser.add_argument('--swec-info-file', type=str, default=None,
                    help='Path to SWEC _info.mat file for seizure exclusion (optional)')
parser.add_argument('--window-len', type=float, default=2.0,
                    help='Window length in seconds for expert dataset (default: 2.0)')
parser.add_argument('--stride', type=float, default=1.0,
                    help='Stride in seconds for expert dataset (default: 1.0)')
parser.add_argument('--max-samples', type=int, default=None,
                    help='Max number of chunks to extract from SWEC data (optional)')
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
print(f"Configuration: TRAIN_EXPERT={args.train_expert}, NUM_EPOCHS={args.num_epochs}")

TRAIN_EXPERT = args.train_expert

# Trial index for U-Net training and evaluation
TRAIN_TRIAL_IDX = 0

# 1. LOAD DATA FOR U-NET (MIXED DATA WITH ARTIFACTS)
data_handler = DataHandler()
data_obj_dict = data_handler.load_npz_data('../../data/added_artifacts_swec_data_512_lower_freq_wo0.npz')

data_obj = SignalDataWithGroundTruth(
    raw_data=data_obj_dict['mixed_data'],
    sampling_rate=data_obj_dict['sampling_rate'],
    ground_truth=data_obj_dict['ground_truth'],
    artifacts=data_obj_dict['artifacts'],
    artifact_markers=ArtifactTriggers(starts=data_obj_dict['artifact_markers']),
)

print(f"U-Net Data loaded: {data_obj.raw_data.shape}")

# 2. CONFIGURATION
LEARNING_RATE = 1e-3
BATCH_SIZE = 1
NUM_EPOCHS = args.num_epochs
IN_CHANNELS = data_obj.raw_data.shape[1] + 1 # +1 for stim
OUT_CHANNELS = data_obj.raw_data.shape[1]

# ==============================================================================
# STEP A: PREPARE EXPERT DATA (SWEC CLEAN DATA)
# ==============================================================================

if TRAIN_EXPERT:
    if args.swec_mat_file is None:
        raise ValueError("--swec-mat-file is required when --train-expert is set")
    
    print("\n" + "=" * 60)
    print("Preparing Expert Dataset from SWEC Clean Data")
    print("=" * 60)
    
    expert_ds, expert_loader, expert_mean, expert_std, expert_sampling_rate = prepare_swec_expert_dataset(
        mat_file_path=args.swec_mat_file,
        info_file_path=args.swec_info_file,
        window_len_sec=args.window_len,
        stride_sec=args.stride,
        batch_size=4,
        max_samples=args.max_samples
    )
    
    print(f"Expert Dataset: {len(expert_ds)} chunks")
    print(f"Expert Sampling Rate: {expert_sampling_rate} Hz")
else:
    expert_mean = None
    expert_std = None
    expert_ds = None
    expert_loader = None

# ==============================================================================
# STEP B: PREPARE MAIN DATA (MIXED / CONTAMINATED)
# ==============================================================================
# This is for the U-Net to learn separation
# Train U-Net only on trial 3
main_ds, main_loader, artifact_masks, mixed_mean, mixed_std = prepare_dataset(
    data_obj=data_obj,
    data_obj_dict=data_obj_dict,
    batch_size=BATCH_SIZE,
    use_normalization=True,
    shuffle=True,
    artifact_duration_ms=40,
    trial_indices=[TRAIN_TRIAL_IDX]  # Use only trial 3 for main training
)

# VERIFICATION PRINT
# Check if the stats are compatible (Expert vs Mixed)
print("\n--- Stats Check ---")
if TRAIN_EXPERT and expert_std is not None:
    print(f"Expert (SWEC Clean) Std: {expert_std.mean().item():.4f}")
else:
    print("Expert stats will be loaded from saved model")
print(f"Main (Mixed) Std:   {mixed_std.mean().item():.4f}")

# ==============================================================================
# PHASE 1: TRAIN OR LOAD THE EXPERT
# ==============================================================================
neural_expert = NeuralExpertAE(in_channels=OUT_CHANNELS).to(DEVICE)
EXPERT_EPOCHS = 50

if TRAIN_EXPERT:
    print("Phase 1: Training Expert on GROUND TRUTH...")
    optimizer_expert = optim.Adam(neural_expert.parameters(), lr=1e-3)
    
    for epoch in range(EXPERT_EPOCHS): 
        total_loss = 0
        for batch in expert_loader:
            x_clean = batch.to(DEVICE)
            
            reconstruction = neural_expert(x_clean)
            
            loss = torch.mean((reconstruction - x_clean) ** 2)
            
            optimizer_expert.zero_grad()
            loss.backward()
            optimizer_expert.step()
            total_loss += loss.item()
            
        if epoch % 10 == 0:
            print(f"Expert Epoch {epoch}: MSE {total_loss / len(expert_loader):.6f}")
    
    # Save expert after training
    expert_save_dir = 'saved_models'
    os.makedirs(expert_save_dir, exist_ok=True)
    expert_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    expert_model_path = os.path.join(expert_save_dir, f'neural_expert_{expert_timestamp}.pth')
    torch.save({
        'epoch': EXPERT_EPOCHS,
        'model_state_dict': neural_expert.state_dict(),
        'optimizer_state_dict': optimizer_expert.state_dict(),
        'in_channels': OUT_CHANNELS,
        'expert_mean': expert_mean.cpu(),
        'expert_std': expert_std.cpu(),
    }, expert_model_path)
    print(f"Expert model saved to: {expert_model_path}")
else:
    expert_model_files = glob(os.path.join('saved_models', 'neural_expert_*.pth'))
    if not expert_model_files:
        raise FileNotFoundError("No expert model found in saved_models. Set TRAIN_EXPERT=True to train one.")
    
    expert_model_path = sorted(expert_model_files)[-1]
    print(f"Loading expert model from: {expert_model_path}")
    
    expert_checkpoint = torch.load(expert_model_path, map_location=DEVICE, weights_only=False)
    neural_expert.load_state_dict(expert_checkpoint['model_state_dict'])
    
    if 'expert_mean' in expert_checkpoint and 'expert_std' in expert_checkpoint:
        expert_mean = expert_checkpoint['expert_mean'].to(DEVICE)
        expert_std = expert_checkpoint['expert_std'].to(DEVICE)
        print("Loaded expert normalization stats from checkpoint")
    else:
        raise ValueError("Expert checkpoint missing normalization stats. Please retrain expert with --train-expert")

# Freeze expert
for param in neural_expert.parameters():
    param.requires_grad = False
neural_expert.eval()
print("Expert Frozen.")


# ==============================================================================
# PHASE 2: TRAIN (U-NET)
# ==============================================================================
main_model = UNet1D(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS).to(DEVICE)
optimizer_main = optim.Adam(main_model.parameters(), lr=1e-3)

physics_loss = PhysicsLoss(
    w_cosine=1.0,
    w_rank=1.0,
    w_smooth=1.0,
    w_spectral=1.0,
    sampling_rate=data_obj.sampling_rate,
    f_cutoff=10.0,
).to(DEVICE)

print("Phase 2: Training U-Net on MIXED Data...")
# Track losses for plotting
train_losses = []
train_losses_physics = []
train_losses_anchor = []
train_losses_expert = []

for epoch in range(NUM_EPOCHS):
    model_loss = 0
    epoch_physics_loss = 0
    epoch_anchor_loss = 0
    epoch_expert_loss = 0
    for batch in main_loader:
        x_mixed = batch[0].to(DEVICE)
        x_stim = batch[1].to(DEVICE)
        mask = batch[2].to(DEVICE)
        
        a_pred = main_model(x_mixed, x_stim)
        s_pred = x_mixed - a_pred
        
        is_clean = (mask < 0.1).float()
        is_dirty = (mask > 0.5).float()
        
        # 3. Losses
        if is_clean.sum() > 0:
            loss_anchor = (a_pred ** 2 * is_clean).sum() / (is_clean.sum() + 1e-8)
        else:
            loss_anchor = 0.0
            
        # B. Expert Loss (Dirty Regions mostly)
        s_pred_recon = neural_expert(s_pred)
        if is_dirty.sum() > 0:
            loss_expert_projection = ((s_pred - s_pred_recon) ** 2 * is_dirty).sum() / (is_dirty.sum() + 1e-8)
        else:
            loss_expert_projection = 0.0
            
        # 2. Physics Loss
        physics_loss_dict = physics_loss(s_pred, a_pred)
        physics_loss_value = (
            2 * physics_loss_dict['cosine'] +
            1.5 * physics_loss_dict['rank_a'] +
            0.2 * physics_loss_dict['rank_s_penalty'] +
            3 * physics_loss_dict['spectral'] +
            0.5 * physics_loss_dict['spectral_slope_s']
        )
        final_loss = 1 * physics_loss_value + 1 * loss_anchor + 1 * loss_expert_projection

        
        optimizer_main.zero_grad()
        final_loss.backward()
        optimizer_main.step()
        model_loss += final_loss.item()
        epoch_physics_loss += physics_loss_value.item()
        epoch_anchor_loss += loss_anchor.item() if isinstance(loss_anchor, torch.Tensor) else loss_anchor
        epoch_expert_loss += loss_expert_projection.item() if isinstance(loss_expert_projection, torch.Tensor) else loss_expert_projection

    # Store average losses for this epoch
    avg_loss = model_loss / len(main_loader)
    train_losses.append(avg_loss)
    train_losses_physics.append(epoch_physics_loss / len(main_loader))
    train_losses_anchor.append(epoch_anchor_loss / len(main_loader))
    train_losses_expert.append(epoch_expert_loss / len(main_loader))

    if epoch % 50 == 0:
        print(f"Main Epoch {epoch}: Loss {avg_loss:.6f} (Physics: {train_losses_physics[-1]:.4f}, Anchor: {train_losses_anchor[-1]:.4f}, Expert: {train_losses_expert[-1]:.4f})")

print("\n--- Training Complete ---")

# ==============================================================================
# PLOT TRAINING LOSS
# ==============================================================================
print("\n--- Plotting Training Loss ---")

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

epochs = np.arange(len(train_losses))

# Plot 1: Total loss
axes[0].plot(epochs, train_losses, linewidth=1.5, color='#1f77b4', label='Total Loss')
axes[0].set_ylabel('Loss', fontsize=11)
axes[0].set_title('Training Loss - Total', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].legend(loc='upper right', fontsize=10)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[0].set_yscale('log')

# Plot 2: Individual loss components
axes[1].plot(epochs, train_losses_physics, linewidth=1.5, color='#2ca02c', label='Physics Loss', alpha=0.8)
axes[1].plot(epochs, train_losses_anchor, linewidth=1.5, color='#ff7f0e', label='Anchor Loss', alpha=0.8)
axes[1].plot(epochs, train_losses_expert, linewidth=1.5, color='#d62728', label='Expert Loss', alpha=0.8)
axes[1].set_xlabel('Epoch', fontsize=11)
axes[1].set_ylabel('Loss', fontsize=11)
axes[1].set_title('Training Loss - Components', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].legend(loc='upper right', fontsize=10)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].set_yscale('log')

plt.tight_layout()
loss_plot_path = f'training_loss_trial_{TRAIN_TRIAL_IDX}.png'
plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
print(f"Loss plot saved to: {loss_plot_path}")
plt.show()

# ==============================================================================
# PREPARE EVALUATION DATA (UNSHUFFLED)
# ==============================================================================
eval_ds, eval_loader, _, _, _ = prepare_dataset(
    data_obj=data_obj,
    data_obj_dict=data_obj_dict,
    batch_size=BATCH_SIZE,
    use_normalization=True,
    shuffle=False, 
    artifact_duration_ms=40,
    trial_indices=[TRAIN_TRIAL_IDX]
)

# ==============================================================================
# SAVE MODELS
# ==============================================================================
model_save_dir = 'saved_models'
os.makedirs(model_save_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Save main model (U-Net)
main_model_path = os.path.join(model_save_dir, f'unet1d_expert_guided_{timestamp}.pth')
torch.save({
    'epoch': NUM_EPOCHS,
    'model_state_dict': main_model.state_dict(),
    'optimizer_state_dict': optimizer_main.state_dict(),
    'loss': model_loss / len(main_loader),
    'in_channels': IN_CHANNELS,
    'out_channels': OUT_CHANNELS,
    'sampling_rate': data_obj.sampling_rate,
    'expert_mean': expert_mean.cpu(),
    'expert_std': expert_std.cpu(),
    'mixed_mean': mixed_mean.cpu(),
    'mixed_std': mixed_std.cpu(),
}, main_model_path)
print(f"Main model saved to: {main_model_path}")

# Save expert model (Neural Expert AE) - only if we trained it, otherwise it's already saved
if TRAIN_EXPERT:
    expert_model_path = os.path.join(model_save_dir, f'neural_expert_{timestamp}.pth')
    torch.save({
        'epoch': EXPERT_EPOCHS,
        'model_state_dict': neural_expert.state_dict(),
        'optimizer_state_dict': optimizer_expert.state_dict(),
        'in_channels': OUT_CHANNELS,
        'expert_mean': expert_mean.cpu(),
        'expert_std': expert_std.cpu(),
    }, expert_model_path)
    print(f"Expert model saved to: {expert_model_path}")
else:
    print("Expert model was loaded from saved_models, skipping save (already exists).")

# ==============================================================================
# EVALUATION
# ==============================================================================
print("\n--- Evaluation ---")

main_model.eval()
neural_expert.eval()

# Evaluation trial index - since we're evaluating on trial 3, use index 0 in the filtered dataset
# (which corresponds to original trial 3)
eval_trial_idx = 0  # Index in the filtered dataset (TRAIN_TRIAL_IDX is at index 0)
original_trial_idx = TRAIN_TRIAL_IDX  # Original trial index for ground truth comparison

with torch.no_grad():
    batch = next(iter(eval_loader))
    x_mixed = batch[0].to(DEVICE)
    x_stim = batch[1].to(DEVICE)
    mask = batch[2].to(DEVICE)
    
    print(f"Evaluating on original trial {original_trial_idx} (index {eval_trial_idx} in filtered dataset)")
    
    # Predictions
    a_pred = main_model(x_mixed, x_stim)
    s_pred = x_mixed - a_pred
    
    # Denormalize predictions
    s_pred_denorm = s_pred * mixed_std.to(DEVICE) + mixed_mean.to(DEVICE)
    a_pred_denorm = a_pred * mixed_std.to(DEVICE) + mixed_mean.to(DEVICE)
    x_mixed_denorm = x_mixed * mixed_std.to(DEVICE) + mixed_mean.to(DEVICE)

# Convert to numpy
predicted_neural_np = s_pred_denorm.cpu().numpy()
predicted_artifact_np = a_pred_denorm.cpu().numpy()
mixed_data_np = x_mixed_denorm.cpu().numpy()
ground_truth_neural = data_obj.ground_truth.astype(np.float32)
ground_truth_artifacts = data_obj.artifacts.astype(np.float32)

# Use original_trial_idx to select from full ground truth
trial_idx = original_trial_idx
ground_truth_neural_for_plot = ground_truth_neural[trial_idx:trial_idx+1]  # (1, C, T)
ground_truth_artifacts_for_plot = ground_truth_artifacts[trial_idx:trial_idx+1]  # (1, C, T)

# Soft rank calculation
def _cov(M: torch.Tensor) -> torch.Tensor:
    T, C, N = M.shape  # T=trials, C=channels, N=time_samples
    M_mean = torch.mean(M, dim=-1, keepdim=True)
    M_centered = M - M_mean
    Cov = (M_centered @ M_centered.transpose(-1, -2)) / (N - 1)
    return Cov

# Convert to tensor: numpy arrays are (trials, channels, time)
predicted_artifact_tensor = torch.from_numpy(predicted_artifact_np)  # (trials, C, T)
predicted_neural_tensor = torch.from_numpy(predicted_neural_np)  # (trials, C, T)

cov_a = _cov(predicted_artifact_tensor)
nuc_a = torch.linalg.norm(cov_a, ord='nuc', dim=(-2, -1))
fro_a = torch.linalg.norm(cov_a, ord='fro', dim=(-2, -1))
soft_rank_a = nuc_a / (fro_a + 1e-6)
print(f"Soft rank for artifact (predicted): {torch.mean(soft_rank_a).item():.4f}")

cov_s = _cov(predicted_neural_tensor)
nuc_s = torch.linalg.norm(cov_s, ord='nuc', dim=(-2, -1))
fro_s = torch.linalg.norm(cov_s, ord='fro', dim=(-2, -1))
soft_rank_s = nuc_s / (fro_s + 1e-6)
print(f"Soft rank for neural (predicted): {torch.mean(soft_rank_s).item():.4f}")

# Cosine similarity
cos_sim_per_channel = F.cosine_similarity(
    predicted_neural_tensor, predicted_artifact_tensor, dim=2, eps=1e-8
)
loss_cosine = torch.mean(cos_sim_per_channel**2)
print(f"Cosine similarity (predicted): {loss_cosine.item():.4f}")

# Ground truth soft rank and cosine similarity
ground_truth_artifact_tensor = torch.from_numpy(ground_truth_artifacts_for_plot)  # (trials, C, T)
ground_truth_neural_tensor = torch.from_numpy(ground_truth_neural_for_plot)  # (trials, C, T)

cov_a_gt = _cov(ground_truth_artifact_tensor)
nuc_a_gt = torch.linalg.norm(cov_a_gt, ord='nuc', dim=(-2, -1))
fro_a_gt = torch.linalg.norm(cov_a_gt, ord='fro', dim=(-2, -1))
soft_rank_a_gt = nuc_a_gt / (fro_a_gt + 1e-6)
print(f"Soft rank for artifact (GT): {torch.mean(soft_rank_a_gt).item():.4f}")

cov_s_gt = _cov(ground_truth_neural_tensor)
nuc_s_gt = torch.linalg.norm(cov_s_gt, ord='nuc', dim=(-2, -1))
fro_s_gt = torch.linalg.norm(cov_s_gt, ord='fro', dim=(-2, -1))
soft_rank_s_gt = nuc_s_gt / (fro_s_gt + 1e-6)
print(f"Soft rank for neural (GT): {torch.mean(soft_rank_s_gt).item():.4f}")

# Cosine similarity for ground truth
cos_sim_per_channel_gt = F.cosine_similarity(
    ground_truth_neural_tensor, ground_truth_artifact_tensor, dim=2, eps=1e-8
)
loss_cosine_gt = torch.mean(cos_sim_per_channel_gt**2)
print(f"Cosine similarity (GT): {loss_cosine_gt.item():.4f}")

# Evaluation metrics
evaluator = Evaluator(sampling_rate=data_obj.sampling_rate)
analyzer = NeuralAnalyzer(sampling_rate=data_obj.sampling_rate)
plotter = NeuralPlotter(evaluator)

trial_idx = TRAIN_TRIAL_IDX
channel_idx = 4

# Use the same trial from ground truth for comparison
ground_truth_neural_eval = ground_truth_neural[trial_idx:trial_idx+1]  # (1, C, T)
ground_truth_artifacts_eval = ground_truth_artifacts[trial_idx:trial_idx+1]  # (1, C, T)

noise_before = mixed_data_np - ground_truth_neural_eval
noise_after = predicted_neural_np - ground_truth_neural_eval

snr_before = evaluator.calculate_snr(ground_truth_neural_eval, noise_before)
snr_after = evaluator.calculate_snr(ground_truth_neural_eval, noise_after)
snr_improvement = evaluator.calculate_snr_improvement(
    mixed_data_np, predicted_neural_np, ground_truth_neural_eval
)

print(f"\nSNR Before (Mixed): {snr_before:.2f} dB")
print(f"SNR After (Cleaned): {snr_after:.2f} dB")
print(f"SNR Improvement: {snr_improvement:.2f} dB")

MSE = np.mean((predicted_neural_np - ground_truth_neural_eval) ** 2)
print(f"\nMSE: {MSE:.4f}")

psd_mse = analyzer.calculate_psd_mse(ground_truth_neural_eval, predicted_neural_np)
print(f"\nPSD MSE (max across channels): {np.max(psd_mse):.4f} at channel {np.argmax(psd_mse)}")
print(f"PSD MSE (min across channels): {np.min(psd_mse):.4f} at channel {np.argmin(psd_mse)}")
print(f"PSD MSE (median across channels): {np.median(psd_mse):.4f}")

coherence_neural = analyzer.calculate_spectral_coherence(ground_truth_neural_eval, predicted_neural_np)
print(f"\nMinimum Spectral Coherence: {np.min(coherence_neural):.4f} at channel {np.argmin(coherence_neural)}")
print(f"Maximum Spectral Coherence: {np.max(coherence_neural):.4f} at channel {np.argmax(coherence_neural)}")
print(f"Median Spectral Coherence: {np.median(coherence_neural):.4f}")

# Spectral analysis
def perform_spectral_analysis(data_3d, analyzer, signal_type, plot=False):
    def spectral_flatness(log_psd: np.ndarray) -> np.ndarray:
        mean_log_psd = np.mean(log_psd, axis=-1)
        geom_mean = np.exp(mean_log_psd)
        psd = np.exp(log_psd)
        arith_mean = np.mean(psd, axis=-1)
        flatness = geom_mean / (arith_mean + 1e-8)
        return np.mean(flatness)

    if data_3d is None or data_3d.ndim != 3:
        print(f"{signal_type}: Invalid 3D data for analysis.")
        return None, None
        
    w, c, n = data_3d.shape
    nperseg = min(n, 256)
    freqs, psd = analyzer.compute_psd(data_3d, nperseg=nperseg)
    
    mean_psd = np.mean(psd, axis=0)
    cumulative_power = np.cumsum(mean_psd)
    
    total_power = cumulative_power[-1]
    if total_power == 0:
        print(f"{signal_type}: Total power is zero.")
        return None, None
    
    threshold_95 = 0.95 * total_power
    idx_95 = np.searchsorted(cumulative_power, threshold_95)
    freq_95 = freqs[idx_95] if idx_95 < len(freqs) else freqs[-1]
    print(f"{signal_type}: 95% of total power is below {freq_95:.2f} Hz")
    
    idx_200 = np.searchsorted(freqs, 200)
    if idx_200 < len(cumulative_power):
        power_below_200 = cumulative_power[idx_200]
    else:
        power_below_200 = cumulative_power[-1]
    percent_below_200 = 100 * power_below_200 / total_power
    print(f"{signal_type}: {percent_below_200:.2f}% of total power is below 200 Hz")

    spectral_flatness_value = spectral_flatness(np.log(mean_psd + 1e-10))
    print(f"{signal_type}: Spectral flatness: {spectral_flatness_value:.4f}")
    
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
        plt.savefig(f'expert_psd_{signal_type.lower().replace(" ", "_")}.png', dpi=150)
        plt.show()
    
    return freqs, mean_psd

print("\n--- Spectral Analysis ---")
perform_spectral_analysis(ground_truth_neural_eval, analyzer, "GT Neural", plot=True)
perform_spectral_analysis(predicted_neural_np, analyzer, "Predicted Neural", plot=True)
perform_spectral_analysis(ground_truth_artifacts_eval, analyzer, "GT Artifact", plot=True)
perform_spectral_analysis(predicted_artifact_np, analyzer, "Predicted Artifact", plot=True)

# Plot signal comparisons
print("\n--- Plotting Signal Comparisons ---")

plotter.plot_cleaned_comparison(
    ground_truth=ground_truth_neural_for_plot,
    mixed_data=mixed_data_np,
    cleaned_data=predicted_neural_np,
    trial_idx=0,  # Use 0 since we only have 1 trial in the arrays
    channel_idx=channel_idx,
    title=f"Neural Signal Comparison - Trial {trial_idx}, Channel {channel_idx}",
    time_axis=True
)

plotter.plot_trace_comparison(
    cleaned=predicted_artifact_np,
    mixed_data=ground_truth_artifacts_for_plot,
    trial_idx=0,  # Use 0 since we only have 1 trial in the arrays
    channel_idx=channel_idx,
    title=f"Artifact Comparison: Predicted vs Ground Truth - Trial {trial_idx}, Channel {channel_idx}",
    time_axis=True
)

# Plot all four signals in one figure
fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

x_axis = np.arange(predicted_neural_np.shape[2]) / data_obj.sampling_rate

# Use index 0 since we only have 1 trial in predictions
plot_trial_idx = 0

# Plot 1: Ground Truth Neural Signal
axs[0].plot(x_axis, ground_truth_neural_for_plot[plot_trial_idx, channel_idx, :], 
            linewidth=1.5, color='#2ca02c', label='GT Neural')
axs[0].set_ylabel('Amplitude (µV)')
axs[0].set_title(f'Ground Truth Neural Signal - Trial {trial_idx}, Channel {channel_idx}')
axs[0].grid(True, alpha=0.3)
axs[0].legend()
axs[0].spines['top'].set_visible(False)
axs[0].spines['right'].set_visible(False)

# Plot 2: Cleaned (Predicted) Neural Signal
axs[1].plot(x_axis, predicted_neural_np[plot_trial_idx, channel_idx, :], 
            linewidth=1.5, color='#1f77b4', label='Predicted Neural')
axs[1].set_ylabel('Amplitude (µV)')
axs[1].set_title(f'Predicted Neural Signal - Trial {trial_idx}, Channel {channel_idx}')
axs[1].grid(True, alpha=0.3)
axs[1].legend()
axs[1].spines['top'].set_visible(False)
axs[1].spines['right'].set_visible(False)

# Plot 3: Ground Truth Artifact
axs[2].plot(x_axis, ground_truth_artifacts_for_plot[plot_trial_idx, channel_idx, :], 
            linewidth=1.5, color='#ff7f0e', label='GT Artifact')
axs[2].set_ylabel('Amplitude (µV)')
axs[2].set_title(f'Ground Truth Artifact - Trial {trial_idx}, Channel {channel_idx}')
axs[2].grid(True, alpha=0.3)
axs[2].legend()
axs[2].spines['top'].set_visible(False)
axs[2].spines['right'].set_visible(False)

# Plot 4: Predicted Artifact
axs[3].plot(x_axis, predicted_artifact_np[plot_trial_idx, channel_idx, :], 
            linewidth=1.5, color='#d62728', label='Predicted Artifact')
axs[3].set_xlabel('Time (s)')
axs[3].set_ylabel('Amplitude (µV)')
axs[3].set_title(f'Predicted Artifact - Trial {trial_idx}, Channel {channel_idx}')
axs[3].grid(True, alpha=0.3)
axs[3].legend()
axs[3].spines['top'].set_visible(False)
axs[3].spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('expert_signal_comparison.png', dpi=150)
plt.show()

print("\n--- Evaluation Complete ---")