import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from glob import glob

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import UNet1D, NeuralExpertAE
from data_utils import prepare_dataset
from sparc import DataHandler
from sparc.core.signal_data import ArtifactTriggers, SignalDataWithGroundTruth

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Trial index for evaluation (must match training)
TRAIN_TRIAL_IDX = 0

# ==============================================================================
# LOAD DATA
# ==============================================================================
data_handler = DataHandler()
data_obj_dict = data_handler.load_npz_data('../../data/added_artifacts_swec_data_512_lower_freq_wo0.npz')

data_obj = SignalDataWithGroundTruth(
    raw_data=data_obj_dict['mixed_data'],
    sampling_rate=data_obj_dict['sampling_rate'],
    ground_truth=data_obj_dict['ground_truth'],
    artifacts=data_obj_dict['artifacts'],
    artifact_markers=ArtifactTriggers(starts=data_obj_dict['artifact_markers']),
)

# Print statistics for each trial and channel
# print("\n" + "="*80)
# print("DATA STATISTICS")
# print("="*80)
# for trial_idx in range(data_obj.ground_truth.shape[0]):
#     print(f"\nTrial {trial_idx}:")
#     for channel_idx in range(data_obj.ground_truth.shape[1]):
#         channel_data = data_obj.ground_truth[trial_idx, channel_idx, :]
#         print(f"  Channel {channel_idx}: "
#               f"min={np.min(channel_data):.4f}, "
#               f"max={np.max(channel_data):.4f}, "
#               f"median={np.median(channel_data):.4f}, "
#               f"mean={np.mean(channel_data):.4f}")
# print("="*80 + "\n")


print(f"Data loaded: {data_obj.raw_data.shape}")

# ==============================================================================
# LOAD MODELS
# ==============================================================================
saved_models_dir = 'saved_models'

# Find the most recent model files
main_model_files = glob(os.path.join(saved_models_dir, 'unet1d_expert_guided_*.pth'))
expert_model_files = glob(os.path.join(saved_models_dir, 'neural_expert_*.pth'))

if not main_model_files:
    raise FileNotFoundError(f"No main model files found in {saved_models_dir}")
if not expert_model_files:
    raise FileNotFoundError(f"No expert model files found in {saved_models_dir}")

# Get most recent models (sorted by filename which includes timestamp)
main_model_path = sorted(main_model_files)[-1]
expert_model_path  = sorted(expert_model_files)[-1]

print(f"Loading main model from: {main_model_path}")
print(f"Loading expert model from: {expert_model_path}")

# Load main model checkpoint
main_checkpoint = torch.load(main_model_path, map_location=DEVICE, weights_only=False)
IN_CHANNELS = main_checkpoint['in_channels']
OUT_CHANNELS = main_checkpoint['out_channels']
mixed_mean = main_checkpoint['mixed_mean'].to(DEVICE)
mixed_std = main_checkpoint['mixed_std'].to(DEVICE)

# Create and load main model
main_model = UNet1D(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS).to(DEVICE)
main_model.load_state_dict(main_checkpoint['model_state_dict'])
main_model.eval()
print(f"Main model loaded: {IN_CHANNELS} input channels, {OUT_CHANNELS} output channels")

# Load expert model checkpoint
expert_checkpoint = torch.load(expert_model_path, map_location=DEVICE, weights_only=False)
expert_in_channels = expert_checkpoint['in_channels']
expert_mean = expert_checkpoint['expert_mean'].to(DEVICE)
expert_std = expert_checkpoint['expert_std'].to(DEVICE)

# Create and load expert model
neural_expert = NeuralExpertAE(in_channels=expert_in_channels).to(DEVICE)
neural_expert.load_state_dict(expert_checkpoint['model_state_dict'])
neural_expert.eval()
print(f"Expert model loaded: {expert_in_channels} input channels")

# ==============================================================================
# PREPARE DATA
# ==============================================================================
# Use trial_indices=[3] to match training, and use saved normalization stats
# Note: We'll use the saved mixed_mean and mixed_std from checkpoint for denormalization
main_ds, main_loader, artifact_masks, computed_mixed_mean, computed_mixed_std = prepare_dataset(
    data_obj=data_obj,
    data_obj_dict=data_obj_dict,
    batch_size=1,
    use_normalization=True,
    shuffle=False,
    artifact_duration_ms=40,
    trial_indices=[TRAIN_TRIAL_IDX]  # Match training: use trial 3
)

# Verify normalization stats match (they should be very close)
print(f"\nNormalization stats check:")
print(f"  Saved mixed_mean: {mixed_mean.mean().item():.6f}")
print(f"  Computed mixed_mean: {computed_mixed_mean.mean().item():.6f}")
print(f"  Saved mixed_std: {mixed_std.mean().item():.6f}")
print(f"  Computed mixed_std: {computed_mixed_std.mean().item():.6f}")
print(f"  Mean diff: {torch.abs(mixed_mean - computed_mixed_mean.to(DEVICE)).mean().item():.6f}")
print(f"  Std diff: {torch.abs(mixed_std - computed_mixed_std.to(DEVICE)).mean().item():.6f}")

# Use saved stats for denormalization (they were used during training)
# But the dataset normalization uses computed stats, which should be the same

# ==============================================================================
# INFERENCE
# ==============================================================================
print("\n--- Running Inference ---")

with torch.no_grad():
    # Get first batch
    batch = next(iter(main_loader))
    x_mixed = batch[0].to(DEVICE)
    x_stim = batch[1].to(DEVICE)
    mask = batch[2].to(DEVICE)
    
    # Predictions
    a_pred = main_model(x_mixed, x_stim)
    s_pred = x_mixed - a_pred
    
    # Denormalize predictions
    s_pred_denorm = s_pred * mixed_std + mixed_mean
    a_pred_denorm = a_pred * mixed_std + mixed_mean
    x_mixed_denorm = x_mixed * mixed_std + mixed_mean

# Convert to numpy
predicted_neural_np = s_pred_denorm.cpu().numpy()  # (1, C, T)
predicted_artifact_np = a_pred_denorm.cpu().numpy()  # (1, C, T)
mixed_data_np = x_mixed_denorm.cpu().numpy()  # (1, C, T)

# Get all ground truth data (all trials)
ground_truth_neural_all = data_obj.ground_truth.astype(np.float32)  # (N, C, T)
ground_truth_artifacts_all = data_obj.artifacts.astype(np.float32)  # (N, C, T)

print(f"Predicted neural shape: {predicted_neural_np.shape}")
print(f"Ground truth neural shape (all trials): {ground_truth_neural_all.shape}")

# ==============================================================================
# PLOTTING
# ==============================================================================
# Since we're using trial_indices=[TRAIN_TRIAL_IDX], the first batch is trial TRAIN_TRIAL_IDX
# But we need to use trial_idx=TRAIN_TRIAL_IDX to index into the full ground truth array
trial_idx = TRAIN_TRIAL_IDX  # Original trial index in full dataset
channel_idx = 85

# Select the corresponding trial from ground truth for comparison
ground_truth_neural = ground_truth_neural_all[trial_idx:trial_idx+1]  # (1, C, T)
ground_truth_artifacts = ground_truth_artifacts_all[trial_idx:trial_idx+1]  # (1, C, T)

# Use index 0 since we only have 1 trial in predictions
plot_trial_idx = 0

# Create time axis
time_axis = np.arange(predicted_neural_np.shape[2]) / data_obj.sampling_rate

# Create figure with subplots
fig, axs = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

# Plot 1: Ground Truth Neural Signal
axs[0].plot(time_axis, ground_truth_neural[plot_trial_idx, channel_idx, :], 
            linewidth=1.5, color='#2ca02c', label='Ground Truth Neural', alpha=0.8)
axs[0].set_ylabel('Amplitude (µV)', fontsize=11)
axs[0].set_title(f'Ground Truth Neural Signal - Trial {trial_idx}, Channel {channel_idx}', fontsize=12, fontweight='bold')
axs[0].grid(True, alpha=0.3)
axs[0].legend(loc='upper right', fontsize=10)
axs[0].spines['top'].set_visible(False)
axs[0].spines['right'].set_visible(False)

# Plot 2: Predicted Neural Signal
axs[1].plot(time_axis, predicted_neural_np[plot_trial_idx, channel_idx, :], 
            linewidth=1.5, color='#1f77b4', label='Predicted Neural', alpha=0.8)
axs[1].set_ylabel('Amplitude (µV)', fontsize=11)
axs[1].set_title(f'Predicted Neural Signal - Trial {trial_idx}, Channel {channel_idx}', fontsize=12, fontweight='bold')
axs[1].grid(True, alpha=0.3)
axs[1].legend(loc='upper right', fontsize=10)
axs[1].spines['top'].set_visible(False)
axs[1].spines['right'].set_visible(False)

# Plot 3: Ground Truth Artifact
axs[2].plot(time_axis, ground_truth_artifacts[plot_trial_idx, channel_idx, :], 
            linewidth=1.5, color='#ff7f0e', label='Ground Truth Artifact', alpha=0.8)
axs[2].set_ylabel('Amplitude (µV)', fontsize=11)
axs[2].set_title(f'Ground Truth Artifact - Trial {trial_idx}, Channel {channel_idx}', fontsize=12, fontweight='bold')
axs[2].grid(True, alpha=0.3)
axs[2].legend(loc='upper right', fontsize=10)
axs[2].spines['top'].set_visible(False)
axs[2].spines['right'].set_visible(False)

# Plot 4: Predicted Artifact
axs[3].plot(time_axis, predicted_artifact_np[plot_trial_idx, channel_idx, :], 
            linewidth=1.5, color='#d62728', label='Predicted Artifact', alpha=0.8)
axs[3].set_xlabel('Time (s)', fontsize=11)
axs[3].set_ylabel('Amplitude (µV)', fontsize=11)
axs[3].set_title(f'Predicted Artifact - Trial {trial_idx}, Channel {channel_idx}', fontsize=12, fontweight='bold')
axs[3].grid(True, alpha=0.3)
axs[3].legend(loc='upper right', fontsize=10)
axs[3].spines['top'].set_visible(False)
axs[3].spines['right'].set_visible(False)

plt.tight_layout()
output_path = 'expert_model_predictions.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nPlot saved to: {output_path}")
plt.show()

# ==============================================================================
# COMPARISON PLOT: Predicted vs Ground Truth Overlaid
# ==============================================================================
fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Plot 1: Neural Signal Comparison
axs[0].plot(time_axis, ground_truth_neural[plot_trial_idx, channel_idx, :], 
            linewidth=2, color='#2ca02c', label='Ground Truth Neural', alpha=0.7)
axs[0].plot(time_axis, predicted_neural_np[plot_trial_idx, channel_idx, :], 
            linewidth=1.5, color='#1f77b4', label='Predicted Neural', alpha=0.7, linestyle='--')
axs[0].set_ylabel('Amplitude (µV)', fontsize=11)
axs[0].set_title(f'Neural Signal: Predicted vs Ground Truth - Trial {trial_idx}, Channel {channel_idx}', 
                 fontsize=12, fontweight='bold')
axs[0].grid(True, alpha=0.3)
axs[0].legend(loc='upper right', fontsize=10)
axs[0].spines['top'].set_visible(False)
axs[0].spines['right'].set_visible(False)

# Plot 2: Artifact Comparison
axs[1].plot(time_axis, ground_truth_artifacts[plot_trial_idx, channel_idx, :], 
            linewidth=2, color='#ff7f0e', label='Ground Truth Artifact', alpha=0.7)
axs[1].plot(time_axis, predicted_artifact_np[plot_trial_idx, channel_idx, :], 
            linewidth=1.5, color='#d62728', label='Predicted Artifact', alpha=0.7, linestyle='--')
axs[1].set_xlabel('Time (s)', fontsize=11)
axs[1].set_ylabel('Amplitude (µV)', fontsize=11)
axs[1].set_title(f'Artifact: Predicted vs Ground Truth - Trial {trial_idx}, Channel {channel_idx}', 
                fontsize=12, fontweight='bold')
axs[1].grid(True, alpha=0.3)
axs[1].legend(loc='upper right', fontsize=10)
axs[1].spines['top'].set_visible(False)
axs[1].spines['right'].set_visible(False)

plt.tight_layout()
output_path_comparison = 'expert_model_comparison.png'
plt.savefig(output_path_comparison, dpi=150, bbox_inches='tight')
print(f"Comparison plot saved to: {output_path_comparison}")
plt.show()

# ==============================================================================
# CALCULATE METRICS
# ==============================================================================
from sparc.core.evaluator import Evaluator

evaluator = Evaluator(sampling_rate=data_obj.sampling_rate)

noise_before = mixed_data_np - ground_truth_neural
noise_after = predicted_neural_np - ground_truth_neural

snr_before = evaluator.calculate_snr(ground_truth_neural, noise_before)
snr_after = evaluator.calculate_snr(ground_truth_neural, noise_after)
snr_improvement = evaluator.calculate_snr_improvement(
    mixed_data_np, predicted_neural_np, ground_truth_neural
)

mse_neural = np.mean((predicted_neural_np - ground_truth_neural) ** 2)
mse_artifact = np.mean((predicted_artifact_np - ground_truth_artifacts) ** 2)

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
ground_truth_artifact_tensor = torch.from_numpy(ground_truth_artifacts)  # (trials, C, T)
ground_truth_neural_tensor = torch.from_numpy(ground_truth_neural)  # (trials, C, T)

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

print("\n--- Evaluation Metrics ---")
print(f"SNR Before (Mixed): {snr_before:.2f} dB")
print(f"SNR After (Cleaned): {snr_after:.2f} dB")
print(f"SNR Improvement: {snr_improvement:.2f} dB")
print(f"MSE (Neural): {mse_neural:.4f}")
print(f"MSE (Artifact): {mse_artifact:.4f}")

print("\n--- Done ---")

