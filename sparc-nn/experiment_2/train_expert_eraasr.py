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

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import UNet1D, NeuralExpertAE
from data_utils import compute_robust_stats, ArtifactDataset
from torch.utils.data import DataLoader
from sparc import DataHandler
from sparc.core.signal_data import SignalData
from sparc.core.neural_analyzer import NeuralAnalyzer
from sparc.core.plotting import NeuralPlotter
from loss import PhysicsLoss

parser = argparse.ArgumentParser(description='Train expert-guided U-Net on ERAASR dataset')
parser.add_argument('--num-epochs', type=int, default=400,
                    help='Number of epochs for main model training (default: 400)')
parser.add_argument('--eraasr-file', type=str, default='../../data/eraasr_1000.npz',
                    help='Path to ERAASR .npz file (default: ../../data/eraasr_1000.npz)')
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
print(f"Configuration: NUM_EPOCHS={args.num_epochs}")

# Trial index for U-Net training and evaluation
TRAIN_TRIAL_IDX = 0

# ==============================================================================
# LOAD ERAASR DATA (NO GROUND TRUTH)
# ==============================================================================
print("\n" + "=" * 60)
print("Loading ERAASR Data")
print("=" * 60)

data_handler = DataHandler()
eraasr_data = data_handler.load_npz_data(args.eraasr_file)

# ERAASR data is stored as 'arr_0' in npz file
if 'arr_0' in eraasr_data:
    orig_data = eraasr_data['arr_0']
else:
    # Try to get the first array if 'arr_0' doesn't exist
    keys = list(eraasr_data.keys())
    if len(keys) > 0:
        orig_data = eraasr_data[keys[0]]
    else:
        raise ValueError(f"Could not find data in ERAASR file: {args.eraasr_file}")

print(f"Raw ERAASR data shape: {orig_data.shape}")

# Ensure data is in (trials, channels, time) format
if orig_data.ndim == 2:
    # If 2D, assume (channels, time) and add trial dimension
    orig_data = orig_data[np.newaxis, :, :]
    print(f"Added trial dimension, new shape: {orig_data.shape}")
elif orig_data.ndim == 3:
    # Check if it's (trials, time, channels) and transpose to (trials, channels, time)
    if orig_data.shape[1] > orig_data.shape[2]:  # Likely (trials, time, channels)
        orig_data = orig_data.transpose(0, 2, 1)  # (trials, channels, time)
        print(f"Transposed to (trials, channels, time), new shape: {orig_data.shape}")

# Create SignalData object (no ground truth, no artifact markers)
# ERAASR doesn't have artifact indices, so we'll create dummy masks
data_obj = SignalData(
    raw_data=orig_data,
    sampling_rate=1000,  # Default ERAASR sampling rate
    artifact_markers=None  # No artifact markers available
)

print(f"ERAASR Data loaded: {data_obj.raw_data.shape}")
print(f"Sampling rate: {data_obj.sampling_rate} Hz")
print(f"Number of trials: {data_obj.raw_data.shape[0]}")

# ==============================================================================
# CONFIGURATION
# ==============================================================================
LEARNING_RATE = 1e-3
BATCH_SIZE = 1
NUM_EPOCHS = args.num_epochs
OUT_CHANNELS = data_obj.raw_data.shape[1]
IN_CHANNELS = OUT_CHANNELS + 1  # +1 for dummy stim_trace (model concatenates x and stim_trace)

# ==============================================================================
# LOAD EXPERT MODEL (DO NOT TRAIN)
# ==============================================================================
print("\n" + "=" * 60)
print("Loading Expert Model")
print("=" * 60)

expert_model_files = glob(os.path.join('saved_models', 'neural_expert_*.pth'))
if not expert_model_files:
    raise FileNotFoundError("No expert model found in saved_models. Please train expert first using train_expert.py")

expert_model_path = sorted(expert_model_files)[-1]
print(f"Loading expert model from: {expert_model_path}")

expert_checkpoint = torch.load(expert_model_path, map_location=DEVICE, weights_only=False)
expert_in_channels = expert_checkpoint['in_channels']
expert_mean = expert_checkpoint['expert_mean'].to(DEVICE)
expert_std = expert_checkpoint['expert_std'].to(DEVICE)

# Check if expert channel count matches ERAASR data
eraasr_channels = OUT_CHANNELS
if expert_in_channels != eraasr_channels:
    print(f"\n⚠️  WARNING: Expert model expects {expert_in_channels} channels, but ERAASR data has {eraasr_channels} channels.")
    print(f"   Expert loss will be disabled. Training will use physics loss and anchor loss only.")
    USE_EXPERT = False
    neural_expert = None
else:
    USE_EXPERT = True
    neural_expert = NeuralExpertAE(in_channels=expert_in_channels).to(DEVICE)
    neural_expert.load_state_dict(expert_checkpoint['model_state_dict'])
    
    # Freeze expert
    for param in neural_expert.parameters():
        param.requires_grad = False
    neural_expert.eval()
    print("Expert model loaded and frozen.")

# ==============================================================================
# PREPARE ERAASR DATASET
# ==============================================================================
print("\n" + "=" * 60)
print("Preparing ERAASR Dataset")
print("=" * 60)

# ERAASR doesn't have ground truth or stim_trace, so we need to create them
mixed_data = data_obj.raw_data.astype(np.float32)

# Select specific trial if needed
if TRAIN_TRIAL_IDX is not None:
    mixed_data = mixed_data[TRAIN_TRIAL_IDX:TRAIN_TRIAL_IDX+1]
    print(f"Using trial {TRAIN_TRIAL_IDX} for training")

# Calculate robust statistics
print("Calculating Robust Statistics...")
data_mean, data_std = compute_robust_stats(mixed_data)
print(f"Robust Norm Stats - Mean: {data_mean.mean():.4f}, Std: {data_std.mean():.4f}")

# Create dummy artifact masks (all zeros = no artifacts marked)
# Since ERAASR doesn't have artifact indices, we create uniform masks
# Option: all zeros (no artifacts) or all ones (everything potentially artifact)
# Using all zeros means the model will rely on physics loss and expert loss only
n_trials, n_channels, n_samples = mixed_data.shape
artifact_masks = torch.zeros((n_trials, n_samples), dtype=torch.float32)
print(f"Created dummy artifact masks: shape {artifact_masks.shape} (all zeros - no artifact regions marked)")

# ERAASR doesn't have stim_trace, so we'll pass None/zeros to the model
# But since model expects it, create dummy zeros (though model won't use it if IN_CHANNELS = OUT_CHANNELS)
n_trials, n_channels, n_samples = mixed_data.shape
stim_trace = np.zeros((n_trials, 1, n_samples), dtype=np.float32)
stim_trace_tensor = torch.from_numpy(stim_trace).float()

# Create dataset
dataset = ArtifactDataset(
    data=mixed_data,
    stim_trace=stim_trace_tensor,
    artifact_masks=artifact_masks,
    data_mean=data_mean,
    data_std=data_std,
    use_normalization=True
)

data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"Dataset prepared: {len(dataset)} samples")

# ==============================================================================
# TRAIN U-NET
# ==============================================================================
main_model = UNet1D(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS).to(DEVICE)
optimizer_main = optim.Adam(main_model.parameters(), lr=LEARNING_RATE)

physics_loss = PhysicsLoss(
    w_cosine=1.0,
    w_rank=1.0,
    w_smooth=1.0,
    w_spectral=1.0,
    sampling_rate=data_obj.sampling_rate,
    f_cutoff=10.0,
).to(DEVICE)

print("\n" + "=" * 60)
print("Training U-Net on ERAASR Data")
print("=" * 60)

for epoch in range(NUM_EPOCHS):
    model_loss = 0
    for batch in data_loader:
        x_mixed = batch[0].to(DEVICE)
        x_stim = batch[1].to(DEVICE)  # Dummy zeros, not used if IN_CHANNELS = OUT_CHANNELS
        mask = batch[2].to(DEVICE)
        
        # Model concatenates x_mixed and x_stim, so IN_CHANNELS = OUT_CHANNELS + 1
        # x_stim is dummy zeros since ERAASR doesn't have real stim
        a_pred = main_model(x_mixed, x_stim)
        s_pred = x_mixed - a_pred
        
        # Since ERAASR has no artifact markers, mask is all zeros
        # This means: is_clean = all ones, is_dirty = all zeros
        is_clean = (mask < 0.1).float()
        is_dirty = (mask > 0.5).float()
        
        # Anchor loss: artifacts should be small everywhere (since we don't know artifact locations)
        # Apply anchor loss to all regions (is_clean will be all ones)
        if is_clean.sum() > 0:
            loss_anchor = (a_pred ** 2 * is_clean).sum() / (is_clean.sum() + 1e-8)
        else:
            loss_anchor = 0.0
        
        # Expert loss: neural signal should be reconstructible by expert everywhere
        # Since we don't have artifact markers, apply expert loss to all regions
        if USE_EXPERT and neural_expert is not None:
            s_pred_recon = neural_expert(s_pred)
            # Use all regions for expert loss (not just dirty regions)
            loss_expert_projection = torch.mean((s_pred - s_pred_recon) ** 2)
        else:
            loss_expert_projection = 0.0
        
        # Physics loss
        physics_loss_dict = physics_loss(s_pred, a_pred)
        physics_loss_value = (
            2 * physics_loss_dict['cosine'] +
            1.5 * physics_loss_dict['rank_a'] +
            0.2 * physics_loss_dict['rank_s_penalty'] +
            3 * physics_loss_dict['spectral'] +
            0.5 * physics_loss_dict['spectral_slope_s']
        )
        
        # Adjust loss weights based on whether expert is available
        if USE_EXPERT and neural_expert is not None:
            final_loss = 1 * physics_loss_value + 1 * loss_anchor + 1 * loss_expert_projection
        else:
            # Without expert, rely more on physics loss
            final_loss = 1.5 * physics_loss_value + 1 * loss_anchor
        
        optimizer_main.zero_grad()
        final_loss.backward()
        optimizer_main.step()
        model_loss += final_loss.item()
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch}: Loss {model_loss / len(data_loader):.6f}")

print("\n--- Training Complete ---")

# ==============================================================================
# SAVE MODEL
# ==============================================================================
model_save_dir = 'saved_models'
os.makedirs(model_save_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
main_model_path = os.path.join(model_save_dir, f'unet1d_expert_guided_eraasr_{timestamp}.pth')

torch.save({
    'epoch': NUM_EPOCHS,
    'model_state_dict': main_model.state_dict(),
    'optimizer_state_dict': optimizer_main.state_dict(),
    'loss': model_loss / len(data_loader),
    'in_channels': IN_CHANNELS,
    'out_channels': OUT_CHANNELS,
    'sampling_rate': data_obj.sampling_rate,
    'expert_mean': expert_mean.cpu() if USE_EXPERT else None,
    'expert_std': expert_std.cpu() if USE_EXPERT else None,
    'use_expert': USE_EXPERT,
    'mixed_mean': data_mean.cpu(),
    'mixed_std': data_std.cpu(),
    'dataset': 'eraasr',
    'trial_idx': TRAIN_TRIAL_IDX,
}, main_model_path)

print(f"Model saved to: {main_model_path}")

# ==============================================================================
# EVALUATION (NO GROUND TRUTH - LIMITED METRICS)
# ==============================================================================
print("\n--- Evaluation (No Ground Truth Available) ---")

main_model.eval()
if USE_EXPERT and neural_expert is not None:
    neural_expert.eval()

# Prepare evaluation data (unshuffled)
eval_dataset = ArtifactDataset(
    data=mixed_data,
    stim_trace=stim_trace_tensor,
    artifact_masks=artifact_masks,
    data_mean=data_mean,
    data_std=data_std,
    use_normalization=True
)
eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)

with torch.no_grad():
    batch = next(iter(eval_loader))
    x_mixed = batch[0].to(DEVICE)
    x_stim = batch[1].to(DEVICE)
    mask = batch[2].to(DEVICE)
    
    # Predictions
    a_pred = main_model(x_mixed, x_stim, mask)
    s_pred = x_mixed - a_pred
    
    # Denormalize
    s_pred_denorm = s_pred * data_std.to(DEVICE) + data_mean.to(DEVICE)
    a_pred_denorm = a_pred * data_std.to(DEVICE) + data_mean.to(DEVICE)
    x_mixed_denorm = x_mixed * data_std.to(DEVICE) + data_mean.to(DEVICE)

# Convert to numpy
predicted_neural_np = s_pred_denorm.cpu().numpy()
predicted_artifact_np = a_pred_denorm.cpu().numpy()
mixed_data_np = x_mixed_denorm.cpu().numpy()

# Soft rank calculation
def _cov(M: torch.Tensor) -> torch.Tensor:
    T, C, N = M.shape
    M_mean = torch.mean(M, dim=-1, keepdim=True)
    M_centered = M - M_mean
    Cov = (M_centered @ M_centered.transpose(-1, -2)) / (N - 1)
    return Cov

predicted_artifact_tensor = torch.from_numpy(predicted_artifact_np)
predicted_neural_tensor = torch.from_numpy(predicted_neural_np)

cov_a = _cov(predicted_artifact_tensor)
nuc_a = torch.linalg.norm(cov_a, ord='nuc', dim=(-2, -1))
fro_a = torch.linalg.norm(cov_a, ord='fro', dim=(-2, -1))
soft_rank_a = nuc_a / (fro_a + 1e-6)
print(f"Soft rank for artifact: {torch.mean(soft_rank_a).item():.4f}")

cov_s = _cov(predicted_neural_tensor)
nuc_s = torch.linalg.norm(cov_s, ord='nuc', dim=(-2, -1))
fro_s = torch.linalg.norm(cov_s, ord='fro', dim=(-2, -1))
soft_rank_s = nuc_s / (fro_s + 1e-6)
print(f"Soft rank for neural: {torch.mean(soft_rank_s).item():.4f}")

# Cosine similarity
cos_sim_per_channel = F.cosine_similarity(
    predicted_neural_tensor, predicted_artifact_tensor, dim=2, eps=1e-8
)
loss_cosine = torch.mean(cos_sim_per_channel**2)
print(f"Cosine similarity: {loss_cosine.item():.4f}")

# Spectral analysis
analyzer = NeuralAnalyzer(sampling_rate=data_obj.sampling_rate)

# PSD analysis
freqs, psd_neural = analyzer.compute_psd(predicted_neural_np)
freqs, psd_artifact = analyzer.compute_psd(predicted_artifact_np)
freqs, psd_mixed = analyzer.compute_psd(mixed_data_np)

mean_psd_neural = np.mean(psd_neural, axis=0)
mean_psd_artifact = np.mean(psd_artifact, axis=0)
mean_psd_mixed = np.mean(psd_mixed, axis=0)

print("\n--- Spectral Analysis ---")
print(f"Neural signal: 95% power below {np.percentile(freqs, 95):.2f} Hz")
print(f"Artifact signal: 95% power below {np.percentile(freqs, 95):.2f} Hz")

# ==============================================================================
# PLOTTING
# ==============================================================================
print("\n--- Plotting Results ---")

channel_idx = 0
plot_trial_idx = 0

time_axis = np.arange(predicted_neural_np.shape[2]) / data_obj.sampling_rate

# Four-panel plot
fig, axs = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

# Plot 1: Mixed Signal
axs[0].plot(time_axis, mixed_data_np[plot_trial_idx, channel_idx, :], 
            linewidth=1.5, color='#9467bd', label='Mixed Signal', alpha=0.8)
axs[0].set_ylabel('Amplitude (µV)', fontsize=11)
axs[0].set_title(f'Mixed Signal - Trial {TRAIN_TRIAL_IDX}, Channel {channel_idx}', fontsize=12, fontweight='bold')
axs[0].grid(True, alpha=0.3)
axs[0].legend(loc='upper right', fontsize=10)
axs[0].spines['top'].set_visible(False)
axs[0].spines['right'].set_visible(False)

# Plot 2: Predicted Neural Signal
axs[1].plot(time_axis, predicted_neural_np[plot_trial_idx, channel_idx, :], 
            linewidth=1.5, color='#1f77b4', label='Predicted Neural', alpha=0.8)
axs[1].set_ylabel('Amplitude (µV)', fontsize=11)
axs[1].set_title(f'Predicted Neural Signal - Trial {TRAIN_TRIAL_IDX}, Channel {channel_idx}', fontsize=12, fontweight='bold')
axs[1].grid(True, alpha=0.3)
axs[1].legend(loc='upper right', fontsize=10)
axs[1].spines['top'].set_visible(False)
axs[1].spines['right'].set_visible(False)

# Plot 3: Predicted Artifact
axs[2].plot(time_axis, predicted_artifact_np[plot_trial_idx, channel_idx, :], 
            linewidth=1.5, color='#d62728', label='Predicted Artifact', alpha=0.8)
axs[2].set_ylabel('Amplitude (µV)', fontsize=11)
axs[2].set_title(f'Predicted Artifact - Trial {TRAIN_TRIAL_IDX}, Channel {channel_idx}', fontsize=12, fontweight='bold')
axs[2].grid(True, alpha=0.3)
axs[2].legend(loc='upper right', fontsize=10)
axs[2].spines['top'].set_visible(False)
axs[2].spines['right'].set_visible(False)

# Plot 4: PSD Comparison
axs[3].semilogy(freqs, mean_psd_mixed[channel_idx, :], 
                linewidth=1.5, color='#9467bd', label='Mixed', alpha=0.7)
axs[3].semilogy(freqs, mean_psd_neural[channel_idx, :], 
                linewidth=1.5, color='#1f77b4', label='Neural', alpha=0.7)
axs[3].semilogy(freqs, mean_psd_artifact[channel_idx, :], 
                linewidth=1.5, color='#d62728', label='Artifact', alpha=0.7)
axs[3].set_xlabel('Frequency (Hz)', fontsize=11)
axs[3].set_ylabel('Power Spectral Density', fontsize=11)
axs[3].set_title(f'Power Spectral Density - Trial {TRAIN_TRIAL_IDX}, Channel {channel_idx}', fontsize=12, fontweight='bold')
axs[3].grid(True, alpha=0.3)
axs[3].legend(loc='upper right', fontsize=10)
axs[3].spines['top'].set_visible(False)
axs[3].spines['right'].set_visible(False)

plt.tight_layout()
output_path = f'eraasr_results_trial_{TRAIN_TRIAL_IDX}.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Plot saved to: {output_path}")
plt.show()

print("\n--- Evaluation Complete ---")

