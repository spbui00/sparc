import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import json
import torch.nn.functional as F
from datetime import datetime
from glob import glob
from pathlib import Path
from sparc import DataHandler, SignalDataWithGroundTruth
from sparc.core.signal_data import ArtifactTriggers
from sparc.core.plotting import NeuralPlotter
from sparc.core.evaluator import Evaluator
from sparc.core.neural_analyzer import NeuralAnalyzer
from models import UNet1D, NeuralExpertAE
from loss import PhysicsLoss
from uncertainty_loss import UncertaintyWeightedLoss
from analyze_sweep import analyze_sweep, parse_results_file, parse_config_from_dirname, compute_composite_score

LEARNING_RATE = 1e-3
BATCH_SIZE = 1
NUM_EPOCHS = 200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

SWEEP_1_DIR = "sweep_1"
SWEEP_2_DIR = "sweep_2"
os.makedirs(SWEEP_2_DIR, exist_ok=True)

# Load data
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
artifact_indices_raw = data_obj.artifact_markers.starts

N_SAMPLES = mixed_data.shape[2]
N_DATA_CHANNELS = mixed_data.shape[1]
IN_CHANNELS = N_DATA_CHANNELS + 1
OUT_CHANNELS = N_DATA_CHANNELS
N_TRIALS = mixed_data.shape[0]

print("Precomputing artifact masks...")
artifact_masks = []
for trial_idx in range(N_TRIALS):
    if artifact_indices_raw.ndim == 3:
        trial_indices = artifact_indices_raw[trial_idx].flatten()
    elif artifact_indices_raw.ndim == 2:
        trial_indices = artifact_indices_raw[trial_idx]
    else:
        trial_indices = artifact_indices_raw
    
    trial_indices = trial_indices[trial_indices >= 0]
    trial_durations = [ARTIFACT_DURATION] * len(trial_indices)
    mask = create_soft_artifact_mask(trial_indices, trial_durations, N_SAMPLES, blur_radius=5)
    artifact_masks.append(mask)

artifact_masks = torch.stack(artifact_masks)
stim_trace_tensor = torch.from_numpy(data_obj_dict['stim_trace']).float()

data_mean = torch.mean(torch.from_numpy(mixed_data).float(), dim=(0, 2), keepdim=True)
data_std = torch.std(torch.from_numpy(mixed_data).float(), dim=(0, 2), keepdim=True)

dataset = ArtifactDataset(mixed_data, stim_trace_tensor, artifact_masks, data_mean, data_std)
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Load expert model
print("\nLoading expert model...")
expert_model_files = glob(os.path.join('../saved_models', 'neural_expert_*.pth'))
if not expert_model_files:
    raise FileNotFoundError(
        "No expert model found in saved_models/. "
        "Please train an expert first using train_expert.py"
    )
expert_model_path = sorted(expert_model_files)[-1]
print(f"Using most recent expert model: {expert_model_path}")

expert_checkpoint = torch.load(expert_model_path, map_location=DEVICE, weights_only=False)
expert_in_channels = expert_checkpoint.get('in_channels', OUT_CHANNELS)

neural_expert = NeuralExpertAE(in_channels=expert_in_channels).to(DEVICE)
neural_expert.load_state_dict(expert_checkpoint['model_state_dict'])

# Freeze expert
for param in neural_expert.parameters():
    param.requires_grad = False
neural_expert.eval()
print(f"Expert model loaded and frozen: {expert_in_channels} input channels")

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

def config_to_string(base_config, w_expert, w_anchor):
    """Create config string including expert weights"""
    parts = []
    for key, val in sorted(base_config.items()):
        if isinstance(val, float):
            parts.append(f"{key}_{val:.2f}")
        else:
            parts.append(f"{key}_{val}")
    parts.append(f"w_expert_{w_expert:.2f}")
    parts.append(f"w_anchor_{w_anchor:.2f}")
    return "_".join(parts)

def config_exists(config_str):
    config_dir = os.path.join(SWEEP_2_DIR, config_str)
    results_file = os.path.join(config_dir, "results.txt")
    return os.path.exists(results_file)

def run_training(base_config, w_expert, w_anchor):
    config_str = config_to_string(base_config, w_expert, w_anchor)
    config_dir = os.path.join(SWEEP_2_DIR, config_str)
    os.makedirs(config_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Running config: {config_str}")
    print(f"Base config: {base_config}")
    print(f"w_expert: {w_expert}, w_anchor: {w_anchor}")
    print(f"{'='*80}\n")
    
    model = UNet1D(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS).to(DEVICE)
    
    model_info = get_model_architecture_info(model)
    with open(os.path.join(config_dir, "model_architecture.json"), 'w') as f:
        json.dump(model_info, f, indent=2)
    
    criterion = PhysicsLoss(
        f_cutoff=base_config['f_cutoff'],
        sampling_rate=data_obj.sampling_rate,
    ).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    loss_history = []
    results_output = []
    
    def log_and_save(msg):
        print(msg)
        results_output.append(msg)
    
    log_and_save(f"Base config: {base_config}")
    log_and_save(f"w_expert: {w_expert}, w_anchor: {w_anchor}")
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
            
            # Compute physics loss
            physics_loss_value = (
                base_config['w_cosine'] * raw_loss_dict['cosine'] +
                base_config['w_rank_s'] * raw_loss_dict['rank_s_penalty'] +
                base_config['w_spectral'] * raw_loss_dict['spectral'] +
                base_config['w_spectral_slope'] * raw_loss_dict['spectral_slope_s'] +
                base_config['w_rank_a'] * raw_loss_dict['rank_a']
            )
            
            # Expert and anchor losses
            is_clean = (artifact_mask_batch < 0.1).float()
            is_dirty = (artifact_mask_batch > 0.5).float()
            
            loss_anchor = torch.tensor(0.0, device=DEVICE)
            if is_clean.sum() > 0:
                loss_anchor = (a_pred_batch ** 2 * is_clean).sum() / (is_clean.sum() + 1e-8)
            
            loss_expert_projection = torch.tensor(0.0, device=DEVICE)
            if is_dirty.sum() > 0:
                s_pred_recon = neural_expert(s_pred_batch)
                loss_expert_projection = ((s_pred_batch - s_pred_recon) ** 2 * is_dirty).sum() / (is_dirty.sum() + 1e-8)
            
            # Final loss
            final_loss = physics_loss_value + w_anchor * loss_anchor + w_expert * loss_expert_projection
            
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
    
    # Evaluation
    model.eval()
    eval_dataset = ArtifactDataset(mixed_data, stim_trace_tensor, artifact_masks, data_mean, data_std)
    eval_loader = DataLoader(eval_dataset, batch_size=N_TRIALS, shuffle=False)
    
    with torch.no_grad():
        new_mixed_signal_norm, new_stim_trace, artifact_mask_batch = next(iter(eval_loader))
        new_mixed_signal_norm_dev = new_mixed_signal_norm.to(DEVICE)
        new_stim_trace_dev = new_stim_trace.to(DEVICE)
        artifact_mask_batch_dev = artifact_mask_batch.to(DEVICE)
        data_mean_dev = data_mean.to(DEVICE)
        data_std_dev = data_std.to(DEVICE)
        
        predicted_artifact_norm = model(new_mixed_signal_norm_dev, new_stim_trace_dev)
        predicted_neural_signal_norm = new_mixed_signal_norm_dev - predicted_artifact_norm
        
        predicted_artifact = (predicted_artifact_norm * data_std_dev) + data_mean_dev
        predicted_neural_signal_unblended = (predicted_neural_signal_norm * data_std_dev) + data_mean_dev
        new_mixed_signal = (new_mixed_signal_norm_dev * data_std_dev) + data_mean_dev
        
        artifact_mask_expanded = artifact_mask_batch_dev.expand_as(new_mixed_signal)
        predicted_neural_signal = (1 - artifact_mask_expanded) * new_mixed_signal + \
                                  artifact_mask_expanded * predicted_neural_signal_unblended
    
    # Calculate metrics (same as hyperparameter_sweep.py)
    evaluator = Evaluator(sampling_rate=data_obj.sampling_rate)
    analyzer = NeuralAnalyzer(sampling_rate=data_obj.sampling_rate)
    
    ground_truth_neural = data_obj.ground_truth.astype(np.float32)
    ground_truth_artifacts = data_obj.artifacts.astype(np.float32)
    predicted_neural_np = predicted_neural_signal.cpu().numpy()
    mixed_data_np = new_mixed_signal.cpu().numpy()
    predicted_artifact_np = predicted_artifact.cpu().numpy()
    
    trial_idx = 0
    channel_idx = 0
    
    noise_before = mixed_data_np - ground_truth_neural
    noise_after = predicted_neural_np - ground_truth_neural
    
    snr_before = evaluator.calculate_snr(ground_truth_neural, noise_before)
    snr_after = evaluator.calculate_snr(ground_truth_neural, noise_after)
    snr_improvement = evaluator.calculate_snr_improvement(mixed_data_np, predicted_neural_np, ground_truth_neural)
    
    log_and_save(f"\nSNR Before (Mixed): {snr_before:.2f} dB")
    log_and_save(f"SNR After (Cleaned): {snr_after:.2f} dB")
    log_and_save(f"SNR Improvement: {snr_improvement:.2f} dB")
    
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
    
    # Save model
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
        'base_hyperparameters': base_config,
        'w_expert': w_expert,
        'w_anchor': w_anchor,
        'loss_history': loss_history,
    }, os.path.join(config_dir, 'model.pth'))
    
    with open(os.path.join(config_dir, "results.txt"), 'w') as f:
        f.write("\n".join(results_output))
    
    with open(os.path.join(config_dir, "config.json"), 'w') as f:
        json.dump({**base_config, 'w_expert': w_expert, 'w_anchor': w_anchor}, f, indent=2)
    
    print(f"\nConfig {config_str} completed. Results saved to {config_dir}\n")

# Get top 10 configurations from sweep_1
print("\n" + "="*80)
print("Analyzing sweep_1 to get top 10 configurations...")
print("="*80)

top_10_results, _ = analyze_sweep(SWEEP_1_DIR, top_n=10, w_snr=1.0, w_mse=0.3, w_coherence=0.3)

if not top_10_results:
    raise ValueError("No results found in sweep_1. Please run hyperparameter_sweep.py first.")

print(f"\nFound {len(top_10_results)} top configurations from sweep_1:")
for i, (config, metrics, score, dirname) in enumerate(top_10_results, 1):
    print(f"  {i}. Score: {score:.4f}, SNR: {metrics['snr_after']:.2f} dB, MSE: {metrics['mse']:.4f}")

# Expert hyperparameter ranges
expert_hyperparameter_ranges = {
    'w_expert': [0.1, 0.5, 1.0, 2.0],
    'w_anchor': [0.1, 0.5, 1.0, 2.0],
}

# Generate configurations: for each top 10 base config, vary w_expert and w_anchor
configs = []
for base_config, _, _, _ in top_10_results:
    for w_expert in expert_hyperparameter_ranges['w_expert']:
        for w_anchor in expert_hyperparameter_ranges['w_anchor']:
            configs.append({
                'base_config': base_config,
                'w_expert': w_expert,
                'w_anchor': w_anchor,
            })

print(f"\nTotal configurations to run: {len(configs)}")
print(f"  - {len(top_10_results)} base configurations from sweep_1")
print(f"  - {len(expert_hyperparameter_ranges['w_expert'])} w_expert values")
print(f"  - {len(expert_hyperparameter_ranges['w_anchor'])} w_anchor values")

completed = 0
skipped = 0

for i, config_item in enumerate(configs):
    base_config = config_item['base_config']
    w_expert = config_item['w_expert']
    w_anchor = config_item['w_anchor']
    
    config_str = config_to_string(base_config, w_expert, w_anchor)
    
    if config_exists(config_str):
        print(f"Skipping config {i+1}/{len(configs)}: {config_str} (already exists)")
        skipped += 1
        continue
    
    try:
        run_training(base_config, w_expert, w_anchor)
        completed += 1
    except Exception as e:
        print(f"Error running config {config_str}: {e}")
        import traceback
        traceback.print_exc()
        continue

print(f"\n{'='*80}")
print(f"Sweep 2 (Expert-guided) complete!")
print(f"Completed: {completed}")
print(f"Skipped: {skipped}")
print(f"Total: {len(configs)}")
print(f"{'='*80}")

