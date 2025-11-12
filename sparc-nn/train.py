import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sparc import DataHandler, SignalDataWithGroundTruth
from sparc.core.signal_data import ArtifactTriggers, SimulatedData
from sparc.core.plotting import NeuralPlotter
from sparc.core.evaluator import Evaluator

from model import UNet1D
from attention_model import AttentionNet1D
from loss import PhysicsLoss

LEARNING_RATE = 1e-3
BATCH_SIZE = 8
NUM_EPOCHS = 200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

data_handler = DataHandler()
# data_obj_dict = data_handler.load_npz_data('../data/simulated_data_2x64_30000.npz')
data_obj_dict = data_handler.load_npz_data('../data/added_artifacts_swec_data_512_lower_freq.npz')

# data_obj = SimulatedData(
#                 raw_data=data_obj_dict['raw_data'],
#                 sampling_rate=data_obj_dict['sampling_rate'],
#                 ground_truth=data_obj_dict['ground_truth'],
#                 artifacts=data_obj_dict['artifacts'],
#                 artifact_markers=ArtifactTriggers(starts=data_obj_dict['artifact_markers']),  
#                 firing_rate=data_obj_dict.get('firing_rate'),
#                 spike_train=data_obj_dict.get('spike_train'),
#                 lfp=data_obj_dict.get('lfp'),
#                 stim_params=None,
#                 snr=data_obj_dict.get('snr'),
#             )
data_obj = SignalDataWithGroundTruth(
            raw_data=data_obj_dict['mixed_data'],
            sampling_rate=data_obj_dict['sampling_rate'],
            ground_truth=data_obj_dict['ground_truth'],
            artifacts=data_obj_dict['artifacts'],
            artifact_markers=ArtifactTriggers(starts=data_obj_dict['artifact_markers']),
        )
mixed_data = data_obj.raw_data.astype(np.float32)

N_SAMPLES = mixed_data.shape[2]
IN_CHANNELS = mixed_data.shape[1]
OUT_CHANNELS = IN_CHANNELS  
N_TRIALS = mixed_data.shape[0]

dataset = TensorDataset(torch.from_numpy(mixed_data))
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
print(f"Data loaded: {len(data_loader)} batches of size {BATCH_SIZE}")

# model = UNet1D(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS).to(DEVICE)
model = AttentionNet1D(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, 
                       d_model=32, n_heads=8, n_layers=6, 
                       dim_feedforward=512, dropout=0.1).to(DEVICE)
print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters.")

criterion = PhysicsLoss(
    w_cosine=0.1,
    w_rank=1,
    w_smooth=0.6,
    w_spectral=1,
    f_cutoff=200.0,
    w_wavelet_energy=0,
    w_wavelet_sparsity=0,
    w_wavelet_entropy=0,
    sampling_rate=data_obj.sampling_rate
).to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("\n--- Starting Training ---")
loss_history = []

early_stopping_patience = 20
early_stopping_delta = 1e-6
best_loss = float('inf')
patience_counter = 0

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0.0
    
    for (x_mixed_batch,) in data_loader: # Note: (x,) unwraps the tuple from dataset
        x_mixed_batch = x_mixed_batch.to(DEVICE)
        s_pred_batch = model(x_mixed_batch)
        a_pred_batch = x_mixed_batch - s_pred_batch
        loss = criterion(s_pred_batch, a_pred_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(data_loader)
    loss_history.append(avg_loss)
    
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.6f}")
    
    if avg_loss < best_loss - early_stopping_delta:
        best_loss = avg_loss
        patience_counter = 0
        best_state_dict = model.state_dict()
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            model.load_state_dict(best_state_dict)
            break

print("--- Training Complete ---")

model.eval()
with torch.no_grad():
    (new_mixed_signal,) = next(iter(data_loader))
    new_mixed_signal = new_mixed_signal.to(DEVICE)
    
    predicted_neural_signal = model(new_mixed_signal)    
    predicted_artifact = new_mixed_signal - predicted_neural_signal

evaluator = Evaluator(sampling_rate=data_obj.sampling_rate)
plotter = NeuralPlotter(evaluator)

ground_truth_neural = data_obj.ground_truth.astype(np.float32)
predicted_neural_np = predicted_neural_signal.cpu().numpy()
mixed_data_np = new_mixed_signal.cpu().numpy()
predicted_artifact_np = predicted_artifact.cpu().numpy()

trial_idx = 0
channel_idx = 0

print("\n--- Computing SNR Metrics ---")
noise_before = mixed_data_np - ground_truth_neural
noise_after = predicted_neural_np - ground_truth_neural

snr_before = evaluator.calculate_snr(ground_truth_neural, noise_before)
snr_after = evaluator.calculate_snr(predicted_neural_np, noise_after)
snr_improvement = evaluator.calculate_snr_improvement(mixed_data_np, predicted_neural_np, ground_truth_neural)

print(f"SNR Before (Mixed): {snr_before:.2f} dB")
print(f"SNR After (Cleaned): {snr_after:.2f} dB")
print(f"SNR Improvement: {snr_improvement:.2f} dB")

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(range(1, len(loss_history) + 1), loss_history, linewidth=2, color='#1f77b4')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Training Loss')
ax.grid(True, alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('training_loss.png', dpi=150)
plt.show()

print("\n--- Plotting Signal Comparisons ---")

plotter.plot_cleaned_comparison(
    ground_truth=ground_truth_neural,
    mixed_data=mixed_data_np,
    cleaned_data=predicted_neural_np,
    trial_idx=trial_idx,
    channel_idx=channel_idx,
    title=f"Neural Signal Comparison - Trial {trial_idx}, Channel {channel_idx}",
    time_axis=True
)

plotter.plot_trace_comparison(
    cleaned=predicted_artifact_np,
    mixed_data=data_obj.artifacts,
    trial_idx=trial_idx,
    channel_idx=channel_idx,
    title=f"Artifact Comparison: Predicted vs Ground Truth - Trial {trial_idx}, Channel {channel_idx}",
    time_axis=True
)

# Plot all four signals in one figure
fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

x_axis = np.arange(ground_truth_neural.shape[2]) / data_obj.sampling_rate

# Plot 1: Ground Truth Neural Signal
axs[0].plot(x_axis, ground_truth_neural[trial_idx, channel_idx, :], 
            linewidth=1.5, color='#2ca02c', label='GT Neural')
axs[0].set_ylabel('Amplitude (µV)')
axs[0].set_title(f'Ground Truth Neural Signal - Trial {trial_idx}, Channel {channel_idx}')
axs[0].grid(True, alpha=0.3)
axs[0].legend()
axs[0].spines['top'].set_visible(False)
axs[0].spines['right'].set_visible(False)

# Plot 2: Cleaned (Predicted) Neural Signal
axs[1].plot(x_axis, predicted_neural_np[trial_idx, channel_idx, :], 
            linewidth=1.5, color='#1f77b4', label='Cleaned Neural')
axs[1].set_ylabel('Amplitude (µV)')
axs[1].set_title(f'Cleaned Neural Signal - Trial {trial_idx}, Channel {channel_idx}')
axs[1].grid(True, alpha=0.3)
axs[1].legend()
axs[1].spines['top'].set_visible(False)
axs[1].spines['right'].set_visible(False)

# Plot 3: Ground Truth Artifact
axs[2].plot(x_axis, data_obj.artifacts[trial_idx, channel_idx, :], 
            linewidth=1.5, color='#ff7f0e', label='GT Artifact')
axs[2].set_ylabel('Amplitude (µV)')
axs[2].set_title(f'Ground Truth Artifact - Trial {trial_idx}, Channel {channel_idx}')
axs[2].grid(True, alpha=0.3)
axs[2].legend()
axs[2].spines['top'].set_visible(False)
axs[2].spines['right'].set_visible(False)

# Plot 4: Cleaned (Predicted) Artifact
axs[3].plot(x_axis, predicted_artifact_np[trial_idx, channel_idx, :], 
            linewidth=1.5, color='#d62728', label='Cleaned Artifact')
axs[3].set_ylabel('Amplitude (µV)')
axs[3].set_xlabel('Time (s)')
axs[3].set_title(f'Cleaned Artifact - Trial {trial_idx}, Channel {channel_idx}')
axs[3].grid(True, alpha=0.3)
axs[3].legend()
axs[3].spines['top'].set_visible(False)
axs[3].spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('all_signals_comparison.png', dpi=150)
plt.show()
