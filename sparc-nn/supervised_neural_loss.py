import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from sparc import DataHandler, SignalDataWithGroundTruth
from sparc.core.signal_data import ArtifactTriggers
from sparc.core.evaluator import Evaluator

import torch.nn as nn
import torch.nn.functional as F
from model import UNet1D

class IndexedDataset(Dataset):
    def __init__(self, mixed_data, neural_data):
        self.mixed_data = torch.from_numpy(mixed_data)
        self.neural_data = torch.from_numpy(neural_data)
    
    def __len__(self):
        return len(self.mixed_data)
    
    def __getitem__(self, idx):
        return self.mixed_data[idx], self.neural_data[idx], idx

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LEARNING_RATE = 1e-3
BATCH_SIZE = 8
NUM_EPOCHS = 1000
print(f"Using device: {DEVICE}")

data_handler = DataHandler()
data_obj_dict = data_handler.load_npz_data('../data/added_artifacts_swec_data_512.npz')
data_obj = SignalDataWithGroundTruth(
            raw_data=data_obj_dict['mixed_data'],
            sampling_rate=data_obj_dict['sampling_rate'],
            ground_truth=data_obj_dict['ground_truth'],
            artifacts=data_obj_dict['artifacts'],
            artifact_markers=ArtifactTriggers(starts=data_obj_dict['artifact_markers']),
        )
mixed_data = data_obj.raw_data.astype(np.float32)
neural_data = data_obj.ground_truth.astype(np.float32)

N_SAMPLES = mixed_data.shape[2]
IN_CHANNELS = mixed_data.shape[1]
OUT_CHANNELS = IN_CHANNELS  
N_TRIALS = mixed_data.shape[0]

dataset = IndexedDataset(mixed_data, neural_data)
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
print(f"Data loaded: {len(data_loader)} batches of size {BATCH_SIZE}")

model = UNet1D(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS).to(DEVICE)
print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

class SupervisedNeuralLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, s_pred, s_true):
        return F.mse_loss(s_pred, s_true)

criterion = SupervisedNeuralLoss().to(DEVICE)
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
    
    for x_mixed_batch, s_true_batch, _ in data_loader:
        x_mixed_batch = x_mixed_batch.to(DEVICE)
        s_true_batch = s_true_batch.to(DEVICE)
        
        s_pred_batch = model(x_mixed_batch)
        
        loss = criterion(s_pred_batch, s_true_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(data_loader)
    loss_history.append(avg_loss)
    
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.6f}")

    if avg_loss < best_loss - early_stopping_delta:
        best_loss = avg_loss
        patience_counter = 0
        # You could save the model here if desired
        best_state_dict = model.state_dict()
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            # Restore best weights before stopping
            model.load_state_dict(best_state_dict)
            break

print("--- Training Complete ---")

# Evaluate the model
model.eval()
with torch.no_grad():
    x_mixed_batch, s_true_batch, batch_indices = next(iter(data_loader))
    x_mixed_batch = x_mixed_batch.to(DEVICE)
    batch_indices = batch_indices.numpy()
    
    predicted_neural_signal = model(x_mixed_batch)    
    predicted_artifact = x_mixed_batch - predicted_neural_signal
    
    print(f"\nInference complete.")
    print(f"Input shape: {x_mixed_batch.shape}")
    print(f"S_pred shape: {predicted_neural_signal.shape}")
    print(f"A_pred shape: {predicted_artifact.shape}")

# Convert to numpy for evaluation
x_mixed_np = x_mixed_batch.cpu().numpy()
predicted_neural_np = predicted_neural_signal.cpu().numpy()
predicted_artifact_np = predicted_artifact.cpu().numpy()
s_true_np = s_true_batch.numpy()

# Calculate metrics
evaluator = Evaluator(sampling_rate=data_obj.sampling_rate)

# Neural signal prediction accuracy
neural_mse = np.mean((predicted_neural_np - s_true_np) ** 2)
print(f"\nNeural Signal Prediction MSE: {neural_mse:.6f}")

# Artifact reconstruction quality
ground_truth_artifact = data_obj.artifacts.astype(np.float32)
gt_artifact_batch = ground_truth_artifact[batch_indices]

artifact_mse = np.mean((predicted_artifact_np - gt_artifact_batch) ** 2)
print(f"Artifact Reconstruction MSE: {artifact_mse:.6f}")

# Plot results
trial_idx = 0
channel_idx = 0

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: True vs Predicted Neural Signal
time_axis = np.arange(predicted_neural_np.shape[2]) / data_obj.sampling_rate
axes[0,0].plot(time_axis, s_true_np[trial_idx, channel_idx, :], 
               label='True Neural', alpha=0.7)
axes[0,0].plot(time_axis, predicted_neural_np[trial_idx, channel_idx, :], 
               label='Predicted Neural', alpha=0.7)
axes[0,0].set_title('Neural Signal: True vs Predicted')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# Plot 2: Artifact Comparison
axes[0,1].plot(time_axis, gt_artifact_batch[trial_idx, channel_idx, :], 
               label='True Artifact', alpha=0.7)
axes[0,1].plot(time_axis, predicted_artifact_np[trial_idx, channel_idx, :], 
               label='Predicted Artifact', alpha=0.7)
axes[0,1].plot(time_axis, x_mixed_np[trial_idx, channel_idx, :], 
               label='Mixed Signal', alpha=0.5)
axes[0,1].set_title('Artifact Comparison')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# Plot 3: Training Loss
axes[1,0].plot(loss_history)
axes[1,0].set_title('Training Loss')
axes[1,0].set_xlabel('Epoch')
axes[1,0].set_ylabel('Loss')
axes[1,0].grid(True, alpha=0.3)

# Plot 4: Reconstruction quality
axes[1,1].plot(time_axis, x_mixed_np[trial_idx, channel_idx, :], 
               label='Mixed', alpha=0.5)
axes[1,1].plot(time_axis, predicted_neural_np[trial_idx, channel_idx, :] + predicted_artifact_np[trial_idx, channel_idx, :], 
               label='Reconstructed', alpha=0.8)
axes[1,1].set_title('Reconstruction: Mixed vs Reconstructed')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('supervised_neural_training_results.png', dpi=150)
plt.show()

print("\n--- Summary ---")
print(f"Final Neural Signal MSE: {neural_mse:.6f}")
print(f"Final Artifact MSE: {artifact_mse:.6f}")

# Test reconstruction on a few samples
print(f"\nReconstruction test:")
for i in range(min(3, BATCH_SIZE)):
    reconstruction_error = np.mean(
        (x_mixed_np[i, channel_idx, :] - 
         (predicted_neural_np[i, channel_idx, :] + predicted_artifact_np[i, channel_idx, :])) ** 2
    )
    print(f"Sample {i} reconstruction MSE: {reconstruction_error:.8f}")

