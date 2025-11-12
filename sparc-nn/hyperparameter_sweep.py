import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sparc import DataHandler, SignalDataWithGroundTruth
from sparc.core.signal_data import ArtifactTriggers
from sparc.core.evaluator import Evaluator
import json
import os
from datetime import datetime
from itertools import product

from model import UNet1D
from attention_model import AttentionNet1D
from loss import PhysicsLoss

LEARNING_RATE = 1e-3
BATCH_SIZE = 8
NUM_EPOCHS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
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

N_SAMPLES = mixed_data.shape[2]
IN_CHANNELS = mixed_data.shape[1]
OUT_CHANNELS = IN_CHANNELS  
N_TRIALS = mixed_data.shape[0]

dataset = TensorDataset(torch.from_numpy(mixed_data))
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
print(f"Data loaded: {len(data_loader)} batches of size {BATCH_SIZE}")

hyperparameter_grid = {
    'w_cosine': [0.1, 0.2, 1.0],
    'w_rank': [0.1, 0.2, 1.0],
    'w_smooth': [0.01, 0.05, 0.1],
    'w_spectral': [0.01, 0.2, 0.5, 1],
    'w_wavelet_energy': [0.01, 0.05, 0.1],
    'w_wavelet_sparsity': [0.01, 0.05, 0.1],
    'w_wavelet_entropy': [0.01, 0.05, 0.1],
}

f_cutoff = 200.0

results_dir = f"sweep_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(results_dir, exist_ok=True)
print(f"\nResults will be saved to: {results_dir}")

ground_truth_neural = data_obj.ground_truth.astype(np.float32)
evaluator = Evaluator(sampling_rate=data_obj.sampling_rate)

all_results = []

param_combinations = list(product(*hyperparameter_grid.values()))
param_names = list(hyperparameter_grid.keys())
total_combinations = len(param_combinations)

print(f"\nTotal hyperparameter combinations to test: {total_combinations}")
print(f"Estimated total epochs: {total_combinations * NUM_EPOCHS}\n")

# for idx, params in enumerate(param_combinations):
#     param_dict = dict(zip(param_names, params))
    
#     print(f"\n{'='*80}")
#     print(f"Experiment {idx+1}/{total_combinations}")
#     print(f"Parameters: {param_dict}")
#     print(f"{'='*80}")
    
#     # model = UNet1D(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS).to(DEVICE)
#     model = AttentionNet1D(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, 
#                            d_model=128, n_heads=8, n_layers=6, 
#                            dim_feedforward=512, dropout=0.1).to(DEVICE)
    
#     criterion = PhysicsLoss(
#         w_cosine=param_dict['w_cosine'], 
#         w_rank=param_dict['w_rank'], 
#         w_smooth=param_dict['w_smooth'],
#         w_spectral=param_dict['w_spectral'],
#         f_cutoff=f_cutoff,
#         sampling_rate=data_obj.sampling_rate
#     ).to(DEVICE)
    
#     optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
#     loss_history = []
    
#     for epoch in range(NUM_EPOCHS):
#         model.train()
#         total_loss = 0.0
        
#         for (x_mixed_batch,) in data_loader:
#             x_mixed_batch = x_mixed_batch.to(DEVICE)
#             a_pred_batch = model(x_mixed_batch)
#             s_pred_batch = x_mixed_batch - a_pred_batch
#             loss = criterion(s_pred_batch, a_pred_batch)
            
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
            
#             total_loss += loss.item()
        
#         avg_loss = total_loss / len(data_loader)
#         loss_history.append(avg_loss)
        
#         if (epoch + 1) % 10 == 0:
#             print(f"  Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.6f}")
    
#     model.eval()
#     with torch.no_grad():
#         all_predictions = []
#         all_mixed = []
        
#         for (x_mixed_batch,) in data_loader:
#             x_mixed_batch = x_mixed_batch.to(DEVICE)
#             a_pred_batch = model(x_mixed_batch)
#             s_pred_batch = x_mixed_batch - a_pred_batch
#             all_predictions.append(s_pred_batch.cpu().numpy())
#             all_mixed.append(x_mixed_batch.cpu().numpy())
        
#         predicted_neural_np = np.concatenate(all_predictions, axis=0)
#         mixed_data_np = np.concatenate(all_mixed, axis=0)
    
#     noise_before = mixed_data_np - ground_truth_neural
#     noise_after = predicted_neural_np - ground_truth_neural
    
#     snr_before = evaluator.calculate_snr(ground_truth_neural, noise_before)
#     snr_after = evaluator.calculate_snr(predicted_neural_np, noise_after)
#     snr_improvement = evaluator.calculate_snr_improvement(mixed_data_np, predicted_neural_np, ground_truth_neural)
    
#     print(f"  SNR Before: {snr_before:.2f} dB")
#     print(f"  SNR After: {snr_after:.2f} dB")
#     print(f"  SNR Improvement: {snr_improvement:.2f} dB")
#     print(f"  Final Loss: {loss_history[-1]:.6f}")
    
#     result = {
#         'experiment_id': idx,
#         'parameters': param_dict,
#         'snr_before': float(snr_before),
#         'snr_after': float(snr_after),
#         'snr_improvement': float(snr_improvement),
#         'final_loss': float(loss_history[-1]),
#         'loss_history': [float(l) for l in loss_history],
#     }
#     all_results.append(result)
    
#     model_filename = f"{results_dir}/model_{idx}.pt"
#     torch.save({
#         'model_state_dict': model.state_dict(),
#         'parameters': param_dict,
#         'metrics': {
#             'snr_before': snr_before,
#             'snr_after': snr_after,
#             'snr_improvement': snr_improvement,
#             'final_loss': loss_history[-1],
#         }
#     }, model_filename)
    
#     fig, ax = plt.subplots(figsize=(8, 5))
#     ax.plot(loss_history, linewidth=2)
#     ax.set_xlabel('Epoch')
#     ax.set_ylabel('Loss')
#     ax.set_title(f'Training Loss - Experiment {idx}')
#     ax.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.savefig(f"{results_dir}/loss_curve_{idx}.png", dpi=100)
#     plt.close()

with open(f"{results_dir}/all_results.json", 'w') as f:
    json.dump(all_results, f, indent=2)

sorted_by_snr_improvement = sorted(all_results, key=lambda x: x['snr_improvement'], reverse=True)
sorted_by_snr_after = sorted(all_results, key=lambda x: x['snr_after'], reverse=True)
sorted_by_final_loss = sorted(all_results, key=lambda x: x['final_loss'])

print(f"\n{'='*80}")
print("SUMMARY - Top 5 by SNR Improvement:")
print(f"{'='*80}")
for i, result in enumerate(sorted_by_snr_improvement[:5]):
    print(f"\nRank {i+1}:")
    print(f"  Experiment ID: {result['experiment_id']}")
    print(f"  Parameters: {result['parameters']}")
    print(f"  SNR Improvement: {result['snr_improvement']:.2f} dB")
    print(f"  SNR After: {result['snr_after']:.2f} dB")
    print(f"  Final Loss: {result['final_loss']:.6f}")

print(f"\n{'='*80}")
print("SUMMARY - Top 5 by SNR After:")
print(f"{'='*80}")
for i, result in enumerate(sorted_by_snr_after[:5]):
    print(f"\nRank {i+1}:")
    print(f"  Experiment ID: {result['experiment_id']}")
    print(f"  Parameters: {result['parameters']}")
    print(f"  SNR After: {result['snr_after']:.2f} dB")
    print(f"  SNR Improvement: {result['snr_improvement']:.2f} dB")
    print(f"  Final Loss: {result['final_loss']:.6f}")

print(f"\n{'='*80}")
print("SUMMARY - Top 5 by Final Loss (lowest):")
print(f"{'='*80}")
for i, result in enumerate(sorted_by_final_loss[:5]):
    print(f"\nRank {i+1}:")
    print(f"  Experiment ID: {result['experiment_id']}")
    print(f"  Parameters: {result['parameters']}")
    print(f"  Final Loss: {result['final_loss']:.6f}")
    print(f"  SNR Improvement: {result['snr_improvement']:.2f} dB")
    print(f"  SNR After: {result['snr_after']:.2f} dB")

summary = {
    'total_experiments': total_combinations,
    'best_snr_improvement': sorted_by_snr_improvement[0],
    'best_snr_after': sorted_by_snr_after[0],
    'lowest_loss': sorted_by_final_loss[0],
}

with open(f"{results_dir}/summary.json", 'w') as f:
    json.dump(summary, f, indent=2)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

snr_improvements = [r['snr_improvement'] for r in all_results]
snr_after_values = [r['snr_after'] for r in all_results]
final_losses = [r['final_loss'] for r in all_results]
experiment_ids = [r['experiment_id'] for r in all_results]

axes[0, 0].scatter(experiment_ids, snr_improvements, alpha=0.6)
axes[0, 0].set_xlabel('Experiment ID')
axes[0, 0].set_ylabel('SNR Improvement (dB)')
axes[0, 0].set_title('SNR Improvement across Experiments')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].scatter(experiment_ids, snr_after_values, alpha=0.6, color='green')
axes[0, 1].set_xlabel('Experiment ID')
axes[0, 1].set_ylabel('SNR After (dB)')
axes[0, 1].set_title('SNR After across Experiments')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].scatter(experiment_ids, final_losses, alpha=0.6, color='red')
axes[1, 0].set_xlabel('Experiment ID')
axes[1, 0].set_ylabel('Final Loss')
axes[1, 0].set_title('Final Loss across Experiments')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].scatter(final_losses, snr_improvements, alpha=0.6, color='purple')
axes[1, 1].set_xlabel('Final Loss')
axes[1, 1].set_ylabel('SNR Improvement (dB)')
axes[1, 1].set_title('SNR Improvement vs Final Loss')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{results_dir}/overview.png", dpi=150)
plt.close()

print(f"\nAll results saved to {results_dir}/")
print(f"  - all_results.json: Complete results for all experiments")
print(f"  - summary.json: Summary of best performing configurations")
print(f"  - model_<id>.pt: Saved model checkpoints")
print(f"  - loss_curve_<id>.png: Training loss curves")
print(f"  - overview.png: Overview plots of all experiments")

