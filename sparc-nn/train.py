import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from datetime import datetime
from sparc import DataHandler, SignalDataWithGroundTruth
from sparc.core.signal_data import ArtifactTriggers
from sparc.core.plotting import NeuralPlotter
from sparc.core.evaluator import Evaluator
from sparc.core.neural_analyzer import NeuralAnalyzer
from models import UNet1D, NeuralExpertAE
from loss import PhysicsLoss
from uncertainty_loss import UncertaintyWeightedLoss
from data_utils import prepare_dataset
from glob import glob
from eval import compute_metrics, perform_spectral_analysis
import random

def parse_args():
    parser = argparse.ArgumentParser(description='Train U-Net for neural signal separation')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs (default: 200)')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size (default: 1)')
    parser.add_argument('--swec-mat-file', type=str, default='../data/added_artifacts_swec_data_512_lower_freq_wo0.npz',
                       help='Path to training data file (default: ../data/added_artifacts_swec_data_512_lower_freq_wo0.npz)')
    parser.add_argument('--artifact-duration-ms', type=int, default=40, help='Artifact duration in milliseconds (default: 40)')
    parser.add_argument('--trial-indices', type=str, default=None, 
                       help='Comma-separated trial indices (e.g., "0,1,2"). If None, uses all trials.')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate (default: 1e-3)')
    parser.add_argument('--use-uncertainty-loss', action='store_true', 
                       help='Use UncertaintyWeightedLoss instead of manual weights')
    parser.add_argument('--w-cosine', type=float, default=5.0, help='Weight for cosine loss (default: 5.0)')
    parser.add_argument('--w-rank-a', type=float, default=1.5, help='Weight for artifact rank loss (default: 1.5)')
    parser.add_argument('--w-rank-s', type=float, default=1.0, help='Weight for neural rank penalty (default: 1.0)')
    parser.add_argument('--w-spectral', type=float, default=1.2, help='Weight for spectral loss (default: 1.2)')
    parser.add_argument('--w-spectral-slope', type=float, default=5.0, help='Weight for spectral slope loss (default: 5.0)')
    parser.add_argument('--f-cutoff', type=float, default=10.0, help='Frequency cutoff for spectral loss (default: 10.0)')
    parser.add_argument('--predict-neural', action='store_true', 
                       help='Predict neural signal instead of artifact (default: False)')
    parser.add_argument('--output-dir', type=str, default='saved_models', help='Directory to save models (default: saved_models)')
    parser.add_argument('--use-expert', action='store_true',
                       help='Use expert-guided training with NeuralExpertAE (default: False)')
    parser.add_argument('--expert-model-path', type=str, default=None,
                       help='Path to expert model checkpoint. If not provided, uses most recent in saved_models/')
    parser.add_argument('--w-expert', type=float, default=1.0, help='Weight for expert loss (default: 1.0)')
    parser.add_argument('--w-anchor', type=float, default=1.0, help='Weight for anchor loss (default: 1.0)')
    parser.add_argument('--save-plots', action='store_true', help='Save plots to files (default: False, only display)')
    parser.add_argument('--plot-channel-idx', type=int, default=0, help='Channel index for plotting (default: 0)')
    return parser.parse_args()

def main():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    args = parse_args()
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")
    
    # Parse trial indices
    trial_indices = None
    if args.trial_indices:
        trial_indices = [int(x.strip()) for x in args.trial_indices.split(',')]
        print(f"Using trial indices: {trial_indices}")
    
    # Load data
    data_handler = DataHandler()
    data_obj_dict = data_handler.load_npz_data(args.swec_mat_file)
    
    data_obj = SignalDataWithGroundTruth(
        raw_data=data_obj_dict['mixed_data'],
        sampling_rate=data_obj_dict['sampling_rate'],
        ground_truth=data_obj_dict['ground_truth'],
        artifacts=data_obj_dict['artifacts'],
        artifact_markers=ArtifactTriggers(starts=data_obj_dict['artifact_markers']),
    )
    
    print(f"Data loaded: {data_obj.raw_data.shape}")
    
    mixed_data = data_obj.raw_data.astype(np.float32)
    N_SAMPLES = mixed_data.shape[2]
    N_DATA_CHANNELS = mixed_data.shape[1]
    IN_CHANNELS = N_DATA_CHANNELS + 1  # +1 for stim trace
    OUT_CHANNELS = N_DATA_CHANNELS
    N_TRIALS = mixed_data.shape[0]
    
    # Prepare dataset
    dataset, data_loader, artifact_masks, data_mean, data_std = prepare_dataset(
        data_obj=data_obj,
        data_obj_dict=data_obj_dict,
        batch_size=args.batch_size,
        use_normalization=True,
        shuffle=True,
        artifact_duration_ms=args.artifact_duration_ms,
        trial_indices=trial_indices
    )
    
    # Create model
    model = UNet1D(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS).to(DEVICE)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters.")
    
    # Load expert model if requested
    neural_expert = None
    if args.use_expert:
        if args.expert_model_path is None:
            # Find most recent expert model
            expert_model_files = glob(os.path.join('saved_models', 'neural_expert_*.pth'))
            if not expert_model_files:
                raise FileNotFoundError(
                    "No expert model found in saved_models/. "
                    "Please train an expert first using train_expert.py or specify --expert-model-path"
                )
            expert_model_path = sorted(expert_model_files)[-1]
            print(f"Using most recent expert model: {expert_model_path}")
        else:
            expert_model_path = args.expert_model_path
        
        print(f"Loading expert model from: {expert_model_path}")
        expert_checkpoint = torch.load(expert_model_path, map_location=DEVICE, weights_only=False)
        expert_in_channels = expert_checkpoint.get('in_channels', OUT_CHANNELS)
        
        neural_expert = NeuralExpertAE(in_channels=expert_in_channels).to(DEVICE)
        neural_expert.load_state_dict(expert_checkpoint['model_state_dict'])
        
        # Freeze expert
        for param in neural_expert.parameters():
            param.requires_grad = False
        neural_expert.eval()
        print(f"Expert model loaded and frozen: {expert_in_channels} input channels")
    
    # Create loss function
    criterion = PhysicsLoss(
        f_cutoff=args.f_cutoff,
        sampling_rate=data_obj.sampling_rate,
    ).to(DEVICE)
    
    # Setup optimizer
    uncertainty_loss = None
    if args.use_uncertainty_loss:
        uncertainty_loss = UncertaintyWeightedLoss(num_losses=5).to(DEVICE)
        optimizer = optim.Adam([
            {'params': model.parameters()},
            {'params': uncertainty_loss.parameters(), 'lr': args.learning_rate}
        ], lr=args.learning_rate)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    print("\n--- Starting Training ---")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    if args.use_uncertainty_loss:
        print("Using UncertaintyWeightedLoss")
    else:
        print(f"Loss weights: cosine={args.w_cosine}, rank_a={args.w_rank_a}, rank_s={args.w_rank_s}, "
              f"spectral={args.w_spectral}, spectral_slope={args.w_spectral_slope}")
    if args.use_expert:
        print(f"Expert-guided training enabled: w_expert={args.w_expert}, w_anchor={args.w_anchor}")
    loss_history = []
    
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        
        for batch in data_loader:
            x_mixed_batch = batch[0].to(DEVICE)
            x_stim_batch = batch[1].to(DEVICE)
            artifact_mask_batch = batch[2].to(DEVICE)
            
            # Prediction mode
            if args.predict_neural:
                s_pred_batch = model(x_mixed_batch, x_stim_batch)
                a_pred_batch = x_mixed_batch - s_pred_batch
            else:
                a_pred_batch = model(x_mixed_batch, x_stim_batch)
                s_pred_batch = x_mixed_batch - a_pred_batch
            
            raw_loss_dict = criterion(s_pred_batch, a_pred_batch)
            
            if epoch % 50 == 0 and epoch > 0:
                print(f"Epoch {epoch}: Raw Loss Scales ---")
                for key, val in raw_loss_dict.items():
                    print(f"  {key}: {val.item():.4f}")
                print("------------------------------\n")
            
            # Compute physics loss
            if args.use_uncertainty_loss:
                physics_loss_value = uncertainty_loss(raw_loss_dict)
            else:
                physics_loss_value = (
                    args.w_cosine * raw_loss_dict['cosine'] +
                    args.w_rank_s * raw_loss_dict['rank_s_penalty'] +
                    args.w_spectral * raw_loss_dict['spectral'] +
                    args.w_spectral_slope * raw_loss_dict['spectral_slope_s'] +
                    args.w_rank_a * raw_loss_dict['rank_a']
                )
            
            # Add expert and anchor losses if using expert
            loss_anchor = torch.tensor(0.0, device=DEVICE)
            loss_expert_projection = torch.tensor(0.0, device=DEVICE)
            
            if args.use_expert:
                is_clean = (artifact_mask_batch < 0.1).float()
                is_dirty = (artifact_mask_batch > 0.5).float()
                
                # Anchor loss: encourage artifact to be small in clean regions
                if is_clean.sum() > 0:
                    loss_anchor = (a_pred_batch ** 2 * is_clean).sum() / (is_clean.sum() + 1e-8)
                else:
                    loss_anchor = torch.tensor(0.0, device=DEVICE)
                
                # Expert loss: encourage neural signal to be consistent with expert in dirty regions
                if is_dirty.sum() > 0:
                    s_pred_recon = neural_expert(s_pred_batch)
                    loss_expert_projection = ((s_pred_batch - s_pred_recon) ** 2 * is_dirty).sum() / (is_dirty.sum() + 1e-8)
                else:
                    loss_expert_projection = torch.tensor(0.0, device=DEVICE)
            
            # Final loss
            final_loss = physics_loss_value + args.w_anchor * loss_anchor + args.w_expert * loss_expert_projection
            
            optimizer.zero_grad()
            final_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            if epoch == 0 and args.use_uncertainty_loss:
                print("\n--- DEBUG: Raw Loss Scales ---")
                for key, val in raw_loss_dict.items():
                    print(f"{key}: {val.item():.4f}")
                weights = torch.exp(-uncertainty_loss.log_vars).detach().cpu().numpy()
                print(f"Learned Weights: {weights}")
                print("------------------------------\n")
            
            total_loss += final_loss.item()
        
        avg_loss = total_loss / len(data_loader)
        loss_history.append(avg_loss)
        
        if epoch % 100 == 0 and epoch > 0:
            print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {avg_loss:.6f}")
    
    print("--- Training Complete ---")
    
    # Save model
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = os.path.join(args.output_dir, f'unet1d_{timestamp}.pth')
    
    loss_weights = {
        'w_cosine': args.w_cosine,
        'w_rank_a': args.w_rank_a,
        'w_rank_s': args.w_rank_s,
        'w_spectral': args.w_spectral,
        'w_spectral_slope': args.w_spectral_slope,
    }
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'in_channels': IN_CHANNELS,
            'out_channels': OUT_CHANNELS,
        },
        'training_config': {
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
            'num_epochs': args.epochs,
            'artifact_duration_ms': args.artifact_duration_ms,
            'sampling_rate': data_obj.sampling_rate,
            'predict_neural': args.predict_neural,
        },
        'loss_config': {
            'f_cutoff': args.f_cutoff,
            'use_uncertainty_loss': args.use_uncertainty_loss,
            'loss_weights': loss_weights,
            'use_expert': args.use_expert,
            'w_expert': args.w_expert if args.use_expert else None,
            'w_anchor': args.w_anchor if args.use_expert else None,
        },
        'loss_history': loss_history,
        'data_mean': data_mean.cpu(),
        'data_std': data_std.cpu(),
    }
    
    if args.use_uncertainty_loss:
        checkpoint['uncertainty_loss_state_dict'] = uncertainty_loss.state_dict()
    
    torch.save(checkpoint, model_path)
    print(f"Model saved to: {model_path}")
    
    # Evaluation
    model.eval()
    eval_dataset, eval_loader, _, _, _ = prepare_dataset(
        data_obj=data_obj,
        data_obj_dict=data_obj_dict,
        batch_size=N_TRIALS if trial_indices is None else len(trial_indices),
        use_normalization=True,
        shuffle=False,
        artifact_duration_ms=args.artifact_duration_ms,
        trial_indices=trial_indices
    )
    
    with torch.no_grad():
        new_mixed_signal_norm, new_stim_trace, artifact_mask_batch = next(iter(eval_loader))
        new_mixed_signal_norm_dev = new_mixed_signal_norm.to(DEVICE)
        new_stim_trace_dev = new_stim_trace.to(DEVICE)
        artifact_mask_batch_dev = artifact_mask_batch.to(DEVICE)
        data_mean_dev = data_mean.to(DEVICE)
        data_std_dev = data_std.to(DEVICE)
        
        if args.predict_neural:
            predicted_neural_signal_norm = model(new_mixed_signal_norm_dev, new_stim_trace_dev)
            predicted_artifact_norm = new_mixed_signal_norm_dev - predicted_neural_signal_norm
        else:
            predicted_artifact_norm = model(new_mixed_signal_norm_dev, new_stim_trace_dev)
            predicted_neural_signal_norm = new_mixed_signal_norm_dev - predicted_artifact_norm
        
        predicted_artifact = (predicted_artifact_norm * data_std_dev) + data_mean_dev
        predicted_neural_signal_unblended = (predicted_neural_signal_norm * data_std_dev) + data_mean_dev
        new_mixed_signal = (new_mixed_signal_norm_dev * data_std_dev) + data_mean_dev
        
        artifact_mask_expanded = artifact_mask_batch_dev.expand_as(new_mixed_signal)
        predicted_neural_signal = (1 - artifact_mask_expanded) * new_mixed_signal + \
                                  artifact_mask_expanded * predicted_neural_signal_unblended
        
        if args.use_uncertainty_loss:
            weights = torch.exp(-uncertainty_loss.log_vars)
            print(f"Final Learned Weights: {weights.cpu().numpy()}")
    
    # Calculate metrics
    evaluator = Evaluator(sampling_rate=data_obj.sampling_rate)
    analyzer = NeuralAnalyzer(sampling_rate=data_obj.sampling_rate)
    
    ground_truth_neural = data_obj.ground_truth.astype(np.float32)
    ground_truth_artifacts = data_obj.artifacts.astype(np.float32)
    predicted_neural_np = predicted_neural_signal.cpu().numpy()
    mixed_data_np = new_mixed_signal.cpu().numpy()
    predicted_artifact_np = predicted_artifact.cpu().numpy()
    
    # Calculate SNR per trial and average (matching hyperparameter_sweep.py)
    n_trials_eval = predicted_neural_np.shape[0]
    snr_before_per_trial = []
    snr_after_per_trial = []
    snr_improvement_per_trial = []
    
    for trial_idx in range(n_trials_eval):
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
    
    print(f"\nSNR Before (Mixed): {snr_before:.2f} dB (averaged over {n_trials_eval} trials)")
    print(f"SNR After (Cleaned): {snr_after:.2f} dB (averaged over {n_trials_eval} trials)")
    print(f"SNR Improvement: {snr_improvement:.2f} dB (averaged over {n_trials_eval} trials)")
    
    # Select trial for plotting - ensure all arrays have shape (1, C, T) for plotting
    eval_trial_idx = 0
    if trial_indices:
        # Use first trial from selected trials
        original_trial_idx = trial_indices[0]
        ground_truth_neural_eval = ground_truth_neural[original_trial_idx:original_trial_idx+1]
        ground_truth_artifacts_eval = ground_truth_artifacts[original_trial_idx:original_trial_idx+1]
    else:
        original_trial_idx = eval_trial_idx
        ground_truth_neural_eval = ground_truth_neural[eval_trial_idx:eval_trial_idx+1]
        ground_truth_artifacts_eval = ground_truth_artifacts[eval_trial_idx:eval_trial_idx+1]
    
    # For plotting, use first trial from batch
    predicted_neural_np_plot = predicted_neural_np[0:1]  # (1, C, T)
    mixed_data_np_plot = mixed_data_np[0:1]  # (1, C, T)
    predicted_artifact_np_plot = predicted_artifact_np[0:1]  # (1, C, T)
    
    # Calculate other metrics using all trials (for consistency with hyperparameter_sweep.py)
    metrics = compute_metrics(
        ground_truth_neural=ground_truth_neural,
        ground_truth_artifacts=ground_truth_artifacts,
        predicted_neural_np=predicted_neural_np,
        predicted_artifact_np=predicted_artifact_np,
        analyzer=analyzer,
        evaluator=evaluator,
        mixed_data_np=mixed_data_np,
        print_results=True
    )
    
    # Print spectral slope loss comparison
    print(f"\n--- Spectral Slope Loss Comparison (Welch) ---")
    print(f"GT Neural: {metrics['spectral_slope_gt_neural']:.6f}")
    print(f"Predicted Neural: {metrics['spectral_slope_pred_neural']:.6f}")
    print(f"GT Artifact: {metrics['spectral_slope_gt_artifact']:.6f}")
    print(f"Predicted Artifact: {metrics['spectral_slope_pred_artifact']:.6f}")
    
    # Print artifact suppression
    print(f"\n--- Artifact Suppression ---")
    print(f"Suppression (Amplitude): {metrics['suppression_amplitude_db']:.2f} dB")
    print(f"Suppression (Power): {metrics['suppression_power_db']:.2f} dB")
    
    # Plot training loss
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(loss_history) + 1), loss_history, linewidth=2, color='#1f77b4')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    if args.save_plots:
        plt.savefig('training_loss.png', dpi=150)
        print("Training loss plot saved to: training_loss.png")
    plt.show()
    
    # Spectral analysis
    print("\n--- Spectral Analysis ---")
    perform_spectral_analysis(ground_truth_neural_eval, analyzer, "GT Neural", plot=True, save=args.save_plots, save_path=None)
    perform_spectral_analysis(predicted_neural_np_plot, analyzer, "Predicted Neural", plot=True, save=args.save_plots, save_path=None)
    perform_spectral_analysis(ground_truth_artifacts_eval, analyzer, "GT Artifact", plot=True, save=args.save_plots, save_path=None)
    perform_spectral_analysis(predicted_artifact_np_plot, analyzer, "Predicted Artifact", plot=True, save=args.save_plots, save_path=None)
    
    # Plot signal comparisons
    print("\n--- Plotting Signal Comparisons ---")
    plotter = NeuralPlotter(evaluator)
    channel_idx = args.plot_channel_idx
    trial_idx_for_title = original_trial_idx
    
    # All arrays now have shape (1, C, T) for plotting
    ground_truth_neural_for_plot = ground_truth_neural_eval
    ground_truth_artifacts_for_plot = ground_truth_artifacts_eval
    
    plotter.plot_cleaned_comparison(
        ground_truth=ground_truth_neural_for_plot,
        mixed_data=mixed_data_np_plot,
        cleaned_data=predicted_neural_np_plot,
        trial_idx=0,
        channel_idx=channel_idx,
        title=f"Neural Signal Comparison - Trial {trial_idx_for_title}, Channel {channel_idx}",
        time_axis=True
    )
    
    plotter.plot_trace_comparison(
        cleaned=predicted_artifact_np_plot,
        mixed_data=ground_truth_artifacts_for_plot,
        trial_idx=0,
        channel_idx=channel_idx,
        title=f"Artifact Comparison: Predicted vs Ground Truth - Trial {trial_idx_for_title}, Channel {channel_idx}",
        time_axis=True
    )
    
    # Plot all four signals in one figure
    fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    
    x_axis = np.arange(predicted_neural_np_plot.shape[2]) / data_obj.sampling_rate
    plot_trial_idx = 0
    
    # Plot 1: Ground Truth Neural Signal
    axs[0].plot(x_axis, ground_truth_neural_eval[plot_trial_idx, channel_idx, :], 
                linewidth=1.5, color='#2ca02c', label='GT Neural')
    axs[0].set_ylabel('Amplitude (µV)')
    axs[0].set_title(f'Ground Truth Neural Signal - Trial {trial_idx_for_title}, Channel {channel_idx}')
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    
    # Plot 2: Predicted Neural Signal
    axs[1].plot(x_axis, predicted_neural_np_plot[plot_trial_idx, channel_idx, :], 
                linewidth=1.5, color='#1f77b4', label='Predicted Neural')
    axs[1].set_ylabel('Amplitude (µV)')
    axs[1].set_title(f'Predicted Neural Signal - Trial {trial_idx_for_title}, Channel {channel_idx}')
    axs[1].grid(True, alpha=0.3)
    axs[1].legend()
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    
    # Plot 3: Ground Truth Artifact
    axs[2].plot(x_axis, ground_truth_artifacts_eval[plot_trial_idx, channel_idx, :], 
                linewidth=1.5, color='#ff7f0e', label='GT Artifact')
    axs[2].set_ylabel('Amplitude (µV)')
    axs[2].set_title(f'Ground Truth Artifact - Trial {trial_idx_for_title}, Channel {channel_idx}')
    axs[2].grid(True, alpha=0.3)
    axs[2].legend()
    axs[2].spines['top'].set_visible(False)
    axs[2].spines['right'].set_visible(False)
    
    # Plot 4: Predicted Artifact
    axs[3].plot(x_axis, predicted_artifact_np_plot[plot_trial_idx, channel_idx, :], 
                linewidth=1.5, color='#d62728', label='Predicted Artifact')
    axs[3].set_xlabel('Time (s)')
    axs[3].set_ylabel('Amplitude (µV)')
    axs[3].set_title(f'Predicted Artifact - Trial {trial_idx_for_title}, Channel {channel_idx}')
    axs[3].grid(True, alpha=0.3)
    axs[3].legend()
    axs[3].spines['top'].set_visible(False)
    axs[3].spines['right'].set_visible(False)
    
    plt.tight_layout()
    if args.save_plots:
        plt.savefig('signal_comparison.png', dpi=150)
        print("Signal comparison plot saved to: signal_comparison.png")
    plt.show()
    
    print("\n--- Plotting Complete ---")

if __name__ == '__main__':
    main()
