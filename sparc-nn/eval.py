import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from glob import glob
from sparc import DataHandler
from sparc.core.signal_data import ArtifactTriggers, SignalDataWithGroundTruth
from sparc.core.evaluator import Evaluator
from sparc.core.neural_analyzer import NeuralAnalyzer
from sparc.core.plotting import NeuralPlotter
from models import UNet1D
from data_utils import prepare_dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _cov(M: torch.Tensor) -> torch.Tensor:
    """Calculate covariance between channels"""
    T, C, N = M.shape
    M_mean = torch.mean(M, dim=-1, keepdim=True)
    M_centered = M - M_mean
    Cov = (M_centered @ M_centered.transpose(-1, -2)) / (N - 1)
    return Cov

def compute_metrics(ground_truth_neural, ground_truth_artifacts, predicted_neural_np, 
                    predicted_artifact_np, analyzer, evaluator, mixed_data_np=None, print_results=True):
    """Compute evaluation metrics including SNR, MSE, soft rank, cosine similarity, PSD, and coherence"""
    import torch.nn.functional as F
    
    noise_after = predicted_neural_np - ground_truth_neural
    
    if mixed_data_np is not None:
        noise_before = mixed_data_np - ground_truth_neural
        snr_before = evaluator.calculate_snr(ground_truth_neural, noise_before)
        snr_improvement = evaluator.calculate_snr_improvement(mixed_data_np, predicted_neural_np, ground_truth_neural)
    else:
        noise_before = None
        snr_before = None
        snr_improvement = None
    
    if noise_before is not None:
        snr_before = evaluator.calculate_snr(ground_truth_neural, noise_before)
    else:
        snr_before = None
    
    snr_after = evaluator.calculate_snr(ground_truth_neural, noise_after)
    
    mse_neural = np.mean((predicted_neural_np - ground_truth_neural) ** 2)
    mse_artifact = np.mean((predicted_artifact_np - ground_truth_artifacts) ** 2)
    
    # Soft rank calculation
    predicted_artifact_tensor = torch.from_numpy(predicted_artifact_np)
    predicted_neural_tensor = torch.from_numpy(predicted_neural_np)
    
    cov_a = _cov(predicted_artifact_tensor)
    nuc_a = torch.linalg.norm(cov_a, ord='nuc', dim=(-2, -1))
    fro_a = torch.linalg.norm(cov_a, ord='fro', dim=(-2, -1))
    soft_rank_a = nuc_a / (fro_a + 1e-6)
    
    cov_s = _cov(predicted_neural_tensor)
    nuc_s = torch.linalg.norm(cov_s, ord='nuc', dim=(-2, -1))
    fro_s = torch.linalg.norm(cov_s, ord='fro', dim=(-2, -1))
    soft_rank_s = nuc_s / (fro_s + 1e-6)
    
    # Cosine similarity
    cos_sim_per_channel = F.cosine_similarity(
        predicted_neural_tensor, predicted_artifact_tensor, dim=2, eps=1e-8
    )
    loss_cosine = torch.mean(cos_sim_per_channel**2)
    
    # Ground truth metrics
    ground_truth_artifact_tensor = torch.from_numpy(ground_truth_artifacts)
    ground_truth_neural_tensor = torch.from_numpy(ground_truth_neural)
    
    cov_a_gt = _cov(ground_truth_artifact_tensor)
    nuc_a_gt = torch.linalg.norm(cov_a_gt, ord='nuc', dim=(-2, -1))
    fro_a_gt = torch.linalg.norm(cov_a_gt, ord='fro', dim=(-2, -1))
    soft_rank_a_gt = nuc_a_gt / (fro_a_gt + 1e-6)
    
    cov_s_gt = _cov(ground_truth_neural_tensor)
    nuc_s_gt = torch.linalg.norm(cov_s_gt, ord='nuc', dim=(-2, -1))
    fro_s_gt = torch.linalg.norm(cov_s_gt, ord='fro', dim=(-2, -1))
    soft_rank_s_gt = nuc_s_gt / (fro_s_gt + 1e-6)
    
    cos_sim_per_channel_gt = F.cosine_similarity(
        ground_truth_neural_tensor, ground_truth_artifact_tensor, dim=2, eps=1e-8
    )
    loss_cosine_gt = torch.mean(cos_sim_per_channel_gt**2)
    
    # PSD metrics
    psd_mse = analyzer.calculate_psd_mse(ground_truth_neural, predicted_neural_np)
    coherence_neural = analyzer.calculate_spectral_coherence(ground_truth_neural, predicted_neural_np)
    
    metrics = {
        'snr_before': snr_before,
        'snr_after': snr_after,
        'snr_improvement': snr_improvement,
        'mse_neural': mse_neural,
        'mse_artifact': mse_artifact,
        'soft_rank_a': torch.mean(soft_rank_a).item(),
        'soft_rank_s': torch.mean(soft_rank_s).item(),
        'soft_rank_a_gt': torch.mean(soft_rank_a_gt).item(),
        'soft_rank_s_gt': torch.mean(soft_rank_s_gt).item(),
        'cosine_similarity': loss_cosine.item(),
        'cosine_similarity_gt': loss_cosine_gt.item(),
        'psd_mse': psd_mse,
        'coherence_neural': coherence_neural,
    }
    
    if print_results:
        if snr_before is not None:
            print(f"SNR Before (Mixed): {snr_before:.2f} dB")
        print(f"SNR After (Cleaned): {snr_after:.2f} dB")
        if snr_improvement is not None:
            print(f"SNR Improvement: {snr_improvement:.2f} dB")
        
        print(f"MSE (Neural): {mse_neural:.4f}")
        print(f"MSE (Artifact): {mse_artifact:.4f}")
        
        print(f"Soft rank for artifact (predicted): {metrics['soft_rank_a']:.4f}")
        print(f"Soft rank for neural (predicted): {metrics['soft_rank_s']:.4f}")
        print(f"Cosine similarity (predicted): {metrics['cosine_similarity']:.4f}")
        
        print(f"Soft rank for artifact (GT): {metrics['soft_rank_a_gt']:.4f}")
        print(f"Soft rank for neural (GT): {metrics['soft_rank_s_gt']:.4f}")
        print(f"Cosine similarity (GT): {metrics['cosine_similarity_gt']:.4f}")
        
        print(f"\nPSD MSE (max across channels): {np.max(psd_mse):.4f} at channel {np.argmax(psd_mse)}")
        print(f"PSD MSE (min across channels): {np.min(psd_mse):.4f} at channel {np.argmin(psd_mse)}")
        print(f"PSD MSE (median across channels): {np.median(psd_mse):.4f}")
        
        print(f"\nMinimum Spectral Coherence: {np.min(coherence_neural):.4f} at channel {np.argmin(coherence_neural)}")
        print(f"Maximum Spectral Coherence: {np.max(coherence_neural):.4f} at channel {np.argmax(coherence_neural)}")
        print(f"Median Spectral Coherence: {np.median(coherence_neural):.4f}")
    
    return metrics

def perform_spectral_analysis(data_3d, analyzer, signal_type, plot=False, save=False, save_path=None):
    """Perform spectral analysis on 3D data array"""
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
        if save:
            if save_path is None:
                filename = f'psd_{signal_type.lower().replace(" ", "_")}.png'
            else:
                filename = os.path.join(save_path, f'psd_{signal_type.lower().replace(" ", "_")}.png')
            plt.savefig(filename, dpi=150)
            print(f"PSD plot saved to: {filename}")
        plt.show()
    
    return freqs, mean_psd

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate trained U-Net model')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to model checkpoint. If not provided, uses most recent in saved_models/')
    parser.add_argument('--data-path', type=str, default='../data/added_artifacts_swec_data_512_lower_freq_wo0.npz',
                       help='Path to data file (default: ../data/added_artifacts_swec_data_512_lower_freq_wo0.npz)')
    parser.add_argument('--trial-idx', type=int, default=0, help='Trial index for evaluation (default: 0)')
    parser.add_argument('--channel-idx', type=int, default=0, help='Channel index for plotting (default: 0)')
    parser.add_argument('--output-dir', type=str, default='.', help='Directory to save plots (default: current directory)')
    parser.add_argument('--artifact-duration-ms', type=int, default=40, help='Artifact duration in milliseconds (default: 40)')
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"Using device: {DEVICE}")
    
    # Find model path
    if args.model_path is None:
        saved_models_dir = 'saved_models'
        model_files = glob(os.path.join(saved_models_dir, 'unet1d_*.pth'))
        if not model_files:
            raise FileNotFoundError(f"No model files found in {saved_models_dir}. Please specify --model-path")
        model_path = sorted(model_files)[-1]
        print(f"Using most recent model: {model_path}")
    else:
        model_path = args.model_path
    
    # Load model
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    
    model_config = checkpoint.get('model_config', {})
    training_config = checkpoint.get('training_config', {})
    loss_config = checkpoint.get('loss_config', {})
    
    IN_CHANNELS = model_config.get('in_channels')
    OUT_CHANNELS = model_config.get('out_channels')
    
    if IN_CHANNELS is None or OUT_CHANNELS is None:
        raise ValueError("Model checkpoint missing model_config. Please retrain model.")
    
    # Get normalization stats
    if 'data_mean' in checkpoint and 'data_std' in checkpoint:
        data_mean = checkpoint['data_mean'].to(DEVICE)
        data_std = checkpoint['data_std'].to(DEVICE)
    else:
        print("Warning: Checkpoint missing normalization stats. Will compute from data.")
        data_mean = None
        data_std = None
    
    # Create and load model
    model = UNet1D(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    print(f"Model loaded with {sum(p.numel() for p in model.parameters())} parameters.")
    
    # Load data
    print(f"Loading data from: {args.data_path}")
    data_handler = DataHandler()
    data_obj_dict = data_handler.load_npz_data(args.data_path)
    
    data_obj = SignalDataWithGroundTruth(
        raw_data=data_obj_dict['mixed_data'],
        sampling_rate=data_obj_dict['sampling_rate'],
        ground_truth=data_obj_dict['ground_truth'],
        artifacts=data_obj_dict['artifacts'],
        artifact_markers=ArtifactTriggers(starts=data_obj_dict['artifact_markers']),
    )
    
    print(f"Data loaded: {data_obj.raw_data.shape}")
    
    # Prepare evaluation dataset
    predict_neural = training_config.get('predict_neural', False)
    artifact_duration_ms = training_config.get('artifact_duration_ms', args.artifact_duration_ms)
    
    eval_dataset, eval_loader, _, computed_mean, computed_std = prepare_dataset(
        data_obj=data_obj,
        data_obj_dict=data_obj_dict,
        batch_size=1,
        use_normalization=True,
        shuffle=False,
        artifact_duration_ms=artifact_duration_ms,
        trial_indices=[args.trial_idx]
    )
    
    # Use saved stats if available, otherwise use computed
    if data_mean is None:
        data_mean = computed_mean.to(DEVICE)
    if data_std is None:
        data_std = computed_std.to(DEVICE)
    
    # Run inference
    print("\n--- Running Inference ---")
    with torch.no_grad():
        batch = next(iter(eval_loader))
        x_mixed = batch[0].to(DEVICE)
        x_stim = batch[1].to(DEVICE)
        mask = batch[2].to(DEVICE)
        
        if predict_neural:
            s_pred = model(x_mixed, x_stim)
            a_pred = x_mixed - s_pred
        else:
            a_pred = model(x_mixed, x_stim)
            s_pred = x_mixed - a_pred
        
        # Denormalize
        s_pred_denorm = s_pred * data_std + data_mean
        a_pred_denorm = a_pred * data_std + data_mean
        x_mixed_denorm = x_mixed * data_std + data_mean
        
        # Blend with original in clean regions
        mask_expanded = mask.expand_as(x_mixed_denorm)
        s_pred_final = (1 - mask_expanded) * x_mixed_denorm + mask_expanded * s_pred_denorm
    
    # Convert to numpy
    predicted_neural_np = s_pred_final.cpu().numpy()
    predicted_artifact_np = a_pred_denorm.cpu().numpy()
    mixed_data_np = x_mixed_denorm.cpu().numpy()
    
    # Get ground truth
    ground_truth_neural = data_obj.ground_truth[args.trial_idx:args.trial_idx+1].astype(np.float32)
    ground_truth_artifacts = data_obj.artifacts[args.trial_idx:args.trial_idx+1].astype(np.float32)
    
    # Calculate metrics
    print("\n--- Evaluation Metrics ---")
    evaluator = Evaluator(sampling_rate=data_obj.sampling_rate)
    analyzer = NeuralAnalyzer(sampling_rate=data_obj.sampling_rate)
    
    compute_metrics(
        ground_truth_neural=ground_truth_neural,
        ground_truth_artifacts=ground_truth_artifacts,
        predicted_neural_np=predicted_neural_np,
        predicted_artifact_np=predicted_artifact_np,
        analyzer=analyzer,
        evaluator=evaluator,
        mixed_data_np=mixed_data_np,
        print_results=True
    )
    
    # Plotting
    print("\n--- Generating Plots ---")
    plotter = NeuralPlotter(evaluator)
    os.makedirs(args.output_dir, exist_ok=True)
    
    plot_trial_idx = 0  # Index in the filtered dataset (only 1 trial)
    time_axis = np.arange(predicted_neural_np.shape[2]) / data_obj.sampling_rate
    
    # Comparison plot: Predicted vs Ground Truth Overlaid
    fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Plot 1: Neural Signal Comparison
    axs[0].plot(time_axis, ground_truth_neural[plot_trial_idx, args.channel_idx, :], 
                linewidth=2, color='#2ca02c', label='Ground Truth Neural', alpha=0.7)
    axs[0].plot(time_axis, predicted_neural_np[plot_trial_idx, args.channel_idx, :], 
                linewidth=1.5, color='#1f77b4', label='Predicted Neural', alpha=0.7, linestyle='--')
    axs[0].set_ylabel('Amplitude (µV)', fontsize=11)
    axs[0].set_title(f'Neural Signal: Predicted vs Ground Truth - Trial {args.trial_idx}, Channel {args.channel_idx}', 
                     fontsize=12, fontweight='bold')
    axs[0].grid(True, alpha=0.3)
    axs[0].legend(loc='upper right', fontsize=10)
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    
    # Plot 2: Artifact Comparison
    axs[1].plot(time_axis, ground_truth_artifacts[plot_trial_idx, args.channel_idx, :], 
                linewidth=2, color='#ff7f0e', label='Ground Truth Artifact', alpha=0.7)
    axs[1].plot(time_axis, predicted_artifact_np[plot_trial_idx, args.channel_idx, :], 
                linewidth=1.5, color='#d62728', label='Predicted Artifact', alpha=0.7, linestyle='--')
    axs[1].set_xlabel('Time (s)', fontsize=11)
    axs[1].set_ylabel('Amplitude (µV)', fontsize=11)
    axs[1].set_title(f'Artifact: Predicted vs Ground Truth - Trial {args.trial_idx}, Channel {args.channel_idx}', 
                    fontsize=12, fontweight='bold')
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(loc='upper right', fontsize=10)
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    
    plt.tight_layout()
    comparison_path = os.path.join(args.output_dir, 'model_comparison.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    print(f"Comparison plot saved to: {comparison_path}")
    plt.close()
    
    # Individual signal plots
    fig, axs = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    
    # Plot 1: Ground Truth Neural Signal
    axs[0].plot(time_axis, ground_truth_neural[plot_trial_idx, args.channel_idx, :], 
                linewidth=1.5, color='#2ca02c', label='Ground Truth Neural', alpha=0.8)
    axs[0].set_ylabel('Amplitude (µV)', fontsize=11)
    axs[0].set_title(f'Ground Truth Neural Signal - Trial {args.trial_idx}, Channel {args.channel_idx}', 
                     fontsize=12, fontweight='bold')
    axs[0].grid(True, alpha=0.3)
    axs[0].legend(loc='upper right', fontsize=10)
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    
    # Plot 2: Predicted Neural Signal
    axs[1].plot(time_axis, predicted_neural_np[plot_trial_idx, args.channel_idx, :], 
                linewidth=1.5, color='#1f77b4', label='Predicted Neural', alpha=0.8)
    axs[1].set_ylabel('Amplitude (µV)', fontsize=11)
    axs[1].set_title(f'Predicted Neural Signal - Trial {args.trial_idx}, Channel {args.channel_idx}', 
                     fontsize=12, fontweight='bold')
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(loc='upper right', fontsize=10)
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    
    # Plot 3: Ground Truth Artifact
    axs[2].plot(time_axis, ground_truth_artifacts[plot_trial_idx, args.channel_idx, :], 
                linewidth=1.5, color='#ff7f0e', label='Ground Truth Artifact', alpha=0.8)
    axs[2].set_ylabel('Amplitude (µV)', fontsize=11)
    axs[2].set_title(f'Ground Truth Artifact - Trial {args.trial_idx}, Channel {args.channel_idx}', 
                     fontsize=12, fontweight='bold')
    axs[2].grid(True, alpha=0.3)
    axs[2].legend(loc='upper right', fontsize=10)
    axs[2].spines['top'].set_visible(False)
    axs[2].spines['right'].set_visible(False)
    
    # Plot 4: Predicted Artifact
    axs[3].plot(time_axis, predicted_artifact_np[plot_trial_idx, args.channel_idx, :], 
                linewidth=1.5, color='#d62728', label='Predicted Artifact', alpha=0.8)
    axs[3].set_xlabel('Time (s)', fontsize=11)
    axs[3].set_ylabel('Amplitude (µV)', fontsize=11)
    axs[3].set_title(f'Predicted Artifact - Trial {args.trial_idx}, Channel {args.channel_idx}', 
                     fontsize=12, fontweight='bold')
    axs[3].grid(True, alpha=0.3)
    axs[3].legend(loc='upper right', fontsize=10)
    axs[3].spines['top'].set_visible(False)
    axs[3].spines['right'].set_visible(False)
    
    plt.tight_layout()
    signals_path = os.path.join(args.output_dir, 'model_predictions.png')
    plt.savefig(signals_path, dpi=150, bbox_inches='tight')
    print(f"Signal plots saved to: {signals_path}")
    plt.close()
    
    print("\n--- Evaluation Complete ---")

if __name__ == '__main__':
    main()
