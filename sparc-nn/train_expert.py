import torch
import torch.optim as optim
import os
import argparse
from datetime import datetime
from models import NeuralExpertAE
from data_utils import prepare_swec_expert_dataset

def parse_args():
    parser = argparse.ArgumentParser(description='Train Neural Expert Autoencoder on clean SWEC data')
    parser.add_argument('--swec-mat-file', type=str, required=True,
                       help='Path to SWEC .mat file for expert training')
    parser.add_argument('--swec-info-file', type=str, default=None,
                       help='Path to SWEC _info.mat file for seizure exclusion (optional)')
    parser.add_argument('--window-len', type=float, default=2.0,
                       help='Window length in seconds for expert dataset (default: 2.0)')
    parser.add_argument('--stride', type=float, default=1.0,
                       help='Stride in seconds for expert dataset (default: 1.0)')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Max number of chunks to extract from SWEC data (optional)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size (default: 4)')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate (default: 1e-3)')
    parser.add_argument('--output-dir', type=str, default='saved_models', 
                       help='Directory to save expert model (default: saved_models)')
    parser.add_argument('--out-channels', type=int, default=None,
                       help='Number of output channels (inferred from data if not provided)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")
    
    # Prepare expert dataset
    print("\n" + "=" * 60)
    print("Preparing Expert Dataset from SWEC Clean Data")
    print("=" * 60)
    
    expert_ds, expert_loader, expert_mean, expert_std, expert_sampling_rate = prepare_swec_expert_dataset(
        mat_file_path=args.swec_mat_file,
        info_file_path=args.swec_info_file,
        window_len_sec=args.window_len,
        stride_sec=args.stride,
        batch_size=args.batch_size,
        max_samples=args.max_samples
    )
    
    print(f"Expert Dataset: {len(expert_ds)} chunks")
    print(f"Expert Sampling Rate: {expert_sampling_rate} Hz")
    
    # Get number of channels from first batch
    sample_batch = next(iter(expert_loader))
    if isinstance(sample_batch, torch.Tensor):
        out_channels = sample_batch.shape[1]
    else:
        out_channels = sample_batch[0].shape[1] if len(sample_batch) > 0 else args.out_channels
    
    if args.out_channels is not None:
        out_channels = args.out_channels
    
    if out_channels is None:
        raise ValueError("Could not determine number of channels. Please provide --out-channels")
    
    print(f"Number of channels: {out_channels}")
    
    # Create expert model
    neural_expert = NeuralExpertAE(in_channels=out_channels).to(DEVICE)
    print(f"Expert model created with {sum(p.numel() for p in neural_expert.parameters())} parameters.")
    
    # Setup optimizer
    optimizer_expert = optim.Adam(neural_expert.parameters(), lr=args.learning_rate)
    
    print("\n--- Starting Expert Training ---")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, Learning rate: {args.learning_rate}")
    
    loss_history = []
    
    # Training loop
    for epoch in range(args.epochs):
        neural_expert.train()
        total_loss = 0.0
        
        for batch in expert_loader:
            if isinstance(batch, torch.Tensor):
                x_clean = batch.to(DEVICE)
            else:
                x_clean = batch[0].to(DEVICE)
            
            reconstruction = neural_expert(x_clean)
            loss = torch.mean((reconstruction - x_clean) ** 2)
            
            optimizer_expert.zero_grad()
            loss.backward()
            optimizer_expert.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(expert_loader)
        loss_history.append(avg_loss)
        
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"Expert Epoch [{epoch+1}/{args.epochs}], MSE Loss: {avg_loss:.6f}")
    
    print("--- Expert Training Complete ---")
    
    # Save expert model
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    expert_model_path = os.path.join(args.output_dir, f'neural_expert_{timestamp}.pth')
    
    checkpoint = {
        'epoch': args.epochs,
        'model_state_dict': neural_expert.state_dict(),
        'optimizer_state_dict': optimizer_expert.state_dict(),
        'in_channels': out_channels,
        'expert_mean': expert_mean.cpu(),
        'expert_std': expert_std.cpu(),
        'sampling_rate': expert_sampling_rate,
        'loss_history': loss_history,
        'training_config': {
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
            'num_epochs': args.epochs,
            'window_len_sec': args.window_len,
            'stride_sec': args.stride,
        }
    }
    
    torch.save(checkpoint, expert_model_path)
    print(f"Expert model saved to: {expert_model_path}")

if __name__ == '__main__':
    main()


