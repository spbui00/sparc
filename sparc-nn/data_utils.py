import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import scipy.io
import os
import re

class ArtifactDataset(Dataset):
    def __init__(self, data, stim_trace, artifact_masks, data_mean, data_std, use_normalization=True, eps=1e-8):
        self.data = torch.from_numpy(data)
        self.stim_trace = stim_trace.float()
        self.artifact_masks = artifact_masks
        # Ensure mean/std are tensors and correct shape for broadcasting
        self.data_mean = data_mean if torch.is_tensor(data_mean) else torch.tensor(data_mean)
        self.data_std = data_std if torch.is_tensor(data_std) else torch.tensor(data_std)
        self.use_normalization = use_normalization
        self.eps = eps
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]  # Shape: (C, T)
        
        if self.use_normalization:
            # data_mean/data_std are (1, C, 1), squeeze to (C, 1) for proper broadcasting
            mean = self.data_mean.squeeze(0)  # (C, 1)
            std = self.data_std.squeeze(0)    # (C, 1)
            norm_data = (x - mean) / (std + self.eps)  # (C, T) - (C, 1) = (C, T)
            while norm_data.dim() > 2 and norm_data.shape[0] == 1:
                norm_data = norm_data.squeeze(0)
        else:
            norm_data = x
        
        stim = self.stim_trace[idx]
        if stim.dim() == 1:
            stim = stim.unsqueeze(0)  # (T) -> (1, T)
        elif stim.dim() == 3:
            stim = stim.squeeze(0) if stim.shape[0] == 1 else stim  # (1, 1, T) -> (1, T)
            
        return norm_data, stim, self.artifact_masks[idx]

def compute_robust_stats(data_tensor):
    if not torch.is_tensor(data_tensor):
        data_tensor = torch.from_numpy(data_tensor).float()
        
    channels = data_tensor.shape[1]
    flat_data = data_tensor.transpose(0, 1).reshape(channels, -1)
    
    # 2. Median
    median = torch.nanquantile(flat_data, 0.5, dim=1, keepdim=True).float()
    
    # 3. Robust Std (IQR / 1.3489)
    q75 = torch.nanquantile(flat_data, 0.75, dim=1, keepdim=True)
    q25 = torch.nanquantile(flat_data, 0.25, dim=1, keepdim=True)
    iqr = q75 - q25
    robust_std = iqr / 1.3489
    
    # Safety clamp
    robust_std = torch.clamp(robust_std, min=1e-6).float()
    
    # Reshape to (1, C, 1) for easy broadcasting later
    return median.unsqueeze(0), robust_std.unsqueeze(0)

def prepare_dataset(data_obj, data_obj_dict, batch_size=1, use_normalization=True, shuffle=True, artifact_duration_ms=40, trial_indices=None):
    """
    Prepare a dataset for training the main model.
    
    Args:
        data_obj: SignalDataWithGroundTruth object
        data_obj_dict: Dictionary containing data (for stim_trace)
        batch_size: Batch size for DataLoader
        use_normalization: Whether to normalize the data
        shuffle: Whether to shuffle the dataset
        artifact_duration_ms: Duration of artifacts in milliseconds
        trial_indices: List of trial indices to use (e.g., [3]). If None, uses all trials.
    
    Returns:
        dataset, data_loader, artifact_masks, data_mean, data_std
    """
    mixed_data = data_obj.raw_data.astype(np.float32)
    
    # Select specific trials if specified
    if trial_indices is not None:
        mixed_data = mixed_data[trial_indices]
        print(f"Using mixed data for main training (trials {trial_indices})")
    else:
        print("Using mixed data for main training (all trials)")
    
    # Calculate ROBUST stats (The fix)
    print("Calculating Robust Statistics (ignoring artifacts)...")
    data_mean, data_std = compute_robust_stats(mixed_data)
    
    print(f"Robust Norm Stats - Mean: {data_mean.mean():.4f}, Std: {data_std.mean():.4f}")

    # Prepare Masks
    # If trial_indices specified, we need to pass the original trial indices for artifact markers
    artifact_masks = prepare_artifact_masks(
        artifact_markers=data_obj.artifact_markers,
        n_trials=mixed_data.shape[0],
        n_samples=mixed_data.shape[2],
        artifact_duration_ms=artifact_duration_ms,
        sampling_rate=data_obj.sampling_rate,
        trial_indices=trial_indices
    )
    
    # Select corresponding stim traces if trial_indices specified
    stim_trace_tensor = torch.from_numpy(data_obj_dict['stim_trace']).float()
    if trial_indices is not None:
        stim_trace_tensor = stim_trace_tensor[trial_indices]
    
    dataset = ArtifactDataset(
        data=mixed_data,
        stim_trace=stim_trace_tensor,
        artifact_masks=artifact_masks,
        data_mean=data_mean,
        data_std=data_std,
        use_normalization=use_normalization
    )
    
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return dataset, data_loader, artifact_masks, data_mean, data_std


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


def prepare_artifact_masks(artifact_markers, n_trials, n_samples, artifact_duration_ms, sampling_rate, blur_radius=5, trial_indices=None):
    """
    Prepare artifact masks for the specified trials.
    
    Args:
        artifact_markers: ArtifactTriggers object with artifact start indices
        n_trials: Number of trials in the filtered dataset
        n_samples: Number of time samples per trial
        artifact_duration_ms: Duration of artifacts in milliseconds
        sampling_rate: Sampling rate in Hz
        blur_radius: Radius for soft mask blurring
        trial_indices: List of original trial indices (e.g., [3] if using trial 3). 
                      If None, assumes trials 0..n_trials-1
    
    Returns:
        Stacked artifact masks tensor
    """
    artifact_duration = int(artifact_duration_ms / 1000 * sampling_rate)
    artifact_indices_raw = artifact_markers.starts
    
    print("Precomputing artifact masks...")
    artifact_masks = []
    
    if trial_indices is None:
        trial_indices = list(range(n_trials))
    
    for i, original_trial_idx in enumerate(trial_indices):
        if artifact_indices_raw.ndim == 3:
            trial_indices_data = artifact_indices_raw[original_trial_idx].flatten()
        elif artifact_indices_raw.ndim == 2:
            trial_indices_data = artifact_indices_raw[original_trial_idx]
        else:
            trial_indices_data = artifact_indices_raw
        
        trial_indices_data = trial_indices_data[trial_indices_data >= 0]
        trial_durations = [artifact_duration] * len(trial_indices_data)
        
        mask = create_soft_artifact_mask(trial_indices_data, trial_durations, n_samples, blur_radius=blur_radius)
        artifact_masks.append(mask)
    
    artifact_masks = torch.stack(artifact_masks)
    return artifact_masks


class CleanDataset(Dataset):
    def __init__(self, data, data_mean, data_std, use_normalization=True, eps=1e-8):
        self.data = data  # List of clean chunks, each is (C, T_chunk)
        self.data_mean = data_mean if torch.is_tensor(data_mean) else torch.tensor(data_mean)
        self.data_std = data_std if torch.is_tensor(data_std) else torch.tensor(data_std)
        self.use_normalization = use_normalization
        self.eps = eps
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]  # Shape: (C, T_chunk)
        
        if self.use_normalization:
            mean = self.data_mean.squeeze(0)  # (C, 1)
            std = self.data_std.squeeze(0)    # (C, 1)
            norm_data = (x - mean) / (std + self.eps)
            while norm_data.dim() > 2 and norm_data.shape[0] == 1:
                norm_data = norm_data.squeeze(0)
        else:
            norm_data = x
            
        return norm_data


def prepare_clean_dataset(data_obj, batch_size=4, use_normalization=True, shuffle=True, trial_indices=None):
    """
    Prepare a dataset using ground truth as clean data for training the neural expert.
    
    Args:
        data_obj: SignalDataWithGroundTruth object
        batch_size: Batch size for DataLoader
        use_normalization: Whether to normalize the data
        shuffle: Whether to shuffle the dataset
        trial_indices: List of trial indices to use (e.g., [0, 1, 2]). If None, uses all trials.
    
    Returns:
        dataset, data_loader, data_mean, data_std
    """
    if not hasattr(data_obj, 'ground_truth') or data_obj.ground_truth is None:
        raise ValueError("ground_truth is required for prepare_clean_dataset")
    
    clean_data = data_obj.ground_truth.astype(np.float32)
    
    # Select specific trials if specified
    if trial_indices is not None:
        clean_data = clean_data[trial_indices]
        print(f"Using ground truth as clean data for expert training (trials {trial_indices})")
    else:
        print("Using ground truth as clean data for expert training (all trials)")
    
    # Compute statistics on clean data
    print("Calculating statistics for clean data...")
    data_mean, data_std = compute_robust_stats(clean_data)
    print(f"Clean Data Stats - Median: {data_mean.mean():.4f}, Robust Std: {data_std.mean():.4f}")
    
    # Convert to list of tensors (one per trial)
    clean_data_list = [torch.from_numpy(clean_data[i]).float() for i in range(clean_data.shape[0])]
    
    dataset = CleanDataset(
        data=clean_data_list,
        data_mean=data_mean,
        data_std=data_std,
        use_normalization=use_normalization
    )
    
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return dataset, data_loader, data_mean, data_std


def prepare_swec_expert_dataset(mat_file_path, info_file_path=None, window_len_sec=2.0, stride_sec=1.0, batch_size=4, max_samples=None):
    """
    Loads a SWEC-ETHZ .mat file, chops it into clean windows, and returns a CleanDataset.
    """
    print(f"Loading SWEC data from {mat_file_path}...")
    
    # 1. Parse File Index to determine time offset
    # e.g. ID01_16h.mat -> index 16 -> starts at (16-1)*3600 seconds
    # e.g. concatenated_hours_15_18.mat -> index 15 -> starts at (15-1)*3600 seconds
    filename = os.path.basename(mat_file_path)
    
    # Try concatenated format first: concatenated_hours_15_18.mat
    match_concatenated = re.search(r'concatenated_hours_(\d+)_(\d+)\.mat', filename, re.IGNORECASE)
    if match_concatenated:
        start_hour = int(match_concatenated.group(1))
        end_hour = int(match_concatenated.group(2))
        start_offset_sec = (start_hour - 1) * 3600
        print(f"✔ Parsed File Index (concatenated): {start_hour} to {end_hour}")
        print(f"✔ Global Time Offset: {start_offset_sec} seconds (Hours {start_hour} to {end_hour})")
    else:
        # Try single hour format: 16h.mat or ID01_16h.mat
        match = re.search(r'(\d+)h\.mat', filename, re.IGNORECASE)
        if match:
            file_idx = int(match.group(1))
            start_offset_sec = (file_idx - 1) * 3600
            print(f"✔ Parsed File Index: {file_idx}")
            print(f"✔ Global Time Offset: {start_offset_sec} seconds (Hour {file_idx-1} to {file_idx})")
        else:
            print("Warning: Could not determine file index from name. Assuming 0 offset.")
            start_offset_sec = 0

    # 2. Load Data
    try:
        mat_data = scipy.io.loadmat(mat_file_path)
    except NotImplementedError:
        import h5py
        with h5py.File(mat_file_path, 'r') as f:
            raw_data = np.array(f['EEG']) # Usually (Channels, Time) for v7.3
            if raw_data.shape[0] > raw_data.shape[1]: # Check if transposed
                 raw_data = raw_data.T
    else:
        raw_data = mat_data['EEG']

    # Ensure (Channels, Time) format
    if raw_data.shape[0] > raw_data.shape[1]:
        raw_data = raw_data.T
        
    # Determine Sampling Rate
    if 'fs' in mat_data:
        sampling_rate = int(mat_data['fs'][0][0])
    elif info_file_path: # Try to get it from info file
         info = scipy.io.loadmat(info_file_path)
         if 'fs' in info:
             sampling_rate = int(info['fs'][0][0])
         else:
             sampling_rate = 512
    else:
        sampling_rate = 512 # Fallback

    print(f"Raw Data Shape: {raw_data.shape} | Sampling Rate: {sampling_rate}Hz")
    
    n_channels = raw_data.shape[0]
    n_total_samples = raw_data.shape[1]
    window_samples = int(window_len_sec * sampling_rate)
    stride_samples = int(stride_sec * sampling_rate)
    
    # 3. Create Valid Mask (Exclude Seizures)
    valid_mask = np.ones(n_total_samples, dtype=bool)
    
    if info_file_path:
        print(f"Loading Seizure Info from {info_file_path}...")
        info_data = scipy.io.loadmat(info_file_path)
        
        if 'seizure_begin' in info_data:
            starts = info_data['seizure_begin'].flatten()
            ends = info_data['seizure_end'].flatten()

            print(f"Seizure starts: {starts}")
            print(f"Seizure ends: {ends}")
            
            buffer_sec = 900 # 15 mins buffer
            
            seizures_in_file = 0
            for s, e in zip(starts, ends):
                # Convert global seizure time to local file time
                s_local = s - start_offset_sec
                e_local = e - start_offset_sec
                
                # Check if seizure overlaps with this file (0 to 3600s)
                file_duration = n_total_samples / sampling_rate
                if e_local + buffer_sec < 0 or s_local - buffer_sec > file_duration:
                    continue # Seizure is not in this file
                
                seizures_in_file += 1
                
                # Calculate indices in current file
                s_idx = max(0, int((s_local - buffer_sec) * sampling_rate))
                e_idx = min(n_total_samples, int((e_local + buffer_sec) * sampling_rate))
                
                valid_mask[s_idx:e_idx] = False
                
            print(f"Excluded {seizures_in_file} seizure regions overlapping with this file.")
            
    # 4. Chunking & Platinum Filtering
    clean_chunks = []
    print("Chunking and filtering for artifacts (>500uV)...")
    
    for start_idx in range(0, n_total_samples - window_samples, stride_samples):
        if max_samples and len(clean_chunks) >= max_samples:
            break
            
        end_idx = start_idx + window_samples
        
        if not np.all(valid_mask[start_idx:end_idx]):
            continue
            
        chunk = raw_data[:, start_idx:end_idx]
        
        if np.max(np.abs(chunk)) > 500.0:
            continue
            
        clean_chunks.append(torch.from_numpy(chunk).float())

    print(f"Extracted {len(clean_chunks)} clean windows.")
    
    if len(clean_chunks) == 0:
        raise ValueError("No clean chunks found!")

    # 5. Robust Stats
    subset_size = min(len(clean_chunks), 1000)
    indices = np.random.choice(len(clean_chunks), subset_size, replace=False)
    subset_tensor = torch.stack([clean_chunks[i] for i in indices])
    
    print("Calculating Robust Stats...")
    data_mean, data_std = compute_robust_stats(subset_tensor)
    print(f"SWEC Clean Stats - Median: {data_mean.mean():.4f}, Std: {data_std.mean():.4f}")
    
    # 6. Create Dataset
    dataset = CleanDataset(
        data=clean_chunks,
        data_mean=data_mean,
        data_std=data_std,
        use_normalization=True
    )
    
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataset, data_loader, data_mean, data_std, sampling_rate

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import argparse
    
    parser = argparse.ArgumentParser(description='Test prepare_swec_expert_dataset and plot signals')
    parser.add_argument('--mat-file', type=str, required=True, help='Path to SWEC .mat file')
    parser.add_argument('--info-file', type=str, default=None, help='Path to _info.mat file (optional)')
    parser.add_argument('--chunk-idx', type=int, default=0, help='Chunk index to plot (default: 0)')
    parser.add_argument('--channel-idx', type=int, default=0, help='Channel index to plot (default: 0)')
    parser.add_argument('--num-chunks', type=int, default=10, help='Number of chunks to plot (default: 5)')
    parser.add_argument('--window-len', type=float, default=2.0, help='Window length in seconds (default: 2.0)')
    parser.add_argument('--stride', type=float, default=1.0, help='Stride in seconds (default: 1.0)')
    parser.add_argument('--max-samples', type=int, default=None, help='Max number of chunks to extract (for testing)')
    args = parser.parse_args()
    
    # Load dataset
    print("=" * 60)
    print("Testing prepare_swec_expert_dataset")
    print("=" * 60)
    
    dataset, data_loader, data_mean, data_std, sampling_rate = prepare_swec_expert_dataset(
        mat_file_path=args.mat_file,
        info_file_path=args.info_file,
        window_len_sec=args.window_len,
        stride_sec=args.stride,
        batch_size=4,
        max_samples=args.max_samples
    )
    
    print(f"\nDataset size: {len(dataset)} chunks")
    print(f"Data mean shape: {data_mean.shape}")
    print(f"Data std shape: {data_std.shape}")
    print(f"Sampling rate: {sampling_rate} Hz")
    
    # Get a batch to inspect
    batch = next(iter(data_loader))
    print(f"Batch shape: {batch.shape}")
    print(f"Expected: (batch_size, channels, time_samples)")
    
    # Get time samples from a sample chunk
    sample_chunk = dataset[0]
    if isinstance(sample_chunk, torch.Tensor):
        time_samples = sample_chunk.shape[1]
    else:
        time_samples = len(sample_chunk[0]) if hasattr(sample_chunk, '__len__') else 1024
    
    print(f"Time samples per chunk: {time_samples}")
    
    # Plot 1: Single chunk, single channel
    print(f"\nPlotting chunk {args.chunk_idx}, channel {args.channel_idx}...")
    chunk_data = dataset[args.chunk_idx]
    
    if isinstance(chunk_data, torch.Tensor):
        chunk_np = chunk_data.numpy()
    else:
        chunk_np = chunk_data
    
    # Handle denormalization if needed
    if dataset.use_normalization:
        mean = dataset.data_mean.squeeze(0).numpy()
        std = dataset.data_std.squeeze(0).numpy()
        chunk_denorm = chunk_np * std + mean
    else:
        chunk_denorm = chunk_np
    
    time_axis = np.arange(chunk_denorm.shape[1]) / sampling_rate
    
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot selected channel
    axs[0].plot(time_axis, chunk_denorm[args.channel_idx, :], 
                linewidth=1.5, color='#2ca02c', label=f'Channel {args.channel_idx}')
    axs[0].set_ylabel('Amplitude (µV)', fontsize=11)
    axs[0].set_title(f'SWEC Clean Chunk {args.chunk_idx} - Channel {args.channel_idx}', fontsize=12, fontweight='bold')
    axs[0].grid(True, alpha=0.3)
    axs[0].legend(loc='upper right', fontsize=10)
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    
    # Plot all channels (overlaid, with transparency)
    axs[1].plot(time_axis, chunk_denorm.T, alpha=0.3, linewidth=0.5)
    axs[1].plot(time_axis, chunk_denorm[args.channel_idx, :], 
                linewidth=2, color='#d62728', label=f'Channel {args.channel_idx} (highlighted)')
    axs[1].set_xlabel('Time (s)', fontsize=11)
    axs[1].set_ylabel('Amplitude (µV)', fontsize=11)
    axs[1].set_title(f'All Channels - Chunk {args.chunk_idx}', fontsize=12, fontweight='bold')
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(loc='upper right', fontsize=10)
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('swec_test_single_chunk.png', dpi=150, bbox_inches='tight')
    print(f"Saved plot to: swec_test_single_chunk.png")
    plt.show()
    
    # Plot 2: Multiple chunks, same channel
    print(f"\nPlotting {args.num_chunks} chunks, channel {args.channel_idx}...")
    num_chunks_to_plot = min(args.num_chunks, len(dataset))
    
    fig, axs = plt.subplots(num_chunks_to_plot, 1, figsize=(12, 2*num_chunks_to_plot), sharex=True)
    if num_chunks_to_plot == 1:
        axs = [axs]
    
    for i in range(num_chunks_to_plot):
        chunk_data = dataset[i]
        if isinstance(chunk_data, torch.Tensor):
            chunk_np = chunk_data.numpy()
        else:
            chunk_np = chunk_data
        
        if dataset.use_normalization:
            mean = dataset.data_mean.squeeze(0).numpy()
            std = dataset.data_std.squeeze(0).numpy()
            chunk_denorm = chunk_np * std + mean
        else:
            chunk_denorm = chunk_np
        
        time_axis = np.arange(chunk_denorm.shape[1]) / sampling_rate
        
        axs[i].plot(time_axis, chunk_denorm[args.channel_idx, :], 
                   linewidth=1.5, color=f'C{i}', label=f'Chunk {i}')
        axs[i].set_ylabel('Amplitude (µV)', fontsize=10)
        axs[i].set_title(f'Chunk {i} - Channel {args.channel_idx}', fontsize=11)
        axs[i].grid(True, alpha=0.3)
        axs[i].legend(loc='upper right', fontsize=9)
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
    
    axs[-1].set_xlabel('Time (s)', fontsize=11)
    plt.tight_layout()
    plt.savefig('swec_test_multiple_chunks.png', dpi=150, bbox_inches='tight')
    print(f"Saved plot to: swec_test_multiple_chunks.png")
    plt.show()
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Dataset Statistics:")
    print("=" * 60)
    all_chunks = []
    for i in range(min(100, len(dataset))):  # Sample first 100 chunks for stats
        chunk = dataset[i]
        if isinstance(chunk, torch.Tensor):
            all_chunks.append(chunk.numpy())
        else:
            all_chunks.append(chunk)
    
    if all_chunks:
        all_data = np.stack(all_chunks, axis=0)
        print(f"Sampled chunks shape: {all_data.shape}")
        print(f"Mean amplitude: {np.mean(all_data):.4f} µV")
        print(f"Std amplitude: {np.std(all_data):.4f} µV")
        print(f"Min amplitude: {np.min(all_data):.4f} µV")
        print(f"Max amplitude: {np.max(all_data):.4f} µV")
        print(f"Max abs amplitude: {np.max(np.abs(all_data)):.4f} µV")
    
    print("\nTest complete!")
    