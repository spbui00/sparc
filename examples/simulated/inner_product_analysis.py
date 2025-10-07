import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from sparc.core.data_handler import DataHandler
from sparc.core.signal_data import SimulatedData, ArtifactTriggers


def inner_product(x, y):
    if x.ndim == 1 and y.ndim == 1:
        return np.dot(x, y)
    elif x.ndim == 2 and y.ndim == 2:
        return np.sum(x * y)
    else:
        return np.dot(x.flatten(), y.flatten())


def is_orthogonal(x, y, tolerance=1e-10):
    inner_prod = inner_product(x, y)
    return abs(inner_prod) < tolerance


def analyze_artifact_ground_truth_inner_products(data_obj):
    n_trials = data_obj.raw_data.shape[0]
    n_channels = data_obj.raw_data.shape[1]
    
    print(f"Data shape: {data_obj.raw_data.shape}")
    print(f"Number of trials: {n_trials}")
    print(f"Number of channels: {n_channels}")
    
    trial_inner_products = []
    channel_inner_products = np.zeros((n_trials, n_channels))
    
    print("\n=== Trial-wise Analysis ===")
    for trial_idx in range(n_trials):
        ground_truth = data_obj.ground_truth[trial_idx]  # (channels, timesteps)
        artifacts = data_obj.artifacts[trial_idx]  # (channels, timesteps)
        
        # Inner product for entire trial (all channels flattened)
        trial_inner_prod = inner_product(ground_truth, artifacts)
        trial_inner_products.append(trial_inner_prod)
        
        print(f"Trial {trial_idx}: Inner product = {trial_inner_prod:.6f}")
    
    print("\n=== Channel-wise Analysis ===")
    for trial_idx in range(n_trials):
        ground_truth = data_obj.ground_truth[trial_idx]  # (channels, timesteps)
        artifacts = data_obj.artifacts[trial_idx]  # (channels, timesteps)
        
        print(f"\nTrial {trial_idx}:")
        for channel_idx in range(n_channels):
            # Inner product for specific channel
            channel_inner_prod = inner_product(ground_truth[channel_idx], artifacts[channel_idx])
            channel_inner_products[trial_idx, channel_idx] = channel_inner_prod
            
            # Check orthogonality
            is_orth = is_orthogonal(ground_truth[channel_idx], artifacts[channel_idx])
            
            print(f"  Channel {channel_idx}: Inner product = {channel_inner_prod:.6f}, Orthogonal = {is_orth}")
    
    print("\n=== Summary Statistics ===")
    print(f"Trial-wise inner products:")
    print(f"  Mean: {np.mean(trial_inner_products):.6f}")
    print(f"  Std: {np.std(trial_inner_products):.6f}")
    print(f"  Min: {np.min(trial_inner_products):.6f}")
    print(f"  Max: {np.max(trial_inner_products):.6f}")
    
    print(f"\nChannel-wise inner products (across all trials):")
    print(f"  Mean: {np.mean(channel_inner_products):.6f}")
    print(f"  Std: {np.std(channel_inner_products):.6f}")
    print(f"  Min: {np.min(channel_inner_products):.6f}")
    print(f"  Max: {np.max(channel_inner_products):.6f}")
    
    # Check how many are orthogonal
    orthogonal_count = 0
    total_count = n_trials * n_channels
    
    for trial_idx in range(n_trials):
        for channel_idx in range(n_channels):
            if is_orthogonal(data_obj.ground_truth[trial_idx, channel_idx], 
                           data_obj.artifacts[trial_idx, channel_idx]):
                orthogonal_count += 1
    
    print(f"\nOrthogonality:")
    print(f"  Orthogonal channels: {orthogonal_count}/{total_count} ({100*orthogonal_count/total_count:.1f}%)")
    
    return {
        'trial_inner_products': trial_inner_products,
        'channel_inner_products': channel_inner_products,
        'orthogonal_count': orthogonal_count,
        'total_count': total_count
    }


def main():
    data_handler = DataHandler()
    data_obj = data_handler.load_npz_data('../../data/simulated_data_2x64_1000.npz')
    
    data_obj = SimulatedData(
        raw_data=data_obj['raw_data'],
        sampling_rate=data_obj['sampling_rate'],
        ground_truth=data_obj['ground_truth'],
        artifacts=data_obj['artifacts'],
        artifact_markers=ArtifactTriggers(starts=data_obj['artifact_markers']),  
        firing_rate=data_obj['firing_rate'],
        spike_train=data_obj['spike_train'],
        lfp=data_obj['lfp'],
        stim_params=None,
        snr=data_obj['snr'],
    )
    
    results = analyze_artifact_ground_truth_inner_products(data_obj)
    
    return results


if __name__ == "__main__":
    results = main()