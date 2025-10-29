import numpy as np


def extract_windows(data_obj, window_size_ms):
    window_size_samples = int(window_size_ms * data_obj.sampling_rate / 1000)
    
    clean_neural = data_obj.ground_truth
    artifacts = data_obj.artifacts
    num_trials, num_channels, num_timepoints = clean_neural.shape
    num_windows = int(num_timepoints // window_size_samples)
    
    clean_neural_windows = []
    artifact_windows = []
    
    for trial in range(num_trials):
        for ch in range(num_channels):
            for w in range(num_windows):
                start_idx = int(w * window_size_samples)
                end_idx = int(start_idx + window_size_samples)
                
                clean_window = clean_neural[trial, ch, start_idx:end_idx]
                if clean_window.shape[0] == window_size_samples:
                    clean_neural_windows.append(clean_window)
                
                artifact_window = artifacts[trial, ch, start_idx:end_idx]
                if artifact_window.shape[0] == window_size_samples:
                    artifact_windows.append(artifact_window)
    
    clean_neural_windows = np.array(clean_neural_windows)
    artifact_windows = np.array(artifact_windows)
    
    return clean_neural_windows, artifact_windows