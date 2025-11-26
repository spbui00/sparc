import torch
import numpy as np

def create_stim_trace(
    n_trials, 
    n_samples, 
    artifact_markers, 
    amplitudes_array, 
    sampling_rate,
    t_phase_ms,
    t_inter_ms
):
    """
    Generates a tensor (N_TRIALS, 1, N_SAMPLES) representing the input current.
    Draws a biphasic square wave: [Negative Phase] -> [Gap] -> [Positive Phase]
    """
    # This function needs to output a NumPy array for the data gen script
    # but a Torch tensor for the (old) train script. Let's make it output torch.
    stim_trace = np.zeros((n_trials, 1, n_samples), dtype=np.float32)
    
    phase_samples = int((t_phase_ms / 1000) * sampling_rate)
    gap_samples = int((t_inter_ms / 1000) * sampling_rate)
    
    # Simple check for the 512 Hz bug
    if phase_samples == 0:
        print(f"WARNING: Pulse phase width (0.17ms) is shorter than one sample")
        print(f"at {sampling_rate} Hz. This will result in a zero trace.")
        
    global_pulse_counter = 0
    total_amps = len(amplitudes_array)
    
    starts = artifact_markers.starts
    
    for trial_idx in range(n_trials):
        if isinstance(starts, list) or (isinstance(starts, np.ndarray) and starts.ndim == 2):
            trial_starts = starts[trial_idx]
        else:
            trial_starts = starts
            
        trial_starts = trial_starts[trial_starts >= 0]
        
        for start_idx in trial_starts:
            start_idx = int(start_idx)
            
            if global_pulse_counter >= total_amps:
                break 
            current_amp = amplitudes_array[global_pulse_counter]
            global_pulse_counter += 1
            
            p1_start = start_idx
            p1_end = min(p1_start + phase_samples, n_samples)
            stim_trace[trial_idx, 0, p1_start:p1_end] = -current_amp
            
            p2_start = start_idx + phase_samples + gap_samples
            p2_end = min(p2_start + phase_samples, n_samples)
            
            if p2_start < n_samples:
                stim_trace[trial_idx, 0, p2_start:p2_end] = current_amp

    max_val = np.max(np.abs(stim_trace))
    if max_val > 0:
        stim_trace = stim_trace / max_val

    # Return a torch tensor as your original train.py script expected
    return torch.from_numpy(stim_trace)
