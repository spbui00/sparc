import numpy as np
import scipy.io
import h5py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sparc.core.data_handler import DataHandler

data_handler = DataHandler()

pickle_file = '../research/datasets/SWEC-ETHZ/selected_clips_pat123.pkl'
mat_file = '../data/concatenated_hours_15_18.mat'

print("=" * 60)
print("Loading pickle file...")
print("=" * 60)

pickle_data = data_handler.load_pickle_data(pickle_file)
print(f"Pickle keys: {list(pickle_data.keys())}")

patient_id = '01'
print(f"\nChecking for patient_id: '{patient_id}'")
if patient_id not in pickle_data:
    print(f"✗ Patient {patient_id} not found in pickle file")
    print(f"Available patients: {list(pickle_data.keys())}")
    exit(1)

print(f"✓ Patient {patient_id} found in pickle file")

if 'seizure_clips' not in pickle_data[patient_id]:
    print(f"✗ 'seizure_clips' not found for patient {patient_id}")
    exit(1)

if 'non_seizure_clips' not in pickle_data[patient_id]:
    print(f"✗ 'non_seizure_clips' not found for patient {patient_id}")
    exit(1)

seizure_clips = np.array(pickle_data[patient_id]['seizure_clips'])
non_seizure_clips = np.array(pickle_data[patient_id]['non_seizure_clips'])

print(f"\nPatient {patient_id} data:")
print(f"  Seizure clips shape: {seizure_clips.shape}")
print(f"  Non-seizure clips shape: {non_seizure_clips.shape}")

trials, channels, segments, samples = non_seizure_clips.shape
print(f"  Trials: {trials}, Channels: {channels}, Segments: {segments}, Samples per segment: {samples}")

non_seizure_reshaped = non_seizure_clips.reshape(trials, channels, segments * samples)
print(f"\nNon-seizure reshaped shape: {non_seizure_reshaped.shape}")

total_samples_pickle = non_seizure_reshaped.shape[2]
sampling_rate = 512
duration_pickle = total_samples_pickle / sampling_rate
print(f"Total samples per trial: {total_samples_pickle}")
print(f"Duration per trial: {duration_pickle:.2f} seconds ({duration_pickle/3600:.4f} hours)")

print("\n" + "=" * 60)
print("Loading concatenated mat file...")
print("=" * 60)

try:
    mat_data = data_handler.load_matlab_data(mat_file)
    eeg_mat = mat_data['EEG']
    if eeg_mat.shape[0] > eeg_mat.shape[1]:
        eeg_mat = eeg_mat.T
    
    if 'fs' in mat_data:
        fs_mat = int(mat_data['fs'][0][0]) if hasattr(mat_data['fs'], '__len__') else int(mat_data['fs'])
    else:
        fs_mat = 512
        print("⚠ No fs in mat file, assuming 512 Hz")
except Exception as e:
    print(f"Error loading mat file: {e}")
    exit(1)

print(f"Mat file EEG shape: {eeg_mat.shape} (Channels, Time)")
print(f"Mat file sampling rate: {fs_mat} Hz")

n_channels_mat = eeg_mat.shape[0]
n_samples_mat = eeg_mat.shape[1]
duration_mat = n_samples_mat / fs_mat
print(f"Total samples: {n_samples_mat}")
print(f"Total duration: {duration_mat:.2f} seconds ({duration_mat/3600:.4f} hours)")

print("\n" + "=" * 60)
print("Comparison...")
print("=" * 60)

if n_channels_mat != channels:
    print(f"✗ Channel count mismatch: mat={n_channels_mat}, pickle={channels}")
else:
    print(f"✓ Channel count matches: {channels} channels")

if fs_mat != sampling_rate:
    print(f"✗ Sampling rate mismatch: mat={fs_mat} Hz, pickle={sampling_rate} Hz")
else:
    print(f"✓ Sampling rate matches: {sampling_rate} Hz")

expected_total_samples = total_samples_pickle * trials
print(f"\nPickle data: {trials} trials × {total_samples_pickle} samples = {expected_total_samples} total samples")
print(f"Mat file: {n_samples_mat} samples")

if n_samples_mat >= expected_total_samples:
    print(f"✓ Mat file has enough samples (or more)")
    
    print("\n" + "=" * 60)
    print("Attempting to find pickle data in mat file...")
    print("=" * 60)
    
    pickle_concatenated = non_seizure_reshaped.reshape(channels, -1)
    print(f"Pickle data concatenated shape: {pickle_concatenated.shape}")
    
    if pickle_concatenated.shape[1] <= n_samples_mat:
        max_diff = np.max(np.abs(eeg_mat[:, :pickle_concatenated.shape[1]] - pickle_concatenated))
        mean_diff = np.mean(np.abs(eeg_mat[:, :pickle_concatenated.shape[1]] - pickle_concatenated))
        
        print(f"\nComparing first {pickle_concatenated.shape[1]} samples:")
        print(f"  Max absolute difference: {max_diff:.6f}")
        print(f"  Mean absolute difference: {mean_diff:.6f}")
        
        if max_diff < 1e-6:
            print("  ✓ Data appears to be identical!")
        elif max_diff < 1.0:
            print("  ⚠ Data is similar but not identical (possible rounding differences)")
        else:
            print("  ✗ Data differs significantly")
            
        sample_idx = min(1000, pickle_concatenated.shape[1] - 1)
        channel_idx = 0
        print(f"\nSample comparison at index {sample_idx}, channel {channel_idx}:")
        print(f"  Pickle: {pickle_concatenated[channel_idx, sample_idx]:.6f}")
        print(f"  Mat: {eeg_mat[channel_idx, sample_idx]:.6f}")
        print(f"  Difference: {abs(pickle_concatenated[channel_idx, sample_idx] - eeg_mat[channel_idx, sample_idx]):.6f}")
        
        print("\n" + "=" * 60)
        print("Checking if data appears at different positions...")
        print("=" * 60)
        
        search_window = min(10000, n_samples_mat - pickle_concatenated.shape[1])
        best_match_pos = None
        best_match_diff = float('inf')
        
        for offset in range(0, search_window, 1000):
            end_pos = offset + pickle_concatenated.shape[1]
            if end_pos > n_samples_mat:
                break
            
            diff = np.mean(np.abs(eeg_mat[:, offset:end_pos] - pickle_concatenated))
            if diff < best_match_diff:
                best_match_diff = diff
                best_match_pos = offset
        
        if best_match_pos is not None:
            print(f"Best match found at offset {best_match_pos} samples ({best_match_pos/fs_mat:.2f} seconds)")
            print(f"  Mean difference: {best_match_diff:.6f}")
            
            if best_match_diff < 1e-6:
                print("  ✓ Perfect match found!")
            elif best_match_diff < 1.0:
                print("  ⚠ Close match found")
            else:
                print("  ✗ No good match found")
    else:
        print(f"✗ Pickle data ({pickle_concatenated.shape[1]} samples) is longer than mat file ({n_samples_mat} samples)")
else:
    print(f"✗ Mat file has fewer samples than expected from pickle")

print("\n" + "=" * 60)
print("Summary:")
print("=" * 60)
print(f"Patient ID: {patient_id}")
print(f"Pickle file (patient {patient_id}) contains {trials} trials of {channels} channels")
print(f"Each trial: {total_samples_pickle} samples ({duration_pickle:.2f} seconds)")
print(f"Total pickle data: {expected_total_samples} samples ({expected_total_samples/fs_mat/3600:.4f} hours)")
print(f"\nMat file: {n_channels_mat} channels, {n_samples_mat} samples ({duration_mat:.2f} seconds, {duration_mat/3600:.4f} hours)")

