import numpy as np
import scipy.io
import h5py
import os
import re
from glob import glob

def load_mat_file(filepath):
    try:
        mat_data = scipy.io.loadmat(filepath)
        eeg = mat_data['EEG']
        if 'fs' in mat_data:
            fs = int(mat_data['fs'][0][0]) if hasattr(mat_data['fs'], '__len__') else int(mat_data['fs'])
        else:
            fs = None
    except NotImplementedError:
        with h5py.File(filepath, 'r') as f:
            eeg = np.array(f['EEG'])
            if 'fs' in f:
                fs = int(np.array(f['fs'])[0])
            else:
                fs = None
    
    if eeg.shape[0] > eeg.shape[1]:
        eeg = eeg.T
    
    return eeg, fs

def extract_hour_number(filename):
    match = re.search(r'(\d+)h\.mat', filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None

data_dir = '../data'

hour_files = sorted(glob(os.path.join(data_dir, '*h.mat')), key=extract_hour_number)

if hour_files:
    start_hour = extract_hour_number(os.path.basename(hour_files[0]))
    end_hour = extract_hour_number(os.path.basename(hour_files[-1]))
    output_file = f'../data/concatenated_hours_{start_hour}_{end_hour}.mat'
else:
    output_file = '../data/concatenated_hours.mat'

if not hour_files:
    print(f"No hour files found in {data_dir}")
    exit(1)

print(f"Found {len(hour_files)} hour files:")
for f in hour_files:
    print(f"  {os.path.basename(f)}")

print("\n" + "=" * 60)
print("Loading files...")
print("=" * 60)

eeg_list = []
fs_list = []
shapes = []

for filepath in hour_files:
    eeg, fs = load_mat_file(filepath)
    eeg_list.append(eeg)
    fs_list.append(fs)
    shapes.append(eeg.shape)
    print(f"{os.path.basename(filepath)}: shape {eeg.shape}, fs={fs} Hz")

fs_set = set(fs_list) - {None}
if len(fs_set) > 1:
    print("\n⚠ Warning: Files have different sampling rates!")
    print(f"Sampling rates: {fs_set}")
    exit(1)

if None in fs_list:
    print("\n⚠ Warning: Some files missing sampling rate, using default 512 Hz")
    fs = 512
else:
    fs = fs_list[0]

if len(set(s[0] for s in shapes)) > 1:
    print("\n⚠ Warning: Files have different channel counts!")
    print(f"Channel counts: {set(s[0] for s in shapes)}")
    exit(1)

n_channels = shapes[0][0]
print(f"\nAll files: {n_channels} channels, {fs} Hz")

print("\n" + "=" * 60)
print("Concatenating...")
print("=" * 60)

concatenated = np.concatenate(eeg_list, axis=1)

print(f"Concatenated shape: {concatenated.shape}")
print(f"Total duration: {concatenated.shape[1] / fs:.2f} seconds ({concatenated.shape[1] / fs / 3600:.4f} hours)")

print("\n" + "=" * 60)
print(f"Saving to {output_file}...")
print("=" * 60)

scipy.io.savemat(output_file, {
    'EEG': concatenated,
    'fs': np.array([[fs]])
}, do_compression=True)

print(f"✓ Saved concatenated data to {output_file}")
print(f"  Shape: {concatenated.shape}")
print(f"  Sampling rate: {fs} Hz")

