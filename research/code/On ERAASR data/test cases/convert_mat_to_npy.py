import scipy.io
import numpy as np
import os

### Convert mat to npy #############################################################################
# Load the .mat file
read_folder_name = '/net/inltitan1/scratch2/Xiaoyong/Artifact_cancellation/ethz_data/interp/'
save_folder_name = '/net/inltitan1/scratch2/Xiaoyong/Artifact_cancellation/ethz_data/interp/'
# data_clean = scipy.io.loadmat('clean2_15trials_allch.mat')
data_artifact = scipy.io.loadmat(read_folder_name + 'interpolation_nonseizure_amp73.mat')

# Extract the variable
# my_clean = data_clean['Ain_clean2_reshaped']
my_artifact = data_artifact['Dtemp_1'] #Dout_clean, Dtemp_1, reconstructed_signal

# Ensure it's float64 (double precision)
# my_clean = my_clean.astype(np.float64)
my_artifact = my_artifact.astype(np.float64)
my_artifact = my_artifact.transpose((0, 2, 1))  # Transpose to (trials, channels, time)
# Save as .npy file
# np.save('clean2_15trials_allch.npy', my_clean)
np.save(save_folder_name + 'interpolation_nonseizure_amp73.npy', my_artifact)


# folder_name = 'svd_window_size/N10/'
# file_names = [f'artifact_alltrials_allch_svdN10_window{i}.mat' for i in range(10, 100, 10)]
# num_files = len(file_names)  # Correct way to get the number of files


# for i in range(num_files):
#     # Construct full file path
#     file_path = os.path.join(folder_name, file_names[i])

#     # Load the .mat file
#     data_artifact = scipy.io.loadmat(file_path)

#     my_artifact = data_artifact['Ain_artifact_reshaped']

#     # Ensure it's float64 (double precision)
#     my_artifact = my_artifact.astype(np.float64)

#     # Save as .npy file (replacing .mat with .npy in filename)
#     npy_filename = file_names[i].replace('.mat', '.npy')
#     np.save(os.path.join(folder_name, npy_filename), my_artifact)

### Convert npy to mat #############################################################################
# Define paths
# npy_file_path = '/net/inltitan1/scratch2/yuhxie/ethz_data/amp73/mixed_nonseizure1.npy'
# mat_file_path = '/home/ni/Documents/artifact-cancellation/datasets/SWEC-ETHZ-iEEG/high_amp_73/mixed_nonseizure1_rate2kHz.mat'

# # Load the .npy file
# my_artifact = np.load(npy_file_path)

# # Save as .mat file
# scipy.io.savemat(mat_file_path, {'mixed_nonseizure': my_artifact})

# # Define paths
# npy_file_path = '/net/inltitan1/scratch2/yuhxie/ethz_data/amp73/clean_nonseizure1.npy'
# mat_file_path = '/home/ni/Documents/artifact-cancellation/datasets/SWEC-ETHZ-iEEG/high_amp_73/clean_nonseizure1_rate2kHz.mat'

# # Load the .npy file
# my_clean = np.load(npy_file_path)

# # Save as .mat file
# scipy.io.savemat(mat_file_path, {'signal_nonseizure': my_clean})