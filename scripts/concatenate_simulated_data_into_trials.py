import numpy as np
import os
import time
from typing import List, Dict, Any

from sparc import DataHandler


def concatenate_simulated_files(input_dir: str, num_files: int, output_path: str):
    print(f"Starting concatenation of {num_files} files from '{input_dir}'...")
    handler = DataHandler()

    all_raw_data: List[np.ndarray] = []
    all_ground_truth: List[np.ndarray] = []
    all_artifacts: List[np.ndarray] = []
    all_firing_rates: List[np.ndarray] = []
    all_spike_trains: List[np.ndarray] = []
    all_lfp: List[np.ndarray] = []
    all_artifact_markers_starts: List[List[np.ndarray]] = []
    metadata = {}
    
    for i in range(1, num_files + 1):
        base_filename = f"SimulatedData_2x64_30000_{i}"
        filepath_npz = os.path.join(input_dir, f"{base_filename}.npz")
        filepath_mat = os.path.join(input_dir, f"{base_filename}.mat")

        if os.path.exists(filepath_npz):
            filepath = filepath_npz
        elif os.path.exists(filepath_mat):
            filepath = filepath_mat
        else:
            print(f"Warning: Could not find file for index {i}. Skipping.")
            continue

        print(f"Processing file {i}/{num_files}: {os.path.basename(filepath)}")
        
        sim_data = handler.load_simulated_data(filepath, sampling_rate=30000)

        all_raw_data.append(sim_data.raw_data)
        all_ground_truth.append(sim_data.ground_truth)
        all_artifacts.append(sim_data.artifacts)
        all_firing_rates.append(sim_data.firing_rate)
        all_spike_trains.append(sim_data.spike_train)
        all_lfp.append(sim_data.lfp)
        all_artifact_markers_starts.extend(sim_data.artifact_markers.starts)

        if i == 1:
            metadata['AllSNR'] = sim_data.snr
            
    if not all_raw_data:
        print("Error: No data files were found or processed. Aborting.")
        return

    print("\nConcatenating all trials...")
    
    final_data_dict: Dict[str, Any] = {
        'SimCombined': np.concatenate(all_raw_data, axis=0),
        'SimBB': np.concatenate(all_ground_truth, axis=0),
        'SimArtifact': np.concatenate(all_artifacts, axis=0),
        'SimFR': np.concatenate(all_firing_rates, axis=0),
        'SimSpikeTrain': np.concatenate(all_spike_trains, axis=0),
        'SimLFP': np.concatenate(all_lfp, axis=0),
        'artifact_markers_starts': all_artifact_markers_starts,
        'AllSNR': metadata['AllSNR']
    }
    
    print(f"Saving concatenated data for {final_data_dict['SimCombined'].shape[0]} trials to '{output_path}'...")
    start_time = time.time()
    np.savez_compressed(output_path, **final_data_dict)
    end_time = time.time()
    
    print(f"Successfully saved in {end_time - start_time:.2f} seconds.")
    print(f"Final raw data shape: {final_data_dict['SimCombined'].shape} (trials, channels, timesteps)")

if __name__ == "__main__":
    INPUT_DATA_DIRECTORY = '../data/'
    OUTPUT_NPZ_PATH = '../data/SimulatedData_2x64_30000_10trials.npz'
    NUMBER_OF_FILES = 10

    concatenate_simulated_files(INPUT_DATA_DIRECTORY, NUMBER_OF_FILES, OUTPUT_NPZ_PATH)
