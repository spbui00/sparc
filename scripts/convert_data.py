from sparc.core.data_handler import DataHandler
import numpy as np
import time

MAT_FILE_PATH = '../../research/generate_dataset/SimulatedData_1_1024.mat'
NPZ_FILE_PATH = '../../research/generate_dataset/SimulatedData_1_1024.npz'

if __name__ == "__main__":
    start_time = time.time()
    handler = DataHandler()
    mat_data = handler.load_matlab_data(MAT_FILE_PATH)
    end_time = time.time()
    print(f"Loaded MATLAB .mat file in {end_time - start_time:.2f} seconds.")

    start_time = time.time()
    print(f"Saving data to the fast .npz format at '{NPZ_FILE_PATH}'...")
    end_time = time.time()
    print(f"Data saved in {end_time - start_time:.2f} seconds.")

    np.savez_compressed(NPZ_FILE_PATH, **mat_data)
    print("saved at " + NPZ_FILE_PATH)

