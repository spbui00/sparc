import numpy as np
import scipy.io
import pickle
from sparc.core.signal_data import SignalData, SignalDataWithGroundTruth

class DataHandler:
    """
    Handles loading and preprocessing of neural data for SPARC.
    """
    
    def _ensure_2d(self, arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 1:
            return arr.reshape(-1, 1)
        return arr

    def load_matlab_data(self, filepath: str) -> dict:
        return scipy.io.loadmat(filepath)
    
    def load_pickle_data(self, filepath: str) -> dict:
        """Load data from pickle file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def load_simulated_data(self, filepath: str) -> SignalDataWithGroundTruth:
        try:
            data = self.load_matlab_data(filepath)

            gt = self._ensure_2d(data['SimBB'].squeeze())
            artifacts = self._ensure_2d(data['SimArtifact'].squeeze())
            mixed = self._ensure_2d(data['SimCombined'].squeeze())
            sampling_rate = 30000  # read from matlab script

            return SignalDataWithGroundTruth(
                raw_data=mixed,
                sampling_rate=sampling_rate,
                ground_truth_spikes=gt,
                artifacts=artifacts
            )
        
        except FileNotFoundError:
            print(f"Error: The file '{filepath}' was not found.")
            raise
        except KeyError as e:
            print(f"Error: A required key is missing from the .mat file: {e}")
            raise

    def load_eraasr_data(self, filepath: str) -> SignalData:
        try:
            data = self.load_matlab_data(filepath)
            
            mixed_data = data['data_trials_by_time_by_channels']
            sampling_rate = 30000  # 30 kHz
            
            return SignalData(
                raw_data=mixed_data,
                sampling_rate=sampling_rate
            )
        except FileNotFoundError:
            print(f"Error: The file '{filepath}' was not found.")
            raise
        except KeyError as e:
            print(f"Error: A required key is missing from the .mat file: {e}")
            raise
    
