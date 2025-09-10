import numpy as np
import scipy.io
import pickle

class NeuralDataHandler:
    """
    Handles loading and preprocessing of neural data for SAC.
    """
    
    def __init__(self, sampling_rate: float):
        self.sampling_rate = sampling_rate
        
    def load_matlab_data(self, filepath: str, data_key: str = 'data_trials_by_time_by_channels') -> np.ndarray:
        """
        Load neural data from MATLAB .mat file.
        
        Args:
            filepath: Path to .mat file
            data_key: Key for the data in the .mat file
            
        Returns:
            Data array of shape (trials, timesteps, channels)
        """
        data = scipy.io.loadmat(filepath)
        return data[data_key]
    
    def load_pickle_data(self, filepath: str) -> dict:
        """Load data from pickle file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
