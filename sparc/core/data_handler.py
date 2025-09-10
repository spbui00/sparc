import numpy as np
import scipy.io
from scipy import signal
from typing import Optional
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
    
    def detect_artifacts(self, data: np.ndarray, threshold: float = 5.0, 
                        min_duration: int = 10) -> np.ndarray:
        """
        Detect artifact locations based on amplitude thresholding.
        
        Args:
            data: Neural data array (timesteps, channels)
            threshold: Amplitude threshold for artifact detection
            min_duration: Minimum artifact duration in samples
            
        Returns:
            Boolean array indicating artifact locations
        """
        # Simple threshold-based detection
        artifact_mask = np.abs(data) > threshold
        
        # Remove short artifacts
        for ch in range(data.shape[1]):
            ch_mask = artifact_mask[:, ch]
            # Find continuous regions
            diff = np.diff(np.concatenate(([False], ch_mask, [False])).astype(int))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            
            for start, end in zip(starts, ends):
                if end - start < min_duration:
                    ch_mask[start:end] = False
                    
        return artifact_mask
    
    def preprocess_data(self, data: np.ndarray, 
                       filter_low: Optional[float] = None,
                       filter_high: Optional[float] = None,
                       notch_freq: Optional[float] = None) -> np.ndarray:
        """
        Preprocess neural data with filtering.
        
        Args:
            data: Raw neural data
            filter_low: Low-pass filter cutoff (Hz)
            filter_high: High-pass filter cutoff (Hz)
            notch_freq: Notch filter frequency (Hz)
            
        Returns:
            Preprocessed data
        """
        processed_data = data.copy()
        
        # Apply filters if specified
        if filter_high is not None:
            sos = signal.butter(4, filter_high, btype='high', fs=self.sampling_rate, output='sos')
            processed_data = signal.sosfilt(sos, processed_data, axis=0)
            
        if filter_low is not None:
            sos = signal.butter(4, filter_low, btype='low', fs=self.sampling_rate, output='sos')
            processed_data = signal.sosfilt(sos, processed_data, axis=0)
            
        if notch_freq is not None:
            # 50/60 Hz notch filter
            sos = signal.iirnotch(notch_freq, 30, fs=self.sampling_rate, output='sos')
            processed_data = signal.sosfilt(sos, processed_data, axis=0)
            
        return processed_data