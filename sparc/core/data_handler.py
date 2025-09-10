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
            
            artifact_indices = None
            if 'AllStimIdx' in data:
                # AllStimIdx is 1-indexed in MATLAB, convert to 0-indexed for Python
                raw_indices = data['AllStimIdx'].squeeze().astype(int)
                artifact_indices = raw_indices - 1
                
                # Validate indices are within bounds
                max_samples = mixed.shape[0]
                valid_mask = (artifact_indices >= 0) & (artifact_indices < max_samples)
                if not np.all(valid_mask):
                    print(f"Warning: {np.sum(~valid_mask)} artifact indices are out of bounds, filtering them out")
                    artifact_indices = artifact_indices[valid_mask]
                
                print(f"Loaded {len(artifact_indices)} artifact indices from MATLAB file")

            return SignalDataWithGroundTruth(
                raw_data=mixed,
                sampling_rate=sampling_rate,
                ground_truth=gt,
                artifacts=artifacts,
                artifact_indices=artifact_indices,
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
            
            # Check if artifact indices are available
            artifact_indices = None
            if 'artifact_indices' in data:
                artifact_indices = data['artifact_indices'].squeeze().astype(int)
                if artifact_indices.max() > 0:  # Check if 1-indexed (MATLAB style)
                    artifact_indices = artifact_indices - 1  # Convert to 0-indexed
                print(f"Loaded {len(artifact_indices)} artifact indices from ERAASR file")
            
            return SignalData(
                raw_data=mixed_data,
                sampling_rate=sampling_rate,
                artifact_indices=artifact_indices
            )
        except FileNotFoundError:
            print(f"Error: The file '{filepath}' was not found.")
            raise
        except KeyError as e:
            print(f"Error: A required key is missing from the .mat file: {e}")
            raise
    
