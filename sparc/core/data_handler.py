from typing import Optional
import numpy as np
import scipy.io
import pickle
from sparc.core.signal_data import SignalData, SignalDataWithGroundTruth, SimulatedData

class DataHandler:
    def _ensure_2d(self, arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 1:
            return arr.reshape(-1, 1)
        return arr

    def load_matlab_data(self, filepath: str) -> dict:
        return scipy.io.loadmat(filepath)
    
    def load_pickle_data(self, filepath: str) -> dict:
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def load_simulated_data(self, filepath: str, sampling_rate: Optional[int] = None) -> SimulatedData:
        """Load simulated data from a MATLAB .mat file from review paper"""
        try:
            data = self.load_matlab_data(filepath)

            gt = self._ensure_2d(data['SimBB'].squeeze())
            artifacts = self._ensure_2d(data['SimArtifact'].squeeze())
            mixed = self._ensure_2d(data['SimCombined'].squeeze())
            sampling_rate = sampling_rate if sampling_rate is not None else 30000
            
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

            # New fields for SimulatedData
            firing_rate = self._ensure_2d(data['SimFR'].squeeze())
            spike_train = self._ensure_2d(data['SimSpikeTrain'].squeeze())
            lfp = self._ensure_2d(data['SimLFP'].squeeze())
            stim_params = data['StimParam']
            snr = self._ensure_2d(data['AllSNR'].squeeze())

            return SimulatedData(
                raw_data=mixed,
                sampling_rate=sampling_rate,
                ground_truth=gt,
                artifacts=artifacts,
                artifact_indices=artifact_indices,
                firing_rate=firing_rate,
                spike_train=spike_train,
                lfp=lfp,
                stim_params=stim_params,
                snr=snr,
            )
        except FileNotFoundError:
            print(f"Error: The file '{filepath}' was not found.")
            raise
        except KeyError as e:
            print(f"Error: A required key is missing from the .mat file: {e}")
            raise

    def load_eraasr_data(self, filepath: str, sampling_rate: Optional[int] = None) -> SignalData:
        try:
            data = self.load_matlab_data(filepath)
            
            mixed_data = data['data_trials_by_time_by_channels']
            sampling_rate = sampling_rate if sampling_rate is not None else 30000 
            
            # Check if artifact indices are available
            artifact_indices = None
            if 'artifact_indices' in data:
                indices = data['artifact_indices'].squeeze()
                if indices.size > 0:
                    artifact_indices = np.atleast_1d(indices).astype(int)
                    if np.max(artifact_indices) > 0:  # Check if 1-indexed (MATLAB style)
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
    
