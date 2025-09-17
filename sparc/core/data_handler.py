from typing import Optional
import numpy as np
import scipy.io
from scipy import signal
import h5py
import pickle
from sparc.core.signal_data import SignalData, SimulatedData, ArtifactTriggers, ArtifactWindows
from sparc.utils.generate_artifacts import generate_synthetic_artifacts, resample_signal, generate_sine_exp_decay_artifact


class DataHandler:
    def _to_2d_array(self, value) -> np.ndarray:
        arr = np.array(value)
        
        if arr.ndim == 0:
            return arr.reshape(1, 1)
        if arr.ndim == 1:
            return arr.reshape(-1, 1)
        return arr

    def load_npz_data(self, filepath: str) -> dict:
        return np.load(filepath)

    def load_matlab_data(self, filepath: str) -> dict:
        try:
            return scipy.io.loadmat(filepath, squeeze_me=True)
        except NotImplementedError:
            print(f"File '{filepath}' is a v7.3 MATLAB file. Reading with h5py.")
            data = {}
            with h5py.File(filepath, 'r') as f:
                for key, value in f.items():
                    if isinstance(value, h5py.Dataset):
                        loaded_data = value[()]
                        
                        if loaded_data.ndim > 1:
                            data[key] = loaded_data.T
                        else:
                            data[key] = loaded_data
            return data

    def load_pickle_data(self, filepath: str) -> dict:
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def load_simulated_data(self, filepath: str, sampling_rate: Optional[int] = None) -> SimulatedData:
        """Load simulated data from a MATLAB .mat file from review paper"""
        try:
            if filepath.endswith('.npz'):
                data = self.load_npz_data(filepath)
            else:
                data = self.load_matlab_data(filepath)
            print("Loaded keys from .mat file:", list(data.keys()))

            gt = self._to_2d_array(data['SimBB'])
            artifacts = self._to_2d_array(data['SimArtifact'])
            mixed = self._to_2d_array(data['SimCombined'])
            sampling_rate = sampling_rate if sampling_rate is not None else 30000
            
            # AllStimIdx is 1-indexed in MATLAB, convert to 0-indexed for Python
            raw_indices = data['AllStimIdx'].squeeze().astype(int)
            artifact_indices = raw_indices - 1
            artifact_markers = ArtifactTriggers(starts=artifact_indices)
            
            print(f"Loaded {len(artifact_markers.starts)} artifact indices from MATLAB file")

            # New fields for SimulatedData
            firing_rate = self._to_2d_array(data['SimFR'])
            spike_train = self._to_2d_array(data['SimSpikeTrain'])
            lfp = self._to_2d_array(data['SimLFP'])
            snr = self._to_2d_array(data['AllSNR'])
            # stim_params = data['StimParam']

            return SimulatedData(
                raw_data=mixed,
                sampling_rate=sampling_rate,
                ground_truth=gt,
                artifacts=artifacts,
                artifact_markers=artifact_markers,
                firing_rate=firing_rate,
                spike_train=spike_train,
                lfp=lfp,
                stim_params=None,
                snr=snr,
            )
        except FileNotFoundError:
            print(f"Error: The file '{filepath}' was not found.")
            raise
        except KeyError as e:
            print(f"Error: A required key is missing from the .mat file: {e}")
            raise

    def load_eraasr_data(self, filepath: str, sampling_rate: Optional[int] = None) -> SignalData:
        """Load ERAASR example data and add generated artifacts"""
        try:
            data = self.load_matlab_data(filepath)
            
            mixed_data = data['data_trials_by_time_by_channels']
            print(f"Loaded data shape from ERAASR .mat file: {mixed_data.shape}")
            sampling_rate = sampling_rate if sampling_rate is not None else 30000 
            artifact_markers = self._detect_artifacts_eraasr(mixed_data, sampling_rate)
            
            return SignalData(
                raw_data=mixed_data,
                sampling_rate=sampling_rate,
                artifact_markers=artifact_markers
            )
        except FileNotFoundError:
            print(f"Error: The file '{filepath}' was not found.")
            raise
        except KeyError as e:
            print(f"Error: A required key is missing from the .mat file: {e}")
            raise
    
    def _detect_artifacts_eraasr(self, data: np.ndarray, sampling_rate) -> ArtifactWindows:
        # TODO: Make this more robust and flexible
        threshold_channel_idx = 15
        hp_corner_hz = 250
        hp_order = 4
        threshold_uv = -1000
        artifact_duration_ms = 57

        artifact_duration_samples = round(artifact_duration_ms / 1000 * sampling_rate)

        b, a = signal.butter(hp_order, hp_corner_hz, btype='high', fs=sampling_rate)

        artifact_windows = []
        n_trials = data.shape[0]

        for i in range(n_trials):
            trial_signal = data[i, :, threshold_channel_idx]
            filtered_signal = signal.filtfilt(b, a, trial_signal)
            
            crossings = np.where(filtered_signal < threshold_uv)[0]

            if crossings.size > 0:
                start_index = crossings[0]
                end_index = start_index + artifact_duration_samples
                artifact_windows.append((start_index, end_index))
            else:
                artifact_windows.append((None, None))

        return ArtifactWindows(intervals=np.array(artifact_windows))
