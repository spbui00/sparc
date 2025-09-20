from typing import Optional
import numpy as np
import scipy.io
from scipy import signal
import h5py
import pickle
from sparc.core.signal_data import SignalData, SignalDataWithGroundTruth, SimulatedData, ArtifactTriggers, ArtifactWindows


class DataHandler:
    def _standardize_shape(self, arr: np.ndarray, source_format: str) -> np.ndarray:
        arr = np.array(arr)
        arr = arr.squeeze() 

        if source_format == 'simulated':
            if arr.ndim == 0:
                return arr.reshape(1, 1, 1)
            if arr.ndim == 1: # (timesteps,) -> (1, timesteps, 1)
                return arr.reshape(1, -1, 1)
            elif arr.ndim == 2: # (timesteps, channels) -> (1, timesteps, channels)
                return arr[np.newaxis, :, :]
            elif arr.ndim == 3: # (timesteps, d1, d2) -> (1, timesteps, d1*d2)
                reshaped_2d = arr.reshape(arr.shape[0], -1)
                transposed = reshaped_2d.T
                return transposed[np.newaxis, :, :]
            else:
                raise ValueError(f"Unsupported shape for simulated data: {arr.shape}")

        elif source_format == 'swec' or source_format =='eraasr':
            if arr.ndim == 3:
                return arr.transpose(0, 2, 1)
            else:
                raise ValueError(f"data must be 3D. Got shape: {arr.shape}")

        raise ValueError(f"Unknown source format: {source_format}")

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

            gt = self._standardize_shape(data['SimBB'], 'simulated')
            artifacts = self._standardize_shape(data['SimArtifact'], 'simulated')
            mixed = self._standardize_shape(data['SimCombined'], 'simulated')
            firing_rate = self._standardize_shape(data['SimFR'], 'simulated')
            spike_train = self._standardize_shape(data['SimSpikeTrain'], 'simulated')
            lfp = self._standardize_shape(data['SimLFP'], 'simulated')
            snr = self._standardize_shape(data['AllSNR'], 'simulated')

            raw_indices = data['AllStimIdx'].squeeze().astype(int)
            artifact_indices = raw_indices - 1
            num_channels = mixed.shape[1]
            markers_for_one_trial = [artifact_indices for _ in range(num_channels)]
            all_trials_markers = [markers_for_one_trial]
            artifact_markers = ArtifactTriggers(starts=all_trials_markers)
            
            return SimulatedData(
                raw_data=mixed,
                sampling_rate=sampling_rate if sampling_rate is not None else 30000,
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

    
    def load_swec_ethz(
        self,
        mixed_data_path: str,
        ground_truth_path: str,
        artifact_path: str,
        sampling_rate: float,
        stim_rate: float
    ) -> SignalDataWithGroundTruth:
        mixed_data = np.load(mixed_data_path)
        ground_truth_data = np.load(ground_truth_path)
        artifact_data = np.load(artifact_path)

        num_samples = mixed_data.shape[-1]
        stim_period_samples = int(sampling_rate / stim_rate)
        
        start_indices = np.arange(0, num_samples, stim_period_samples)
        
        artifact_markers = ArtifactTriggers(starts=[start_indices]) # TODO: change
        
        return SignalDataWithGroundTruth(
            raw_data=mixed_data,
            sampling_rate=sampling_rate,
            ground_truth=ground_truth_data,
            artifacts=artifact_data,
            artifact_markers=artifact_markers
        )


    def load_eraasr_data(self, filepath: str, sampling_rate: Optional[int] = None) -> SignalData:
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
        print("Detecting artifacts using ERAASR heuristic from paper...")
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
