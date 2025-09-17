import numpy as np
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from typing import Optional, List
from sparc import BaseSACMethod
from sparc.core.signal_data import ArtifactMarkers, ArtifactTriggers, ArtifactWindows


class ERAASR(BaseSACMethod):
    def __init__(self,
                 sampling_rate: Optional[float] = None,
                 hp_corner_hz: float = 200,
                 n_pc_channels: int = 12,
                 n_pc_pulses: int = 6,
                 n_pc_trials: int = 2,
                 omit_bandwidth_channels: int = 3,
                 samples_per_pulse: int = 30,
                 n_pulses: int = 20,
                 **kwargs):
        """
        Args:
            hp_corner_hz (float): Corner frequency for the initial high-pass filter.
            n_pc_channels (int): Number of principal components to use when cleaning across channels.
            n_pc_pulses (int): Number of principal components to use when cleaning across pulses.
            n_pc_trials (int): Number of principal components to use when cleaning across trials.
            omit_bandwidth_channels (int): Number of adjacent channels to exclude when building the PCA model for a given channel.
            samples_per_pulse (int): The number of time samples in a single stimulation pulse.
            n_pulses (int): The number of pulses in a single stimulation train.
        """
        super().__init__(sampling_rate,**kwargs)
        self.hp_corner_hz = hp_corner_hz
        self.n_pc_channels = n_pc_channels
        self.n_pc_pulses = n_pc_pulses
        self.n_pc_trials = n_pc_trials
        self.omit_bandwidth_channels = omit_bandwidth_channels
        self.samples_per_pulse = samples_per_pulse
        self.n_pulses = n_pulses
        self.artifact_markers_ = None

    def fit(self, data: np.ndarray, artifact_markers: Optional[ArtifactMarkers] = None) -> 'ERAASR':
        if self.sampling_rate is None:
            raise ValueError("ERAASR requires a sampling rate.")
        if artifact_markers is None:
            raise ValueError("ERAASR requires 'artifact_markers'.")
            
        if isinstance(artifact_markers, ArtifactTriggers):
            self.artifact_markers_ = np.sort(artifact_markers.starts.flatten())
        elif isinstance(artifact_markers, ArtifactWindows):
            self.artifact_markers_ = np.sort(artifact_markers.intervals[:, 0].flatten())
        else:
            raise TypeError("artifact_markers must be of type ArtifactTriggers or ArtifactWindows")

        self.is_fitted = True
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        if not self.is_fitted or self.artifact_markers_ is None:
            raise RuntimeError("The transform method cannot be called before fit.")
        num_trials, num_samples, num_channels = data.shape
        cleaned_data = data.copy()

        pulse_len = self.samples_per_pulse
        train_len = pulse_len * self.n_pulses
        
        for start_idx in self.artifact_markers_:
            if start_idx + train_len > num_samples:
                print(f"  Warning: Skipping artifact train at {start_idx} as it exceeds data length.")
                continue
            
            artifact_window = np.arange(start_idx, start_idx + train_len)
            data_to_clean = cleaned_data[:, artifact_window, :]

            # High-pass filter the data to be cleaned
            if self.hp_corner_hz > 0:
                data_to_clean = self._high_pass_filter(data_to_clean, self.sampling_rate, self.hp_corner_hz, order=4)
            
            # Reshape into a 4D tensor: (trials, time_in_pulse, pulses, channels)
            segment_tensor = data_to_clean.reshape((num_trials, pulse_len, self.n_pulses, num_channels))
            
            
            # A. Clean across Channels
            if self.n_pc_channels > 0:
                matrix_for_pca = segment_tensor.transpose(1, 2, 0, 3).reshape(-1, num_channels)
                cleaned_matrix, _, _ = self._clean_matrix_via_pca_regression(
                    matrix_for_pca, self.n_pc_channels, self.omit_bandwidth_channels
                )
                segment_tensor = cleaned_matrix.reshape(pulse_len, self.n_pulses, num_trials, num_channels).transpose(2, 0, 1, 3)

            # B. Clean across Pulses
            if self.n_pc_pulses > 0:
                matrix_for_pca = segment_tensor.transpose(1, 0, 3, 2).reshape(-1, self.n_pulses)
                cleaned_matrix, _, _ = self._clean_matrix_via_pca_regression(
                    matrix_for_pca, self.n_pc_pulses
                )
                segment_tensor = cleaned_matrix.reshape(pulse_len, num_trials, num_channels, self.n_pulses).transpose(1, 0, 3, 2)
            
            # C. Clean across Trials
            if self.n_pc_trials > 0:
                matrix_for_pca = segment_tensor.transpose(1, 2, 3, 0).reshape(-1, num_trials)
                cleaned_matrix, _, _ = self._clean_matrix_via_pca_regression(
                    matrix_for_pca, self.n_pc_trials
                )
                segment_tensor = cleaned_matrix.reshape(pulse_len, self.n_pulses, num_channels, num_trials).transpose(3, 0, 1, 2)
                
            # --- Postprocessing ---
            cleaned_segment = segment_tensor.reshape((num_trials, train_len, num_channels))
            
            # Insert the cleaned segment back into the main data array
            cleaned_data[:, artifact_window, :] = cleaned_segment

        return cleaned_data

    def _high_pass_filter(self, data: np.ndarray, fs: float, corner_hz: float, order: int) -> np.ndarray:
        sos = signal.butter(order, corner_hz, btype='high', fs=fs, output='sos')
        return signal.sosfiltfilt(sos, data, axis=1)

    def _reshape_tensor(self, tensor: np.ndarray, dim_groups: List[List[int]]) -> np.ndarray:
        # Python uses 0-based indexing
        flat_dims = [item for sublist in dim_groups for item in sublist]
        permute_order = flat_dims + [i for i in range(tensor.ndim) if i not in flat_dims]
        permuted_tensor = np.transpose(tensor, axes=permute_order)
        
        new_shape = []
        for group in dim_groups:
            size = np.prod([permuted_tensor.shape[i] for i in range(len(group))])
            new_shape.append(size)
            
        final_shape = tuple(new_shape) + permuted_tensor.shape[len(flat_dims):]
        return np.reshape(permuted_tensor, final_shape)

    def _clean_matrix_via_pca_regression(self, matrix: np.ndarray, n_components: int, 
                                        omit_bandwidth: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cleaned_matrix = matrix.copy()
        reconstructed_artifact = np.zeros_like(matrix)
        
        num_features = matrix.shape[1]
        all_feature_indices = np.arange(num_features)

        artifact_pcs = None
        for i in range(num_features):
            omit_indices = np.arange(max(0, i - omit_bandwidth), min(num_features, i + omit_bandwidth + 1))
            model_indices = np.setdiff1d(all_feature_indices, omit_indices)
            
            if not model_indices.any():
                continue
                
            model_data = matrix[:, model_indices]
            
            # Fit PCA on the other features
            pca = PCA(n_components=n_components)
            artifact_pcs = pca.fit_transform(model_data)
            
            # Fit linear regression to predict the current feature from the artifact PCs
            reg = LinearRegression()
            reg.fit(artifact_pcs, matrix[:, i])
            
            # The prediction is the reconstructed artifact
            artifact_for_feature_i = reg.predict(artifact_pcs)
            
            cleaned_matrix[:, i] -= artifact_for_feature_i
            reconstructed_artifact[:, i] = artifact_for_feature_i
            
        # For simplicity, we return the PCs from the last iteration.
        # A full implementation might store all sets of PCs.
        return cleaned_matrix, artifact_pcs, reconstructed_artifact
