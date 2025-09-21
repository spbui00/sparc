import numpy as np
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from typing import Optional, List
from sparc import BaseSACMethod
from sparc.core.signal_data import ArtifactMarkers, ArtifactTriggers, ArtifactWindows


class ERAASR(BaseSACMethod):
    def __init__(self,
                 hp_corner_hz: float = 200.0,
                 hp_order: int = 4,
                 samples_pre_train: int = 0,
                 samples_per_pulse: int = 30,
                 n_pulses: int = 20,
                 n_pc_channels: int = 12,
                 n_pc_pulses: int = 6,
                 n_pc_trials: int = 2,
                 omit_bandwidth_channels: int = 3,
                 omit_bandwidth_pulses: int = 1,
                 omit_bandwidth_trials: int = 1,
                 pca_only_omitted: bool = True, 
                 clean_over_channels_individual_trials: bool = False,
                 clean_over_pulses_individual_channels: bool = False,
                 clean_over_trials_individual_channels: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.hp_corner_hz = hp_corner_hz
        self.hp_order = hp_order
        self.samples_pre_train = samples_pre_train
        self.samples_per_pulse = samples_per_pulse
        self.n_pulses = n_pulses
        self.n_pc_channels = n_pc_channels
        self.n_pc_pulses = n_pc_pulses
        self.n_pc_trials = n_pc_trials
        self.omit_bandwidth_channels = omit_bandwidth_channels
        self.omit_bandwidth_pulses = omit_bandwidth_pulses
        self.omit_bandwidth_trials = omit_bandwidth_trials
        self.pca_only_omitted = pca_only_omitted
        self.clean_over_channels_individual_trials = clean_over_channels_individual_trials
        self.clean_over_pulses_individual_channels = clean_over_pulses_individual_channels
        self.clean_over_trials_individual_channels = clean_over_trials_individual_channels
        self.is_fitted = False

    def fit(self, data: np.ndarray, artifact_markers: Optional[ArtifactMarkers] = None) -> 'ERAASR':
        self.is_fitted = True
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("The transform method cannot be called before fit.")
        if self.sampling_rate is None:
            raise ValueError("Sampling rate must be set.")
        if data.ndim != 3:
            raise ValueError("Input data must be 3D (trials, channels, timesteps).")

        data_to_process = data.transpose(0, 2, 1)
        num_trials, num_samples, num_channels = data_to_process.shape

        self._highpass_filter(data_to_process) 

        pulse_len = self.samples_per_pulse
        train_len = pulse_len * self.n_pulses
        artifact_window = np.arange(self.samples_pre_train, self.samples_pre_train + train_len)

        if self.samples_pre_train + train_len > num_samples:
            raise ValueError("Artifact window (samples_pre_train + train_len) exceeds data length.")

        data_segment = data_to_process[:, artifact_window, :]
        segment_tensor = data_segment.reshape((num_trials, pulse_len, self.n_pulses, num_channels))
        
        # A. Clean across Channels
        if self.n_pc_channels > 0:
            # Reshape to (trials*time*pulses, channels)
            matrix_for_pca = segment_tensor.transpose(0, 1, 2, 3).reshape(-1, num_channels)
            cleaned_matrix, _, _ = self._clean_matrix_via_pca_regression(
                matrix_for_pca, self.n_pc_channels, self.omit_bandwidth_channels, self.pca_only_omitted
            )
            segment_tensor = cleaned_matrix.reshape(num_trials, pulse_len, self.n_pulses, num_channels)

        # B. Clean across Pulses
        if self.n_pc_pulses > 0:
            # Reshape to (trials*time*channels, pulses)
            matrix_for_pca = segment_tensor.transpose(0, 1, 3, 2).reshape(-1, self.n_pulses)
            cleaned_matrix, _, _ = self._clean_matrix_via_pca_regression(
                matrix_for_pca, self.n_pc_pulses, self.omit_bandwidth_pulses, self.pca_only_omitted
            )
            segment_tensor = cleaned_matrix.reshape(num_trials, pulse_len, num_channels, self.n_pulses).transpose(0, 1, 3, 2)
        
        # C. Clean across Trials
        if self.n_pc_trials > 0 and num_trials > 1:
            # Reshape to (time*pulses*channels, trials)
            matrix_for_pca = segment_tensor.transpose(1, 2, 3, 0).reshape(-1, num_trials)
            cleaned_matrix, _, _ = self._clean_matrix_via_pca_regression(
                matrix_for_pca, self.n_pc_trials, self.omit_bandwidth_trials, self.pca_only_omitted
            )
            segment_tensor = cleaned_matrix.reshape(pulse_len, self.n_pulses, num_channels, num_trials).transpose(3, 0, 1, 2)
            
        cleaned_segment = segment_tensor.reshape((num_trials, train_len, num_channels))
        
        final_cleaned_transposed = data_to_process.copy()
        final_cleaned_transposed[:, artifact_window, :] = cleaned_segment

        return final_cleaned_transposed.transpose(0, 2, 1)

    def _clean_matrix_via_pca_regression(self, matrix: np.ndarray, n_components: int, 
                                          omit_bandwidth: int, pca_only_omitted: bool) -> tuple:
        # matrix shape: (samples, features)
        cleaned_matrix = matrix.copy()
        num_features = matrix.shape[1]
        all_feature_indices = np.arange(num_features)
        
        # Center the data column-wise
        mean_vec = np.nanmean(matrix, axis=0)
        matrix_centered = matrix - mean_vec

        for i in range(num_features):
            omit_indices = np.arange(max(0, i - omit_bandwidth), min(num_features, i + omit_bandwidth + 1))
            
            if pca_only_omitted:
                # Build PCA model only from channels outside the omit-bandwidth
                model_indices = np.setdiff1d(all_feature_indices, omit_indices)
                if len(model_indices) < n_components: continue
                model_data = matrix_centered[:, model_indices]
                pca = PCA(n_components=n_components)
                artifact_pcs = pca.fit_transform(model_data)
            else:
                # Build PCA model from all channels, then nullify components from omitted channels
                if num_features < n_components: continue
                pca = PCA(n_components=n_components)
                pca.fit_transform(matrix_centered)
                coeffs = pca.components_.T
                coeffs[omit_indices, :] = 0 # Nullify contributions
                artifact_pcs = matrix_centered @ coeffs

            reg = LinearRegression()
            reg.fit(artifact_pcs, matrix_centered[:, i])
            artifact_for_feature_i = reg.predict(artifact_pcs)
            cleaned_matrix[:, i] = matrix_centered[:, i] - artifact_for_feature_i
        
        # Add the mean back to the cleaned matrix
        cleaned_matrix += mean_vec
        return cleaned_matrix, None, None # Return tuple for compatibility

    def _highpass_filter(self, data: np.ndarray) -> np.ndarray:
        if self.hp_corner_hz <= 0 or self.sampling_rate is None:
            return data
        sos = signal.butter(self.hp_order, self.hp_corner_hz, btype='high', fs=self.sampling_rate, output='sos')
        return signal.sosfiltfilt(sos, data, axis=-1)
