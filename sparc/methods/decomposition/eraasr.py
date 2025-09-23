import numpy as np
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from typing import Optional
from tqdm import tqdm
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
        try:
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
                print(f"Warning: Artifact window exceeds data length. Skipping ERAASR processing.")
                return data  # Return original data

            data_segment = data_to_process[:, artifact_window, :]
            segment_tensor = data_segment.reshape((num_trials, pulse_len, self.n_pulses, num_channels))
        except Exception as e:
            print(f"Warning: ERAASR preprocessing failed: {e}. Returning original data.")
            return data
        
        try:
            # A. Clean across Channels
            if self.n_pc_channels > 0:
                # Reshape to (trials*time*pulses, channels)
                matrix_for_pca = segment_tensor.transpose(0, 1, 2, 3).reshape(-1, num_channels)
                cleaned_matrix, _, _ = self._clean_matrix_via_pca_regression(
                    matrix_for_pca, self.n_pc_channels, self.omit_bandwidth_channels, self.pca_only_omitted,
                    desc="Cleaning across channels"
                )
                segment_tensor = cleaned_matrix.reshape(num_trials, pulse_len, self.n_pulses, num_channels)

            # B. Clean across Pulses
            if self.n_pc_pulses > 0:
                # Reshape to (trials*time*channels, pulses)
                matrix_for_pca = segment_tensor.transpose(0, 1, 3, 2).reshape(-1, self.n_pulses)
                cleaned_matrix, _, _ = self._clean_matrix_via_pca_regression(
                    matrix_for_pca, self.n_pc_pulses, self.omit_bandwidth_pulses, self.pca_only_omitted,
                    desc="Cleaning across pulses"
                )
                segment_tensor = cleaned_matrix.reshape(num_trials, pulse_len, num_channels, self.n_pulses).transpose(0, 1, 3, 2)
            
            # C. Clean across Trials
            if self.n_pc_trials > 0 and num_trials > 1:
                # Reshape to (time*pulses*channels, trials)
                matrix_for_pca = segment_tensor.transpose(1, 2, 3, 0).reshape(-1, num_trials)
                cleaned_matrix, _, _ = self._clean_matrix_via_pca_regression(
                    matrix_for_pca, self.n_pc_trials, self.omit_bandwidth_trials, self.pca_only_omitted,
                    desc="Cleaning across trials"
                )
                segment_tensor = cleaned_matrix.reshape(pulse_len, self.n_pulses, num_channels, num_trials).transpose(3, 0, 1, 2)
                
            cleaned_segment = segment_tensor.reshape((num_trials, train_len, num_channels))
            
            final_cleaned_transposed = data_to_process.copy()
            final_cleaned_transposed[:, artifact_window, :] = cleaned_segment

            return final_cleaned_transposed.transpose(0, 2, 1)
            
        except Exception as e:
            print(f"Warning: ERAASR cleaning failed: {e}. Returning original data.")
            return data

    def _clean_matrix_via_pca_regression(self, matrix: np.ndarray, n_components: int, 
                                      omit_bandwidth: int, pca_only_omitted: bool, desc: str = "Processing") -> tuple:
        try:
            cleaned_matrix = matrix.copy()
            num_features = matrix.shape[1]
            
            # Check basic conditions
            if num_features < 2:
                print(f"Warning: Not enough features ({num_features}) for PCA. Skipping cleaning step.")
                return cleaned_matrix, None, None
                
            if matrix.shape[0] < n_components:
                print(f"Warning: Not enough samples ({matrix.shape[0]}) for {n_components} components. Skipping cleaning step.")
                return cleaned_matrix, None, None
            
            mean_vec = np.nanmean(matrix, axis=0)
            matrix_centered = matrix - mean_vec

            # Pre-compute PCA for non-omitted mode (efficiency)
            if not pca_only_omitted and num_features >= n_components:
                try:
                    pca_all = PCA(n_components=n_components)
                    pca_all.fit(matrix_centered)
                    base_components = pca_all.components_.T  # Shape: (num_features, n_components)
                except Exception as e:
                    print(f"Warning: PCA fitting failed: {e}. Skipping cleaning step.")
                    return cleaned_matrix, None, None

            for i in tqdm(range(num_features), desc=desc, leave=False):
                try:
                    min_omit = max(0, i - (omit_bandwidth - 1) // 2)
                    max_omit = min(num_features - 1, i + omit_bandwidth // 2)
                    omit_indices = np.arange(min_omit, max_omit + 1)
                    
                    if pca_only_omitted:
                        model_indices = np.setdiff1d(np.arange(num_features), omit_indices)
                        if len(model_indices) < n_components: 
                            continue
                        model_data = matrix_centered[:, model_indices]
                        pca = PCA(n_components=n_components)
                        artifact_pcs = pca.fit_transform(model_data)
                    else:
                        if num_features < n_components: 
                            continue
                        components_copy = base_components.copy()
                        components_copy[omit_indices, :] = 0
                        artifact_pcs = matrix_centered @ components_copy

                    # Check if we have valid PCs
                    if artifact_pcs.shape[1] == 0:
                        continue
                        
                    reg = LinearRegression()
                    reg.fit(artifact_pcs, matrix_centered[:, i])
                    artifact_for_feature_i = reg.predict(artifact_pcs)
                    cleaned_matrix[:, i] = matrix_centered[:, i] - artifact_for_feature_i
                    
                except Exception as e:
                    print(f"Warning: Failed to clean feature {i}: {e}. Skipping this feature.")
                    continue
            
            cleaned_matrix += mean_vec
            return cleaned_matrix, None, None
            
        except Exception as e:
            print(f"Warning: PCA regression failed: {e}. Returning original matrix.")
            return matrix, None, None

    def _highpass_filter(self, data: np.ndarray) -> np.ndarray:
        if self.hp_corner_hz <= 0 or self.sampling_rate is None:
            return data
        sos = signal.butter(self.hp_order, self.hp_corner_hz, btype='high', fs=self.sampling_rate, output='sos')
        return signal.sosfiltfilt(sos, data, axis=-1)
