import numpy as np
from typing import Literal, Optional, Any, List, Tuple
from sparc import BaseSACMethod
from sklearn.decomposition import PCA as SklearnPCA
import matplotlib.pyplot as plt
from scipy import signal


class PCA(BaseSACMethod):
    def __init__(self, 
        n_components: int = 2, 
        features_axis: Optional[int] = -1, 
        noise_identify_method: Literal['variance', 'manual']='variance',
        mode: Literal['global', 'targeted']='global',
        pre_ms: float = 5.0,
        post_ms: float = 25,
        highpass_cutoff: Optional[float] = None,
        filter_order: int = 4,
        noise_components: Optional[List[int]] = None,
        **kwargs):
        super().__init__(**kwargs)
        self._features_axis = features_axis
        self.n_components = n_components
        self.pca_ = None
        self.noise_components_ = None
        self.noise_identify_method = noise_identify_method
        self.mode = mode
        self.pre_ms = pre_ms
        self.post_ms = post_ms
        self.highpass_cutoff = highpass_cutoff
        self.filter_order = filter_order
        self.noise_components = noise_components
        self.b_ = None
        self.a_ = None
        if self.sampling_rate is not None:
            self._update_ms_to_samples()
            self._design_filter()
    
    def _update_ms_to_samples(self):
        self.pre_samples = int(np.round(self.pre_ms * self.sampling_rate / 1000))
        self.post_samples = int(np.round(self.post_ms * self.sampling_rate / 1000))

    def _design_filter(self):
        if self.highpass_cutoff is not None and self.sampling_rate is not None:
            fs = float(np.asarray(self.sampling_rate).item())
            nyquist = 0.5 * fs
            if not 0 < self.highpass_cutoff < nyquist:
                raise ValueError(
                    f"High-pass cutoff frequency ({self.highpass_cutoff} Hz) "
                    f"must be between 0 and the Nyquist frequency ({nyquist} Hz)."
                )
            self.b_, self.a_ = signal.butter(
                self.filter_order, self.highpass_cutoff, btype="high", fs=fs
            )
        else:
            self.b_, self.a_ = None, None

    def set_sampling_rate(self, sampling_rate: float):
        super().set_sampling_rate(sampling_rate)
        self._update_ms_to_samples()
        self._design_filter()

    def _build_artifact_windows(self, artifact_markers, num_samples: int) -> List[List[Tuple[int, int, int]]]:
        starts_per_trial = getattr(artifact_markers, "starts", artifact_markers)
        windows_per_trial: List[List[Tuple[int, int, int]]] = []

        for trial_starts in starts_per_trial:
            trial_windows: List[Tuple[int, int, int]] = []

            for channel_idx, channel_starts in enumerate(trial_starts):
                if channel_starts is None:
                    continue

                markers = np.asarray(channel_starts).astype(int).ravel()
                if markers.size == 0:
                    continue

                markers = markers[(markers >= 0) & (markers < num_samples)]
                if markers.size == 0:
                    continue

                markers = np.unique(markers)

                raw_windows: List[Tuple[int, int]] = []
                for marker in markers:
                    start = max(int(marker) - self.pre_samples, 0)
                    end = min(int(marker) + self.post_samples, num_samples)
                    if end > start:
                        raw_windows.append((start, end))

                if not raw_windows:
                    continue

                raw_windows.sort(key=lambda w: w[0])

                merged: List[Tuple[int, int]] = []
                for start, end in raw_windows:
                    if not merged or start > merged[-1][1]:
                        merged.append((start, end))
                    else:
                        merged[-1] = (merged[-1][0], max(merged[-1][1], end))

                for start, end in merged:
                    trial_windows.append((channel_idx, start, end))

            windows_per_trial.append(trial_windows)

        return windows_per_trial

    def _identify_noise_components(self, explained_variance_ratio: np.ndarray) -> List[int]:
        if self.noise_identify_method == 'manual':
            if self.noise_components is None:
                raise ValueError("noise_components must be specified for manual mode")
            return self.noise_components
        
        elif self.noise_identify_method == 'variance':
            n_noise = min(self.n_components // 2, len(explained_variance_ratio) - 1)
            return list(range(n_noise))
        
        else:
            raise ValueError(f"Unknown noise_identify_method: {self.noise_identify_method}")

    def _apply_filter(self, data: np.ndarray) -> np.ndarray:
        if self.b_ is not None and self.a_ is not None:
            return signal.filtfilt(self.b_, self.a_, data, axis=-1)
        return data

    def _reshape_for_pca(self, data: np.ndarray) -> np.ndarray:
        data_moved = np.moveaxis(data, self._features_axis, -1)
        original_shape = data_moved.shape
        return data_moved.reshape(-1, original_shape[-1])

    def fit(self, data: np.ndarray, artifact_markers: Optional[np.ndarray] = None, **kwargs) -> 'PCA':
        data_to_fit = self._apply_filter(data)

        if self.mode == 'targeted':
            if artifact_markers is None:
                raise ValueError("artifact_markers must be provided for 'targeted' mode fit.")

            if self._features_axis != 1:
                raise ValueError("In targeted mode, features_axis must be 1 (channels) to keep PCA features consistent.")

            num_samples = data_to_fit.shape[-1]
            windows_per_trial = self._build_artifact_windows(artifact_markers, num_samples)
            self.artifact_locations_per_trial_ = windows_per_trial

            all_artifact_epochs: List[np.ndarray] = []
            for trial_idx, trial_windows in enumerate(windows_per_trial):
                for _, start, end in trial_windows:
                    if end <= start:
                        continue
                    epoch_2d = data_to_fit[trial_idx, :, start:end]  # (channels, epoch_len)
                    if epoch_2d.shape[-1] == 0:
                        continue
                    all_artifact_epochs.append(epoch_2d)

            if not all_artifact_epochs:
                raise ValueError("No artifact epochs were extracted for fitting.")

            concatenated_epochs = np.hstack(all_artifact_epochs)  # (channels, total_len)
            fit_data = concatenated_epochs[np.newaxis, ...]  # (1, channels, total_len)

            data_moved = np.moveaxis(fit_data, self._features_axis, -1)
            num_features = data_moved.shape[-1]
            X = data_moved.reshape(-1, num_features)

            n_components = self.n_components if self.n_components is not None else num_features
            if n_components > num_features:
                raise ValueError(f"n_components ({n_components}) cannot be greater than the number of features ({num_features}).")

            self.pca_ = SklearnPCA(n_components=n_components)
            self.pca_.fit(X)
            
            self.noise_components_ = self._identify_noise_components(
                self.pca_.explained_variance_ratio_
            )
        else:
            self.pca_ = None
            self.noise_components_ = None
        
        self.is_fitted = True
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Method must be fitted before transforming data.")
        
        data_to_transform = self._apply_filter(data)

        if self.mode == 'global':
            return self._process_trial_by_trial(data_to_transform)
        else: 
            return self._process_targeted_windows(data_to_transform)

    def _process_trial_by_trial(self, data: np.ndarray) -> np.ndarray:
        n_trials = data.shape[0]
        cleaned_data = np.zeros_like(data)
        
        for trial_idx in range(n_trials):
            trial_data = data[trial_idx:trial_idx+1]
            
            data_moved = np.moveaxis(trial_data, self._features_axis, -1)
            num_features = data_moved.shape[-1]
            X = data_moved.reshape(-1, num_features)
            
            n_components = self.n_components if self.n_components is not None else num_features
            if n_components > num_features:
                n_components = num_features
            
            trial_pca = SklearnPCA(n_components=n_components)
            trial_pca.fit(X)
            
            noise_components = self._identify_noise_components(
                trial_pca.explained_variance_ratio_
            )
            
            pca_data = trial_pca.transform(X)
            cleaned_pca_data = pca_data.copy()
            if noise_components:
                cleaned_pca_data[:, noise_components] = 0
            cleaned_X = trial_pca.inverse_transform(cleaned_pca_data)
            
            cleaned_data_moved = cleaned_X.reshape(data_moved.shape)
            cleaned_trial = np.moveaxis(cleaned_data_moved, -1, self._features_axis)
            cleaned_data[trial_idx] = cleaned_trial[0]
        
        return cleaned_data

    def _apply_pca_transform(self, data: np.ndarray) -> np.ndarray:
        pca_data = self.pca_.transform(data)
        
        # Remove noise components
        cleaned_pca_data = pca_data.copy()
        cleaned_pca_data[:, self.noise_components_] = 0
        
        # Transform back to original space
        cleaned_data = self.pca_.inverse_transform(cleaned_pca_data)
        return cleaned_data

    def _process_targeted_windows(self, data: np.ndarray) -> np.ndarray:
        if not hasattr(self, 'artifact_locations_per_trial_'):
            raise ValueError("Fit must be called with artifact_markers in targeted mode before transform.")
        
        n_trials = data.shape[0]
        if self._features_axis != 1:
            raise ValueError("In targeted mode, features_axis must be 1 (channels) to keep PCA features consistent.")

        n_channels = data.shape[1]

        final_cleaned_data = data.copy()

        for trial_idx, trial_windows in enumerate(self.artifact_locations_per_trial_):
            if not trial_windows:
                continue

            trial_artifact_epochs: List[np.ndarray] = []
            for _, start, end in trial_windows:
                if end <= start:
                    continue
                epoch_2d = data[trial_idx, :, start:end]  # (channels, epoch_len)
                trial_artifact_epochs.append(epoch_2d)

            if not trial_artifact_epochs:
                continue

            concatenated_epochs = np.hstack(trial_artifact_epochs)  # (channels, total_len)
            transform_data = concatenated_epochs[np.newaxis, ...]  # (1, channels, total_len)

            data_moved = np.moveaxis(transform_data, self._features_axis, -1)
            output_shape = data_moved.shape
            reshaped_data = data_moved.reshape(-1, output_shape[-1])

            pca_data = self.pca_.transform(reshaped_data)
            cleaned_pca_data = pca_data.copy()
            if self.noise_components_:
                cleaned_pca_data[:, self.noise_components_] = 0
            cleaned_reshaped_data = self.pca_.inverse_transform(cleaned_pca_data)

            cleaned_concatenated_epochs = cleaned_reshaped_data.reshape(output_shape)
            cleaned_concatenated_epochs = np.moveaxis(cleaned_concatenated_epochs, -1, self._features_axis)
            cleaned_concatenated_epochs = np.squeeze(cleaned_concatenated_epochs)

            current_pos_in_concat = 0
            for _, start, end in trial_windows:
                epoch_len = end - start
                cleaned_segment = cleaned_concatenated_epochs[:, current_pos_in_concat: current_pos_in_concat + epoch_len]
                final_cleaned_data[trial_idx, :, start:end] = cleaned_segment
                current_pos_in_concat += epoch_len

        return final_cleaned_data

    def plot_components(self, data: np.ndarray, trial_idx: int = 0, channel_idx: int = 0, 
                       title: str = "PCA Components Analysis", save_path: Optional[str] = None):
        if not self.is_fitted:
            raise ValueError("Method must be fitted first.")
        
        if self.mode == 'targeted':
            print("Warning: plot_components is not supported in targeted mode. Skipping.")
            return
        
        if data.ndim != 3:
            raise ValueError("Data must be 3D (trials, channels, timesteps)")
        
        trial_data = data[trial_idx, channel_idx, :]
        
        filtered_data = self._apply_filter(data[trial_idx:trial_idx+1])
        data_moved = np.moveaxis(filtered_data, self._features_axis, -1)
        output_shape = data_moved.shape
        reshaped_data = data_moved.reshape(-1, output_shape[-1])
        
        if self.mode == 'global':
            num_features = reshaped_data.shape[-1]
            n_components = self.n_components if self.n_components is not None else num_features
            if n_components > num_features:
                n_components = num_features
            trial_pca = SklearnPCA(n_components=n_components)
            trial_pca.fit(reshaped_data)
            pca_data = trial_pca.transform(reshaped_data)
            explained_variance = trial_pca.explained_variance_ratio_
            pca_model = trial_pca
        else:
            if self.pca_ is None:
                raise ValueError("PCA model not available for plotting.")
            pca_data = self.pca_.transform(reshaped_data)
            explained_variance = self.pca_.explained_variance_ratio_
            pca_model = self.pca_
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"{title} - Trial {trial_idx}, Channel {channel_idx}")
        
        # Original signal
        time_axis = np.arange(len(trial_data)) / self.sampling_rate
        axes[0, 0].plot(time_axis, trial_data)
        axes[0, 0].set_title('Original Signal')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].grid(True)
        
        # PCA components
        for i in range(min(self.n_components, 4)):
            row, col = (i + 1) // 2, (i + 1) % 2
            if row < 2 and col < 2:
                # Create a matrix with only component i active
                component_pca_data = np.zeros_like(pca_data)
                component_pca_data[:, i] = pca_data[:, i]
                
                # Transform back to original space
                component_reshaped = pca_model.inverse_transform(component_pca_data)
                
                # Reshape back to original data shape
                component_data_moved = component_reshaped.reshape(output_shape)
                component_data = np.moveaxis(component_data_moved, -1, self._features_axis)
                
                # Extract the specific channel
                component_signal = component_data[0, channel_idx, :]
                
                axes[row, col].plot(time_axis, component_signal)
                axes[row, col].set_title(f'Component {i+1} (Var: {explained_variance[i]:.3f})')
                axes[row, col].set_xlabel('Time (s)')
                axes[row, col].set_ylabel('Amplitude')
                axes[row, col].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()