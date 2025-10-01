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
        noise_identify_method: Literal['variance', 'explained_variance_ratio', 'manual']='explained_variance_ratio',
        mode: Literal['global', 'targeted']='global',
        pre_ms: float = 5.0,
        post_ms: float = 25,
        highpass_cutoff: Optional[float] = None,
        filter_order: int = 4,
        noise_components: Optional[List[int]] = None,
        variance_threshold: float = 0.01,
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
        self.variance_threshold = variance_threshold
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
        if not hasattr(artifact_markers, 'starts'):
            raise ValueError("artifact_markers must have 'starts' attribute")
        
        windows_per_trial = []
        
        for trial_idx, trial_markers in enumerate(artifact_markers.starts):
            trial_windows = []
            
            for channel_idx, channel_markers in enumerate(trial_markers):
                if len(channel_markers) == 0:
                    continue
                
                for marker in channel_markers:
                    start = marker + self.pre_samples
                    end = marker + self.post_samples
                    
                    if start >= 0 and end <= num_samples:
                        trial_windows.append((channel_idx, start, end))
            
            windows_per_trial.append(trial_windows)
        
        return windows_per_trial

    def _identify_noise_components(self, explained_variance_ratio: np.ndarray) -> List[int]:
        if self.noise_identify_method == 'manual':
            if self.noise_components is None:
                raise ValueError("noise_components must be specified for manual mode")
            return self.noise_components
        
        elif self.noise_identify_method == 'variance':
            sorted_indices = np.argsort(explained_variance_ratio)[::-1]
            n_noise = min(self.n_components // 2, len(explained_variance_ratio) - 1)
            return sorted_indices[:n_noise].tolist()
        
        elif self.noise_identify_method == 'explained_variance_ratio':
            noise_components = []
            for i, ratio in enumerate(explained_variance_ratio):
                if ratio < self.variance_threshold:
                    noise_components.append(i)
            return noise_components
        
        else:
            raise ValueError(f"Unknown noise_identify_method: {self.noise_identify_method}")

    def fit(self, data: np.ndarray, artifact_markers: Optional[np.ndarray] = None, **kwargs) -> 'PCA':
        if self.b_ is not None and self.a_ is not None:
            data_to_fit = signal.filtfilt(self.b_, self.a_, data, axis=-1)
        else:
            data_to_fit = data

        if self.mode == 'targeted':
            if artifact_markers is None:
                raise ValueError("artifact_markers must be provided for 'targeted' mode fit.")
            
            num_samples = data_to_fit.shape[-1]
            windows_per_trial = self._build_artifact_windows(artifact_markers, num_samples)
            self.artifact_locations_per_trial_ = windows_per_trial

            all_artifact_epochs = []
            expected_window_size = self.post_samples - self.pre_samples
            
            for trial_idx, trial_windows in enumerate(windows_per_trial):
                for channel_idx, start, end in trial_windows:
                    if end <= start:
                        continue
                    
                    # Extract only the specific channel's data for this window
                    if self._features_axis in [-1, 2]:
                        # Time is last axis, extract specific channel
                        epoch = data_to_fit[trial_idx, channel_idx, start:end]
                    else:
                        # Channel is last axis
                        epoch = data_to_fit[trial_idx, start:end, channel_idx]
                    
                    if epoch.size != expected_window_size:
                        continue
                    
                    if epoch.size > 0:
                        # Reshape to (1, n_features) for stacking
                        all_artifact_epochs.append(epoch.reshape(1, -1))

            if not all_artifact_epochs:
                raise ValueError("No valid artifact epochs found for PCA fitting. Check that your artifact windows are not all at the edges of the signal.")
            
            # Stack all epochs
            X = np.vstack(all_artifact_epochs)
            
        else:  # global mode
            data_moved = np.moveaxis(data_to_fit, self._features_axis, -1)
            original_shape = data_moved.shape
            X = data_moved.reshape(-1, original_shape[-1])

        self.pca_ = SklearnPCA(n_components=self.n_components)
        self.pca_.fit(X)
        
        self.noise_components_ = self._identify_noise_components(
            self.pca_.explained_variance_ratio_
        )
        
        self.is_fitted = True
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Method must be fitted before transforming data.")
        
        if self.b_ is not None and self.a_ is not None:
            data_to_transform = signal.filtfilt(self.b_, self.a_, data, axis=-1)
        else:
            data_to_transform = data

        if self.mode == 'global':
            data_moved = np.moveaxis(data_to_transform, self._features_axis, -1)
            output_shape = data_moved.shape
            reshaped_data = data_moved.reshape(-1, output_shape[-1])
            
            pca_data = self.pca_.transform(reshaped_data)
            
            cleaned_pca_data = pca_data.copy()
            cleaned_pca_data[:, self.noise_components_] = 0
            
            cleaned_reshaped_data = self.pca_.inverse_transform(cleaned_pca_data)
            
            cleaned_data_moved = cleaned_reshaped_data.reshape(output_shape)
            cleaned_data = np.moveaxis(cleaned_data_moved, -1, self._features_axis)
            
            return cleaned_data
        
        else:  # targeted mode
            if not hasattr(self, 'artifact_locations_per_trial_'):
                raise ValueError("Fit must be called with artifact_markers in targeted mode before transform.")
            
            final_cleaned_data = data_to_transform.copy()
            expected_window_size = self.post_samples - self.pre_samples
            
            for trial_idx, trial_locations in enumerate(self.artifact_locations_per_trial_):
                if not trial_locations:
                    continue
                
                for channel_idx, start, end in trial_locations:
                    if end <= start:
                        continue
                    
                    if self._features_axis in [-1, 2]:
                        artifact_window = data_to_transform[trial_idx, channel_idx, start:end]
                    else:
                        # Channel is last axis
                        artifact_window = data_to_transform[trial_idx, start:end, channel_idx]
                    
                    # Skip windows that are not the expected size (edge cases)
                    if artifact_window.size != expected_window_size:
                        continue
                    
                    # Apply PCA to this window
                    window_reshaped = artifact_window.reshape(1, -1)
                    pca_window = self.pca_.transform(window_reshaped)
                    
                    # Remove noise components
                    cleaned_pca_window = pca_window.copy()
                    cleaned_pca_window[:, self.noise_components_] = 0
                    
                    # Transform back
                    cleaned_window = self.pca_.inverse_transform(cleaned_pca_window)
                    cleaned_window_reshaped = cleaned_window.reshape(artifact_window.shape)
                    
                    # Put back into final data
                    if self._features_axis in [-1, 2]:
                        final_cleaned_data[trial_idx, channel_idx, start:end] = cleaned_window_reshaped
                    else:
                        final_cleaned_data[trial_idx, start:end, channel_idx] = cleaned_window_reshaped
            
            return final_cleaned_data

    def get_explained_variance_ratio(self) -> np.ndarray:
        """Get explained variance ratio for each component."""
        if not self.is_fitted:
            raise ValueError("Method must be fitted first.")
        return self.pca_.explained_variance_ratio_

    def get_noise_components(self) -> List[int]:
        """Get indices of components identified as noise."""
        if not self.is_fitted:
            raise ValueError("Method must be fitted first.")
        return self.noise_components_

    def plot_components(self, data: np.ndarray, trial_idx: int = 0, channel_idx: int = 0, 
                       title: str = "PCA Components Analysis", save_path: Optional[str] = None):
        if not self.is_fitted:
            raise ValueError("Method must be fitted first.")
        
        if data.ndim != 3:
            raise ValueError("Data must be 3D (trials, channels, timesteps)")
        
        trial_data = data[trial_idx, channel_idx, :]
        
        if self.mode == 'global':
            data_moved = np.moveaxis(data[trial_idx:trial_idx+1], self._features_axis, -1)
            output_shape = data_moved.shape
            reshaped_data = data_moved.reshape(-1, output_shape[-1])
        else:
            data_moved = np.moveaxis(data[trial_idx:trial_idx+1], self._features_axis, -1)
            output_shape = data_moved.shape
            reshaped_data = data_moved.reshape(-1, output_shape[-1])
        
        pca_data = self.pca_.transform(reshaped_data)
        
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
                component_reshaped = self.pca_.inverse_transform(component_pca_data)
                
                # Reshape back to original data shape
                component_data_moved = component_reshaped.reshape(output_shape)
                component_data = np.moveaxis(component_data_moved, -1, self._features_axis)
                
                # Extract the specific channel
                component_signal = component_data[0, channel_idx, :]
                
                axes[row, col].plot(time_axis, component_signal)
                axes[row, col].set_title(f'Component {i+1} (Var: {self.pca_.explained_variance_ratio_[i]:.3f})')
                axes[row, col].set_xlabel('Time (s)')
                axes[row, col].set_ylabel('Amplitude')
                axes[row, col].grid(True)
        
        # Explained variance ratio
        axes[1, 1].bar(range(len(self.pca_.explained_variance_ratio_)), 
                      self.pca_.explained_variance_ratio_)
        axes[1, 1].set_title('Explained Variance Ratio')
        axes[1, 1].set_xlabel('Component')
        axes[1, 1].set_ylabel('Explained Variance Ratio')
        axes[1, 1].grid(True)
        
        # Highlight noise components
        for i, comp_idx in enumerate(self.noise_components_):
            if comp_idx < len(self.pca_.explained_variance_ratio_):
                axes[1, 1].bar(comp_idx, self.pca_.explained_variance_ratio_[comp_idx], 
                              color='red', alpha=0.7, label='Noise' if i == 0 else "")
        
        if self.noise_components_:
            axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def get_reconstruction_error(self, data: np.ndarray) -> float:
        """Calculate reconstruction error after noise removal."""
        if not self.is_fitted:
            raise ValueError("Method must be fitted first.")
        
        cleaned_data = self.transform(data)
        mse = np.mean((data - cleaned_data) ** 2)
        return mse
