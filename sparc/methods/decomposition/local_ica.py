import numpy as np
from typing import Optional
from sparc import BaseSACMethod
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt


class LocalICA(BaseSACMethod):
    def __init__(self, n_components: int = 2, features_axis: Optional[int] = -1,
                 stim_channel: int = 0, local_radius: int = 5, **kwargs):
        super().__init__(**kwargs)
        self._features_axis = features_axis
        self.n_components = n_components
        self.stim_channel = stim_channel
        self.local_radius = local_radius
        self.ica_local_ = None
        self.S_local_ = None
        self.artifact_idx_ = None
        self.spatial_map_ = None
        self.local_channels_ = None
        self._data_moved_shape_template = None  # To infer shape structure

    def _kurtosis(self, x: np.ndarray) -> float:
        if len(x) == 0:
            return 0.0
        mu4 = np.mean(x ** 4)
        var = np.var(x)
        if var == 0:
            return 0.0
        return (mu4 / (var ** 2)) - 3

    def _identify_artifact_component_idx(self, sources: np.ndarray) -> int:
        kurtoses = np.array([self._kurtosis(sources[:, i]) for i in range(sources.shape[1])])
        artifact_index = np.argmax(np.abs(kurtoses))
        return artifact_index

    def fit(self, 
            data: np.ndarray, 
            **kwargs) -> 'LocalICA':
        if self.stim_channel is None:
            raise ValueError("stim_channel must be specified in __init__.")

        data_moved = np.moveaxis(data, self._features_axis, -1)
        self._data_moved_shape_template = data_moved.shape 
        num_features = self._data_moved_shape_template[-1]
        
        if self.stim_channel < 0 or self.stim_channel >= num_features:
            raise ValueError(f"stim_channel ({self.stim_channel}) out of range for {num_features} features.")

        local_start = max(0, self.stim_channel-self.local_radius)
        local_end = min(num_features, self.stim_channel+self.local_radius+1)
        self.local_channels_ = np.arange(local_start, local_end)
        n_local = len(self.local_channels_)

        reshaped_data = data_moved.reshape(-1, num_features)
        local_data = reshaped_data[:, self.local_channels_] # only extract local channels
 
        n_components = self.n_components if self.n_components is not None else n_local
        if n_components > n_local:
            raise ValueError(f"n_components ({n_components}) cannot be greater than local features ({n_local}).")

        self.ica_local_ = FastICA(n_components=n_components, **kwargs)
        self.S_local_ = self.ica_local_.fit_transform(local_data)
        self.artifact_idx_ = self._identify_artifact_component_idx(self.S_local_)

        # Extract spatial map for artifact component (column of mixing_)
        artifact_spatial_local = self.ica_local_.mixing_[:, self.artifact_idx_]
        self.spatial_map_ = np.zeros(num_features)
        self.spatial_map_[self.local_channels_] = artifact_spatial_local

        self.is_fitted = True
        return self
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Method must be fitted before transforming data.")

        data_moved = np.moveaxis(data, self._features_axis, -1)
        target_shape = data_moved.shape
        num_features = target_shape[-1]
        reshaped_data = data_moved.reshape(-1, num_features)
        
        local_data_new = reshaped_data[:, self.local_channels_]
        sources_new_local = self.ica_local_.transform(local_data_new)
        S_artifact_new = sources_new_local[:, self.artifact_idx_]
        
        artifact_estimate = np.outer(S_artifact_new, self.spatial_map_)
        
        cleaned_reshaped = reshaped_data - artifact_estimate
        cleaned_data_moved = cleaned_reshaped.reshape(target_shape)
        cleaned_data = np.moveaxis(cleaned_data_moved, -1, self._features_axis)

        return cleaned_data

    def plot_components(self, n_samples_to_plot: int = 1000):
        if not self.is_fitted:
            raise ValueError("Method must be fitted before plotting components.")
            
        n_components = self.S_local_.shape[1]
        plot_data = self.S_local_[:n_samples_to_plot, :]
        fig, axes = plt.subplots(n_components + 1, 1, figsize=(12, 2 * (n_components + 1)), sharex=True)
        if n_components + 1 == 1: 
            axes = [axes]
        else:
            axes = axes.flatten()
        fig.suptitle('Local Independent Components (Sources) and Spatial Map', fontsize=16)
        
        for i, ax in enumerate(axes[:-1]):
            ax.plot(plot_data[:, i])
            title = f'Local Component {i}'
            if i == self.artifact_idx_:
                title += ' (Identified as Artifact)'
            ax.set_title(title)
            ax.grid(True, linestyle='--', alpha=0.6)
        
        # Plot spatial map
        axes[-1].plot(self.spatial_map_)
        axes[-1].set_title('Artifact Spatial Map (across all channels)')
        axes[-1].set_xlabel('Channel Index')
        axes[-1].grid(True, linestyle='--', alpha=0.6)
        axes[-1].axvline(x=self.stim_channel, color='red', linestyle='--', label='Stim Channel')
        axes[-1].legend()
        
        axes[0].set_ylabel('Amplitude')  # For sources
        axes[-2].set_ylabel('Spatial Weight')  # Before spatial
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
