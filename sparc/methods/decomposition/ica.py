import numpy as np
from typing import Literal, Optional, Any, List, Tuple
from sparc import BaseSACMethod
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt
from numpy import intp
from scipy import signal


class ICA(BaseSACMethod):
    def __init__(self, 
        n_components: int = 2, 
        features_axis: Optional[int] = -1, 
        artifact_identify_method: Literal['variance', 'kurtosis_max', 'kurtosis_min']='variance', 
        mode: Literal['global', 'targeted']='targeted',
        pre_ms: float = 5.0,
        post_ms: float = 25,
        highpass_cutoff: Optional[float] = None,
        filter_order: int = 4,
        **kwargs):
        super().__init__(**kwargs)
        self._features_axis = features_axis
        self.n_components = n_components
        self.ica_ = None
        self.S_ = None
        self.artifact_idx_ = None
        self.artifact_identify_method = artifact_identify_method
        self.mode = mode
        self.pre_ms = pre_ms
        self.post_ms = post_ms  
        self.highpass_cutoff = highpass_cutoff
        self.filter_order = filter_order
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
            # Ensure sampling_rate is a scalar
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
            print("No highpass filter applied")
            self.b_, self.a_ = None, None

    def set_sampling_rate(self, sampling_rate: float):
        super().set_sampling_rate(sampling_rate)
        self._update_ms_to_samples()
        self._design_filter()

    def _build_artifact_windows(
        self,
        artifact_markers,
        num_samples: int
    ) -> List[List[Tuple[int, int, int]]]:
        windows_per_trial: List[List[Tuple[int, int, int]]] = []
        starts_per_trial = getattr(artifact_markers, "starts", artifact_markers)

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

                # Build initial windows
                raw_windows: List[Tuple[int, int]] = []
                for marker in markers:
                    start = max(int(marker) - self.pre_samples, 0)
                    end = min(int(marker) + self.post_samples, num_samples)
                    if end > start:
                        raw_windows.append((start, end))

                if not raw_windows:
                    continue

                raw_windows.sort(key=lambda w: w[0])

                # Merge overlaps
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

    def _max_variance(self, sources: np.ndarray) -> intp:
        variances = np.var(sources, axis=0)
        return np.argmax(variances)

    def _find_kurtosis_idx(self, x: np.ndarray, method: Literal['max', 'min']) -> intp:
        def _kurtosis(x: np.ndarray) -> float | Any:
            if len(x) == 0: return 0.0
            mu4 = np.mean(x ** 4)
            var = np.var(x)
            if var == 0: return 0.0
            return (mu4 / (var ** 2)) - 3
            
        kurtoses = np.array([_kurtosis(x[:, i]) for i in range(x.shape[1])])

        if method == 'max':
            return np.argmax(np.abs(kurtoses))
        else: 
            return np.argmin(np.abs(kurtoses))

    def _identify_artifact_component_idx(self, sources: np.ndarray) -> intp:
        if self.artifact_identify_method == 'variance':
            return self._max_variance(sources)
        elif self.artifact_identify_method == 'kurtosis_max':
            return self._find_kurtosis_idx(sources, 'max')
        elif self.artifact_identify_method == 'kurtosis_min':
            return self._find_kurtosis_idx(sources, 'min')
        else:
            raise ValueError(f"Unknown artifact_identify_method: {self.artifact_identify_method}")

    def fit(self, data: np.ndarray, artifact_markers: Optional[np.ndarray] = None, **kwargs) -> 'ICA':
        if self.b_ is not None and self.a_ is not None:
            data_to_fit = signal.filtfilt(self.b_, self.a_, data, axis=-1)
        else:
            data_to_fit = data

        if self.mode == 'targeted':
            if artifact_markers is None:
                raise ValueError("`artifact_markers` must be provided for 'targeted' mode fit.")
            
            num_samples = data_to_fit.shape[-1]
            windows_per_trial = self._build_artifact_windows(artifact_markers, num_samples)
            self.artifact_locations_per_trial_ = windows_per_trial

            all_artifact_epochs = []

            for trial_idx, trial_windows in enumerate(windows_per_trial):
                for channel_idx, start, end in trial_windows:
                    if end <= start:
                        continue
                    if self._features_axis in [-1, 2]:
                        epoch = data_to_fit[trial_idx, ..., start:end]
                    else:
                        epoch = data_to_fit[trial_idx, :, start:end]
                    if epoch.shape[-1] == 0:
                        continue
                    all_artifact_epochs.append(epoch)

            if not all_artifact_epochs:
                raise ValueError("No artifact epochs were extracted for fitting.")
            
            # Concatenate all artifact epochs for ICA training
            concatenated_epochs = np.hstack(all_artifact_epochs)
            fit_data = concatenated_epochs[np.newaxis, ...] if data.ndim == 3 else concatenated_epochs
        else:
            fit_data = data_to_fit

        data_moved = np.moveaxis(fit_data, self._features_axis, -1)
        num_features = data_moved.shape[-1]
        reshaped_data = data_moved.reshape(-1, num_features)

        n_components = self.n_components if self.n_components is not None else num_features
        if n_components > num_features:
            raise ValueError(f"n_components ({n_components}) cannot be greater than the number of features ({num_features}).")

        self.ica_ = FastICA(n_components=n_components, **kwargs)
        self.S_ = self.ica_.fit_transform(reshaped_data)
        self.artifact_idx_ = self._identify_artifact_component_idx(self.S_)

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
            
            sources = self.ica_.transform(reshaped_data)
            sources[:, self.artifact_idx_] = 0
            cleaned_reshaped_data = self.ica_.inverse_transform(sources)

            cleaned_data_moved = cleaned_reshaped_data.reshape(output_shape)
            cleaned_data = np.moveaxis(cleaned_data_moved, -1, self._features_axis)
            return cleaned_data
        else:
            if not hasattr(self, 'artifact_locations_per_trial_'):
                raise ValueError("Fit must be called with artifact_markers in targeted mode before transform.")

            final_cleaned_data = data_to_transform.copy()
            
            for trial_idx, trial_locations in enumerate(self.artifact_locations_per_trial_):
                if not trial_locations:
                    continue  # Skip trials with no artifacts
                
                # Collect artifact epochs for this trial
                trial_artifact_epochs = []
                for channel_idx, start, end in trial_locations:
                    if self._features_axis in [-1, 2]:
                        epoch = data_to_transform[trial_idx, ..., start:end]
                    else:
                        epoch = data_to_transform[trial_idx, :, start:end]
                    trial_artifact_epochs.append(epoch)
                
                if not trial_artifact_epochs:
                    continue
                
                # Concatenate epochs for this trial
                concatenated_epochs = np.hstack(trial_artifact_epochs)
                transform_data = concatenated_epochs[np.newaxis, ...] if data.ndim == 3 else concatenated_epochs

                # Apply ICA
                data_moved = np.moveaxis(transform_data, self._features_axis, -1)
                output_shape = data_moved.shape
                reshaped_data = data_moved.reshape(-1, output_shape[-1])

                sources = self.ica_.transform(reshaped_data)
                sources[:, self.artifact_idx_] = 0
                cleaned_reshaped_data = self.ica_.inverse_transform(sources)
                
                cleaned_concatenated_epochs = cleaned_reshaped_data.reshape(output_shape)
                cleaned_concatenated_epochs = np.moveaxis(cleaned_concatenated_epochs, -1, self._features_axis)
                cleaned_concatenated_epochs = np.squeeze(cleaned_concatenated_epochs)

                # Put cleaned segments back into the original data
                current_pos_in_concat = 0
                for channel_idx, start, end in trial_locations:
                    epoch_len = end - start
                    cleaned_segment = cleaned_concatenated_epochs[..., current_pos_in_concat : current_pos_in_concat + epoch_len]
                    if self._features_axis in [-1, 2]:
                        final_cleaned_data[trial_idx, ..., start:end] = cleaned_segment
                    else:
                        final_cleaned_data[trial_idx, :, start:end] = cleaned_segment
                    current_pos_in_concat += epoch_len
            
            return final_cleaned_data

    def plot_components(self, n_samples_to_plot: int = 1000):
        if not self.is_fitted:
            raise ValueError("Method must be fitted before plotting components.")
            
        n_components = self.S_.shape[1]
        plot_data = self.S_[:n_samples_to_plot, :]
        fig, axes = plt.subplots(n_components, 1, figsize=(12, 2 * n_components), sharex=True)
        if n_components == 1: axes = [axes]
        fig.suptitle('Independent Components (Sources)', fontsize=16)
        
        for i, ax in enumerate(axes):
            ax.plot(plot_data[:, i])
            title = f'Component {i}'
            if i == self.artifact_idx_:
                title += ' (Identified as Artifact)'
            ax.set_title(title)
            ax.grid(True, linestyle='--', alpha=0.6)
            
        axes[-1].set_xlabel('Time (samples)')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()


def main():
    np.random.seed(0)
    n_samples = 2000
    time = np.linspace(0, 8, n_samples)
    s1 = np.sin(2 * np.pi * 5 * time)
    s2 = np.zeros_like(time)
    s2[::200] = 5.0
    s2[1::200] = -5.0
    S = np.c_[s1, s2]
    A = np.array([[1, 1], [0.5, 2], [1.5, 0.5]])
    X_mixed = S.dot(A.T)
    X_3d = X_mixed.T.reshape(1, 3, n_samples)

    ica_model = ICA(n_components=2, features_axis=1)
    ica_model.fit(X_3d, random_state=0, max_iter=500)
    ica_model.plot_components(n_samples_to_plot=2000)
    cleaned_data = ica_model.transform(X_3d)

    channel_to_plot = 1
    ground_truth_neural = A[channel_to_plot, 0] * s1
    plt.figure(figsize=(12, 6))
    plt.plot(time, X_3d[0, channel_to_plot, :], label='Original Mixed Signal', color='orange', alpha=0.7)
    plt.plot(time, ground_truth_neural, label='Ground Truth Neural Signal', color='gray', linestyle='--', linewidth=2)
    plt.plot(time, cleaned_data[0, channel_to_plot, :], label='Cleaned Signal (ICA Output)', color='blue', alpha=0.8)
    plt.title(f'Comparison on Channel {channel_to_plot} (Full View)', fontsize=16)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
