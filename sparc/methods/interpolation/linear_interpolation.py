import numpy as np
from typing import Optional
from sparc import BaseSACMethod
from sparc.core.signal_data import ArtifactMarkers, ArtifactTriggers


class LinearInterpolation(BaseSACMethod):
    def __init__(self, artifact_duration_ms: float = 1.0, margin_ms: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.artifact_duration_ms = artifact_duration_ms
        self.margin_ms = margin_ms
        self.artifact_markers_ = None

    def fit(self, data: np.ndarray, artifact_markers: Optional[ArtifactMarkers] = None) -> 'LinearInterpolation':
        if artifact_markers is None:
            raise ValueError("LinearInterpolation requires 'artifact_markers' to be provided during fit.")
        
        if not isinstance(artifact_markers, ArtifactTriggers):
            raise TypeError("LinearInterpolation requires artifact_markers to be of type ArtifactTriggers")

        self.artifact_markers_ = np.sort(artifact_markers.starts.flatten())
        self.is_fitted = True
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("The transform method cannot be called before fit.")
        if self.sampling_rate is None:
            raise ValueError("Sampling rate must be set before calling transform.")
        if self.artifact_markers_ is None:
            raise ValueError("Artifact markers are not set. Please fit the model with artifact markers first.")

        was_2d = False
        if data.ndim == 2:
            data = data[np.newaxis, :, :]
            was_2d = True

        cleaned_data = data.copy()
        num_trials, num_samples, num_channels = data.shape

        duration_samples = int(self.artifact_duration_ms / 1000 * self.sampling_rate)
        margin_samples = int(self.margin_ms / 1000 * self.sampling_rate)

        for trial_idx in range(num_trials):
            for trigger_index in self.artifact_markers_:
                # Define the interpolation window boundaries
                t_start = trigger_index - margin_samples
                t_end = trigger_index + duration_samples + margin_samples

                if t_start < 0 or t_end >= num_samples:
                    continue

                xp = np.array([t_start, t_end])
                
                for ch_idx in range(num_channels):
                    fp = cleaned_data[trial_idx, xp, ch_idx]
                    interp_indices = np.arange(t_start, t_end + 1)
                    cleaned_data[trial_idx, interp_indices, ch_idx] = np.interp(interp_indices, xp, fp)

        if was_2d:
            return cleaned_data.squeeze(axis=0)
        
        return cleaned_data
