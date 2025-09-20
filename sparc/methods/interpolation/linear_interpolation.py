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
        self.margin_samples = None
        self.artifact_duration_samples = None
        if self.sampling_rate is not None:
            self._update_ms_to_samples()

    def _update_ms_to_samples(self):
        if self.sampling_rate is None:
            raise ValueError("Sampling rate must be set before updating ms to samples.")
        self.artifact_duration_samples = int(self.artifact_duration_ms / 1000 * self.sampling_rate)
        if self.artifact_duration_samples == 0:
            raise ValueError("artifact_duration_ms is too small for the given sampling rate, resulting in zero samples.")
        self.margin_samples = int(self.margin_ms / 1000 * self.sampling_rate)
        if self.margin_samples == 0:
            raise ValueError("margin_ms is too small for the given sampling rate, resulting in zero samples.")

    def set_sampling_rate(self, sampling_rate: float):
        self.sampling_rate = sampling_rate
        self._update_ms_to_samples()
        return self

    def fit(self, data: np.ndarray, artifact_markers: Optional[ArtifactMarkers] = None) -> 'LinearInterpolation':
        if artifact_markers is None:
            raise ValueError("LinearInterpolation requires 'artifact_markers' to be provided during fit.")
        
        if not isinstance(artifact_markers, ArtifactTriggers):
            raise TypeError("LinearInterpolation requires artifact_markers to be of type ArtifactTriggers")

        self.artifact_markers_ = artifact_markers.starts
        self.is_fitted = True
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("The transform method cannot be called before fit.")
        if self.sampling_rate is None:
            raise ValueError("Sampling rate must be set before calling transform.")
        if self.artifact_markers_ is None:
            raise ValueError("Artifact markers are not set. Please fit the model with artifact markers first.")
        if self.artifact_duration_samples is None or self.margin_samples is None:
            raise ValueError("artifact_duration_samples and margin_samples must be set. Please ensure sampling_rate is provided.")

        cleaned_data = data.copy()
        num_trials, num_channels, num_samples = data.shape

        for trial_idx in range(num_trials):
            for ch_idx in range(num_channels):
                channel_triggers = self.artifact_markers_[trial_idx][ch_idx]
                
                for trigger_index in channel_triggers:
                    t_start = trigger_index - self.margin_samples
                    t_end = trigger_index + self.artifact_duration_samples + self.margin_samples

                    if t_start < 0 or t_end >= num_samples:
                        continue

                    xp = np.array([t_start, t_end])
                    fp = cleaned_data[trial_idx, ch_idx, xp]
                    interp_indices = np.arange(t_start, t_end + 1)
                    cleaned_data[trial_idx, ch_idx, interp_indices] = np.interp(interp_indices, xp, fp)
        
        return cleaned_data
