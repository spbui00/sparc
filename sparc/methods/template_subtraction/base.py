import numpy as np
from typing import Optional
import warnings
from abc import ABC, abstractmethod
from ...core.signal_data import ArtifactMarkers, ArtifactTriggers
from ...core.base_method import BaseSACMethod


class BaseTemplateSubtraction(BaseSACMethod, ABC):
    def __init__(self,
                 sampling_rate: Optional[float] = None,
                 template_length_ms: float = 5.0,
                 pre_ms: float = 0.8,
                 post_ms: float = 1.0,
                 onset_threshold: float = 1.5,
                 detection_method: str = 'gradient',  # 'gradient' or 'amplitude'
                 **kwargs):
        super().__init__(sampling_rate, **kwargs)
        self.template_length_ms = template_length_ms
        self.pre_ms = pre_ms
        self.post_ms = post_ms
        self.onset_threshold = onset_threshold
        self.detection_method = detection_method

        if self.sampling_rate:
            self._update_samples_from_ms()

        self.templates_ = None 
        self.template_indices_ = None
        self.is_fitted = False

    def set_sampling_rate(self, sampling_rate: float):
        super().set_sampling_rate(sampling_rate)
        self._update_samples_from_ms()
        return self

    def _update_samples_from_ms(self):
        if not self.sampling_rate:
            raise ValueError("Sampling rate must be set before updating sample values.")
        calculated_samples = self.template_length_ms * self.sampling_rate / 1000
        
        if calculated_samples < 1.0:
            warnings.warn(
                f"template_length_ms ({self.template_length_ms}ms) is shorter than one sample "
                f"at {self.sampling_rate}Hz. Clamping to a minimum of 1 sample."
            )
            self.template_length_samples = 1
        else:
            self.template_length_samples = int(calculated_samples)

        self.template_length_samples = int(self.template_length_ms * self.sampling_rate / 1000)
        self.pre_samples = int(self.pre_ms * self.sampling_rate / 1000)
        self.post_samples = int(self.post_ms * self.sampling_rate / 1000)

    def fit(self, data: np.ndarray, artifact_markers: ArtifactMarkers) -> 'BaseTemplateSubtraction':
        if not isinstance(artifact_markers, ArtifactTriggers):
            raise TypeError("This method requires ArtifactTriggers")
        self.template_indices_ = artifact_markers.starts
        
        self.templates_ = self._learn_templates(data)
        self.is_fitted = True
        return self

    @abstractmethod
    def _learn_templates(self, data: np.ndarray):
        raise NotImplementedError

    def transform(self, data: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Method must be fitted before transforming data.")
        if data.ndim != 3:
            raise ValueError("Input data must be a 3D numpy array (trials, channels, timepoints).")

        cleaned_data = np.zeros_like(data)
        for trial_idx in range(data.shape[0]):
            cleaned_data[trial_idx] = self._apply_template_subtraction_single_trial(data[trial_idx], trial_idx)
        return cleaned_data

    @abstractmethod
    def _apply_template_subtraction_single_trial(self, data: np.ndarray, trial_idx: int) -> np.ndarray:
        raise NotImplementedError
