import numpy as np
from scipy import signal
from typing import Optional, Tuple, Union, List
from abc import ABC, abstractmethod
from ...core.base_method import BaseSACMethod

class BaseTemplateSubtraction(BaseSACMethod, ABC):
    def __init__(self,
                 sampling_rate: float,
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

        self.template_length_samples = int(self.template_length_ms * self.sampling_rate / 1000)
        self.pre_samples = int(self.pre_ms * self.sampling_rate / 1000)
        self.post_samples = int(self.post_ms * self.sampling_rate / 1000)

        self.templates_ = None 
        self.template_indices_ = None
        self.is_fitted = False

    def fit(self, data: np.ndarray, artifact_indices: Optional[np.ndarray] = None) -> 'BaseTemplateSubtraction':
        if artifact_indices is not None:
            self.template_indices_ = artifact_indices
        else:
            self.template_indices_ = self._detect_artifacts(data)
        
        self.templates_ = self._learn_templates(data)
        self.is_fitted = True
        return self

    def _detect_artifacts(self, data: np.ndarray) -> Union[List[np.ndarray], np.ndarray]:
        if data.ndim == 2: # (timesteps, channels)
            return self._detect_artifacts_single_trial(data)
        elif data.ndim == 3: # (trials, timesteps, channels)
            artifact_indices = []
            for trial_idx in range(data.shape[0]):
                artifact_indices.append(self._detect_artifacts_single_trial(data[trial_idx]))
            return artifact_indices
        else:
            raise ValueError(f"Unsupported data dimension: {data.ndim}")

    def _detect_artifacts_single_trial(self, data: np.ndarray) -> np.ndarray:
        timesteps, channels = data.shape
        artifact_mask = np.zeros((timesteps, channels), dtype=bool)
        
        for ch in range(channels):
            signal_ch = data[:, ch]
            
            if self.detection_method == 'amplitude':
                artifact_regions = self._find_artifact_regions_amplitude(signal_ch)
            else:  # gradient method (default)
                artifact_regions = self._find_artifact_regions_gradient(signal_ch)
            
            for start, end in artifact_regions:
                expanded_start = max(0, start - self.pre_samples)
                expanded_end = min(timesteps, end + self.post_samples)
                artifact_mask[expanded_start:expanded_end, ch] = True
        return artifact_mask

    def _find_artifact_regions_gradient(self, signal_ch: np.ndarray) -> List[Tuple[int, int]]:
        """Find artifact regions using gradient-based detection."""
        gradient = np.gradient(signal_ch)
        if len(signal_ch) > 5:
            gradient_smooth = signal.savgol_filter(gradient, window_length=5, polyorder=2)
        else:
            gradient_smooth = gradient
        gradient_std = np.std(gradient_smooth)
        threshold = self.onset_threshold * gradient_std
        return self._find_artifact_region_gradient(gradient_smooth, threshold)
    
    def _find_artifact_regions_amplitude(self, signal_ch: np.ndarray) -> List[Tuple[int, int]]:
        """Find artifact regions using amplitude-based detection."""
        signal_std = np.std(signal_ch)
        signal_median = np.median(np.abs(signal_ch))
        
        # Use the smaller of std-based or median-based threshold for better detection
        threshold_std = self.onset_threshold * signal_std
        threshold_median = self.onset_threshold * signal_median * 3  # 3x median as alternative
        threshold = min(threshold_std, threshold_median)
        
        # Find regions where signal exceeds threshold
        artifact_mask = np.abs(signal_ch) > threshold
        return self._find_artifact_regions_from_mask(artifact_mask)
    
    def _find_artifact_region_gradient(self, gradient: np.ndarray, threshold: float) -> List[Tuple[int, int]]:
        """Helper method for gradient-based region finding."""
        regions = []
        in_artifact = False
        start_idx = 0
        for i in range(1, len(gradient)):
            if abs(gradient[i]) > threshold and not in_artifact:
                start_idx = i
                in_artifact = True
            elif abs(gradient[i]) <= threshold and in_artifact:
                regions.append((start_idx, i))
                in_artifact = False
        if in_artifact:
            regions.append((start_idx, len(gradient)))
        return regions
    
    def _find_artifact_regions_from_mask(self, mask: np.ndarray) -> List[Tuple[int, int]]:
        """Convert boolean mask to list of (start, end) regions."""
        regions = []
        in_artifact = False
        start_idx = 0
        for i, is_artifact in enumerate(mask):
            is_artifact_bool = bool(is_artifact)
            if is_artifact_bool and not in_artifact:
                start_idx = i
                in_artifact = True
            elif not is_artifact_bool and in_artifact:
                regions.append((start_idx, i))
                in_artifact = False
        if in_artifact:
            regions.append((start_idx, len(mask)))
        return regions

    def _find_artifact_regions(self, mask: np.ndarray) -> List[Tuple[int, int]]:
        regions = []
        in_artifact = False
        start_idx = 0
        for i, is_artifact in enumerate(mask):
            is_artifact_bool = bool(is_artifact)
            if is_artifact_bool and not in_artifact:
                start_idx = i
                in_artifact = True
            elif not is_artifact_bool and in_artifact:
                regions.append((start_idx, i))
                in_artifact = False
        if in_artifact:
            regions.append((start_idx, len(mask)))
        return regions

    @abstractmethod
    def _learn_templates(self, data: np.ndarray) -> any:
        raise NotImplementedError

    def transform(self, data: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Method must be fitted before transforming data.")

        if data.ndim == 2:  # (timesteps, channels)
            return self._apply_template_subtraction_single_trial(data, 0)
        elif data.ndim == 3:  # (trials, timesteps, channels)
            cleaned_data = np.zeros_like(data)
            for trial_idx in range(data.shape[0]):
                cleaned_data[trial_idx] = self._apply_template_subtraction_single_trial(data[trial_idx], trial_idx)
            return cleaned_data
        else:
            raise ValueError(f"Unsupported data dimension: {data.ndim}")

    @abstractmethod
    def _apply_template_subtraction_single_trial(self, data: np.ndarray, trial_idx: int) -> np.ndarray:
        raise NotImplementedError
