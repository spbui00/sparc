import numpy as np
from typing import Optional
from scipy import interpolate
from sparc.core.base_method import BaseSACMethod


class LinearInterpolation(BaseSACMethod):
    def __init__(self, 
                 sampling_rate: float,
                 positive_threshold: float = 5.0,
                 negative_threshold: float = -5.0, 
                 flag_threshold: float = 200.0,
                 normalize_signal: bool = False,
                 **kwargs):
        super().__init__(sampling_rate, **kwargs)
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold
        self.flag_threshold = flag_threshold
        self.normalize_signal = normalize_signal
        
        self.data_min = None
        self.data_max = None
        
    def fit(self, data: np.ndarray, artifact_indices: Optional[np.ndarray] = None) -> 'LinearInterpolation':
        self.is_3d = data.ndim == 3
        
        if self.normalize_signal:
            self.data_min = np.min(data)
            self.data_max = np.max(data)
            
        self.is_fitted = True
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Method must be fitted before transform. Call fit() first.")
            
        if data.ndim == 2:
            data_3d = data[np.newaxis, :, :]
        else:
            data_3d = data.copy()
            
        cleaned_data = self._process_data(data_3d)
        
        if data.ndim == 2:
            return cleaned_data[0]
        else:
            return cleaned_data
            
    def _process_data(self, data: np.ndarray) -> np.ndarray:
        n_trials, n_timesteps, n_channels = data.shape
        cleaned_data = data.copy()
        
        if self.normalize_signal:
            if self.data_min is not None and self.data_max is not None:
                data_min, data_max = self.data_min, self.data_max
            else:
                data_min = np.min(data)
                data_max = np.max(data)
                
            cleaned_data = 2 * (cleaned_data - data_min) / (data_max - data_min) - 1
            
            pos_thresh = self.positive_threshold if not self.normalize_signal else 0.01
            neg_thresh = self.negative_threshold if not self.normalize_signal else -0.001
            flag_thresh = self.flag_threshold if not self.normalize_signal else 0.02
        else:
            pos_thresh = self.positive_threshold
            neg_thresh = self.negative_threshold
            flag_thresh = self.flag_threshold
            
        for trial in range(n_trials):
            for ch in range(n_channels):
                signal = cleaned_data[trial, :, ch]
                
                interpolated_signal = self._detect_and_interpolate(
                    signal, pos_thresh, neg_thresh, flag_thresh
                )
                
                cleaned_data[trial, :, ch] = interpolated_signal
                
        if self.normalize_signal and self.data_min is not None:
            cleaned_data = ((cleaned_data + 1) / 2) * (data_max - data_min) + data_min
            
        return cleaned_data
        
    def _detect_and_interpolate(self, 
                                signal: np.ndarray,
                                pos_thresh: float,
                                neg_thresh: float,
                                flag_thresh: float) -> np.ndarray:
        n_samples = len(signal)
        artifact_mask = np.zeros(n_samples, dtype=bool)
        
        # Initialize flags
        flag_decrease = np.zeros(n_samples, dtype=bool) 
        flag_flat = np.zeros(n_samples, dtype=bool) 
        
        # Calculate signal differences
        signal_diff = np.diff(signal, prepend=signal[0])
        
        # Detect artifacts based on signal derivatives
        for i in range(1, n_samples):
            if signal_diff[i] < neg_thresh:
                flag_decrease[i] = True
                
                # If previous sample was in flat region and signal is within bounds
                if flag_flat[i-1] and -flag_thresh < signal[i-1] < flag_thresh:
                    artifact_mask[i] = False  # End of artifact, use original value
                    flag_flat[i] = False
                else:
                    artifact_mask[i] = True  # Mark as artifact
                    
            # Check for flat/slow changing region
            elif neg_thresh < signal_diff[i] < pos_thresh:
                flag_flat[i] = True
                
                # If previous sample had rapid decrease and current value is within bounds
                if flag_decrease[i-1] and -flag_thresh < signal[i] < flag_thresh:
                    flag_decrease[i] = False  # Reset decrease flag
                    
                # Continue marking as artifact if in artifact region
                if flag_decrease[i-1] or (i > 0 and artifact_mask[i-1]):
                    artifact_mask[i] = True
                    
        # Apply linear interpolation over artifact regions
        interpolated_signal = signal.copy()
        
        # Find artifact and non-artifact indices
        artifact_indices = np.where(artifact_mask)[0]
        valid_indices = np.where(~artifact_mask)[0]
        
        # Perform interpolation if there are both artifact and valid regions
        if len(artifact_indices) > 0 and len(valid_indices) > 1:
            # Use scipy's interp1d for linear interpolation
            try:
                interp_func = interpolate.interp1d(
                    valid_indices, 
                    signal[valid_indices],
                    kind='linear',
                    fill_value='extrapolate',
                    assume_sorted=True
                )
                interpolated_signal[artifact_indices] = interp_func(artifact_indices)
            except Exception:
                # If interpolation fails, use nearest valid values
                for idx in artifact_indices:
                    # Find nearest valid samples
                    before_idx = valid_indices[valid_indices < idx]
                    after_idx = valid_indices[valid_indices > idx]
                    
                    if len(before_idx) > 0 and len(after_idx) > 0:
                        # Linear interpolation between nearest valid points
                        x1, x2 = before_idx[-1], after_idx[0]
                        y1, y2 = signal[x1], signal[x2]
                        alpha = (idx - x1) / (x2 - x1)
                        interpolated_signal[idx] = y1 + alpha * (y2 - y1)
                    elif len(before_idx) > 0:
                        # Use last valid value before artifact
                        interpolated_signal[idx] = signal[before_idx[-1]]
                    elif len(after_idx) > 0:
                        # Use first valid value after artifact
                        interpolated_signal[idx] = signal[after_idx[0]]
                        
        return interpolated_signal
        
    def get_artifact_mask(self, data: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            self.fit(data)
            
        # Handle both 2D and 3D data
        if data.ndim == 2:
            data_3d = data[np.newaxis, :, :]
        else:
            data_3d = data.copy()
            
        n_trials, n_timesteps, n_channels = data_3d.shape
        artifact_mask = np.zeros_like(data_3d, dtype=bool)
        
        # Normalize if needed
        if self.normalize_signal:
            if self.data_min is not None:
                data_min, data_max = self.data_min, self.data_max
            else:
                data_min, data_max = np.min(data_3d), np.max(data_3d)
            data_norm = 2 * (data_3d - data_min) / (data_max - data_min) - 1
            pos_thresh = 0.01
            neg_thresh = -0.001
            flag_thresh = 0.02
        else:
            data_norm = data_3d
            pos_thresh = self.positive_threshold
            neg_thresh = self.negative_threshold
            flag_thresh = self.flag_threshold
            
        # Detect artifacts for each trial and channel
        for trial in range(n_trials):
            for channel in range(n_channels):
                signal = data_norm[trial, :, channel]
                artifact_mask[trial, :, channel] = self._detect_artifacts(
                    signal, pos_thresh, neg_thresh, flag_thresh
                )
                
        # Return in original shape
        if data.ndim == 2:
            return artifact_mask[0]
        else:
            return artifact_mask
            
    def _detect_artifacts(self,
                         signal: np.ndarray,
                         pos_thresh: float,
                         neg_thresh: float,
                         flag_thresh: float) -> np.ndarray:
        n_samples = len(signal)
        artifact_mask = np.zeros(n_samples, dtype=bool)
        
        flag_decrease = np.zeros(n_samples, dtype=bool)
        flag_flat = np.zeros(n_samples, dtype=bool)
        
        signal_diff = np.diff(signal, prepend=signal[0])
        
        for i in range(1, n_samples):
            if signal_diff[i] < neg_thresh:
                flag_decrease[i] = True
                if flag_flat[i-1] and -flag_thresh < signal[i-1] < flag_thresh:
                    artifact_mask[i] = False
                    flag_flat[i] = False
                else:
                    artifact_mask[i] = True
                    
            elif neg_thresh < signal_diff[i] < pos_thresh:
                flag_flat[i] = True
                if flag_decrease[i-1] and -flag_thresh < signal[i] < flag_thresh:
                    flag_decrease[i] = False
                if flag_decrease[i-1] or (i > 0 and artifact_mask[i-1]):
                    artifact_mask[i] = True
                    
        return artifact_mask
