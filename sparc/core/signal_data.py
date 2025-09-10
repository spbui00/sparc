from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class SignalData:
    """Represents basic signal data."""
    raw_data: np.ndarray
    sampling_rate: float

@dataclass
class SignalDataWithGroundTruth(SignalData):
    """Represents signal data with ground truth for evaluation."""
    ground_truth: np.ndarray
    artifacts: np.ndarray
    artifact_indices: Optional[np.ndarray] = None