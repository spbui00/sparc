from dataclasses import dataclass, field
import numpy as np
from typing import Any


@dataclass
class SignalData:
    """Represents basic signal data."""
    raw_data: np.ndarray
    sampling_rate: float
    artifact_indices: np.ndarray | None = field(default=None, kw_only=True)


@dataclass
class SignalDataWithGroundTruth(SignalData):
    """Represents signal data with ground truth for evaluation."""
    ground_truth: np.ndarray
    artifacts: np.ndarray


@dataclass
class SimulatedData(SignalDataWithGroundTruth):
    """Represents simulated data with additional metadata for detailed evaluation."""
    firing_rate: np.ndarray
    spike_train: np.ndarray
    lfp: np.ndarray
    stim_params: Any
    snr: np.ndarray
    
