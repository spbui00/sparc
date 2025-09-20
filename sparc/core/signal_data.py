from dataclasses import dataclass, field
import numpy as np
from typing import Any, Optional, List
from abc import ABC


@dataclass
class ArtifactMarkers(ABC):
    """Abstract base class for artifact location data."""
    pass

@dataclass
class ArtifactTriggers(ArtifactMarkers):
    """Represents artifacts as a 1D array of trigger points (start times)."""
    starts: List[List[np.ndarray]] # Shape: (n_artifacts,)

@dataclass
class ArtifactWindows(ArtifactMarkers):
    """Represents artifacts as a 2D array of [start, end] intervals."""
    intervals: List[List[np.ndarray]] # Shape: (n_artifacts, 2)


@dataclass
class SignalData:
    """Represents basic signal data."""
    raw_data: np.ndarray
    sampling_rate: float
    artifact_markers: ArtifactMarkers | None = field(default=None, kw_only=True)

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
    stim_params: Optional[Any]
    snr: np.ndarray
    
