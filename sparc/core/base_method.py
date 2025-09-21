from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Optional
import inspect

from .signal_data import ArtifactMarkers


class BaseSACMethod(ABC):
    """
    Abstract base class for Stimulation Artifact Cancellation (SAC) methods.
    
    All SAC methods should inherit from this class and implement the required methods.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the SAC method.
        
        Args:
            sampling_rate: Sampling rate of the neural data (Hz)
            **kwargs: Method-specific parameters
        """
        self.sampling_rate = None
        self.params = kwargs
        self.is_fitted = False

    def set_sampling_rate(self, sampling_rate: float):
        """Set the sampling rate."""
        self.sampling_rate = sampling_rate
        return self
        
    @abstractmethod
    def fit(self, data: np.ndarray, artifact_markers: Optional[ArtifactMarkers]) -> 'BaseSACMethod':
        """
        Fit the method to the data (e.g., learn templates, compute parameters).
        
        Args:
            data: Neural data array of shape (timesteps, channels) or (trials, timesteps, channels)
            artifact_indices: Optional array indicating artifact locations
            
        Returns:
            self for method chaining
        """
        pass
    
    @abstractmethod
    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Apply artifact cancellation to the data.
        
        Args:
            data: Neural data array of shape (timesteps, channels) or (trials, timesteps, channels)
            
        Returns:
            Cleaned data array with same shape as input
        """
        pass
    
    def fit_transform(self, data: np.ndarray, artifact_indices: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fit the method and transform the data in one step.
        
        Args:
            data: Neural data array
            artifact_indices: Optional array indicating artifact locations
            
        Returns:
            Cleaned data array
        """
        return self.fit(data, artifact_indices).transform(data)
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get method parameters.
        Introspects the constructor and returns all arguments as a dictionary.
        """
        init_signature = inspect.signature(self.__class__.__init__)
        params = {}
        for param in init_signature.parameters.values():
            if param.name != 'self' and param.kind != param.VAR_KEYWORD and param.kind != param.VAR_POSITIONAL:
                if hasattr(self, param.name):
                    params[param.name] = getattr(self, param.name)
        if hasattr(self, 'params'):
            params.update(self.params)
        return params
