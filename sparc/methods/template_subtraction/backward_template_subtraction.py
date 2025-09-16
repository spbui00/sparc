import numpy as np
from typing import Dict
from .base import BaseTemplateSubtraction


class BackwardTemplateSubtraction(BaseTemplateSubtraction):
    def __init__(self, *args, num_templates_for_avg=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_templates_for_avg = num_templates_for_avg

    def _learn_templates(self, data: np.ndarray) -> Dict:
        return {}

    def _apply_template_subtraction_single_trial(self, data: np.ndarray, trial_idx: int) -> np.ndarray:
        cleaned_data = data.copy()
        template_length = self.template_length_samples
        
        if template_length == 0:
            return cleaned_data

        template_length = self.template_length_samples
        artifact_indices = self.template_indices_
        
        if isinstance(artifact_indices, list):
            # If we have multiple trials, use the appropriate one
            # For now, assuming single trial or using first trial's indices
            artifact_indices = artifact_indices[0] if len(artifact_indices) > 0 else artifact_indices
        
        if not isinstance(artifact_indices, np.ndarray):
            artifact_indices = np.array(artifact_indices)
        
        for ch in range(data.shape[1]):
            signal_ch = data[:, ch]
            
            # Process each artifact location
            for i, artifact_idx in enumerate(artifact_indices):
                if artifact_idx + template_length > len(signal_ch):
                    continue
                
                if i >= self.num_templates_for_avg:
                    templates = []
                    for k in range(self.num_templates_for_avg):
                        prev_idx = artifact_indices[i - k - 1]
                        if prev_idx + template_length <= len(signal_ch):
                            templates.append(signal_ch[prev_idx:prev_idx + template_length])
                    
                    if templates:
                        avg_template = np.mean(np.array(templates), axis=0)
                        # Subtract the average template at the current artifact location
                        cleaned_data[artifact_idx:artifact_idx + template_length, ch] -= avg_template
    
        return cleaned_data
