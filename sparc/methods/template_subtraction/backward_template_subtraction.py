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
        artifact_markers = self.template_indices_
        
        if isinstance(artifact_markers, list):
            # If we have multiple trials, use the appropriate one
            # For now, assuming single trial or using first trial's indices
            artifact_markers = artifact_markers[0] if len(artifact_markers) > 0 else artifact_markers
        
        if not isinstance(artifact_markers, np.ndarray):
            artifact_markers = np.array(artifact_markers)
        
        for ch in range(data.shape[1]):
            signal_ch = data[:, ch]
            
            # Process each artifact location
            for i, artifact_idx in enumerate(artifact_markers):
                if artifact_idx + template_length > len(signal_ch):
                    continue
                
                if i >= self.num_templates_for_avg:
                    templates = []
                    for k in range(self.num_templates_for_avg):
                        prev_idx = artifact_markers[i - k - 1]
                        if prev_idx + template_length <= len(signal_ch):
                            templates.append(signal_ch[prev_idx:prev_idx + template_length])
                    
                    if templates:
                        avg_template = np.mean(np.array(templates), axis=0)
                        # Subtract the average template at the current artifact location
                        cleaned_data[artifact_idx:artifact_idx + template_length, ch] -= avg_template
    
        return cleaned_data
