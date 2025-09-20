import numpy as np
from typing import Dict

from sparc.core.signal_data import ArtifactMarkers
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
            print("Template length is zero, skipping template subtraction.")
            return cleaned_data

        try:
            artifact_markers = self.template_indices_[trial_idx]
        except IndexError:
            print(f"Warning: trial_idx {trial_idx} is out of bounds for artifact markers. Returning trial unmodified.")
            return cleaned_data
        
        for ch in range(data.shape[0]):
            signal_ch = data[ch, :]
            this_channel_markers = artifact_markers[ch]

            if this_channel_markers.size == 0:
                continue
            
            for i, artifact_idx in enumerate(this_channel_markers):
                if i == 0:
                    continue

                if artifact_idx + template_length > len(signal_ch):
                    continue
                
                num_available_templates = min(i, self.num_templates_for_avg)
                
                templates = []
                for k in range(num_available_templates):
                    prev_idx = this_channel_markers[i - k - 1]
                    if prev_idx + template_length <= len(signal_ch):
                        templates.append(signal_ch[prev_idx:prev_idx + template_length])
                
                if templates:
                    avg_template = np.mean(np.array(templates), axis=0)
                    
                    cleaned_data[ch, artifact_idx:artifact_idx + template_length] -= avg_template
                
        return cleaned_data
