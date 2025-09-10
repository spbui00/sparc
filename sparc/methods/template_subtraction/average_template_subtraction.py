import numpy as np
from typing import Dict, List
from .base import BaseTemplateSubtraction


class AverageTemplateSubtraction(BaseTemplateSubtraction):
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

        for ch in range(data.shape[1]):
            signal_ch = data[:, ch]
            num_cycles = len(signal_ch) // template_length
            last_avg_template = np.zeros(template_length)

            if num_cycles > self.num_templates_for_avg:
                for i in range(num_cycles - self.num_templates_for_avg):
                    start_idx = i * template_length
                    
                    templates = []
                    for k in range(self.num_templates_for_avg):
                        template_start = start_idx + k * template_length
                        template_end = template_start + template_length
                        templates.append(signal_ch[template_start:template_end])
                    
                    avg_template = np.mean(np.array(templates), axis=0)
                    last_avg_template = avg_template

                    cleaned_data[start_idx:start_idx+template_length, ch] -= avg_template

                for i in range(num_cycles - self.num_templates_for_avg, num_cycles):
                    start_idx = i * template_length
                    cleaned_data[start_idx:start_idx+template_length, ch] -= last_avg_template
        
        return cleaned_data
