from typing import Dict

import numpy as np


class AtlasSimulator:
    def __init__(self, config: Dict):
        self.config = config
        self.param_bounds = config['parameter_space']
        self.reset()

    def reset(self):
        self.job_counter = 0
        self.rng = np.random.RandomState(None)

    def generate_next_job_params(self) -> Dict:
        self.job_counter += 1

        params = self._sample_random_params()
        processing_time = self._sample_reaction_time()

        return {
            'S4_processing_time': processing_time,
            'temperature': params['temperature'],
            'pressure': params['pressure'],
            'flow_rate': params['flow_rate'],
            'job_id': self.job_counter
        }

    def _sample_random_params(self) -> Dict:
        return {
            'temperature': np.random.uniform(*self.param_bounds['temperature']),
            'pressure': np.random.uniform(*self.param_bounds['pressure']),
            'flow_rate': np.random.uniform(*self.param_bounds['flow_rate'])
        }

    def _sample_reaction_time(self) -> float:
        return float(np.clip(self.rng.normal(13.0, 2.75), 10.0, 17.0))
