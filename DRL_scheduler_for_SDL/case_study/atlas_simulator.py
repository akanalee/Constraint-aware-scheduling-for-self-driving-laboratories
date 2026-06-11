import numpy as np
from typing import Dict, List
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel


class AtlasSimulator:
    """
    Atlas
    S4
    """

    def __init__(self, config: Dict):
        self.config = config
        self.reset()  #   reset
        self.history = []  # [(params, observed_quality), ...]
        self.best_params = None
        self.best_quality = -np.inf

        # GP
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            alpha=1e-6,
            normalize_y=True
        )

        # 
        self.param_bounds = config['parameter_space']

        # 
        self.exploration_schedule = config['exploration_schedule']

        self.job_counter = 0
        #np.random.seed(config['seed'])
        #self.rng = np.random.RandomState(config['seed'])
        self.rng = np.random.RandomState(None)

    def reset(self):
        """ """
        self.history = []
        self.best_params = None
        self.best_quality = -np.inf
        self.job_counter = 0

        #  GP 
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            alpha=1e-6,
            normalize_y=True
        )

        # episode
        self.rng = np.random.RandomState(None)

    def generate_next_job_params(self) -> Dict:
        """
        job
        Returns: {'S4_processing_time': float, 'temperature': float, ...}
        """
        self.job_counter += 1

        # 
        phase = self._get_current_phase()
        variation = phase['param_variation']

        if len(self.history) < 5:
            # 
            params = self._sample_random_params()
        else:
            # GP
            params = self._sample_gp_guided_params(variation)

        # S4
        processing_time = self._params_to_processing_time(params)

        return {
            'S4_processing_time': processing_time,
            'temperature': params['temperature'],
            'pressure': params['pressure'],
            'flow_rate': params['flow_rate'],
            'job_id': self.job_counter
        }

    def update_with_result(self, params: Dict, quality: float):
        """
        GP
        quality: makespan + 
        """
        self.history.append((params, quality))

        if quality > self.best_quality:
            self.best_quality = quality
            self.best_params = params

        # GP5
        if len(self.history) % 5 == 0 and len(self.history) >= 10:
            X = np.array([[p['temperature'], p['pressure'], p['flow_rate']]
                          for p, _ in self.history])
            y = np.array([q for _, q in self.history])
            self.gp.fit(X, y)

    def _get_current_phase(self) -> Dict:
        """job"""
        for phase_name, phase_config in self.exploration_schedule.items():
            start, end = phase_config['jobs_range']
            if start <= self.job_counter < end:
                return phase_config
        # 
        return list(self.exploration_schedule.values())[-1]

    def _sample_random_params(self) -> Dict:
        """"""
        return {
            'temperature': np.random.uniform(*self.param_bounds['temperature']),
            'pressure': np.random.uniform(*self.param_bounds['pressure']),
            'flow_rate': np.random.uniform(*self.param_bounds['flow_rate'])
        }

    def _sample_gp_guided_params(self, variation: float) -> Dict:
        """
        GP
        variation: 0-1
        """
        if self.best_params is None:
            return self._sample_random_params()

        # 
        params = {}
        for key in ['temperature', 'pressure', 'flow_rate']:
            lower, upper = self.param_bounds[key]
            best_val = self.best_params[key]
            range_size = (upper - lower) * variation

            new_val = np.random.normal(best_val, range_size / 3)  # 3-sigma
            new_val = np.clip(new_val, lower, upper)
            params[key] = new_val

        return params

    def _params_to_processing_time(self, params: Dict) -> float:
        """
        
        time = f(temperature, pressure, flow_rate)
        """
        # 
        base_time = 11.0  # minutes

        # 
        temp_factor = 1.0 - (params['temperature'] - 400) / (600 - 400) * 0.3

        # 
        pressure_factor = 1.0 + 0.2 * abs(params['pressure'] - 5.5) / 4.5

        # 
        flow_factor = 1.0 - (params['flow_rate'] - 10) / (100 - 10) * 0.2

        # 
        noise = self.rng.normal(0, 0.1)

        processing_time = base_time * temp_factor * pressure_factor * flow_factor * (1 + noise)

        # 
        return np.clip(processing_time, 10.0, 17.0)

    def get_statistics(self) -> Dict:
        """"""
        if not self.history:
            return {}

        times = [self._params_to_processing_time(p) for p, _ in self.history]
        qualities = [q for _, q in self.history]

        return {
            'n_jobs_completed': self.job_counter,
            'best_quality': self.best_quality,
            'best_params': self.best_params,
            'avg_processing_time': np.mean(times),
            'std_processing_time': np.std(times),
            'quality_improvement': (qualities[-1] - qualities[0]) if len(qualities) > 1 else 0
        }