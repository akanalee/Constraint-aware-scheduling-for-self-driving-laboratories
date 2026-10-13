import os
from typing import Dict

import numpy as np
import yaml

from case_study.atlas_simulator import AtlasSimulator


class DatasetGenerator:
    def __init__(self, config_path: str = None):
        if config_path is None:
            _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(_project_root, 'configs', 'sdl_config.yaml')
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.process_steps = self.config['process_steps']
        self.job_types = self.config['job_types']

        self.atlas_simulator = AtlasSimulator(self.config['atlas_config'])

        self.step_map = self._build_step_map()

    def _build_step_map(self) -> Dict[str, str]:
        step_map = {}
        for full_name in self.process_steps.keys():
            short_name = full_name.split('_')[0]
            step_map[short_name] = full_name
        return step_map

    def generate_job(
        self,
        job_id: str,
        job_type: str,
        arrival_time: float
    ) -> Dict:
        job_config = self.job_types[job_type]
        sequence = job_config['sequence']
        material = job_config['material']

        operations = []

        atlas_params = self.atlas_simulator.generate_next_job_params()

        for i, step_name in enumerate(sequence):
            if '_' in step_name:
                short_name = step_name.split('_')[0]
                machine_hint = step_name.split('_')[1]
            else:
                short_name = step_name
                machine_hint = None

            if short_name not in self.step_map:
                print(f"Warning: Step '{short_name}' not found in process_steps")
                continue

            full_step_name = self.step_map[short_name]
            step_info = self.process_steps[full_step_name]

            base_time = step_info['processing_time']['base']
            std = step_info['processing_time']['std']
            distribution = step_info['processing_time'].get('distribution', 'normal')

            processing_times = {}

            for mach_info in step_info['machines']:
                machine_id = mach_info['id']

                if machine_hint:
                    if machine_hint.lower() not in machine_id.lower():
                        continue

                if distribution == 'atlas_modulated' and short_name == 'S4':
                    time = atlas_params.get('S4_processing_time', base_time)
                elif distribution == 'normal':
                    time = np.random.normal(base_time, std)
                else:
                    time = base_time * np.random.uniform(1 - std / base_time, 1 + std / base_time)

                processing_times[machine_id] = max(0.5, time)

            if not processing_times:
                print(f"Warning: no compatible machine for {job_id}_op{i + 1} "
                      f"(step {full_step_name}, machine hint '{machine_hint}')")
                continue

            operations.append({
                'id': f"{job_id}_op{i + 1}",
                'step': short_name,
                'processing_times': processing_times,
                'material': material,
                'op_type': short_name
            })

        return {
            'id': job_id,
            'type': job_type,
            'arrival_time': arrival_time,
            'priority': job_config['priority'],
            'operations': operations
        }
