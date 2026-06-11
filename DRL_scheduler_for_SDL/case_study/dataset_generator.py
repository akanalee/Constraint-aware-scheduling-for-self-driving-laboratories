"""
Dataset Generator - / ()
"""

import os
import numpy as np
import yaml
from pathlib import Path
from typing import List, Dict, Tuple
from case_study.atlas_simulator import AtlasSimulator


class DatasetGenerator:
    """
    SDL
    job
    """

    def __init__(self, config_path: str = None):
        if config_path is None:
            _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(_project_root, 'configs', 'sdl_config.yaml')
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.process_steps = self.config['process_steps']
        self.job_types = self.config['job_types']
        self.tmax_constraints = self.config['tmax_constraints']

        # Atlas
        self.atlas_simulator = AtlasSimulator(self.config['atlas_config'])

        # step_short_namename  
        self.step_map = self._build_step_map()

    def _build_step_map(self) -> Dict[str, str]:
        """
        step
        'S1' -> 'S1_substrate_prep'
         step
        """
        step_map = {}
        for full_name in self.process_steps.keys():
            # short name (e.g., 'S1' from 'S1_substrate_prep')
            short_name = full_name.split('_')[0]
            step_map[short_name] = full_name
        return step_map

    def generate_job(
        self,
        job_id: str,
        job_type: str,
        arrival_time: float
    ) -> Dict:
        """
        V45S1-S7Job
         step
        """
        job_config = self.job_types[job_type]
        sequence = job_config['sequence']
        material = job_config['material']

        operations = []

        # AtlasS4
        atlas_params = self.atlas_simulator.generate_next_job_params()

        for i, step_name in enumerate(sequence):
            #  step_map
            # step_name  'S1', 'S2_robots', 'S4_CVD' 
            if '_' in step_name:
                # machine hint 'S4_CVD'
                short_name = step_name.split('_')[0]  # 'S4'
                machine_hint = step_name.split('_')[1]  # 'CVD'
            else:
                # short name 'S1', 'S2'
                short_name = step_name
                machine_hint = None

            # step name  
            if short_name not in self.step_map:
                print(f" Warning: Step '{short_name}' not found in process_steps")
                continue

            full_step_name = self.step_map[short_name]
            step_info = self.process_steps[full_step_name]

            # processing times
            base_time = step_info['processing_time']['base']
            std = step_info['processing_time']['std']
            distribution = step_info['processing_time'].get('distribution', 'normal')

            processing_times = {}

            for mach_info in step_info['machines']:
                machine_id = mach_info['id']

                #  machine_hint
                if machine_hint:
                    # hint
                    if machine_hint.lower() not in machine_id.lower():
                        continue

                # 
                if distribution == 'atlas_modulated' and short_name == 'S4':
                    time = atlas_params.get('S4_processing_time', base_time)
                elif distribution == 'normal':
                    time = np.random.normal(base_time, std)
                else:
                    time = base_time * np.random.uniform(1 - std / base_time, 1 + std / base_time)

                processing_times[machine_id] = max(0.5, time)
                #print(processing_times)


            if not processing_times:
                print(f"============================================================")
                print(f" : Op {job_id}_op{i + 1} (Step: {full_step_name})  processing_times ")
                print(
                    f"    step_info['machines']  machine_hint='{machine_hint}' ")
                # 
                print(f"    (step_info['machines']): {step_info['machines']}")
                print(f"============================================================")

            #  5material
            if processing_times:  # compatible
                operations.append({
                    'id': f"{job_id}_op{i + 1}",
                    'step': short_name,
                    'processing_times': processing_times,
                    'material': material,  # V45
                    'op_type': short_name
                })

        if operations and operations[0]['id'] == f"{job_id}_op1":
            op1_data = operations[0]
            """
            print(f"\n====================== JOB GENERATION DIAGNOSIS ======================")
            print(f"Job ID: {job_id}, Type: {job_type}")
            print(f"Step Name: {op1_data['step']} -> {self.step_map.get(op1_data['step'], 'Unknown')}")
            print(f"Operation ID: {op1_data['id']}")
            print(f"Processing Times Keys (Compatible Machines): {op1_data['processing_times'].keys()}")
            print(f"Processing Times Dict: {op1_data['processing_times']}")

            # 
            full_step_name = self.step_map.get(op1_data['step'], None)
            if full_step_name:
                step_info = self.process_steps[full_step_name]
                print(f"Original Config Machines (YAML): {[m['id'] for m in step_info['machines']]}")
            print(f"============================================================")"""

        return {
            'id': job_id,
            'type': job_type,
            'arrival_time': arrival_time,
            'priority': job_config['priority'],
            'operations': operations
        }

    def generate_job_arrivals(
        self,
        n_jobs: int,
        load_factor: float,
        seed: int = None
    ) -> List[float]:
        """
        V4Poisson
        V4V4jobAtlas
        
        """
        # V4Poisson
        # V4
        return []

    def generate_dataset(
        self,
        scale: str,
        n_instances: int = 10,
        seed_offset: int = 0
    ) -> List[Dict]:
        """
        

        Args:
            scale: 'small', 'medium', 'large'
            n_instances: 
            seed_offset: train/test

        Returns:
            dataset: List[{
                'instance_id': str,
                'jobs': List[Job],
                'machines': List[Machine],
                'constraints': List[Constraint]
            }]
        """
        scale_config = self.config['experiment_scales'][scale]
        n_jobs = scale_config['n_jobs']
        load_factor = scale_config['load_factor']
        job_type_weights = scale_config.get('job_type_weights', [0.4, 0.4, 0.2])

        dataset = []

        for inst_idx in range(n_instances):
            seed = seed_offset + inst_idx
            #np.random.seed(seed)

            # V4Poisson
            arrival_times = self.generate_job_arrivals(n_jobs, load_factor, seed)

            # arrival_timesV4
            if not arrival_times:
                arrival_times = [float(i * 5) for i in range(n_jobs)]

            # jobs
            jobs = []
            job_type_list = list(self.job_types.keys())

            for i, arrival_time in enumerate(arrival_times):
                # job
                job_type = np.random.choice(job_type_list, p=job_type_weights)
                job_id = f"job_{inst_idx}_{i}"

                job = self.generate_job(job_id, job_type, arrival_time)

                #  joboperations
                if job['operations']:
                    jobs.append(job)

            # 
            machines = self._extract_machines()

            instance = {
                'instance_id': f"{scale}_{inst_idx}",
                'scale': scale,
                'n_jobs': len(jobs),  #  job
                'load_factor': load_factor,
                'jobs': jobs,
                'machines': machines,
                'constraints': self.tmax_constraints,
                'seed': seed
            }

            dataset.append(instance)

        return dataset

    def _extract_machines(self) -> List[Dict]:
        """process_steps"""
        machines = []
        seen_ids = set()  #  

        for step_name, step_info in self.process_steps.items():
            for mach_info in step_info['machines']:
                machine_id = mach_info['id']

                #  
                if machine_id in seen_ids:
                    continue
                seen_ids.add(machine_id)

                machines.append({
                    'id': machine_id,
                    'type': mach_info['type'],
                    'capacity': mach_info.get('capacity', 1),
                    'step': step_name
                })
        return machines

    def save_dataset(
        self,
        dataset: List[Dict],
        output_path: str
    ):
        """"""
        import pickle

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'wb') as f:
            pickle.dump(dataset, f)

        print(f" Dataset saved to {output_path}")
        print(f"  Total instances: {len(dataset)}")
        if dataset:
            print(f"  Scale: {dataset[0]['scale']}")
            print(f"  Jobs per instance: {dataset[0]['n_jobs']}")

    @staticmethod
    def load_dataset(path: str) -> List[Dict]:
        """"""
        import pickle

        with open(path, 'rb') as f:
            dataset = pickle.load(f)

        print(f" Dataset loaded from {path}")
        print(f"  Total instances: {len(dataset)}")

        return dataset


# CLI
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate SDL scheduling dataset')
    parser.add_argument('--scale', type=str, choices=['small', 'medium', 'large'],
                       default='medium', help='Dataset scale')
    parser.add_argument('--n_instances', type=int, default=10,
                       help='Number of instances')
    parser.add_argument('--seed_offset', type=int, default=0,
                       help='Seed offset for train/test split')
    _proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument('--output', type=str,
                       default=os.path.join(_proj_root, 'data', 'dataset.pkl'),
                       help='Output path')

    args = parser.parse_args()

    generator = DatasetGenerator()
    dataset = generator.generate_dataset(
        scale=args.scale,
        n_instances=args.n_instances,
        seed_offset=args.seed_offset
    )
    generator.save_dataset(dataset, args.output)