import os
import yaml
from typing import Dict, List


_ALKOX_TEMPLATE_YAML = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'DRL_scheduler_for_SDL', 'configs', 'sdl_config_alkox.yaml'
))


def _load_base_times_from_yaml(yaml_path: str) -> Dict[str, float]:
    with open(yaml_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    out: Dict[str, float] = {}
    for step_name, step_cfg in (cfg.get('process_steps') or {}).items():
        short = step_name.split('_', 1)[0]
        if isinstance(step_cfg, dict) and 'processing_time' in step_cfg:
            out[short] = float(step_cfg['processing_time']['base'])
    return out


_SEQUENCE = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S4', 'S7']


_MACHINES_PER_STEP: Dict[str, List[str]] = {
    'S1': [f'liquid_handler_buffer_{i}' for i in range(1, 7)],
    'S2': [f'liquid_handler_solution_{i}' for i in range(1, 7)],
    'S3': [f'mixer_{i}' for i in range(1, 7)],
    'S4': [f'robot_arm_{c}' for c in 'AB'],
    'S5': [f'incubator_{i:02d}' for i in range(1, 13)],
    'S6': [f'cooling_chamber_{c}' for c in 'ABCDEF'],
    'S7': [f'hplc_{c}' for c in 'AB'],
}


_BASE_TIMES_MIN: Dict[str, float] = _load_base_times_from_yaml(_ALKOX_TEMPLATE_YAML)


def build_alkox_job(job_id: str,
                    bo_params: Dict,
                    arrival_time: float = 0.0) -> Dict:
    operations: List[Dict] = []
    for op_idx, step in enumerate(_SEQUENCE):
        machines = _MACHINES_PER_STEP[step]
        pt = _BASE_TIMES_MIN[step]
        processing_times = {m: pt for m in machines}
        operations.append({
            'id':                  f"{job_id}_op{op_idx}",
            'op_type':             step,
            'op_index':            op_idx,
            'processing_times':    processing_times,
            'compatible_machines': list(machines),
            'material':            'reaction_vial',
            'step':                step,
        })

    return {
        'id':           job_id,
        'type':         'SDL_experiment',
        'operations':   operations,
        'arrival_time': float(arrival_time),
        'priority':     'normal',
        'bo_params':    dict(bo_params),
    }
