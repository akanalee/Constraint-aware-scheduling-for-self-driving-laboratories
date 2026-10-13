import os
import yaml
from typing import Dict, List


_SUZUKI_TEMPLATE_YAML = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'DRL_scheduler_for_SDL', 'configs', 'sdl_config_suzuki.yaml'
))


def _load_base_times_from_yaml(yaml_path: str) -> Dict[str, float]:
    with open(yaml_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    out: Dict[str, float] = {}
    for step_name, step_cfg in (cfg.get('process_steps') or {}).items():
        short = step_name.split('_', 1)[0]
        if short == 'S4':
            continue
        base = float(step_cfg['processing_time']['base'])
        out[short] = base
    return out


_MACHINES_PER_STEP = {
    'S1': [f"SPU_{i}" for i in range(1, 7)],
    'S2': [f"pre_reactor_{c}" for c in "ABCDEF"],
    'S3': [f"robot_arm_{c}" for c in "AB"],
    'S4': [f"reactor_{i}" for i in range(1, 13)],
    'S5': [f"cooling_chamber_{c}" for c in "ABCDEF"],
    'S6': [f"liquid_handler_{c}" for c in "ABCDEF"],
    'S7': [f"spectrometer_{c}" for c in "AB"],
}

_BASE_TIMES_MIN: Dict[str, float] = _load_base_times_from_yaml(_SUZUKI_TEMPLATE_YAML)

_SEQUENCE = ['S1', 'S2', 'S3', 'S4', 'S5', 'S3', 'S6', 'S3', 'S7']


def build_suzuki_job(job_id: str,
                     bo_params: Dict,
                     arrival_time: float = 0.0) -> Dict:
    ligand = str(bo_params['ligand'])
    res_time = float(bo_params['res_time'])

    operations: List[Dict] = []
    for op_idx, step in enumerate(_SEQUENCE):
        machines = _MACHINES_PER_STEP[step]
        if step == 'S4':
            pt = res_time
        else:
            pt = _BASE_TIMES_MIN[step]
        processing_times = {m: pt for m in machines}
        operations.append({
            'id':                  f"{job_id}_op{op_idx}",
            'op_type':             step,
            'op_index':            op_idx,
            'processing_times':    processing_times,
            'compatible_machines': list(machines),
            'material':            ligand,
            'step':                step,
        })

    return {
        'id':           job_id,
        'type':         'SDL_experiment',
        'operations':   operations,
        'arrival_time': float(arrival_time),
        'priority':     'normal',
        'bo_params':    dict(bo_params),
        'ligand':       ligand,
    }
