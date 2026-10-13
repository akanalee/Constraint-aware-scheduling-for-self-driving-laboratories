import os
import copy
import yaml
from typing import Dict, Any, List


_LIQUID_HANDLER_BUFFER_POOL: List[Dict[str, Any]] = [
    {'id': f'liquid_handler_buffer_{i}', 'type': 'liquid_handler', 'capacity': 1}
    for i in range(1, 7)
]
_LIQUID_HANDLER_SOLUTION_POOL: List[Dict[str, Any]] = [
    {'id': f'liquid_handler_solution_{i}', 'type': 'liquid_handler', 'capacity': 1}
    for i in range(1, 7)
]
_MIXER_POOL: List[Dict[str, Any]] = [
    {'id': f'mixer_{i}', 'type': 'mixer', 'capacity': 1}
    for i in range(1, 7)
]
_ROBOT_POOL: List[Dict[str, Any]] = [
    {'id': f'robot_arm_{c}', 'type': '6dof_robot', 'capacity': 1}
    for c in 'AB'
]
_INCUBATOR_POOL: List[Dict[str, Any]] = [
    {'id': f'incubator_{i:02d}', 'type': 'incubator', 'capacity': 1}
    for i in range(1, 13)
]
_COOLING_POOL: List[Dict[str, Any]] = [
    {'id': f'cooling_chamber_{c}', 'type': 'cooling_chamber', 'capacity': 1}
    for c in 'ABCDEF'
]
_HPLC_POOL: List[Dict[str, Any]] = [
    {'id': f'hplc_{c}', 'type': 'hplc', 'capacity': 1}
    for c in 'AB'
]

_ALKOX_MACHINE_OVERRIDES: Dict[str, List[Dict[str, Any]]] = {
    'S1_buffer_prep':          _LIQUID_HANDLER_BUFFER_POOL,
    'S2_enzyme_solution_prep': _LIQUID_HANDLER_SOLUTION_POOL,
    'S3_substrate_mixing':     _MIXER_POOL,
    'S4_robot_transfer':       _ROBOT_POOL,
    'S5_enzymatic_incubation': _INCUBATOR_POOL,
    'S6_quench_sample_prep':   _COOLING_POOL,
    'S7_HPLC_quantification':  _HPLC_POOL,
}


def build_runtime_config(template_path: str, runtime_path: str = None) -> str:
    if runtime_path is None:
        d = os.path.dirname(os.path.abspath(template_path))
        runtime_path = os.path.join(d, 'sdl_config_alkox_runtime.yaml')

    with open(template_path, 'r', encoding='utf-8') as f:
        template = yaml.safe_load(f)

    runtime = copy.deepcopy(template)

    runtime.setdefault('process_steps', {})
    for step_name, machines in _ALKOX_MACHINE_OVERRIDES.items():
        runtime['process_steps'].setdefault(step_name, {})
        runtime['process_steps'][step_name]['machines'] = copy.deepcopy(machines)

    runtime['drl_runtime_override'] = {'use_config_machines': True}

    with open(runtime_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(runtime, f, sort_keys=False, default_flow_style=False)

    n_machines = len({m['id'] for ms in _ALKOX_MACHINE_OVERRIDES.values() for m in ms})
    print(f"[AlkOxConfigBuilder] Wrote {runtime_path}")
    print(f"  unique machines: {n_machines}, "
          f"types: {sorted({m['type'] for ms in _ALKOX_MACHINE_OVERRIDES.values() for m in ms})}")
    return runtime_path


if __name__ == '__main__':
    import sys
    tpl = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', 'DRL_scheduler_for_SDL', 'configs', 'sdl_config_alkox.yaml'
    )
    build_runtime_config(os.path.normpath(tpl))
