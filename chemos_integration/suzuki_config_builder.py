import os
import copy
import yaml

_BENCHMARK_YAML = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'DRL_scheduler_for_SDL', 'configs', 'sdl_config.yaml'
))

_BENCH_TO_SUZUKI_STEP = {
    'S3_pre_reaction': 'S2_mixing',
    'S5_cooling':      'S5_quench',
}

_BENCH_EXCLUDED_STEPS = {'S2_robotic_transfer', 'S6_spectroscopy', 'S7_mass_spec'}


def build_runtime_config(template_path: str, runtime_path: str = None) -> str:
    if runtime_path is None:
        d = os.path.dirname(os.path.abspath(template_path))
        runtime_path = os.path.join(d, 'sdl_config_suzuki_runtime.yaml')

    with open(template_path, 'r', encoding='utf-8') as f:
        template = yaml.safe_load(f)

    runtime = copy.deepcopy(template)

    if os.path.exists(_BENCHMARK_YAML):
        with open(_BENCHMARK_YAML, 'r', encoding='utf-8') as f:
            bench = yaml.safe_load(f)
        bench_steps = bench.get('process_steps', {})
        runtime.setdefault('process_steps', {})
        for step_name, step_info in bench_steps.items():
            if step_name in _BENCH_EXCLUDED_STEPS:
                continue
            if not isinstance(step_info, dict) or 'machines' not in step_info:
                continue
            suzuki_step = _BENCH_TO_SUZUKI_STEP.get(step_name, step_name)
            runtime['process_steps'].setdefault(suzuki_step, {})
            runtime['process_steps'][suzuki_step]['machines'] = step_info['machines']

    _SUZUKI_MACHINE_OVERRIDES = {
        'S3_robotic_transfer': [
            {'id': f'robot_arm_{c}', 'type': '6dof_robot', 'capacity': 1}
            for c in 'AB'
        ],
        'S6_sample_prep': [
            {'id': f'liquid_handler_{c}', 'type': 'liquid_handler', 'capacity': 1}
            for c in 'ABCDEF'
        ],
        'S7_HPLC_quant': [
            {'id': f'spectrometer_{c}', 'type': 'spectrometer', 'capacity': 1}
            for c in 'AB'
        ],
    }
    runtime.setdefault('process_steps', {})
    for step_name, machines in _SUZUKI_MACHINE_OVERRIDES.items():
        runtime['process_steps'].setdefault(step_name, {})
        runtime['process_steps'][step_name]['machines'] = machines

    with open(runtime_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(runtime, f, sort_keys=False, default_flow_style=False)

    print(f"[SuzukiConfigBuilder] Wrote {runtime_path}")
    return runtime_path


if __name__ == '__main__':
    import sys
    tpl = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', 'DRL_scheduler_for_SDL', 'configs', 'sdl_config_suzuki.yaml'
    )
    build_runtime_config(os.path.normpath(tpl))
