import argparse
import csv
import heapq
import math
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS)
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'DRL_scheduler_for_SDL'))
sys.path.insert(0, _THIS)

from chemos_integration.file_protocol import (
    init_comm_dirs, cleanup_comm_dirs,
    ScheduleCommand, write_schedule_command,
    read_completion_events, mark_event_processed,
    set_drl_ready, is_chemos_ready, set_episode_complete,
    CLOCK_STATE_FILE, read_prev_clock_epoch,
)
from chemos_integration.global_clock import GlobalClock
from chemos_integration.suzuki_config_builder import build_runtime_config
from case_study.suzuki_simulator_bnn import SuzukiReactionSimulator
from chemos_integration.suzuki_bo_optimizer import (
    SuzukiBOOptimizer, _WARMUP_PLAN,
)
from chemos_integration.suzuki_job_factory import (
    build_suzuki_job, _BASE_TIMES_MIN as _FACTORY_BASE_TIMES_MIN,
)


class _NoiselessCachingSuzukiSim:
    _WARMUP_YIELDS: Dict[Tuple, Tuple[float, float, float]] = {}
    _NOISELESS_CACHE: Dict[Tuple, Tuple[float, float, float]] = {}
    _CACHE_PARAM_DECIMALS = 9

    def __init__(self, seed: int):
        self._inner = SuzukiReactionSimulator(seed=seed)
        self._base_seed = int(seed)

    @classmethod
    def _param_key(cls, params: Dict) -> Tuple:
        return (
            str(params['ligand']),
            round(float(params['res_time']),         cls._CACHE_PARAM_DECIMALS),
            round(float(params['temperature']),      cls._CACHE_PARAM_DECIMALS),
            round(float(params['catalyst_loading']), cls._CACHE_PARAM_DECIMALS),
        )

    @classmethod
    def _cache_key(cls, base: int, params: Dict) -> Tuple:
        return (int(base),) + cls._param_key(params)

    @classmethod
    def clear_noiseless_cache(cls) -> None:
        cls._NOISELESS_CACHE.clear()

    @classmethod
    def _ensure_warmup_yields_loaded(cls) -> None:
        if cls._WARMUP_YIELDS:
            return
        for entry in _WARMUP_PLAN:
            key = cls._param_key(entry)
            cls._WARMUP_YIELDS[key] = (
                float(entry['_warmup_yield']),
                0.0,
                float(entry['res_time']),
            )

    def query_noiseless(self, params: Dict) -> Dict:
        pkey = self._param_key(params)
        warm = self._WARMUP_YIELDS.get(pkey)
        if warm is not None:
            y, ton, rt = warm
            return {'yield_pct': y, 'turnover': ton, 'reaction_time_min': rt}
        key = (int(self._base_seed),) + pkey
        cached = self._NOISELESS_CACHE.get(key)
        if cached is not None:
            y, ton, rt = cached
            return {'yield_pct': y, 'turnover': ton, 'reaction_time_min': rt}
        result = self._inner.query_noiseless(params)
        self._NOISELESS_CACHE[key] = (
            float(result['yield_pct']),
            float(result['turnover']),
            float(result['reaction_time_min']),
        )
        return result

    def __getattr__(self, name):
        return getattr(self._inner, name)


_NoiselessCachingSuzukiSim._ensure_warmup_yields_loaded()

import yaml


SUZUKI_TEMPLATE_YAML = os.path.join(
    _PROJECT_ROOT, 'DRL_scheduler_for_SDL', 'configs', 'sdl_config_suzuki.yaml'
)
MODEL_PATH = os.path.join(
    _PROJECT_ROOT, 'DRL_scheduler_for_SDL', 'training',
    'saved_models', 'SUZUKI.pt'
)
STOP_FILE = os.path.join(_PROJECT_ROOT, 'STOP')
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, 'outputs', 'case_study_suzuki')

FIXED_PIPELINE_MIN: float = (
    _FACTORY_BASE_TIMES_MIN['S1'] + _FACTORY_BASE_TIMES_MIN['S2']
    + _FACTORY_BASE_TIMES_MIN['S3'] + _FACTORY_BASE_TIMES_MIN['S5']
    + _FACTORY_BASE_TIMES_MIN['S3'] + _FACTORY_BASE_TIMES_MIN['S6']
    + _FACTORY_BASE_TIMES_MIN['S3'] + _FACTORY_BASE_TIMES_MIN['S7']
)

_NATIVE_MACHINE_IDS: Dict[str, List[str]] = {
    'S1': [f'SPU_{i}' for i in range(1, 7)],
    'S2': [f'pre_reactor_{c}' for c in 'ABCDEF'],
    'S3': [f'robot_arm_{c}' for c in 'AB'],
    'S4': [f'reactor_{i}' for i in range(1, 13)],
    'S5': [f'cooling_chamber_{c}' for c in 'ABCDEF'],
    'S6': [f'liquid_handler_{c}' for c in 'ABCDEF'],
    'S7': [f'spectrometer_{c}' for c in 'AB'],
}
_NATIVE_MACHINE_COUNTS: Dict[str, int] = {
    s: len(ids) for s, ids in _NATIVE_MACHINE_IDS.items()
}

_STAGE_PROC_TIME: Dict[str, Optional[float]] = {
    **_FACTORY_BASE_TIMES_MIN,
    'S4': None,
}

_PIPELINE: List[str] = ['S1', 'S2', 'S3', 'S4', 'S5', 'S3', 'S6', 'S3', 'S7']


def _tmax_steps_from_yaml(yaml_path: str) -> List[Tuple[int, int, float]]:
    with open(yaml_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    constraints = (cfg.get('env_config') or {}).get('tmax_constraints') or []
    out: List[Tuple[int, int, float]] = []
    for c in constraints:
        f_tag, t_tag = c['from_step'], c['to_step']
        try:
            i = _PIPELINE.index(f_tag)
            j = _PIPELINE.index(t_tag, i + 1)
        except ValueError:
            continue
        out.append((i, j, float(c['max_interval'])))
    return out


_TMAX_STEPS: List[Tuple[int, int, float]] = _tmax_steps_from_yaml(SUZUKI_TEMPLATE_YAML)


class SuzukiERCFSSController:
    def __init__(self, runtime_config_path: str, cfg: Dict, time_scale: float):
        self.runtime_config_path = runtime_config_path
        self.cfg = cfg
        self.bo_cfg = cfg['suzuki_bo_config']
        env_cfg = cfg['env_config']
        self.total_budget = int(self.bo_cfg.get('total_budget', env_cfg.get('n_jobs', 30)))
        self.n_warmup = int(self.bo_cfg.get('n_warmup', 12))
        self.time_scale = time_scale

        self.scheduler = None
        self._scheduler_loaded = False

        self.bo: Optional[SuzukiBOOptimizer] = None
        self.sim: Optional[SuzukiReactionSimulator] = None
        self.clock: Optional[GlobalClock] = None
        self.seed: int = 0
        self.episode_idx: int = 0
        self.id_prefix: str = "e0_ERCFSS"
        self.job_counter: int = 0

        self.job_buffer: List[Dict] = []
        self.released_jobs: Dict[str, Dict] = {}
        self.completed_ops = set()
        self.completed_jobs = set()
        self.dispatched_count = 0
        self.active_op_schedule: Dict[str, Tuple[float, str, float]] = {}
        self.op_records: Dict[str, Dict] = {}
        self._schedule_compensation_min = 0.0
        self._scheduled_last_op_end: Dict[str, float] = {}

        self._last_bo_wall_sec: float = 0.0

        self.stats = self._empty_stats()

    @staticmethod
    def _empty_stats() -> Dict:
        return {
            'release_count': 0,
            'total_solve_time': 0.0,
            'total_ppo_inference_time': 0.0,
            'total_wall_time': 0.0,
            'total_bo_wall_time_s': 0.0,
            'time_breakdown': {'hla_decision': 0.0, 'feasibility_check': 0.0,
                               'time_advance': 0.0, 'bo_delay': 0.0},
            'total_makespan': 0.0,
            'update_points': 0,
            'tmax_violations': 0,
        }

    def reset_for_episode(self, ep_idx: int, seed: int,
                          bo: SuzukiBOOptimizer,
                          sim: SuzukiReactionSimulator,
                          id_prefix: str):
        self.episode_idx = ep_idx
        self.seed = seed
        self.bo = bo
        self.sim = sim
        self.id_prefix = id_prefix
        self.job_counter = 0
        new_epoch = read_prev_clock_epoch() + 1
        self.clock = GlobalClock(
            time_scale=self.time_scale,
            state_file=CLOCK_STATE_FILE,
            epoch=new_epoch,
        )

        self.job_buffer.clear()
        self.released_jobs.clear()
        self.completed_ops.clear()
        self.completed_jobs.clear()
        self.dispatched_count = 0
        self.active_op_schedule.clear()
        self.op_records.clear()
        self._schedule_compensation_min = 0.0
        self._scheduled_last_op_end.clear()
        self.stats = self._empty_stats()

        try:
            from chemos_integration.file_protocol import (
                SCHEDULE_CMD_DIR, OP_COMPLETE_DIR,
            )
            for d in (SCHEDULE_CMD_DIR, OP_COMPLETE_DIR):
                if os.path.isdir(d):
                    for fn in os.listdir(d):
                        try:
                            os.remove(os.path.join(d, fn))
                        except OSError:
                            pass
            if os.path.exists(STOP_FILE):
                os.remove(STOP_FILE)
        except Exception as e:
            print(f"[ERC-FSS ep{ep_idx}] IPC dir cleanup warning: {e}")

        if self._scheduler_loaded and self.scheduler is not None:
            try:
                self.scheduler.env.reset(skip_warmstart=True)
                self.scheduler._raw_time_getter = self.clock.get_current_time
                try:
                    self.scheduler.clear_pending_jobs()
                except Exception:
                    pass
                try:
                    self.scheduler.stats = {
                        'total_releases': 0,
                        'total_solve_time': 0.0,
                        'avg_solve_time': 0.0,
                        'total_ops_scheduled': 0,
                        'total_ppo_inference_time': 0.0,
                    }
                except Exception:
                    pass
                print(f"[ERC-FSS ep{ep_idx}] scheduler env.reset OK")
            except Exception as e:
                print(f"[ERC-FSS ep{ep_idx}] env.reset failed ({e}); rebuilding scheduler")
                self._scheduler_loaded = False
                self.scheduler = None

    def _load_scheduler(self):
        if self._scheduler_loaded:
            return
        print("[Suzuki] Loading ERC-FSS scheduler...")
        from chemos_integration.drl_scheduler_wrapper import DRLSchedulerWrapper
        self.scheduler = DRLSchedulerWrapper(
            model_path=MODEL_PATH,
            config_path=self.runtime_config_path,
            current_time_getter=self.clock.get_current_time,
        )
        self._scheduler_loaded = True
        print("[Suzuki] Scheduler ready.")

    def cold_start(self):
        print(f"\n[ERC-FSS ep{self.episode_idx}] Cold start: {self.n_warmup} LHS warmstart jobs")
        for params in self.bo.ask(self.n_warmup):
            self._dispatch_new_job(params, warmstart=True)

    def _dispatch_new_job(self, bo_params: Dict, warmstart: bool = False):
        job_id = f"{self.id_prefix}_job_{self.job_counter:04d}"
        self.job_counter += 1
        bo_params['job_id'] = job_id
        arrival_time_min = self.clock.get_current_time() / 60.0
        job = build_suzuki_job(job_id, bo_params, arrival_time=arrival_time_min)
        job['_warmstart'] = warmstart
        self.job_buffer.append(job)
        self.released_jobs[job_id] = job
        self.dispatched_count += 1

    def _handle_op_complete(self, event):
        op_id = event.op_id
        self.completed_ops.add(op_id)
        self.op_records[op_id] = {
            'job_id': event.job_id,
            'machine': event.machine_id,
            'start': event.start_time,
            'end': event.end_time,
        }
        if op_id in self.active_op_schedule:
            del self.active_op_schedule[op_id]
        if self._scheduler_loaded and self.scheduler:
            try:
                graph = self.scheduler.env.graph
                op = graph.operations.get(op_id)
                if op:
                    op.is_completed = True
                    op.is_scheduled = True
                    op.scheduled_machine = event.machine_id
                    op.scheduled_start = float(event.start_time)
                    op.scheduled_end = float(event.end_time)
                    graph.scheduled_ops.add(op_id)
            except Exception:
                pass
    def _handle_job_complete(self, event):
        if event.job_id in self.completed_jobs:
            return
        self.completed_jobs.add(event.job_id)
        self._on_job_complete_bo(event.job_id)

    def _on_job_complete_bo(self, job_id: str):
        job = self.released_jobs.get(job_id)
        if job is None:
            return
        bo_params = job['bo_params']
        result = self.sim.query_noiseless({
            'ligand': bo_params['ligand'],
            'res_time': bo_params['res_time'],
            'temperature': bo_params['temperature'],
            'catalyst_loading': bo_params['catalyst_loading'],
        })
        yield_pct = float(result['yield_pct'])

        tell_params = dict(bo_params); tell_params['job_id'] = job_id
        self.bo.tell(tell_params, yield_pct)

        last_op_id = f"{job_id}_op{len(_PIPELINE) - 1}"
        if last_op_id in self.op_records:
            virtual_time_min = self.op_records[last_op_id]['end']
        elif job_id in self._scheduled_last_op_end:
            virtual_time_min = self._scheduled_last_op_end[job_id]
            print(f"  [WARN] {last_op_id} not in op_records; "
                  f"using scheduled S7 end {virtual_time_min:.2f}min")
        else:
            job_ends = [v['end'] for k, v in self.op_records.items()
                        if k.startswith(f"{job_id}_op")]
            virtual_time_min = (max(job_ends) if job_ends
                                else self.clock.get_current_time_minutes())
            print(f"  [WARN] No S7 end found for {job_id}; "
                  f"fallback virtual_time={virtual_time_min:.2f}min")

        self.bo.updates[-1]['virtual_time_min'] = virtual_time_min

        if self.dispatched_count >= self.total_budget:
            return

        self.clock.pause()
        t0 = time.time()
        new_params = self.bo.ask(1)[0]
        bo_wall = time.time() - t0
        self.clock.resume_with_delay(bo_wall)
        self._last_bo_wall_sec = bo_wall
        self.stats['total_bo_wall_time_s'] += bo_wall
        self.stats['time_breakdown']['bo_delay'] += bo_wall
        self._dispatch_new_job(new_params, warmstart=False)

    def _has_work_for_scheduler(self) -> bool:
        return len(self.job_buffer) > 0

    def _sync_env_with_active(self):
        if not (self._scheduler_loaded and self.scheduler):
            return
        machine_floor: Dict[str, float] = {}
        for _oid, (_s, _m, _e) in self.active_op_schedule.items():
            if _e > machine_floor.get(_m, 0.0):
                machine_floor[_m] = _e
        for m_obj in self.scheduler.env.machines:
            floor = machine_floor.get(m_obj.id, 0.0)
            if m_obj.available_time < floor:
                m_obj.available_time = floor

    def execute_progressive_release(self) -> Optional[Dict]:
        if not self._has_work_for_scheduler():
            return None
        self._load_scheduler()

        print(f"\n[ERC-FSS ep{self.episode_idx}] Progressive Release "
              f"(t={self.clock.get_current_time() / 60:.2f}min, "
              f"buffer={len(self.job_buffer)})")

        self.scheduler.clear_pending_jobs()
        for job in self.job_buffer:
            self.scheduler.add_job_to_buffer(job)

        self._sync_env_with_active()

        pre_virt_min = self.clock.get_current_time_minutes()
        was_idle = not self.active_op_schedule

        self.clock.pause()
        release_start = time.time()
        try:
            result = self.scheduler.execute_progressive_release()
        except Exception:
            self.clock.resume_with_delay(time.time() - release_start)
            raise
        release_elapsed = time.time() - release_start
        solve_time_sec_for_clock = 0.0 if was_idle else release_elapsed
        if not result.get('success'):
            self.clock.resume_with_delay(solve_time_sec_for_clock)
        self.stats['time_breakdown']['hla_decision'] += release_elapsed
        self.stats['time_breakdown']['feasibility_check'] += result.get('feasibility_check_time', 0.0)
        self.stats['total_solve_time'] += release_elapsed
        self.stats['total_ppo_inference_time'] += result.get('ppo_inference_time', 0.0)

        if not result.get('success'):
            print(f"  Infeasible (solve={release_elapsed:.2f}s)")
            return None

        solve_time_sec = release_elapsed if not was_idle else 0.0
        solve_time_min = solve_time_sec / 60.0
        self._schedule_compensation_min = solve_time_min
        compensated = [
            (start + solve_time_min, op_id, machine, end + solve_time_min)
            for start, op_id, machine, end in result['schedule']
        ]

        in_batch_ids = {op_id for _, op_id, _, _ in compensated}
        machine_floor: Dict[str, float] = {}
        for _oid, (_s, _m, _e) in self.active_op_schedule.items():
            if _oid in in_batch_ids:
                continue
            if _e > machine_floor.get(_m, 0.0):
                machine_floor[_m] = _e

        by_machine: Dict[str, List[Tuple[float, str, str, float]]] = {}
        for tup in compensated:
            by_machine.setdefault(tup[2], []).append(tup)
        for m in by_machine:
            by_machine[m].sort(key=lambda t: (t[0], t[1]))

        clamped_map: Dict[str, Tuple[float, str, str, float]] = {
            t[1]: t for t in compensated
        }
        for _pass in range(8):
            changed = False
            for m, ops_on_m in by_machine.items():
                floor = machine_floor.get(m, 0.0)
                for i, (start, op_id, machine, end) in enumerate(ops_on_m):
                    cur = clamped_map[op_id]
                    duration = cur[3] - cur[0]
                    new_start = max(cur[0], floor)
                    if new_start > cur[0] + 1e-9:
                        new_end = new_start + duration
                        clamped_map[op_id] = (new_start, op_id, machine, new_end)
                        changed = True
                        floor = new_end
                    else:
                        floor = max(floor, cur[3])
                    ops_on_m[i] = clamped_map[op_id]
            for op_id, cur in list(clamped_map.items()):
                start, _oid, machine, end = cur
                duration = end - start
                job_id = op_id.rsplit('_op', 1)[0] if '_op' in op_id else op_id
                try:
                    op_idx = int(op_id.rsplit('_op', 1)[1])
                except (ValueError, IndexError):
                    op_idx = -1
                if op_idx <= 0:
                    continue
                pred_id = f"{job_id}_op{op_idx - 1}"
                if pred_id in clamped_map:
                    pred_end = clamped_map[pred_id][3]
                else:
                    pe = self.active_op_schedule.get(pred_id)
                    pred_end = pe[2] if pe is not None else 0.0
                if pred_end > start + 1e-9:
                    new_start = pred_end
                    new_end = new_start + duration
                    clamped_map[op_id] = (new_start, op_id, machine, new_end)
                    changed = True
                    for i, t in enumerate(by_machine.get(machine, [])):
                        if t[1] == op_id:
                            by_machine[machine][i] = clamped_map[op_id]
                            break
                    by_machine[machine].sort(key=lambda t: (t[0], t[1]))
            if not changed:
                break

        clamped = [clamped_map[op_id] for _, op_id, _, _ in compensated]
        if any(abs(c[0] - o[0]) > 1e-6 for c, o in zip(clamped, compensated)):
            n_shifted = sum(1 for c, o in zip(clamped, compensated)
                            if abs(c[0] - o[0]) > 1e-6)
            max_shift = max(c[0] - o[0] for c, o in zip(clamped, compensated))
            print(f"  [CAPACITY CLAMP] shifted {n_shifted}/{len(clamped)} ops "
                  f"forward (max +{max_shift:.3f}min) to enforce machine capacity")
        compensated = clamped

        if self._scheduler_loaded and self.scheduler is not None:
            graph = self.scheduler.env.graph
            machines = getattr(graph, 'machines', {}) or {}
            for start, op_id, machine, end in compensated:
                op_obj = graph.operations.get(op_id)
                if op_obj is not None:
                    op_obj.scheduled_start = start
                    op_obj.scheduled_end = end
                m_obj = machines.get(machine)
                if m_obj is not None and getattr(m_obj, 'available_time', 0.0) < end:
                    m_obj.available_time = end

        compat_by_op: Dict[str, List[str]] = {}
        for j in self.released_jobs.values():
            for o in j.get('operations', []) or []:
                cm = o.get('compatible_machines')
                if cm:
                    compat_by_op[o['id']] = list(cm)

        bad_assign: List[str] = []
        for start, op_id, machine, end in compensated:
            self.active_op_schedule[op_id] = (start, machine, end)
            cm = compat_by_op.get(op_id)
            if cm is not None and machine not in cm:
                bad_assign.append(f"{op_id} -> {machine} (allowed: {cm})")
            _last_suffix = f"_op{len(_PIPELINE) - 1}"
            if op_id.endswith(_last_suffix):
                self._scheduled_last_op_end[op_id[:-len(_last_suffix)]] = end
        if bad_assign:
            print(f"  [SCHEDULE CONFLICT] {len(bad_assign)} ops assigned to "
                  f"machines outside their compatible pool:")
            for _b in bad_assign[:8]:
                print(f"    {_b}")

        _mach_windows: Dict[str, List[Tuple[float, float, str]]] = {}
        for _oid, (_s, _m, _e) in self.active_op_schedule.items():
            _mach_windows.setdefault(_m, []).append((_s, _e, _oid))
        _conflicts: List[str] = []
        for _m, _wins in _mach_windows.items():
            _wins.sort()
            for _i in range(len(_wins) - 1):
                _s1, _e1, _o1 = _wins[_i]; _s2, _e2, _o2 = _wins[_i + 1]
                if _s2 < _e1 - 1e-6:
                    _conflicts.append(
                        f"{_m}: {_o1}[{_s1:.2f},{_e1:.2f}] vs {_o2}[{_s2:.2f},{_e2:.2f}]"
                    )
        if _conflicts:
            print(f"  [SCHEDULE CONFLICT] {len(_conflicts)} overlapping machine "
                  f"slots across active schedule: capacity violation")
            for _c in _conflicts[:8]:
                print(f"    {_c}")

        released_ids = {j['id'] if isinstance(j, dict) else j.id
                        for j in result['released_jobs']}
        self.job_buffer = [j for j in self.job_buffer if j['id'] not in released_ids]

        cmd = ScheduleCommand(
            schedule=compensated,
            released_jobs=list(released_ids),
            makespan=result['makespan'],
            solve_time=release_elapsed,
        )
        self.clock.resume_with_delay(solve_time_sec)
        write_schedule_command(cmd)

        self.stats['release_count'] += 1
        self.stats['update_points'] += 1
        self.stats['total_makespan'] = max(
            self.stats['total_makespan'], result['makespan']
        )
        print(f"  Released {len(released_ids)}, k={result['k_value']}, "
              f"ms={result['makespan']:.2f}min, solve={release_elapsed:.3f}s "
              f"-> compensation={solve_time_sec:.3f}s "
              f"({solve_time_min:.3f}min)  "
              f"clock={pre_virt_min + solve_time_min:.3f}min")
        return {
            'success': True, 'released_jobs': list(released_ids),
            'schedule': compensated, 'makespan': result['makespan'],
            'solve_time': release_elapsed,
            'command_id': cmd.command_id,
        }

    def advance_time_to_next_event(self):
        current_time_sec = self.clock.get_current_time()
        next_event_sec = None
        for start_min, machine, end_min in self.active_op_schedule.values():
            end_sec = end_min * 60.0
            if end_sec > current_time_sec:
                if next_event_sec is None or end_sec < next_event_sec:
                    next_event_sec = end_sec
        if next_event_sec is not None:
            self.clock.advance_to(next_event_sec)

    def is_episode_complete(self) -> bool:
        return len(self.completed_jobs) >= self.total_budget

    def run_episode(self) -> Dict:
        print(f"\n{'#' * 60}\n# ERC-FSS Episode {self.episode_idx} (seed={self.seed})\n{'#' * 60}")
        self._load_scheduler()
        self.clock.reset()
        self.cold_start()
        episode_start = time.time()

        warmup_releases = 0
        warmup_stuck = 0
        while self._has_work_for_scheduler():
            buf_before = len(self.job_buffer)
            result = self.execute_progressive_release()
            warmup_releases += 1
            if result is None or not result.get('success'):
                warmup_stuck = buf_before
                print(f"  [WARMUP] Burst loop stopped (ERC-FSS infeasible): "
                      f"{warmup_stuck} job(s) left in buffer; will be released "
                      f"on first rolling-horizon trigger.")
                break
            if len(self.job_buffer) >= buf_before:
                warmup_stuck = len(self.job_buffer)
                print(f"  [WARMUP] Burst loop stopped (no progress): "
                      f"buffer {buf_before} -> {len(self.job_buffer)}")
                break
        released_in_burst = self.n_warmup - warmup_stuck
        if warmup_releases > 1 and warmup_stuck == 0:
            print(f"  [WARMUP] All {self.n_warmup} warmstart jobs released in "
                  f"{warmup_releases} progressive-release calls.")
        elif warmup_releases >= 1:
            print(f"  [WARMUP] Burst released {released_in_burst}/{self.n_warmup} "
                  f"warmstart jobs in {warmup_releases} call(s); "
                  f"{warmup_stuck} deferred to rolling-horizon.")

        step = 0
        while not self.is_episode_complete():
            step += 1
            if os.path.exists(STOP_FILE):
                os.remove(STOP_FILE)
                print("[STOP] STOP file detected.")
                break

            events = read_completion_events()
            op_events, job_events, stale_events = [], [], []
            for ev in events:
                if ev.job_id not in self.released_jobs:
                    stale_events.append(ev)
                elif ev.event_type == 'op_complete':
                    op_events.append(ev)
                elif ev.event_type == 'job_complete':
                    job_events.append(ev)

            for ev in stale_events:
                mark_event_processed(ev)

            op_events.sort(key=lambda ev: (float(ev.end_time), ev.op_id))
            released_this_tick = False
            for ev in op_events:
                self._handle_op_complete(ev)
                mark_event_processed(ev)
                if self._has_work_for_scheduler():
                    res = self.execute_progressive_release()
                    if res is not None and res.get('success'):
                        released_this_tick = True

            def _last_op_end(ev):
                last_op_id = f"{ev.job_id}_op{len(_PIPELINE) - 1}"
                if last_op_id in self.op_records:
                    return self.op_records[last_op_id]['end']
                ends = [v['end'] for k, v in self.op_records.items()
                        if k.startswith(f"{ev.job_id}_op")]
                return max(ends) if ends else float('inf')

            job_events.sort(key=_last_op_end)

            n_evs = len(job_events)
            for i, ev in enumerate(job_events):
                if ev.job_id in self.completed_jobs:
                    mark_event_processed(ev); continue
                self._handle_job_complete(ev)
                mark_event_processed(ev)
                if not self._has_work_for_scheduler():
                    continue
                bo_wall_sec = self._last_bo_wall_sec
                if i == n_evs - 1:
                    fire = True
                else:
                    next_ev = job_events[i + 1]
                    gap_virt_min = _last_op_end(next_ev) - _last_op_end(ev)
                    fire = gap_virt_min > bo_wall_sec / 60.0
                if fire:
                    self.execute_progressive_release()
                    released_this_tick = True

            if not released_this_tick and self._has_work_for_scheduler():
                self.execute_progressive_release()

            adv_start = time.time()
            self.advance_time_to_next_event()
            self.stats['time_breakdown']['time_advance'] += time.time() - adv_start
            time.sleep(0.1 * self.time_scale)
        self.stats['total_wall_time'] = time.time() - episode_start

        while True:
            flush = [ev for ev in read_completion_events()
                     if ev.job_id in self.released_jobs]
            if not flush:
                break
            flush.sort(key=lambda ev: (float(ev.end_time), ev.op_id))
            got_new = False
            for ev in flush:
                if ev.event_type == 'op_complete':
                    self._handle_op_complete(ev)
                    got_new = True
                mark_event_processed(ev)
            if not got_new:
                break

        if self.op_records:
            actual_makespan = max(r['end'] for r in self.op_records.values())
        else:
            actual_makespan = 0.0

        tmax_violations: List[Dict] = []
        if self._scheduler_loaded and self.scheduler is not None:
            try:
                tmax_violations = self.scheduler._check_tmax_violations()
            except Exception as e:
                print(f"[ERC-FSS ep{self.episode_idx}] tmax check failed: {e}")

        return {
            'episode': self.episode_idx,
            'mode': 'ERCFSS',
            'seed': self.seed,
            'completed_jobs': len(self.completed_jobs),
            'makespan_min': actual_makespan,
            'ercfss_makespan_min': self.stats['total_makespan'],
            'wall_time_s': self.stats['total_wall_time'],
            'bo_wall_time_s': self.stats['total_bo_wall_time_s'],
            'total_solve_time': self.stats['total_solve_time'],
            'feasibility_check_time': self.stats['time_breakdown']['feasibility_check'],
            'tmax_violations': len(tmax_violations),
            'tmax_violation_details': tmax_violations,
            'aborted_count': 0,
            'bo_summary': self.bo.summary(),
            'op_records': dict(self.op_records),
        }


def _open_csv_safely(preferred_path: str):
    try:
        return open(preferred_path, 'w', newline='', encoding='utf-8'), preferred_path
    except PermissionError:
        root, ext = os.path.splitext(preferred_path)
        alt = f"{root}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"
        print(f"[WARN] {preferred_path} locked; writing to {alt}")
        return open(alt, 'w', newline='', encoding='utf-8'), alt


MEAS_RSD = 0.02

_LHS_WARMSTART: List[Dict] = [
    {'ligand': 'L4', 'res_time': 3.058482, 'temperature': 49.376306, 'catalyst_loading': 2.412417},
    {'ligand': 'L3', 'res_time': 6.725547, 'temperature': 61.193652, 'catalyst_loading': 1.316260},
    {'ligand': 'L5', 'res_time': 4.324363, 'temperature': 69.792701, 'catalyst_loading': 2.510347},
    {'ligand': 'L6', 'res_time': 4.799179, 'temperature': 93.588878, 'catalyst_loading': 1.350583},
    {'ligand': 'L2', 'res_time': 2.207029, 'temperature': 52.987063, 'catalyst_loading': 1.860573},
    {'ligand': 'L1', 'res_time': 1.974286, 'temperature': 59.497357, 'catalyst_loading': 1.541567},
    {'ligand': 'L4', 'res_time': 5.499235, 'temperature': 98.010014, 'catalyst_loading': 2.076071},
    {'ligand': 'L1', 'res_time': 4.229744, 'temperature': 75.191954, 'catalyst_loading': 2.460855},
    {'ligand': 'L2', 'res_time': 2.480499, 'temperature': 91.944313, 'catalyst_loading': 2.047022},
    {'ligand': 'L3', 'res_time': 1.274209, 'temperature': 109.072502, 'catalyst_loading': 1.254491},
    {'ligand': 'L0', 'res_time': 7.510890, 'temperature': 97.521601, 'catalyst_loading': 1.191122},
    {'ligand': 'L6', 'res_time': 2.861497, 'temperature': 98.503397, 'catalyst_loading': 2.228437},
]


class Operator:
    def __init__(self, kind: str, strength: float, label: str):
        self.kind = kind
        self.strength = float(strength)
        self.label = label

    def apply(self, sim, params: Dict, y_true: float, dt: float, tau=None,
              rng=None, z=None) -> float:
        if self.kind == 'none' or dt <= 0.0:
            return y_true
        t = tau if (tau and tau > 0) else max(dt, 1e-9)
        h = 1.0 - math.exp(-dt / t)
        scale = max(abs(y_true), 1.0)
        mean = y_true * (1.0 - self.strength * h)
        sd = MEAS_RSD * scale
        if z is not None:
            eps = sd * float(z)
        elif rng is not None:
            eps = float(rng.normal(0.0, sd))
        else:
            eps = 0.0
        return float(max(0.0, mean + eps))


class SuzukiAddedArm:
    def __init__(self, cfg: Dict, time_scale: float):
        self.cfg = cfg
        self.bo_cfg = cfg['suzuki_bo_config']
        env_cfg = cfg['env_config']
        self.total_budget = int(self.bo_cfg.get('total_budget', env_cfg.get('n_jobs', 40)))
        self.n_warmup = int(self.bo_cfg.get('n_warmup', 12))
        self.time_scale = time_scale

    def run(self, ep, seed, bo, sim, id_prefix, policy, operator) -> Dict:
        assert policy in ('abort', 'noabort')
        op = operator
        noise_rng = np.random.RandomState(int(seed) & 0x7fffffff)
        noise_Z = np.random.RandomState(int(seed) & 0x7fffffff).standard_normal(1024)

        P = _PIPELINE
        last_op = len(P) - 1
        proc_table = _STAGE_PROC_TIME
        mach_ids = _NATIVE_MACHINE_IDS
        mach_counts = _NATIVE_MACHINE_COUNTS
        tmax_steps = _TMAX_STEPS
        sec = [(fi, ti, mx) for fi, ti, mx in tmax_steps if ti == last_op]
        from_sec = sec[0][0] if sec else None
        mx_sec = sec[0][2] if sec else None

        mach = {s: [0.0] * n for s, n in mach_counts.items()}
        job_params: Dict[str, Dict] = {}
        step_end: Dict[str, Dict] = {}
        job_hold: Dict[str, float] = {}
        in_pipeline, aborted_set, warmup_jids = set(), set(), set()
        completed_count = [0]; aborted_count = [0]; violated_completed = [0]
        makespan_min = [0.0]; job_ctr = [0]; tc = [0]
        solve_acc = [0.0]
        eq: List = []; pending: List = []; pending_seq = [0]; records: List[Dict] = []
        op_log: List[Dict] = []

        def push(t, jid, sidx):
            heapq.heappush(eq, (t, tc[0], jid, sidx)); tc[0] += 1

        def push_pending(ready, jid, sidx):
            pending.append((ready, pending_seq[0], jid, sidx)); pending_seq[0] += 1

        def pick_idle_machine(stage, t_now):
            for i, av in enumerate(mach[stage]):
                if av <= t_now + 1e-4:
                    return i
            return None

        def tmax_excess_at_now(jid, sidx, t_now):
            for fi, ti, mx in tmax_steps:
                if ti != sidx:
                    continue
                prev = step_end.get(jid, {}).get(fi)
                if prev is None:
                    continue
                gap = t_now - prev
                if gap > mx + 1e-4:
                    return True, f"{P[fi]}->{P[ti]}", gap - mx
            return False, None, 0.0

        def try_dispatch_op(jid, sidx, t_now):
            stage = P[sidx]
            violated, cname, excess = tmax_excess_at_now(jid, sidx, t_now)
            if violated and policy == 'abort':
                return 'aborted', cname, excess
            best_i = pick_idle_machine(stage, t_now)
            if best_i is None:
                return 'wait', None, 0.0
            if (sidx == last_op and from_sec is not None
                    and from_sec in step_end.get(jid, {})):
                job_hold[jid] = t_now - step_end[jid][from_sec]
            proc = (float(job_params[jid]['res_time']) if stage == 'S4'
                    else proc_table[stage])
            t_done = t_now + proc
            mach[stage][best_i] = t_done
            step_end.setdefault(jid, {})[sidx] = t_done
            push(t_done, jid, sidx)
            op_log.append({
                'job_id': jid, 'op_id': f'{jid}_op{sidx}', 'op_index': sidx,
                'machine': mach_ids[stage][best_i], 'stage': stage,
                'start': float(t_now), 'end': float(t_done)})
            return 'dispatched', None, 0.0

        def dispatch(params, arrival, warmup=False):
            jid = f"{id_prefix}_job_{job_ctr[0]:04d}"; job_ctr[0] += 1
            job_params[jid] = params; step_end[jid] = {}; in_pipeline.add(jid)
            if warmup:
                warmup_jids.add(jid)
            push_pending(arrival, jid, 0)

        def replace_aborted(jid_aborted, t_now):
            if len(in_pipeline) + completed_count[0] >= self.total_budget:
                return
            dispatch(dict(job_params[jid_aborted]), t_now,
                     warmup=(jid_aborted in warmup_jids))

        def maybe_dispatch_next(t_now):
            if len(in_pipeline) + completed_count[0] >= self.total_budget:
                return
            t0 = time.time(); nxt = bo.ask(1)[0]; bw = time.time() - t0
            dispatch(nxt, t_now + bw / 60.0)

        def drain_pending(t_now):
            while True:
                pending.sort(key=lambda x: (x[0], x[1]))
                progress = False; keep = []
                for ready, seq, jid, sidx in pending:
                    if jid in aborted_set or jid not in in_pipeline:
                        continue
                    if ready > t_now + 1e-4:
                        keep.append((ready, seq, jid, sidx)); continue
                    _ts = time.perf_counter()
                    status, _c, _e = try_dispatch_op(jid, sidx, t_now)
                    solve_acc[0] += time.perf_counter() - _ts
                    if status == 'dispatched':
                        progress = True
                    elif status == 'aborted':
                        in_pipeline.discard(jid); aborted_set.add(jid)
                        aborted_count[0] += 1; replace_aborted(jid, t_now); progress = True
                    else:
                        keep.append((ready, seq, jid, sidx))
                pending[:] = keep
                if not progress:
                    break

        def on_job_complete(jid, t_now):
            params = job_params[jid]
            y_true = float(sim.query_noiseless({
                'ligand': params['ligand'], 'res_time': params['res_time'],
                'temperature': params['temperature'],
                'catalyst_loading': params['catalyst_loading']})['yield_pct'])
            hold = float(job_hold.get(jid, 0.0))
            dt = hold
            try:
                _jN = int(jid.rsplit('_', 1)[-1])
            except Exception:
                _jN = len(records)
            y_obs = op.apply(sim, params, y_true, dt, mx_sec, noise_rng,
                             z=noise_Z[_jN % len(noise_Z)])
            if mx_sec is not None and hold > mx_sec:
                violated_completed[0] += 1
            tp = dict(params); tp['job_id'] = jid
            bo.tell(tp, y_obs)
            u = bo.updates[-1]
            u['time'] = t_now
            u['hold_min'] = hold; u['delta_t'] = dt
            best_obs_now = float(u['best_so_far'])
            print(f"  [DONE] {jid}  t={t_now:.2f}min  y_obs={float(y_obs):.2f}%  "
                  f"hold={hold:.1f}min  best={best_obs_now:.2f}%  "
                  f"({completed_count[0]}/{self.total_budget})")
            records.append({
                'step': len(bo.updates), 'job_id': jid, 'time': t_now,
                'ligand': params['ligand'], 'res_time': float(params['res_time']),
                'temperature': float(params['temperature']),
                'catalyst_loading': float(params['catalyst_loading']),
                'hold_min': hold, 'delta_t': dt, 'y_obs': float(y_obs),
                'best_obs': float(u['best_so_far'])})

        print(f"[added-arm:{op.label}] policy={policy}  budget={self.total_budget}  "
              f"warmup={self.n_warmup}  (self-contained DES)")
        for p in bo.ask(self.n_warmup):
            dispatch(p, 0.0, warmup=True)
        drain_pending(0.0)

        while completed_count[0] < self.total_budget and (eq or pending):
            if not eq:
                drain_pending(min(p[0] for p in pending))
                if not eq:
                    break
                continue
            t_now, _, jid, sidx = heapq.heappop(eq)
            if jid in aborted_set or jid not in in_pipeline:
                continue
            if sidx == last_op:
                in_pipeline.discard(jid)
                if t_now > makespan_min[0]:
                    makespan_min[0] = t_now
                completed_count[0] += 1
                on_job_complete(jid, t_now); maybe_dispatch_next(t_now)
            else:
                push_pending(t_now, jid, sidx + 1)
            drain_pending(t_now)

        print(f"[added-arm:{op.label}] done  completed={completed_count[0]}/{self.total_budget}  "
              f"violated(hold>Tmax)={violated_completed[0]}  makespan={makespan_min[0]:.2f}min")
        obs_vals = [r['y_obs'] for r in records]
        best_obs = max(obs_vals) if obs_vals else 0.0
        return {
            'policy': policy, 'operator': op.label, 'episode': ep, 'seed': seed,
            'completed_jobs': completed_count[0],
            'aborted_count': aborted_count[0], 'violated_completed': violated_completed[0],
            'makespan_min': makespan_min[0], 'best_obs': best_obs,
            'total_solve_time': solve_acc[0],
            'records': records, 'op_log': op_log, 'bo_summary': bo.summary()}


class NoiseTellBO:
    def __init__(self, inner, operator, noise_rng, hold_fn, mx_sec, noise_z=None):
        self._inner = inner
        self._op = operator
        self._rng = noise_rng
        self._holdfn = hold_fn
        self._mx = mx_sec
        self._best_obs = -1e30
        self._Z = noise_z

    def tell(self, params, y_clean):
        jid = params.get('job_id') if isinstance(params, dict) else None
        hold = 0.0
        try:
            hold = float(self._holdfn(jid)) if jid is not None else 0.0
        except Exception:
            hold = 0.0
        dt = hold
        zj = None
        if self._Z is not None and jid is not None:
            try:
                zj = float(self._Z[int(jid.rsplit('_', 1)[-1]) % len(self._Z)])
            except Exception:
                zj = None
        y_obs = self._op.apply(None, params, float(y_clean), dt, self._mx, self._rng, z=zj)
        self._inner.tell(params, y_obs)
        self._best_obs = max(self._best_obs, float(y_obs))
        ups = getattr(self._inner, 'updates', None)
        if ups:
            ups[-1]['y_obs'] = float(y_obs)
            ups[-1]['hold_min'] = hold
            ups[-1]['best_so_far'] = self._best_obs

    def __getattr__(self, name):
        return getattr(self._inner, name)


class SuzukiExperimentRunner:
    _SCHED_POLICY_NAME = {
        'ERCFSS': 'ERC-FSS',
        'abort': 'ChemOS built-in dispatch with termination',
        'noabort': 'ChemOS built-in dispatch with no termination',
    }

    def __init__(self, episodes: int, base_seed: int, time_scale: float):
        self.episodes = episodes
        self.base_seed = base_seed
        self.time_scale = time_scale

        self.runtime_config_path = build_runtime_config(SUZUKI_TEMPLATE_YAML)
        with open(self.runtime_config_path, 'r', encoding='utf-8') as f:
            self.cfg = yaml.safe_load(f)
        self.bo_cfg = self.cfg['suzuki_bo_config']

        self.out_dir = OUTPUT_DIR
        os.makedirs(self.out_dir, exist_ok=True)

        self.ercfss = SuzukiERCFSSController(self.runtime_config_path, self.cfg, self.time_scale)
        _NoiselessCachingSuzukiSim.clear_noiseless_cache()

        _ref = _NoiselessCachingSuzukiSim(seed=1)

        def _wy(p):
            return float(_ref.query_noiseless({k: p[k] for k in
                ('ligand', 'res_time', 'temperature', 'catalyst_loading')})['yield_pct'])

        self._warmstart_plan = sorted((dict(p) for p in _LHS_WARMSTART), key=_wy)
        self.added = SuzukiAddedArm(self.cfg, self.time_scale)
        self.corrupt_out_dir = self.out_dir
        self._mx_sec = next((mx for fi, ti, mx in _TMAX_STEPS
                             if ti == len(_PIPELINE) - 1), None)

    def _new_bo(self, seed: int, sim) -> SuzukiBOOptimizer:
        return SuzukiBOOptimizer(self.bo_cfg, seed=seed, simulator=sim,
                                 warmup_plan=[dict(p) for p in self._warmstart_plan])

    def _new_sim(self, seed: int):
        return _NoiselessCachingSuzukiSim(seed=seed)

    def _ercfss_hold(self, jid):
        if jid is None:
            return 0.0
        P = _PIPELINE; last = len(P) - 1
        sec = [(fi, ti, mx) for fi, ti, mx in _TMAX_STEPS if ti == last]
        if not sec:
            return 0.0
        fi = sec[0][0]
        rec = getattr(self.ercfss, 'op_records', {}) or {}
        q = rec.get(f"{jid}_op{fi}")
        h = rec.get(f"{jid}_op{last}")
        if q is None:
            return 0.0
        hs = (h['start'] if h is not None
              else getattr(self.ercfss, '_scheduled_last_op_end', {}).get(jid))
        return max(0.0, float(hs) - float(q['end'])) if hs is not None else 0.0

    def _write_best_curve(self, tag, times, obs_yields, makespan,
                          solve_time=0.0, tmax_viol=0):
        path = os.path.join(self.corrupt_out_dir, f"suzuki_truecurve_{tag}.csv")
        b = -1e30; rows = []
        for t, y in sorted(zip(times, obs_yields), key=lambda z: z[0]):
            b = max(b, float(y)); rows.append((float(t), b))
        f, path = _open_csv_safely(path)
        with f:
            w = csv.writer(f)
            w.writerow(['time', 'best_so_far', 'makespan_min',
                        'total_solving_time_s', 'tmax_violations'])
            for i, (t, bb) in enumerate(rows):
                if i == 0:
                    w.writerow([f"{t:.4f}", f"{bb:.4f}", f"{makespan:.4f}",
                                f"{solve_time:.4f}", int(tmax_viol)])
                else:
                    w.writerow([f"{t:.4f}", f"{bb:.4f}", '', '', ''])
        print(f"[Suzuki] best-obs curve ({len(rows)} pts, final={rows[-1][1]:.2f}) -> {path}")

    def _write_steps_curve(self, tag, rows, makespan, solve_time=0.0, tmax_viol=0):
        path = os.path.join(self.corrupt_out_dir, f"suzuki_steps_{tag}.csv")
        srt = sorted(rows, key=lambda r: r['time'])
        f, path = _open_csv_safely(path)
        with f:
            w = csv.writer(f)
            w.writerow(['step', 'time', 'y_obs', 'best_obs_running',
                        'makespan_min', 'total_solving_time_s', 'tmax_violations'])
            best_yo = -1e30
            for i, r in enumerate(srt):
                yo = float(r['y_obs'])
                best_yo = max(best_yo, yo)
                if i == 0:
                    w.writerow([i + 1, f"{float(r['time']):.4f}", f"{yo:.4f}",
                                f"{best_yo:.4f}", f"{makespan:.4f}",
                                f"{solve_time:.4f}", int(tmax_viol)])
                else:
                    w.writerow([i + 1, f"{float(r['time']):.4f}", f"{yo:.4f}",
                                f"{best_yo:.4f}", '', '', ''])
        print(f"[Suzuki] steps ({len(srt)} pts) best_obs={best_yo:.2f} -> {path}")

    def _export_sched9(self, policy, lbl, ercfss_res=None, op_log=None, makespan=0.0,
                       solve_time=0.0, tmax_viol=0, aborted_count=0):
        import json as _json
        from case_study_export import _ercfss_section, _chemos_abort_section
        human = self._SCHED_POLICY_NAME[policy]
        sched_dir = os.path.join(self.corrupt_out_dir, f"sched9_{human}_{lbl}")
        os.makedirs(sched_dir, exist_ok=True)
        if ercfss_res is not None:
            section = _ercfss_section(ercfss_res)
        else:
            nres = {'makespan_min': makespan, 'op_log': op_log or [],
                    'abort_replace_pairs': [], 'aborted_count': int(aborted_count),
                    'total_solve_time': float(solve_time), 'wall_time_s': 0.0}
            section = _chemos_abort_section(nres)
            section['tmax_violations'] = int(tmax_viol)
        payload = {'timestamp': datetime.now().isoformat(), human: section}
        fpath = os.path.join(sched_dir, 'schedule.json')
        with open(fpath, 'w', encoding='utf-8') as f:
            _json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[Suzuki] schedule -> {fpath}")

    def _write_9grid_summary(self, rows):
        path = os.path.join(self.corrupt_out_dir, 'suzuki_9grid_schedule_summary.csv')
        f, path = _open_csv_safely(path)
        with f:
            w = csv.writer(f)
            w.writerow(['severity', 'policy', 'makespan_min',
                        'total_solving_time_s', 'tmax_violations', 'completed_jobs'])
            for r in rows:
                w.writerow(r)
        print(f"[Suzuki] schedule summary ({len(rows)} rows) -> {path}")

    def run_ninegrid(self):
        levels = [('noise_10pct', 0.10), ('noise_20pct', 0.20), ('noise_30pct', 0.30)]
        seed = self.base_seed
        print(f"\n{'#'*64}\n# Suzuki case study: 3 degradation levels x 3 policies, "
              f"seed={seed}\n{'#'*64}")
        init_comm_dirs()
        print("[Suzuki] Waiting for ChemOS ...")
        wait = 0
        while not is_chemos_ready():
            time.sleep(1); wait += 1
            if wait % 10 == 0:
                print(f"  [Suzuki] waiting ({wait}s) ...")
            if wait > 300:
                print("[Suzuki] ChemOS did not respond in 300s; aborting.")
                cleanup_comm_dirs(); return
        set_drl_ready()
        print("[Suzuki] ChemOS connected.")
        summary_rows = []
        try:
            for lbl, s in levels:
                sev = f"{int(round(s * 100))}%"
                op = Operator('noise', s, lbl)
                print(f"\n{'='*60}\n===== degradation level {sev}: ERC-FSS / with termination / "
                      f"without termination =====\n{'='*60}")
                sim = self._new_sim(seed)
                inner_bo = self._new_bo(seed, sim)
                _Z = np.random.RandomState(seed & 0x7fffffff).standard_normal(1024)
                bo = NoiseTellBO(inner_bo, op, np.random.RandomState(seed & 0x7fffffff),
                                 self._ercfss_hold, self._mx_sec, noise_z=_Z)
                self.ercfss.reset_for_episode(0, seed, bo, sim, f"e0_ERCFSS_{lbl}")
                res = self.ercfss.run_episode()
                ercfss_ms = float(res.get('makespan_min', 0.0))
                ercfss_solve = float(res.get('total_solve_time', 0.0))
                ercfss_viol = int(res.get('tmax_violations', 0))
                times = [u.get('virtual_time_min', 0.0) for u in inner_bo.updates]
                oys = [float(u.get('y_obs', 0.0) or 0.0) for u in inner_bo.updates]
                self._write_best_curve(f"ERCFSS_{lbl}", times, oys, ercfss_ms, ercfss_solve, ercfss_viol)
                ercfss_steps = [{'time': u.get('virtual_time_min', 0.0),
                              'y_obs': float(u.get('y_obs', 0.0) or 0.0)}
                             for u in inner_bo.updates]
                self._write_steps_curve(f"ERCFSS_{lbl}", ercfss_steps, ercfss_ms, ercfss_solve, ercfss_viol)
                self._export_sched9('ERCFSS', lbl, ercfss_res=res)
                summary_rows.append([sev, 'ERC-FSS', f"{ercfss_ms:.3f}",
                                     f"{ercfss_solve:.4f}", ercfss_viol, len(inner_bo.updates)])
                for policy in ('abort', 'noabort'):
                    sim2 = self._new_sim(seed); bo2 = self._new_bo(seed, sim2)
                    r = self.added.run(0, seed, bo2, sim2, f"e0_{policy}_{lbl}", policy, op)
                    ms = float(r['makespan_min'])
                    solve = float(r.get('total_solve_time', 0.0))
                    viol = int(r['aborted_count'] if policy == 'abort'
                               else r['violated_completed'])
                    times2 = [rec['time'] for rec in r['records']]
                    oys2 = [rec['y_obs'] for rec in r['records']]
                    self._write_best_curve(f"{policy}_{lbl}", times2, oys2, ms, solve, viol)
                    self._write_steps_curve(f"{policy}_{lbl}", r['records'], ms, solve, viol)
                    self._export_sched9(policy, lbl, op_log=r['op_log'],
                                        makespan=ms, solve_time=solve, tmax_viol=viol,
                                        aborted_count=r['aborted_count'])
                    pol_disp = self._SCHED_POLICY_NAME[policy]
                    summary_rows.append([sev, pol_disp, f"{ms:.3f}", f"{solve:.4f}",
                                         viol, r['completed_jobs']])
            self._write_9grid_summary(summary_rows)
        finally:
            set_episode_complete()
            cleanup_comm_dirs()
            print("[Suzuki] done; comm dirs cleaned.")

    def run(self):
        self.run_ninegrid()


def main():
    parser = argparse.ArgumentParser(
        description='Suzuki-Miyaura closed-loop case study (ERC-FSS vs ChemOS built-in dispatch)')
    parser.add_argument('--base-seed', type=int, default=1)
    parser.add_argument('--time-scale', type=float, default=0.01,
                        help='Real seconds per virtual second (0.01 = 100x faster than real time)')
    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}")
        return
    if not os.path.exists(SUZUKI_TEMPLATE_YAML):
        print(f"Suzuki template not found: {SUZUKI_TEMPLATE_YAML}")
        return

    runner = SuzukiExperimentRunner(
        episodes=1, base_seed=args.base_seed, time_scale=args.time_scale)
    try:
        runner.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        cleanup_comm_dirs()
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        cleanup_comm_dirs()


if __name__ == '__main__':
    main()
