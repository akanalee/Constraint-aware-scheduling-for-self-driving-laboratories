import argparse
import csv
import heapq
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

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
from chemos_integration.alkox_config_builder import build_runtime_config
from chemos_integration.alkox_bo_optimizer import AlkOxBOOptimizer, _WARMUP_PLAN
from chemos_integration.alkox_job_factory import (
    build_alkox_job, _BASE_TIMES_MIN as _FACTORY_BASE_TIMES_MIN,
)
from case_study.alkox_simulator_bnn import AlkOxReactionSimulator

import yaml

class _NoiselessCachingAlkOxSim:
    _WARMUP_YIELDS: Dict[Tuple, float] = {}
    _CACHE_PARAM_DECIMALS = 9

    def __init__(self, seed: int):
        self._inner = AlkOxReactionSimulator(seed=seed)
        self._base_seed = int(seed)

    @classmethod
    def _param_key(cls, params: Dict) -> Tuple:
        return tuple(
            round(float(params[k]), cls._CACHE_PARAM_DECIMALS)
            for k in ('catalase', 'peroxidase', 'alcohol_oxidase', 'ph')
        )

    @classmethod
    def _ensure_warmup_yields_loaded(cls) -> None:
        if cls._WARMUP_YIELDS:
            return
        for entry in _WARMUP_PLAN:
            cls._WARMUP_YIELDS[cls._param_key(entry)] = float(entry['_warmup_conv'])

    def query_noiseless(self, params: Dict) -> Dict:
        cached = self._WARMUP_YIELDS.get(self._param_key(params))
        if cached is not None:
            return {'conversion_pct': cached}
        return self._inner.query_noiseless(params)

    def __getattr__(self, name):
        return getattr(self._inner, name)


_NoiselessCachingAlkOxSim._ensure_warmup_yields_loaded()


ALKOX_TEMPLATE_YAML = os.path.join(
    _PROJECT_ROOT, 'DRL_scheduler_for_SDL', 'configs', 'sdl_config_alkox.yaml'
)
MODEL_PATH = os.path.join(
    _PROJECT_ROOT, 'DRL_scheduler_for_SDL', 'training',
    'saved_models', 'ALKOX.pt'
)
STOP_FILE = os.path.join(_PROJECT_ROOT, 'STOP')
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, 'outputs', 'case_study_alkox')

_PIPELINE: List[str] = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S4', 'S7']
_LAST_OP_INDEX: int = len(_PIPELINE) - 1

_STAGE_PROC_TIME: Dict[str, float] = dict(_FACTORY_BASE_TIMES_MIN)

_LH_BUFFER_COUNT   = 6
_LH_SOLUTION_COUNT = 6
_MIXER_COUNT       = 6
_ROBOT_COUNT       = 2
_INCUBATOR_COUNT   = 12
_COOLING_COUNT     = 6
_HPLC_COUNT        = 2

_STAGE_TO_POOL: Dict[str, str] = {
    'S1': 'LH_BUF',
    'S2': 'LH_SOL',
    'S3': 'MIX',
    'S4': 'ROBOT',
    'S5': 'INC',
    'S6': 'COOL',
    'S7': 'HPLC',
}
_POOL_COUNTS: Dict[str, int] = {
    'LH_BUF': _LH_BUFFER_COUNT,
    'LH_SOL': _LH_SOLUTION_COUNT,
    'MIX':    _MIXER_COUNT,
    'ROBOT':  _ROBOT_COUNT,
    'INC':    _INCUBATOR_COUNT,
    'COOL':   _COOLING_COUNT,
    'HPLC':   _HPLC_COUNT,
}

_POOL_MACHINE_IDS: Dict[str, List[str]] = {
    'LH_BUF': [f'liquid_handler_buffer_{i}' for i in range(1, _LH_BUFFER_COUNT + 1)],
    'LH_SOL': [f'liquid_handler_solution_{i}' for i in range(1, _LH_SOLUTION_COUNT + 1)],
    'MIX':    [f'mixer_{i}' for i in range(1, _MIXER_COUNT + 1)],
    'ROBOT':  [f'robot_arm_{c}' for c in 'AB'][:_ROBOT_COUNT],
    'INC':    [f'incubator_{i:02d}' for i in range(1, _INCUBATOR_COUNT + 1)],
    'COOL':   [f'cooling_chamber_{c}' for c in 'ABCDEF'][:_COOLING_COUNT],
    'HPLC':   [f'hplc_{c}' for c in 'AB'][:_HPLC_COUNT],
}


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


_TMAX_STEPS: List[Tuple[int, int, float]] = _tmax_steps_from_yaml(ALKOX_TEMPLATE_YAML)


_BO_PARAM_KEYS = ('catalase', 'peroxidase', 'alcohol_oxidase', 'ph')


def _sim_kwargs(bo_params: Dict) -> Dict:
    return {k: float(bo_params[k]) for k in _BO_PARAM_KEYS}


def _format_bo(p: Dict) -> str:
    return (f"cat={float(p['catalase']):.3f} "
            f"per={float(p['peroxidase']):.2f} "
            f"AOx={float(p['alcohol_oxidase']):.2f} "
            f"pH={float(p['ph']):.2f}")


class AlkOxERCFSSController:
    def __init__(self, runtime_config_path: str, cfg: Dict, time_scale: float):
        self.runtime_config_path = runtime_config_path
        self.cfg = cfg
        self.bo_cfg = cfg['alkox_bo_config']
        env_cfg = cfg['env_config']
        self.total_budget = int(self.bo_cfg.get('total_budget', env_cfg.get('n_jobs', 40)))
        self.n_warmup = int(self.bo_cfg.get('n_warmup', 12))
        self.time_scale = time_scale

        self.scheduler = None
        self._scheduler_loaded = False

        self.bo: Optional[AlkOxBOOptimizer] = None
        self.sim: Optional[AlkOxReactionSimulator] = None
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
        self._scheduled_last_end: Dict[str, float] = {}

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
                          bo: AlkOxBOOptimizer,
                          sim: AlkOxReactionSimulator,
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
        self._scheduled_last_end.clear()
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
        print("[AlkOx] Loading ERC-FSS scheduler...")
        from chemos_integration.drl_scheduler_wrapper import DRLSchedulerWrapper
        self.scheduler = DRLSchedulerWrapper(
            model_path=MODEL_PATH,
            config_path=self.runtime_config_path,
            current_time_getter=self.clock.get_current_time,
        )
        self._scheduler_loaded = True
        print("[AlkOx] Scheduler ready.")

    def cold_start(self):
        print(f"\n[ERC-FSS ep{self.episode_idx}] Cold start: {self.n_warmup} LHS warmstart jobs")
        for params in self.bo.ask(self.n_warmup):
            self._dispatch_new_job(params, warmstart=True)

    def _dispatch_new_job(self, bo_params: Dict, warmstart: bool = False):
        job_id = f"{self.id_prefix}_job_{self.job_counter:04d}"
        self.job_counter += 1
        bo_params['job_id'] = job_id
        arrival_time_min = self.clock.get_current_time() / 60.0
        job = build_alkox_job(job_id, bo_params, arrival_time=arrival_time_min)
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
        result = self.sim.query_noiseless(_sim_kwargs(bo_params))
        conversion = float(result['conversion_pct'])

        tell_params = dict(bo_params); tell_params['job_id'] = job_id
        self.bo.tell(tell_params, conversion)

        last_op_id = f"{job_id}_op{_LAST_OP_INDEX}"
        if last_op_id in self.op_records:
            virtual_time_min = self.op_records[last_op_id]['end']
        elif job_id in self._scheduled_last_end:
            virtual_time_min = self._scheduled_last_end[job_id]
            print(f"  [WARN] {last_op_id} not in op_records; "
                  f"using scheduled end {virtual_time_min:.2f}min")
        else:
            job_ends = [v['end'] for k, v in self.op_records.items()
                        if k.startswith(f"{job_id}_op")]
            virtual_time_min = (max(job_ends) if job_ends
                                else self.clock.get_current_time_minutes())
            print(f"  [WARN] No last-op end found for {job_id}; "
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
        last_op_suffix = f"_op{_LAST_OP_INDEX}"
        for start, op_id, machine, end in compensated:
            self.active_op_schedule[op_id] = (start, machine, end)
            cm = compat_by_op.get(op_id)
            if cm is not None and machine not in cm:
                bad_assign.append(f"{op_id} -> {machine} (allowed: {cm})")
            if op_id.endswith(last_op_suffix):
                self._scheduled_last_end[op_id[:-len(last_op_suffix)]] = end
        if bad_assign:
            print(f"  [SCHEDULE CONFLICT] {len(bad_assign)} ops outside compatible pool:")
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
                        f"{_m}: {_o1}[{_s1:.2f},{_e1:.2f}] vs "
                        f"{_o2}[{_s2:.2f},{_e2:.2f}]"
                    )
        if _conflicts:
            print(f"  [SCHEDULE CONFLICT] {len(_conflicts)} overlapping machine slots")
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
              f"-> shift={solve_time_min:.3f}min "
              f"(idle={was_idle})")
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
                      f"{warmup_stuck} job(s) left in buffer.")
                break
            if len(self.job_buffer) >= buf_before:
                warmup_stuck = len(self.job_buffer)
                print(f"  [WARMUP] Burst loop stopped (no progress): "
                      f"buffer {buf_before} -> {len(self.job_buffer)}")
                break
        released_in_burst = self.n_warmup - warmup_stuck
        if warmup_releases > 1 and warmup_stuck == 0:
            print(f"  [WARMUP] All {self.n_warmup} jobs released in "
                  f"{warmup_releases} progressive-release calls.")
        elif warmup_releases >= 1:
            print(f"  [WARMUP] Burst released {released_in_burst}/{self.n_warmup} "
                  f"jobs in {warmup_releases} call(s); "
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
                last_op_id = f"{ev.job_id}_op{_LAST_OP_INDEX}"
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


class AlkOxChemosDispatchBaseline:
    def __init__(self, runtime_config_path: str, cfg: Dict, time_scale: float):
        self.cfg = cfg
        self.bo_cfg = cfg['alkox_bo_config']
        env_cfg = cfg['env_config']
        self.total_budget = int(self.bo_cfg.get('total_budget', env_cfg.get('n_jobs', 40)))
        self.n_warmup = int(self.bo_cfg.get('n_warmup', 12))
        self.time_scale = time_scale

        self.bo: Optional[AlkOxBOOptimizer] = None
        self.sim: Optional[AlkOxReactionSimulator] = None
        self.seed: int = 0
        self.episode_idx: int = 0
        self.id_prefix: str = "e0_ChemOSDispatch"
        self.stats: Dict = {}

    @staticmethod
    def _empty_stats() -> Dict:
        return {'total_wall_time': 0.0, 'total_bo_wall_time_s': 0.0,
                'total_solve_time': 0.0}

    def reset_for_episode(self, ep_idx: int, seed: int,
                          bo: AlkOxBOOptimizer,
                          sim: AlkOxReactionSimulator,
                          id_prefix: str):
        self.episode_idx = ep_idx
        self.seed = seed
        self.bo = bo
        self.sim = sim
        self.id_prefix = id_prefix
        self.stats = self._empty_stats()

    def run_episode(self) -> Dict:
        print(f"\n{'#' * 60}\n# ChemOS built-in dispatch Episode {self.episode_idx} (seed={self.seed})\n{'#' * 60}")
        print(f"[ChemOS-dispatch ep{self.episode_idx}] budget={self.total_budget}  "
              f"warmup={self.n_warmup}  "
              f"tmax=enforced(abort+replace)")
        episode_start = time.time()

        pool: Dict[str, List[float]] = {
            name: [0.0] * cnt for name, cnt in _POOL_COUNTS.items()
        }
        job_params: Dict[str, Dict] = {}
        step_end: Dict[str, Dict] = {}

        in_pipeline: set = set()
        aborted_set: set = set()
        warmup_jids: set = set()
        warmup_aborted_count = [0]
        completed_count = [0]
        aborted_count = [0]
        makespan_min = [0.0]
        job_ctr = [0]
        tc = [0]

        eq: List = []
        pending: List[Tuple[float, int, str, int]] = []
        pending_seq = [0]

        op_log: List[Dict] = []
        abort_replace_pairs: List[Dict] = []

        def push(t: float, jid: str, sidx: int) -> None:
            heapq.heappush(eq, (t, tc[0], jid, sidx))
            tc[0] += 1

        def push_pending(ready: float, jid: str, sidx: int) -> None:
            pending.append((ready, pending_seq[0], jid, sidx))
            pending_seq[0] += 1

        def pick_idle_machine(stage: str,
                              t_now: float) -> Optional[int]:
            pool_name = _STAGE_TO_POOL[stage]
            for i, av in enumerate(pool[pool_name]):
                if av <= t_now + 1e-4:
                    return i
            return None

        def tmax_check_at_now(jid: str, sidx: int,
                              t_now: float) -> Tuple[bool, Optional[str], float]:
            for fi, ti, mx in _TMAX_STEPS:
                if ti != sidx:
                    continue
                prev = step_end.get(jid, {}).get(fi)
                if prev is None:
                    continue
                gap = t_now - prev
                if gap > mx + 1e-4:
                    return True, f"{_PIPELINE[fi]}->{_PIPELINE[ti]}", gap - mx
            return False, None, 0.0

        def try_dispatch_op(jid: str, sidx: int,
                            t_now: float) -> Tuple[str, Optional[str], float]:
            stage = _PIPELINE[sidx]
            violated, cname, excess = tmax_check_at_now(jid, sidx, t_now)
            if violated:
                return 'aborted', cname, excess
            best_i = pick_idle_machine(stage, t_now)
            if best_i is None:
                return 'wait', None, 0.0
            t_start = t_now
            proc = _STAGE_PROC_TIME[stage]
            t_done = t_start + proc
            pool_name = _STAGE_TO_POOL[stage]
            pool[pool_name][best_i] = t_done
            step_end.setdefault(jid, {})[sidx] = t_done
            push(t_done, jid, sidx)
            op_log.append({
                'job_id': jid,
                'op_id': f'{jid}_op{sidx}',
                'op_index': sidx,
                'machine': _POOL_MACHINE_IDS[pool_name][best_i],
                'stage': stage,
                'start': float(t_start),
                'end': float(t_done),
            })
            return 'dispatched', None, 0.0

        def dispatch(params: Dict, arrival: float, warmup: bool = False) -> None:
            jid = f"{self.id_prefix}_job_{job_ctr[0]:04d}"
            job_ctr[0] += 1
            job_params[jid] = params
            step_end[jid] = {}
            in_pipeline.add(jid)
            if warmup:
                warmup_jids.add(jid)
            push_pending(arrival, jid, 0)
            tag = "WARMUP" if warmup else "BO"
            print(f"  [{tag}] {jid}  {_format_bo(params)}  arr={arrival:.2f}min")

        def replace_aborted(jid_aborted: str, t_now: float) -> None:
            if len(in_pipeline) + completed_count[0] >= self.total_budget:
                return
            orig_params = dict(job_params[jid_aborted])
            arrival = t_now
            replacement_jid = f"{self.id_prefix}_job_{job_ctr[0]:04d}"
            abort_replace_pairs.append({
                'terminated_job': jid_aborted,
                'replacement_job': replacement_jid,
            })
            dispatch(orig_params, arrival, warmup=(jid_aborted in warmup_jids))

        def maybe_dispatch_next(t_now: float) -> None:
            if len(in_pipeline) + completed_count[0] >= self.total_budget:
                return
            t0 = time.time()
            nxt = self.bo.ask(1)[0]
            bw = time.time() - t0
            self.stats['total_bo_wall_time_s'] += bw
            arrival = t_now + bw / 60.0
            dispatch(nxt, arrival)

        def drain_pending(t_now: float) -> None:
            _t0 = time.time()
            try:
                while True:
                    pending.sort(key=lambda x: (x[0], x[1]))
                    progress = False
                    keep: List[Tuple[float, int, str, int]] = []
                    for ready, seq, jid, sidx in pending:
                        if jid in aborted_set or jid not in in_pipeline:
                            continue
                        if ready > t_now + 1e-4:
                            keep.append((ready, seq, jid, sidx))
                            continue
                        status, cname, excess = try_dispatch_op(jid, sidx, t_now)
                        if status == 'dispatched':
                            progress = True
                        elif status == 'aborted':
                            in_pipeline.discard(jid)
                            aborted_set.add(jid)
                            aborted_count[0] += 1
                            is_warmup_abort = jid in warmup_jids
                            if is_warmup_abort:
                                warmup_aborted_count[0] += 1
                            cname_s = cname or f"step{sidx}"
                            kind = "WARMUP" if is_warmup_abort else "BO"
                            print(f"  [TMAX ABORT][{kind}] {jid}  constraint={cname_s}  "
                                  f"excess=+{excess:.2f}min  t={t_now:.2f}min")
                            replace_aborted(jid, t_now)
                            progress = True
                        else:
                            keep.append((ready, seq, jid, sidx))
                    pending[:] = keep
                    if not progress:
                        break
            finally:
                self.stats['total_solve_time'] += time.time() - _t0

        warmup_batch = list(self.bo.ask(self.n_warmup))
        warmup_batch.sort(key=lambda p: float(p.get('_warmup_conv', float('inf'))))
        for p in warmup_batch:
            dispatch(p, 0.0, warmup=True)
        drain_pending(0.0)

        while completed_count[0] < self.total_budget and (eq or pending):
            if not eq:
                next_ready = min(p[0] for p in pending)
                drain_pending(next_ready)
                if not eq:
                    break
                continue

            t_now, _, jid, sidx = heapq.heappop(eq)
            if jid in aborted_set or jid not in in_pipeline:
                continue
            params = job_params[jid]

            if sidx == _LAST_OP_INDEX:
                in_pipeline.discard(jid)
                if t_now > makespan_min[0]:
                    makespan_min[0] = t_now
                completed_count[0] += 1
                result = self.sim.query_noiseless(_sim_kwargs(params))
                conv = float(result['conversion_pct'])
                tp = dict(params); tp['job_id'] = jid
                self.bo.tell(tp, conv)
                self.bo.updates[-1]['virtual_time_min'] = t_now
                maybe_dispatch_next(t_now)
            else:
                push_pending(t_now, jid, sidx + 1)

            drain_pending(t_now)

        warmup_aborted_count = warmup_aborted_count[0]
        completed_count = completed_count[0]
        aborted_count = aborted_count[0]
        makespan_min = makespan_min[0]

        self.stats['total_wall_time'] = time.time() - episode_start

        print(f"\n{'=' * 60}")
        print(f"ChemOS built-in dispatch Episode {self.episode_idx} done  (seed={self.seed})")
        print(f"  Completed (non-aborted): {completed_count}/{self.total_budget}")
        print(f"  Tmax violations total: {aborted_count}  "
              f"(warmstart aborts: {warmup_aborted_count}, "
              f"BO aborts: {aborted_count - warmup_aborted_count})")
        print(f"  Total dispatched (incl. replacements): {job_ctr[0]}")
        print(f"  Makespan (excl. aborted ops): {makespan_min:.2f} min")
        print(f"  Wall time: {self.stats['total_wall_time']:.2f}s")
        print(f"{'=' * 60}\n")

        return {
            'episode':                self.episode_idx,
            'mode':                   'ChemOSDispatch',
            'seed':                   self.seed,
            'completed_jobs':         completed_count,
            'makespan_min':           makespan_min,
            'ercfss_makespan_min':       makespan_min,
            'wall_time_s':            self.stats['total_wall_time'],
            'bo_wall_time_s':         self.stats['total_bo_wall_time_s'],
            'total_solve_time':       self.stats['total_solve_time'],
            'tmax_violations':        aborted_count,
            'tmax_violation_details': [],
            'aborted_count':          aborted_count,
            'warmup_aborted_count':   warmup_aborted_count,
            'total_dispatched':       job_ctr[0],
            'bo_summary':             self.bo.summary(),
            'op_log':                 op_log,
            'abort_replace_pairs':    abort_replace_pairs,
        }


def _open_csv_safely(preferred_path: str):
    try:
        return open(preferred_path, 'w', newline='', encoding='utf-8'), preferred_path
    except PermissionError:
        root, ext = os.path.splitext(preferred_path)
        alt = f"{root}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"
        print(f"[WARN] {preferred_path} locked; writing to {alt}")
        return open(alt, 'w', newline='', encoding='utf-8'), alt


def save_conversion_curve(out_dir: str, ep: int, mode: str, seed: int,
                          bo_updates: List[Dict], makespan_min: float,
                          aborted_count: int = 0):
    os.makedirs(out_dir, exist_ok=True)
    preferred = os.path.join(out_dir, f'alkox_conversion_curve_e{ep}_{mode}.csv')
    fields = ['update_step', 'virtual_time_min', 'conversion_pct',
              'best_so_far', 'n_observations',
              'catalase', 'peroxidase', 'alcohol_oxidase', 'ph',
              'makespan_marker_min', 'aborted_count', 'seed', 'mode', 'episode']
    f, path = _open_csv_safely(preferred)
    with f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, u in enumerate(bo_updates, 1):
            w.writerow({
                'update_step': i,
                'virtual_time_min': u.get('virtual_time_min', ''),
                'conversion_pct': u.get('conversion_pct', ''),
                'best_so_far': u.get('best_so_far', ''),
                'n_observations': u.get('n_observations', ''),
                'catalase': u.get('catalase', ''),
                'peroxidase': u.get('peroxidase', ''),
                'alcohol_oxidase': u.get('alcohol_oxidase', ''),
                'ph': u.get('ph', ''),
                'makespan_marker_min': '', 'aborted_count': '',
                'seed': seed, 'mode': mode, 'episode': ep,
            })
        w.writerow({
            'update_step': '', 'virtual_time_min': '',
            'conversion_pct': '', 'best_so_far': '', 'n_observations': '',
            'catalase': '', 'peroxidase': '',
            'alcohol_oxidase': '', 'ph': '',
            'makespan_marker_min': makespan_min,
            'aborted_count': aborted_count,
            'seed': seed, 'mode': mode, 'episode': ep,
        })
    print(f"[ConversionCurve] e{ep}/{mode} ({len(bo_updates)} pts, "
          f"ms={makespan_min:.2f}, aborted={aborted_count}) -> {path}")


def save_comparison_summary(out_dir: str, results: List[Dict]):
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    preferred = os.path.join(out_dir, f'alkox_comparison_summary_{ts}.csv')
    fields = ['episode', 'mode', 'seed', 'completed_jobs', 'makespan_min',
              'ercfss_makespan_min', 'wall_time_s', 'bo_wall_time_s',
              'best_yield', 'final_best_so_far', 'mean_yield',
              'n_ask', 'n_tell', 'tmax_violations', 'aborted_count']
    f, path = _open_csv_safely(preferred)
    with f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            bs = r.get('bo_summary', {}) or {}
            w.writerow({
                'episode': r['episode'], 'mode': r['mode'], 'seed': r['seed'],
                'completed_jobs': r.get('completed_jobs', 0),
                'makespan_min': f"{r.get('makespan_min', 0.0):.3f}",
                'ercfss_makespan_min': f"{r.get('ercfss_makespan_min', 0.0):.3f}",
                'wall_time_s': f"{r.get('wall_time_s', 0.0):.2f}",
                'bo_wall_time_s': f"{r.get('bo_wall_time_s', 0.0):.3f}",
                'best_yield': f"{bs.get('best_yield', 0.0):.3f}",
                'final_best_so_far': f"{bs.get('final_best_so_far', 0.0):.3f}",
                'mean_yield': f"{bs.get('mean_yield', 0.0):.3f}",
                'n_ask': bs.get('n_ask', 0), 'n_tell': bs.get('n_tell', 0),
                'tmax_violations': r.get('tmax_violations', 0),
                'aborted_count': r.get('aborted_count', 0),
            })
    print(f"[Summary] {len(results)} rows -> {path}")
    return path


class AlkOxExperimentRunner:
    def __init__(self, episodes: int, base_seed: int, time_scale: float):
        self.episodes = episodes
        self.base_seed = base_seed
        self.time_scale = time_scale

        self.runtime_config_path = build_runtime_config(ALKOX_TEMPLATE_YAML)
        with open(self.runtime_config_path, 'r', encoding='utf-8') as f:
            self.cfg = yaml.safe_load(f)
        self.bo_cfg = self.cfg['alkox_bo_config']

        self.out_dir = OUTPUT_DIR
        os.makedirs(self.out_dir, exist_ok=True)

        self.ercfss = AlkOxERCFSSController(self.runtime_config_path, self.cfg, self.time_scale)
        self.chemos_dispatch = AlkOxChemosDispatchBaseline(
            self.runtime_config_path, self.cfg, self.time_scale
        )

        self.results: List[Dict] = []

        self._warmup_plan_cache: Dict[int, List[Dict]] = {}

    def _get_or_build_warmup_plan(self, seed: int) -> List[Dict]:
        if seed not in self._warmup_plan_cache:
            print(f"[Runner] Building warmup plan for seed={seed}")
            scratch_bo = AlkOxBOOptimizer(
                self.bo_cfg, seed=seed,
                sim_factory=lambda s=seed: AlkOxReactionSimulator(seed=s),
            )
            self._warmup_plan_cache[seed] = [dict(p) for p in scratch_bo._warmup_plan]
        return [dict(p) for p in self._warmup_plan_cache[seed]]

    def _new_bo(self, seed: int) -> AlkOxBOOptimizer:
        warmup_plan = self._get_or_build_warmup_plan(seed)
        return AlkOxBOOptimizer(
            self.bo_cfg, seed=seed,
            sim_factory=lambda s=seed: AlkOxReactionSimulator(seed=s),
            warmup_plan=warmup_plan,
        )

    def _new_sim(self, seed: int) -> AlkOxReactionSimulator:
        return _NoiselessCachingAlkOxSim(seed=seed)

    def _run_ercfss_episode(self, ep: int, seed: int) -> Dict:
        bo = self._new_bo(seed)
        sim = self._new_sim(seed)
        self.ercfss.reset_for_episode(ep, seed, bo, sim, f"e{ep}_ERCFSS")
        res = self.ercfss.run_episode()
        save_conversion_curve(self.out_dir, ep, 'ERCFSS', seed,
                              bo.updates, res['makespan_min'],
                              aborted_count=res.get('aborted_count', 0))
        return res

    def _run_chemos_dispatch_episode(self, ep: int, seed: int) -> Dict:
        bo = self._new_bo(seed)
        sim = self._new_sim(seed)
        self.chemos_dispatch.reset_for_episode(ep, seed, bo, sim, f"e{ep}_ChemOSDispatch")
        res = self.chemos_dispatch.run_episode()
        save_conversion_curve(self.out_dir, ep, 'ChemOSDispatch', seed,
                              bo.updates, res['makespan_min'],
                              aborted_count=res.get('aborted_count', 0))
        return res

    def run(self):
        print(f"\n{'=' * 60}")
        print(f"AlkOx Comparison Harness")
        print(f"  episodes   = {self.episodes}")
        print(f"  base_seed  = {self.base_seed}")
        print(f"  time_scale = {self.time_scale}")
        print(f"  out_dir    = {self.out_dir}")
        print('=' * 60)

        init_comm_dirs()

        print("\nWaiting for ChemOS ...")
        wait = 0
        while not is_chemos_ready():
            time.sleep(1)
            wait += 1
            if wait % 10 == 0:
                print(f"  Waiting for ChemOS ({wait}s) ...")
            if wait > 300:
                print("ChemOS did not respond in 300s, aborting.")
                cleanup_comm_dirs()
                return
        print("ChemOS connected.")
        set_drl_ready()

        try:
            for ep in range(self.episodes):
                seed = self.base_seed + ep
                print(f"\n\n{'=' * 60}\n=== Episode {ep}/{self.episodes - 1}  "
                      f"seed={seed} ===\n{'=' * 60}")

                ercfss_res = self._run_ercfss_episode(ep, seed)
                self.results.append(ercfss_res)

                dispatch_res = self._run_chemos_dispatch_episode(ep, seed)
                self.results.append(dispatch_res)

                try:
                    from case_study_export import write_episode_json
                    write_episode_json(self.out_dir,
                                       ep, seed, ercfss_res, dispatch_res)
                except Exception as e:
                    print(f"[CaseStudyExport] failed for ep{ep}: {e}")
        finally:
            set_episode_complete()
            save_comparison_summary(self.out_dir, self.results)
            cleanup_comm_dirs()
            print("\nAll episodes done.")


def main():
    parser = argparse.ArgumentParser(description='AlkOx closed-loop SDL validation (ERC-FSS vs ChemOS built-in dispatch)')
    parser.add_argument('--episodes', type=int, default=1,
                        help='Number of episodes (each runs ERC-FSS then ChemOS built-in dispatch).')
    parser.add_argument('--base-seed', type=int,
                        default=3723620,
                        help='Seed for episode 0; episode i uses base_seed + i.')
    parser.add_argument('--time-scale', type=float, default=0.01,
                        help='Real seconds per virtual second (0.01 = 100x faster than real time)')
    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}")
        return
    if not os.path.exists(ALKOX_TEMPLATE_YAML):
        print(f"AlkOx template not found: {ALKOX_TEMPLATE_YAML}")
        return

    runner = AlkOxExperimentRunner(
        episodes=args.episodes,
        base_seed=args.base_seed,
        time_scale=args.time_scale,
    )
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
