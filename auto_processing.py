import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'DRL_scheduler_for_SDL'))

from chemos_integration.file_protocol import (
    init_comm_dirs, cleanup_comm_dirs,
    ScheduleCommand, write_schedule_command, read_completion_events,
    mark_event_processed, set_drl_ready, is_chemos_ready,
    set_episode_complete, CLOCK_STATE_FILE, read_prev_clock_epoch,
)
from chemos_integration.global_clock import GlobalClock

np.set_printoptions(precision=4)

TOTAL_JOBS = 40
POOL_SIZE = 80
INITIAL_JOBS = 12
BO_DELAY_MEAN_SEC = 10.0
BO_DELAY_STD_SEC = 2.5
BO_DELAY_MIN_SEC = 5.0
BO_DELAY_MAX_SEC = 15.0
SOLVE_TIME_MARGIN = 0.001

MODEL_PATH = os.path.join(
    project_root, 'DRL_scheduler_for_SDL', 'training', 'saved_models', 'ERC_FSS.pt',
)
CONFIG_PATH = os.path.join(
    project_root, 'DRL_scheduler_for_SDL', 'configs', 'sdl_config.yaml',
)
OUTPUT_DIR = os.path.join(project_root, 'outputs', 'chemos_validation')

STOP_FILE = os.path.join(project_root, 'STOP')


def pregenerate_jobs(seed: int, n_jobs: int = POOL_SIZE):
    from case_study.dataset_generator import DatasetGenerator
    gen = DatasetGenerator(config_path=CONFIG_PATH)

    np.random.seed(seed)
    gen.atlas_simulator.rng = np.random.RandomState(seed + 3000000)

    jobs = []
    for i in range(n_jobs):
        jobs.append(gen.generate_job(
            job_id=f"job_{i}",
            job_type='SDL_experiment',
            arrival_time=0.0,
        ))

    bo_rng = np.random.RandomState(seed + 9999999)
    bo_delays = {
        f"job_{i}": float(np.clip(bo_rng.normal(BO_DELAY_MEAN_SEC, BO_DELAY_STD_SEC),
                                 BO_DELAY_MIN_SEC, BO_DELAY_MAX_SEC))
        for i in range(INITIAL_JOBS, n_jobs)
    }

    print(f"\n[PRE-GEN] {len(jobs)} jobs (seed={seed})")
    return jobs, bo_delays


class ERCFSSController:
    def __init__(self, seed: int, time_scale: float = 1.0):
        self.seed = seed
        self.time_scale = float(time_scale)
        np.random.seed(seed)

        print(f"\n{'=' * 60}")
        print(f"ERC-FSS controller (seed={seed})")
        print(f"  Experiments: {TOTAL_JOBS} ({INITIAL_JOBS} warm-start)")
        print(f"  Output:      {OUTPUT_DIR}")
        print(f"{'=' * 60}")

        prev_epoch = read_prev_clock_epoch()
        init_comm_dirs()

        self.clock = GlobalClock(
            time_scale=self.time_scale,
            state_file=CLOCK_STATE_FILE,
            epoch=prev_epoch + 1,
        )

        self.pregen_jobs, self.bo_delays_map = pregenerate_jobs(seed, POOL_SIZE)
        self._next_dyn_idx = INITIAL_JOBS

        self.scheduler = None
        self._scheduler_loaded = False

        self.job_buffer: List[Dict] = []
        self.released_jobs: Dict[str, Dict] = {}
        self.completed_jobs = set()
        self.jobs_generated = 0

        self.bo_queue: List[Tuple[float, str, Dict]] = []

        self.op_records: Dict[str, Dict] = {}

        self.active_op_schedule: Dict[str, Tuple[float, str, float]] = {}

        self.schedule_sequence: List[Dict] = []
        self.release_history: List[Dict] = []
        self._step = 0

        self.stats = {
            'release_count': 0,
            'total_solve_time': 0.0,
            'total_ppo_inference_time': 0.0,
            'total_wall_time': 0.0,
            'time_breakdown': {
                'hla_decision': 0.0,
                'feasibility_check': 0.0,
                'scheduling': 0.0,
                'time_advance': 0.0,
            },
            'candidates_checked': 0,
            'candidates_infeasible': 0,
            'final_planned_makespan': 0.0,
        }

    def _load_scheduler(self):
        if self._scheduler_loaded:
            return
        print("[ERC-FSS] Loading scheduler...")
        from chemos_integration.drl_scheduler_wrapper import DRLSchedulerWrapper
        self.scheduler = DRLSchedulerWrapper(
            model_path=MODEL_PATH, config_path=CONFIG_PATH,
            current_time_getter=self.clock.get_current_time,
        )
        self._scheduler_loaded = True

    def cold_start(self):
        print(f"\nCold start: loading first {INITIAL_JOBS} jobs")
        for i in range(INITIAL_JOBS):
            job = dict(self.pregen_jobs[i])
            job['arrival_time'] = 0.0
            self.job_buffer.append(job)
            self.jobs_generated += 1
        print(f"Cold start done: {len(self.job_buffer)} jobs in buffer\n")

    def process_completion_events(self):
        events = read_completion_events()
        for event in events:
            if event.event_type == 'op_complete':
                if event.op_id in self.active_op_schedule:
                    del self.active_op_schedule[event.op_id]
                self.op_records[event.op_id] = {
                    'job_id': event.job_id,
                    'machine': event.machine_id,
                    'start': event.start_time,
                    'end':   event.end_time,
                }
                if self._scheduler_loaded and self.scheduler:
                    op = self.scheduler.env.graph.operations.get(event.op_id)
                    if op is not None:
                        op.is_completed = True
                print(f"  [op_complete] {event.op_id} @ {event.machine_id} "
                      f"[{event.start_time:.2f}, {event.end_time:.2f}]")

            elif event.event_type == 'job_complete':
                jid = event.job_id
                if jid not in self.completed_jobs:
                    self.completed_jobs.add(jid)
                    print(f"  [job_complete] {jid}  "
                          f"({len(self.completed_jobs)}/{TOTAL_JOBS})")
                    self._enqueue_next_dyn_job(jid)

            mark_event_processed(event)

    def _enqueue_next_dyn_job(self, trigger_jid: str):
        if self._next_dyn_idx >= TOTAL_JOBS:
            return
        next_pool = self.pregen_jobs[self._next_dyn_idx]
        delay_sec = self.bo_delays_map.get(next_pool['id'], 10.0)
        ready_sec = self.clock.get_current_time() + delay_sec
        self.bo_queue.append((ready_sec, trigger_jid, next_pool))
        self.bo_queue.sort(key=lambda x: x[0])
        self._next_dyn_idx += 1
        print(f"  [BO] {next_pool['id']} proposed after {trigger_jid} "
              f"(ready at {ready_sec:.2f}s)")

    def process_bo_queue(self):
        if not self.bo_queue:
            return
        now = self.clock.get_current_time()
        DRAIN_TOL = 0.1
        while self.bo_queue and now + DRAIN_TOL >= self.bo_queue[0][0]:
            ready_sec, _trig, pool_job = self.bo_queue.pop(0)
            job = dict(pool_job)
            job['arrival_time'] = ready_sec / 60.0
            self.job_buffer.append(job)
            self.jobs_generated += 1
            print(f"  [BO] {job['id']} arrival={job['arrival_time']:.2f}min")

    def execute_progressive_release(self) -> Optional[Dict]:
        if not self.job_buffer:
            return None
        self._load_scheduler()
        self._step += 1

        print(f"\n  >>> Progressive release  step={self._step}  "
              f"buffer={len(self.job_buffer)}  "
              f"t={self.clock.get_current_time()/60:.2f}min")

        self.scheduler.clear_pending_jobs()
        for job in self.job_buffer:
            self.scheduler.add_job_to_buffer(job)

        release_start = time.time()
        result = self.scheduler.execute_progressive_release()
        release_elapsed = time.time() - release_start

        self.stats['time_breakdown']['feasibility_check'] += result.get('feasibility_check_time', 0.0)
        self.stats['candidates_checked'] += result.get('candidates_checked', 0)
        self.stats['candidates_infeasible'] += result.get('candidates_infeasible', 0)

        if not result['success']:
            print(f"  Progressive release infeasible "
                  f"(solve={release_elapsed:.2f}s)")
            self.stats['total_solve_time'] += release_elapsed
            self.stats['time_breakdown']['hla_decision'] += release_elapsed
            return None

        # computation-time compensation: the measured solver time is added to the
        # scheduled start times before dispatching to ChemOS
        solve_time_min = release_elapsed / 60.0 + SOLVE_TIME_MARGIN

        compensated = [
            (s + solve_time_min, op_id, m, e + solve_time_min)
            for s, op_id, m, e in result['schedule']
        ]

        for s, op_id, m, e in compensated:
            self.active_op_schedule[op_id] = (s, m, e)
            self.schedule_sequence.append({
                'step':    self._step,
                'start':   s,
                'op_id':   op_id,
                'machine': m,
                'end':     e,
            })

        released_ids = set()
        for job in result['released_jobs']:
            jid = job['id'] if isinstance(job, dict) else job.id
            released_ids.add(jid)
            self.released_jobs[jid] = job
        self.job_buffer = [
            j for j in self.job_buffer
            if (j['id'] if isinstance(j, dict) else j.id) not in released_ids
        ]

        self.release_history.append({
            'step':                self._step,
            'current_time':        self.clock.get_current_time() / 60.0,
            'released_jobs':       sorted(released_ids),
            'k_value':             result.get('k_value', 0),
            'makespan':            result['makespan'],
            'combinations_tried':  result.get('combinations_tried', 0),
        })

        cmd = ScheduleCommand(
            schedule=compensated,
            released_jobs=list(released_ids),
            makespan=result['makespan'],
            solve_time=release_elapsed,
        )
        write_schedule_command(cmd)

        self.stats['release_count'] += 1
        self.stats['total_solve_time'] += release_elapsed
        self.stats['time_breakdown']['hla_decision'] += release_elapsed
        self.stats['total_ppo_inference_time'] += \
            result.get('ppo_inference_time', 0.0)
        self.stats['final_planned_makespan'] = result['makespan']

        self.clock.add_delay(solve_time_min * 60.0)

        print(f"  Released k={result['k_value']}  "
              f"jobs={len(released_ids)}  "
              f"makespan={result['makespan']:.2f}min  "
              f"solve={release_elapsed:.2f}s  "
              f"compensation=+{solve_time_min:.2f}min  cmd={cmd.command_id}")
        return {'success': True, 'released_jobs': sorted(released_ids),
                'command_id': cmd.command_id}

    def advance_time_to_next_event(self):
        current_sec = self.clock.get_current_time()
        next_sec: Optional[float] = None

        for _op, (_s_min, _m, end_min) in self.active_op_schedule.items():
            end_sec = end_min * 60.0
            if end_sec > current_sec and (next_sec is None or end_sec < next_sec):
                next_sec = end_sec

        if self.bo_queue:
            br = self.bo_queue[0][0]
            if br > current_sec and (next_sec is None or br < next_sec):
                next_sec = br

        if next_sec is not None:
            print(f"  advance to {next_sec/60:.2f}min")
            self.clock.advance_to(next_sec)
        else:
            self.clock.advance_to(current_sec + 60.0)

    def is_episode_complete(self) -> bool:
        return len(self.completed_jobs) >= TOTAL_JOBS

    def _collect_job_details(self) -> Dict:
        details: Dict[str, Dict] = {}
        for op_id, rec in self.op_records.items():
            jid = rec['job_id']
            try:
                idx = int(op_id.rsplit('_op', 1)[-1])
            except (ValueError, IndexError):
                idx = None
            if jid not in details:
                details[jid] = {'ops': [], 'completed': 0,
                                'scheduled': 0, 'total': 0}
            details[jid]['ops'].append({
                'op_id': op_id,
                'op_index': idx,
                'is_completed': True,
                'is_scheduled': True,
                'scheduled_start': rec['start'],
                'scheduled_end':   rec['end'],
                'scheduled_machine': rec['machine'],
            })
            details[jid]['total'] += 1
            details[jid]['completed'] += 1
            details[jid]['scheduled'] += 1
        for jid, d in details.items():
            d['ops'].sort(key=lambda o: (o['op_index']
                                         if o['op_index'] is not None else 0))
        return details

    def _tmax_violations(self) -> Tuple[int, List[Dict]]:
        out: List[Dict] = []
        if not (self._scheduler_loaded and self.scheduler):
            return 0, out
        try:
            raw = self.scheduler._check_tmax_violations()
        except Exception:
            raw = []
        for v in raw:
            out.append({
                'job_id':       v['job_id'],
                'constraint':   f"{v['from_op']}->{v['to_op']}",
                'max_interval': v['max_interval'],
                'actual':       round(v['actual_interval'], 3),
                'excess':       round(v['violation'], 3),
            })
        return len(out), out

    def _actual_makespan(self) -> float:
        if not self.op_records:
            return 0.0
        return max(r['end'] for r in self.op_records.values())

    def save_results(self) -> str:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        actual_makespan = self._actual_makespan()
        tmax_count, _ = self._tmax_violations()

        episode_result = {
            'episode_idx':  0,
            'seed':         self.seed,
            'timestamp':    datetime.now().isoformat(),
            'ERC-FSS': {
                'makespan':          actual_makespan,
                'job_details':       self._collect_job_details(),
                'tmax_violations':   tmax_count,
                'total_solve_time':  self.stats['total_solve_time'],
                'total_time':        self.stats['total_wall_time'],
                'time_breakdown':    dict(self.stats['time_breakdown']),
                'candidates_checked':    self.stats['candidates_checked'],
                'candidates_infeasible': self.stats['candidates_infeasible'],
                'schedule_sequence': self.schedule_sequence,
                'release_history':   self.release_history,
            },
        }
        fname = f"episode_000_seed_{self.seed}.json"
        fpath = os.path.join(OUTPUT_DIR, fname)
        with open(fpath, 'w', encoding='utf-8') as f:
            json.dump(episode_result, f, indent=2, ensure_ascii=False)
        print(f"\n[Output] Episode JSON saved: {fpath}")
        return fpath

    def print_final_stats(self):
        s = self.stats
        rc = s['release_count']
        ts = s['total_solve_time']
        tw = s['total_wall_time']
        actual_makespan = self._actual_makespan()
        tmax_count, _ = self._tmax_violations()

        print(f"\n{'=' * 60}")
        print(f"Episode complete (seed={self.seed})")
        print(f"{'=' * 60}")
        print(f"  Completed jobs:            {len(self.completed_jobs)}/{TOTAL_JOBS}")
        print(f"  Total releases:            {rc}")
        print(f"  Computation time:          {ts:.2f}s  "
              f"(avg {ts/max(1, rc):.2f}s/release)")
        print(f"  Gurobi feasibility check:  {s['time_breakdown']['feasibility_check']:.2f}s  "
              f"({s['candidates_checked']} checks, {s['candidates_infeasible']} infeasible)")
        print(f"  Makespan:                  {actual_makespan:.2f} min")
        print(f"  Tmax violations:           {tmax_count}")
        print(f"  Total wall-clock time:     {tw:.2f}s  ({tw/60:.2f} min)")
        print(f"{'=' * 60}")

    def run(self, wait_timeout=300):
        print(f"\n{'#' * 60}")
        print(f"ERC-FSS controller started, waiting for ChemOS ...")
        print(f"{'#' * 60}")

        wait = 0
        while not is_chemos_ready():
            time.sleep(1)
            wait += 1
            if wait % 10 == 0:
                print(f"  waiting for ChemOS... ({wait}s)")
            if wait > wait_timeout:
                print(f"ChemOS not responding after {wait_timeout} s, exiting")
                return
        print("ChemOS connected")
        set_drl_ready()

        self.cold_start()

        episode_wall_start = time.time()
        step = 0
        try:
            while not self.is_episode_complete():
                step += 1
                if os.path.exists(STOP_FILE):
                    os.remove(STOP_FILE)
                    print("STOP file detected, shutting down")
                    break

                print(f"\n--- Step {step} ---")
                print(f"  t={self.clock.get_current_time()/60:.2f}min  "
                      f"buffer={len(self.job_buffer)}  "
                      f"bo_queue={len(self.bo_queue)}  "
                      f"completed={len(self.completed_jobs)}/{TOTAL_JOBS}")

                self.process_completion_events()
                self.process_bo_queue()

                if self.job_buffer:
                    self.execute_progressive_release()

                adv_t0 = time.time()
                self.advance_time_to_next_event()
                self.stats['time_breakdown']['time_advance'] += time.time() - adv_t0

                time.sleep(0.5 * self.time_scale)
        finally:
            self.stats['total_wall_time'] = time.time() - episode_wall_start
            set_episode_complete()
            self.print_final_stats()
            self.save_results()
            cleanup_comm_dirs()


def main():
    parser = argparse.ArgumentParser(
        description='ERC-FSS controller for the ChemOS-integrated validation'
    )
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for experiment generation')
    parser.add_argument('--time-scale', type=float, default=1.0,
                        help='Real seconds per virtual second (1.0 = real time)')
    parser.add_argument('--wait-timeout', type=int, default=300,
                        help='Seconds to wait for the start request in ChemOS')
    args = parser.parse_args()

    seed = args.seed
    if seed is None:
        seed = int(np.random.randint(0, 100000) + np.random.randint(0, 1000) * 100000)

    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}")
        return
    if not os.path.exists(CONFIG_PATH):
        print(f"Config not found: {CONFIG_PATH}")
        return

    print("=" * 60)
    print("ChemOS-integrated ERC-FSS scheduling")
    print(f"  Seed:       {seed}")
    print(f"  Time scale: {args.time_scale}")
    print(f"  Model:      {MODEL_PATH}")
    print(f"  Config:     {CONFIG_PATH}")
    print("=" * 60)

    controller = ERCFSSController(seed=seed, time_scale=args.time_scale)
    try:
        controller.run(wait_timeout=args.wait_timeout)
    except KeyboardInterrupt:
        print("\nInterrupted, shutting down...")
        cleanup_comm_dirs()
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        cleanup_comm_dirs()


if __name__ == '__main__':
    main()
