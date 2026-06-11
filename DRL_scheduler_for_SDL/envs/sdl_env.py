import gymnasium as gym
import numpy as np
import torch
from typing import Tuple, Dict
from configs import Config
from masking.mask_manager import MaskManager
from .job_buffer import JobBuffer
from .disjunctive_graph import DisjunctiveGraph, Machine, Operation
import time
import os
import yaml
from ortools.sat.python import cp_model
from ortools.linear_solver import pywraplp
from case_study.dataset_generator import DatasetGenerator
import gurobipy as gp
from gurobipy import GRB

class SDLEnv(gym.Env):
    def __init__(self, config):
        super().__init__()

        #  1:  Config 

        if isinstance(config, dict) and not isinstance(config, Config):
            config = Config(config)

        #  2:  env_config
        if hasattr(config, 'env_config'):
            config = config.env_config
            if isinstance(config, dict) and not isinstance(config, Config):
                config = Config(config)
        elif isinstance(config, dict) and 'env_config' in config:
            config = Config(config['env_config'])
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(project_root, 'configs', 'sdl_config.yaml')
        self.config = config
        self.n_jobs = self._get_conf('n_jobs', 50)
        self.load_factor = self._get_conf('load_factor', 0.8)
        self.initial_jobs = self._get_conf('initial_jobs', 4)
        self.buffer_size = self._get_conf('buffer_size', 100)

        self.job_counter = 0
        self._completed_job_ids = set()
        self.dataset_generator = DatasetGenerator(config_path=config_path)
        from case_study.atlas_simulator import AtlasSimulator
        atlas_config = self._get_conf('atlas_config', {})
        from case_study.sdl_process_config import ATLAS_CONFIG

        if self._get_conf('case_study', 'sdl_standard') == 'sdl_standard':
            #   ATLAS_CONFIG
            full_atlas_config = ATLAS_CONFIG
        else:
            # PPOYAMLCASE_STUDY_CONFIG
            full_atlas_config = atlas_config

        self.atlas_simulator = AtlasSimulator(full_atlas_config)

        #  3: 
        tmax_constraints_raw = self._get_conf('tmax_constraints', [])
        self.tmax_constraints = self._normalize_tmax_constraints(tmax_constraints_raw)

        #  4:  setup_times
        setup_times_config = self._get_conf('setup_times', {})
        self.setup_times_map = self._parse_setup_times(setup_times_config)

        self.job_buffer = JobBuffer(max_size=self.buffer_size)
        self.current_time = 0.0

        #  5:  mask_config
        mask_config = self._get_conf('mask_config', {
            'tmax_lookahead_depth': 2,
            'tmax_time_budget_ms': 10,
            'cache_enabled': True
        })
        if isinstance(mask_config, dict) and not isinstance(mask_config, Config):
            mask_config = Config(mask_config)

        #  6:  MaskManager
        mask_config_with_context = Config({
            'tmax_lookahead_depth': mask_config.get('tmax_lookahead_depth', 2),
            'tmax_time_budget_ms': mask_config.get('tmax_time_budget_ms', 10),
            'cache_enabled': mask_config.get('cache_enabled', True),
            'tmax_constraints': self.tmax_constraints,  #
            'setup_times': setup_times_config
        })

        self.mask_manager = MaskManager(
            mask_config_with_context)

        #  V9
        machine_list = self._get_conf('machines', [])
        if not machine_list:
            machine_list = self._get_conf('machine_list', [])

        #  sdl_config.yaml
        #  machine_list
        machine_list = []

        #
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    case_study_config = yaml.safe_load(f)

                # process_steps
                if 'process_steps' in case_study_config:
                    extracted_machines = []
                    seen_ids = set()

                    for step_info in case_study_config['process_steps'].values():
                        if isinstance(step_info, dict) and 'machines' in step_info:
                            for m in step_info['machines']:
                                if isinstance(m, dict) and 'id' in m:
                                    machine_id = m['id']
                                    if machine_id not in seen_ids:
                                        extracted_machines.append(m)
                                        seen_ids.add(machine_id)

                    if extracted_machines:
                        machine_list = extracted_machines
                else:
                    print(f"[SDLEnv] No process_steps in {config_path}")
            except yaml.YAMLError as e:
                print(f"[SDLEnv] YAML parse error in {config_path}: {e}")
            except Exception as e:
                print(f"[SDLEnv] Config load error: {e}")
        else:
            print(f"[SDLEnv] Config {config_path} not found")

        #
        self.machines = self._init_machines(machine_list)
        self.machines.sort(key=lambda m: m.id)
        self.machines_dict = {m.id: m for m in self.machines}
        # self.arrival_queue = []
        # self.next_arrival_idx = 0
        self.completed_jobs_count = 0
        self.last_step_completed_count = 0
        self.total_steps = 0
        self.graph = None
        self.bo_delays_map = {}  # job_id -> BO delay (min); set externally for reproducible comparisons

        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Dict({})

    @property
    def machine_available_time(self):
        """
         SSOT:  Machine  available_time
         env_adapter  graph
        """
        if not hasattr(self, 'machines') or not self.machines:
            return {}
        return {m.id: m.available_time for m in self.machines}

    def _get_conf(self, key, default):
        if hasattr(self.config, 'get'): return self.config.get(key, default)
        return getattr(self.config, key, default)

    def _init_machines(self, config_list):
        machines = []
        if not config_list:  #
            raise RuntimeError("[SDLEnv] machine_list is empty - check sdl_config.yaml process_steps indentation")
            for i, t in enumerate(types): machines.append(Machine(id=f"M{i}_{t}", machine_type=t))
            return machines

        if isinstance(config_list, dict):  #
            flat = []
            for v in config_list.values():
                if isinstance(v, list):
                    flat.extend(v)
                elif isinstance(v, dict) and 'machines' in v:
                    flat.extend(v['machines'])
            config_list = flat

        for m in config_list:
            mid = m.get('id') if isinstance(m, dict) else m.id
            mtype = m.get('type', 'default') if isinstance(m, dict) else getattr(m, 'type', 'default')
            machines.append(Machine(id=mid, machine_type=mtype))
        return machines

    def _normalize_tmax_constraints(self, constraints_raw):
        """
         TmaxMaskBuilder

         (YAML):
            - from_step: "S3"
              to_step: "S4"
              max_interval: 10.0
              reason: "..."
              severity: "critical"
              penalty_factor: 100.0

         ():
            [(from_id, to_id, max_interval), ...]
            : [("S3", "S4", 10.0), ("S4", "S6", 5.0), ...]
        """
        if not constraints_raw:
            return []

        normalized = []
        for constraint in constraints_raw:
            if isinstance(constraint, dict):
                #
                from_step = constraint.get('from_step') or constraint.get('from_id') or constraint.get('A')
                to_step = constraint.get('to_step') or constraint.get('to_id') or constraint.get('B')
                max_interval = constraint.get('max_interval') or constraint.get('Tmax')

                if from_step and to_step and max_interval is not None:
                    normalized.append((from_step, to_step, max_interval))
            elif isinstance(constraint, (tuple, list)) and len(constraint) == 3:
                #
                normalized.append(tuple(constraint))

        return normalized

    def reset(self, seed=None, options=None, skip_warmstart=False):
        super().reset(seed=seed)
        self._skip_warmstart = skip_warmstart
        # if seed: np.random.seed(seed)

        self.job_buffer.clear()
        self.current_time = 0.0
        self.total_steps = 0
        self.job_counter = 0  # V4
        self._completed_job_ids = set()
        self.execution_history = []
        #   1 Machine
        for m in self.machines:
            m.available_time = 0.0
            m.utilization = 0.0
            m.queue = []
            m.last_op_id = None
            m.last_op_material = None

        #   3 Atlas Simulator
        if hasattr(self, 'atlas_simulator'):
            self.atlas_simulator.reset()  #  atlas_simulator.py  reset

        #  ()
        self.graph = DisjunctiveGraph(
            machines={m.id: m for m in self.machines},
            env=self
        )

        # V4Poisson
        # self.arrival_queue = self._generate_arrivals(self.n_jobs, self.load_factor)  # V4
        # self.next_arrival_idx = 0  # V4

        self.completed_jobs_count = 0

        # V4Warmstart - jobskip_warmstart=True
        if not getattr(self, '_skip_warmstart', False):
            warmstart_jobs = [
                ('SDL_experiment', 0.0),
                ('SDL_experiment', 0.0),
                ('SDL_experiment', 0.0),
                ('SDL_experiment', 0.0),
                ('SDL_experiment', 0.0),
                ('SDL_experiment', 0.0),
                ('SDL_experiment', 0.0),
                ('SDL_experiment', 0.0),
                ('SDL_experiment', 0.0),
                ('SDL_experiment', 0.0),
                ('SDL_experiment', 0.0),
                ('SDL_experiment', 0.0),
                #('SDL_experiment', 0.0),
                #('SDL_experiment', 0.0),
                #('SDL_experiment', 0.0),
                #('SDL_experiment', 0.0),
            ]

            for job_type, arrival_time in warmstart_jobs:
                self._inject_atlas_job(job_type, arrival_time)

        self.machines.sort(key=lambda m: m.id)
        self.machines_dict = {m.id: m for m in self.machines}
        return self._get_hla_state(), {}

    def step_hla(self, action):
        """
         V9
        1. ReleaseCP-SAT
        2.  self.current_time_advance_time
        3.  self.graph.reset()ops
        """
        self.total_steps += 1
        reward = 0.0

        if action == 1:  # Release
            if len(self.job_buffer) > 0:
                graph_snapshot = self.graph.snapshot()
                released_jobs = self.job_buffer.flush()
                self._build_static_subproblem(released_jobs)
                result = self._solve_subproblem_with_lower()

                if result['status'] == 'infeasible':
                    self.graph.restore(graph_snapshot)
                    for job in reversed(released_jobs):
                        self.job_buffer.buffer.appendleft(job)
                    reward = -100.0
                    self.last_cp_schedule = []
                else:
                    for start_time, op_id, machine_id, end_time in result['schedule']:
                        self.graph.schedule_operation(op_id, machine_id, start_time, end_time)
                    self.last_cp_schedule = result['schedule']
                    reward = 0.0
            else:
                reward = 0.0
                self.last_cp_schedule = []

        else:  # Wait
            reward = 0.0
            self.last_cp_schedule = []

        self._advance_time_and_inject_jobs()
        done = self._check_all_jobs_completed()
        return self._get_hla_state(), reward, done, False, {}

    def _solve_subproblem_with_lower(self, rule=None, time_limit=60, mip_gap_abs=None):
        """
        FJSP MILP via Gurobi.

        Fixes vs. previous version:
          1. horizon = greedy SPT (no *2 inflation, called once)
          2. start/end vars bounded by [0, horizon] — tightens LP relaxation
          3. Standard 2-term Big-M disjunctive with z_ab = x_a AND x_b
             linearisation — LP lower bound is now meaningful, not ≈ 0
          4. Per-machine M replaces global horizon
          5. MIPGapAbs (absolute minutes) instead of MIPGap (relative %)
          6. Tmax picks closest predecessor by op_index, not dict-order [0]
        """

        # Phase 0: unschedule future ops & consistency check.
        for op in self.graph.operations.values():
            if not op.is_completed and op.is_scheduled:
                if op.scheduled_start > self.current_time + 1e-4:
                    op.is_scheduled = False
                    op.scheduled_machine = None
                    op.scheduled_start = 0.0
                    op.scheduled_end = 0.0
                    if op.id in self.graph.scheduled_ops:
                        self.graph.scheduled_ops.remove(op.id)

        inconsistent_ops = []
        for op in self.graph.operations.values():
            if op.is_scheduled and not op.is_completed:
                if op.scheduled_start <= self.current_time:
                    for pred_id in op.predecessors:
                        if pred_id in self.graph.operations:
                            pred = self.graph.operations[pred_id]
                            if pred.scheduled_end > op.scheduled_start + 1e-4:
                                inconsistent_ops.append(op.id)
                                break

        if inconsistent_ops:
            for op_id in inconsistent_ops:
                op = self.graph.operations[op_id]
                op.is_scheduled = False
                op.scheduled_machine = None
                op.scheduled_start = 0.0
                op.scheduled_end = 0.0
                if op.id in self.graph.scheduled_ops:
                    self.graph.scheduled_ops.remove(op.id)

        self.graph._update_eligible_ops()

        # Phase 1: collect ops & build Gurobi model.
        all_ops = [op for op in self.graph.operations.values() if not op.is_completed]
        if not all_ops:
            return {'status': 'ok', 'makespan': 0.0, 'schedule': []}

        try:
            model = gp.Model("SDL_FJSP")
            model.setParam('OutputFlag', 0)
            model.setParam('Presolve', 2)
            model.setParam('MIPFocus', 1)          # find good feasible solutions fast
            if time_limit is not None:
                model.setParam('TimeLimit', time_limit)
            # mip_gap_abs is an ABSOLUTE gap in minutes — use MIPGapAbs, not MIPGap
            if mip_gap_abs is not None:
                model.setParam('MIPGapAbs', mip_gap_abs)
        except gp.GurobiError as e:
            print(f"[MILP] Gurobi init error: {e}")
            return {'status': 'error', 'makespan': 0.0, 'schedule': []}

        # ----------------------------------------------------------------
        # Horizon: single greedy SPT pass (no *2 inflation)
        # ----------------------------------------------------------------
        def _greedy_horizon():
            sched_end = {}
            for op in self.graph.operations.values():
                if op.is_completed or (
                    op.is_scheduled and op.scheduled_start <= self.current_time + 1e-4
                ):
                    sched_end[op.id] = op.scheduled_end

            mach_avail = dict(self.machine_available_time)
            remaining = set(op.id for op in all_ops if op.id not in sched_end)

            pred_count = {
                oid: sum(1 for p in self.graph.operations[oid].predecessors
                         if p not in sched_end)
                for oid in remaining
            }
            ready_set = {oid for oid in remaining if pred_count[oid] == 0}

            while remaining:
                if not ready_set:
                    ready_set = set(remaining)

                op_id = min(
                    ready_set,
                    key=lambda oid: self.graph.operations[oid].get_min_processing_time()
                )
                op = self.graph.operations[op_id]
                best_m = min(
                    op.compatible_machines,
                    key=lambda m: mach_avail.get(m, self.current_time)
                )
                pred_end = max(
                    (sched_end[p] for p in op.predecessors if p in sched_end),
                    default=self.current_time
                )
                start = max(pred_end, mach_avail.get(best_m, self.current_time))
                end = start + op.processing_times[best_m]
                sched_end[op_id] = end
                mach_avail[best_m] = end
                remaining.remove(op_id)
                ready_set.discard(op_id)
                for oid in list(remaining):
                    if op_id in self.graph.operations[oid].predecessors:
                        pred_count[oid] -= 1
                        if pred_count[oid] == 0:
                            ready_set.add(oid)

            return max(sched_end.values()) if sched_end else self.current_time

        horizon_spt = _greedy_horizon()  # single call, no *2
        #print(f"  Greedy horizon (UB / variable bound): {horizon:.2f}")
        hint_sched = {}
        for op in self.graph.operations.values():
            if op.is_completed or (
                    op.is_scheduled and op.scheduled_start <= self.current_time + 1e-4
            ):
                hint_sched[op.id] = (op.scheduled_end, op.scheduled_machine)

        hint_mach_avail = dict(self.machine_available_time)
        hint_remaining = [op for op in all_ops if op.id not in hint_sched]
        hint_ordered = []
        visited = set(hint_sched.keys())
        while hint_remaining:
            progress = False
            for op in list(hint_remaining):
                if all(p in visited for p in op.predecessors):
                    hint_ordered.append(op)
                    visited.add(op.id)
                    hint_remaining.remove(op)
                    progress = True
            if not progress:
                hint_ordered.extend(hint_remaining)
                break

        for op in hint_ordered:
            pred_end = max(
                (hint_sched[p][0] for p in op.predecessors if p in hint_sched),
                default=self.current_time
            )
            best_m = min(
                op.compatible_machines,
                key=lambda m: hint_mach_avail.get(m, self.current_time)
            )
            start_h = max(pred_end, hint_mach_avail.get(best_m, self.current_time))
            end_h = start_h + op.processing_times[best_m]
            hint_sched[op.id] = (end_h, best_m)
            hint_mach_avail[best_m] = end_h

        horizon = max((v[0] for v in hint_sched.values()), default=self.current_time)
        # Phase 2: decision variables. start/end bounded by [0, horizon].
        op_vars = {}

        for op in all_ops:
            op_id = op.id
            is_running = (op.is_scheduled and
                          op.scheduled_start <= self.current_time + 1e-4)

            start_var = model.addVar(lb=0.0, ub=horizon, name=f'{op_id}_start')
            end_var   = model.addVar(lb=0.0, ub=horizon, name=f'{op_id}_end')

            if is_running:
                compatible_machines = [m for m in self.machines
                                       if m.id == op.scheduled_machine]
                if not compatible_machines:
                    print(f"[MILP] Machine not found for running op {op.id}")
                    return {'status': 'infeasible', 'makespan': 0.0, 'schedule': []}
            else:
                compatible_machines = [m for m in self.machines
                                       if m.id in op.compatible_machines]

            if not compatible_machines:
                print(f"[MILP] No compatible machines for {op.id}")
                return {'status': 'infeasible', 'makespan': 0.0, 'schedule': []}

            mach_vars = {
                mach.id: model.addVar(vtype=GRB.BINARY, name=f'{op_id}_on_{mach.id}')
                for mach in compatible_machines
            }

            if is_running:
                model.addConstr(start_var == op.scheduled_start)
                model.addConstr(end_var   == op.scheduled_end)
                model.addConstr(mach_vars[op.scheduled_machine] == 1)
            else:
                arrival = op.arrival_time if hasattr(op, 'arrival_time') else 0.0
                earliest_start_candidates = [arrival, self.current_time]

                for pred_id in op.predecessors:
                    if pred_id in self.graph.operations:
                        pred = self.graph.operations[pred_id]
                        if pred.is_completed or pred.is_scheduled:
                            earliest_start_candidates.append(pred.scheduled_end)

                if op.op_index > 0:
                    for hist_op in self.graph.operations.values():
                        if (hist_op.job_id == op.job_id and
                                hist_op.op_index == op.op_index - 1 and
                                (hist_op.is_completed or hist_op.is_scheduled)):
                            earliest_start_candidates.append(hist_op.scheduled_end)
                            break

                min_start = max(earliest_start_candidates)
                model.addConstr(start_var >= min_start)
                model.addConstr(gp.quicksum(mach_vars.values()) == 1)

                proc_expr = gp.quicksum(
                    mach_vars[mach.id] * op.processing_times[mach.id]
                    for mach in compatible_machines
                    if mach.id in op.processing_times
                )
                model.addConstr(end_var == start_var + proc_expr)

            op_vars[op_id] = {
                'start':      start_var,
                'end':        end_var,
                'mach_vars':  mach_vars,
                'is_running': is_running,
                'op_obj':     op,
            }

        # Phase 3: precedence constraints (deduplicated).
        added_prec = set()

        def _add_prec_var(succ_id, pred_end_var):
            key = (succ_id, id(pred_end_var))
            if key not in added_prec:
                added_prec.add(key)
                model.addConstr(op_vars[succ_id]['start'] >= pred_end_var)

        def _add_prec_const(succ_id, pred_end_val, label):
            key = (succ_id, 'const', label, pred_end_val)
            if key not in added_prec:
                added_prec.add(key)
                model.addConstr(op_vars[succ_id]['start'] >= pred_end_val)

        for op in all_ops:
            for pred_id in op.predecessors:
                if pred_id in op_vars:
                    _add_prec_var(op.id, op_vars[pred_id]['end'])
                elif pred_id in self.graph.operations:
                    pred = self.graph.operations[pred_id]
                    if pred.is_completed or pred.is_scheduled:
                        _add_prec_const(op.id, pred.scheduled_end, pred_id)

        job_ops_map = {}
        for op in all_ops:
            job_ops_map.setdefault(op.job_id, []).append(op)

        for job_id, ops in job_ops_map.items():
            ops.sort(key=lambda x: x.op_index)
            for i in range(len(ops) - 1):
                _add_prec_var(ops[i + 1].id, op_vars[ops[i].id]['end'])
            for op in ops:
                exp_prev = op.op_index - 1
                if exp_prev < 0:
                    continue
                in_model = any(
                    o.op_index == exp_prev and o.job_id == job_id for o in all_ops
                )
                if not in_model:
                    hist_prev = next(
                        (h for h in self.graph.operations.values()
                         if h.job_id == job_id and h.op_index == exp_prev),
                        None
                    )
                    if hist_prev and (hist_prev.is_completed or hist_prev.is_scheduled):
                        _add_prec_const(op.id, hist_prev.scheduled_end, hist_prev.id)
                    elif hist_prev is None:
                        print(f"[MILP] Predecessor not found: {op.id} (index={exp_prev})")

        # Phase 4: Tmax constraints. Predecessor = closest op with smaller op_index.
        tmax_count = 0

        for (from_step, to_step, max_interval) in self.tmax_constraints:
            for op in all_ops:
                if op.op_type != to_step:
                    continue
                # closest preceding from_step op for this job
                candidates = [
                    o for o in self.graph.operations.values()
                    if o.job_id == op.job_id
                    and o.op_type == from_step
                    and o.op_index < op.op_index
                ]
                if not candidates:
                    continue
                pred = max(candidates, key=lambda o: o.op_index)

                if pred.id in op_vars:
                    model.addConstr(
                        op_vars[op.id]['start'] - op_vars[pred.id]['end'] <= max_interval
                    )
                    tmax_count += 1
                elif pred.is_completed or pred.is_scheduled:
                    model.addConstr(
                        op_vars[op.id]['start'] <= pred.scheduled_end + max_interval
                    )
                    tmax_count += 1

        # Phase 5: disjunctive (no-overlap) constraints
        #
        # Standard 2-term Big-M FJSP formulation:
        #
        #   z_ab = x_a AND x_b  (linearised: both ops on this machine)
        #   y_ab = 1  =>  op_a before op_b
        #
        #   start_b >= end_a - M*(1-y_ab) - M*(1-z_ab)
        #   start_a >= end_b - M*y_ab     - M*(1-z_ab)
        #
        # LP relaxation quality:
        #   2-term: at y=0.5, z=0.5 -> start_b >= end_a - M  (binding)
        #   old 3-term: at y=x_a=x_b=0.33 -> start_b >= end_a - 3M  (trivial)
        #
        # Per-machine M = available_time + sum of all candidate proc times
        # (much tighter than global horizon)
        total_disj = 0

        for machine in self.machines:
            mid = machine.id
            pot_ops = [op for op in all_ops if mid in op_vars[op.id]['mach_vars']]
            if len(pot_ops) < 2:
                continue

            base_avail = max(
                self.machine_available_time.get(mid, self.current_time),
                self.current_time
            )

            for i in range(len(pot_ops)):
                for j in range(i + 1, len(pot_ops)):
                    op_a = pot_ops[i]
                    op_b = pot_ops[j]

                    x_a = op_vars[op_a.id]['mach_vars'][mid]
                    x_b = op_vars[op_b.id]['mach_vars'][mid]
                    proc_a = op_a.processing_times.get(mid, 0.0)
                    proc_b = op_b.processing_times.get(mid, 0.0)
                    M_pair = horizon

                    z_ab = model.addVar(vtype=GRB.BINARY,
                                        name=f'z_{mid}_{op_a.id}_{op_b.id}')
                    model.addConstr(z_ab <= x_a)
                    model.addConstr(z_ab <= x_b)
                    model.addConstr(z_ab >= x_a + x_b - 1)

                    y_ab = model.addVar(vtype=GRB.BINARY,
                                        name=f'y_{mid}_{op_a.id}_{op_b.id}')

                    model.addConstr(
                        op_vars[op_b.id]['start']
                        >= op_vars[op_a.id]['end'] - M_pair * (1 - y_ab) - M_pair * (1 - z_ab)
                    )
                    model.addConstr(
                        op_vars[op_a.id]['start']
                        >= op_vars[op_b.id]['end'] - M_pair * y_ab - M_pair * (1 - z_ab)
                    )
                    total_disj += 2

        # Phase 6: objective (minimise makespan).
        makespan_var = model.addVar(lb=0.0, ub=horizon, name='makespan')
        for op in all_ops:
            model.addConstr(makespan_var >= op_vars[op.id]['end'])
        model.setObjective(makespan_var, GRB.MINIMIZE)

        # Phase 7: MIP warm start from greedy SPT hint.
        try:
            for op in all_ops:
                if op.id not in hint_sched:
                    continue
                end_h, mach_h = hint_sched[op.id]
                if mach_h not in op_vars[op.id]['mach_vars']:
                    continue
                proc_h = op.processing_times.get(mach_h, 0.0)
                op_vars[op.id]['start'].Start = max(0.0, end_h - proc_h)
                op_vars[op.id]['end'].Start = end_h
                for mach_id, var in op_vars[op.id]['mach_vars'].items():
                    var.Start = 1.0 if mach_id == mach_h else 0.0

            makespan_var.Start = horizon

        except Exception as e:
            print(f"[MILP] MIP start failed (non-fatal): {e}")

        # Phase 8: solve.
        model.update()
        # Incumbent patience callback
        import time as _time
        _cb_state = {
            'best': float('inf'),
            'last_improve_time': _time.time(),
            'patience_seconds': 1000,
            'min_improvement': 0.0000005,
        }

        def _incumbent_callback(model, where):
            if where == GRB.Callback.MIPSOL:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if obj < _cb_state['best'] - _cb_state['min_improvement']:
                    _cb_state['best'] = obj
                    _cb_state['last_improve_time'] = _time.time()

            elif where == GRB.Callback.MIP:
                elapsed = _time.time() - _cb_state['last_improve_time']
                if elapsed > _cb_state['patience_seconds']:
                    best_bd = model.cbGet(GRB.Callback.MIP_OBJBND)
                    incumbent = model.cbGet(GRB.Callback.MIP_OBJBST)
                    print(f"[MILP] patience {_cb_state['patience_seconds']}s exceeded; "
                          f"incumbent={incumbent:.3f} bound={best_bd:.3f} "
                          f"gap={100*(incumbent-best_bd)/incumbent:.1f}%")
                    model.terminate()
        t0 = time.time()
        model.optimize(_incumbent_callback)
        solve_time = time.time() - t0

        # Phase 9: extract solution.
        if model.status == GRB.INFEASIBLE:
            return {'status': 'infeasible', 'makespan': 0.0, 'schedule': []}

        if model.SolCount > 0:
            result_makespan = makespan_var.X
            if model.status == GRB.OPTIMAL:
                print(f"[MILP] OPTIMAL  makespan={result_makespan:.2f}  solve={solve_time:.2f}s")
            elif model.status == GRB.TIME_LIMIT:
                gap = model.MIPGap * 100
                print(f"[MILP] TIME_LIMIT  makespan={result_makespan:.2f}  gap={gap:.1f}%  solve={solve_time:.2f}s")
            else:
                print(f"[MILP] feasible  makespan={result_makespan:.2f}  solve={solve_time:.2f}s")

            schedule = []
            for op in all_ops:
                op_id     = op.id
                start_val = op_vars[op_id]['start'].X
                end_val   = op_vars[op_id]['end'].X
                selected_machine = None
                for mach_id, var in op_vars[op_id]['mach_vars'].items():
                    if var.X > 0.5:
                        selected_machine = mach_id
                        break
                if selected_machine:
                    schedule.append((start_val, op_id, selected_machine, end_val))
                    expected_end = start_val + op.processing_times[selected_machine]
                    if abs(end_val - expected_end) > 1e-3:
                        print(f"[MILP] Duration mismatch on {op_id}: "
                              f"solver={end_val:.2f} expected={expected_end:.2f}")
                else:
                    print(f"[MILP] No machine selected for {op_id}")

            schedule.sort(key=lambda x: x[0])
            return {'status': 'ok', 'makespan': result_makespan, 'schedule': schedule}

        print(f"[MILP] No solution found (status={model.status})")
        return {'status': 'error', 'makespan': 0.0, 'schedule': []}


    def _inject_atlas_job(self, job_type, arrival_time):
        """
        V45dataset_generatorS1-S7Job
        """
        job_id = f"job_{self.job_counter}"
        self.job_counter += 1

        # V45dataset_generatorjob
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(project_root, 'configs', 'sdl_config.yaml')
        # jobS1-S7
        job = self.dataset_generator.generate_job(
            job_id=job_id,
            job_type=job_type,
            arrival_time=arrival_time
        )

        # buffer
        from envs.job_buffer import Job
        job_obj = Job(
            id=job['id'],
            job_type=job['type'],
            operations=job['operations'],
            arrival_time=job['arrival_time'],
            priority=job.get('priority', 'normal')
        )

        self.job_buffer.add_job(job_obj)

    def step_lower(self, op_idx, mach_idx):
        """
         V9 self.current_time
        """
        if op_idx >= len(self.graph.eligible_ops):
            return {}, -10, True, {}

        op_id = self.graph.eligible_ops[op_idx]

        if mach_idx >= len(self.machines):
            return {}, -10, True, {}

        machine = self.machines[mach_idx]

        # makespan
        H_t = self.graph.get_makespan_lower_bound()

        # start_timeschedule_operation
        self.graph.schedule_operation(op_id, machine.id, self.current_time)

        #  current_time
        # self.current_time = max(self.current_time, self.graph.current_time)

        # makespan
        H_t_plus_1 = self.graph.get_makespan_lower_bound()

        # reward
        from rewards.lower_reward import compute_lower_reward
        reward = compute_lower_reward(
            H_t=H_t,
            H_t_plus_1=H_t_plus_1,
            graph=self.graph,
            tmax_constraints=self.tmax_constraints,
            setup_times_map=self.setup_times_map,
            alpha_tmax=self._get_conf('alpha_tmax', 50.0),
            alpha_setup=self._get_conf('alpha_setup', 1.0)
        )

        done = len(self.graph.eligible_ops) == 0
        if done:
            self._update_completed_jobs_stats()

        return self._get_lower_state(), reward, done, {}

    def _advance_time_and_inject_jobs(self):
        """Advance clock to the next op completion / machine idle, then mark
        any due ops as completed and inject new jobs from the dataset."""
        if not self.graph or not self.graph.operations:
            if self.job_counter < self.n_jobs:
                self._try_generate_new_jobs(count=1)
            return

        pending_end_times = []
        past_uncompleted = []
        for op in self.graph.operations.values():
            if op.is_scheduled and not op.is_completed:
                if op.scheduled_end > self.current_time:
                    pending_end_times.append(op.scheduled_end)
                else:
                    past_uncompleted.append(op)

        if pending_end_times:
            target_time = min(pending_end_times)
        else:
            future_idle = [m.available_time for m in self.machines
                           if m.available_time > self.current_time]
            target_time = min(future_idle) if future_idle else self.current_time + 1.0

        if target_time > self.current_time:
            self.current_time = target_time

        completed_count = 0
        epsilon = 1e-4
        for op in self.graph.operations.values():
            if op.is_scheduled and not op.is_completed:
                if (op.scheduled_end <= self.current_time + epsilon and
                        op.scheduled_start <= self.current_time + epsilon):
                    op.is_completed = True
                    completed_count += 1
                    self.execution_history.append({
                        'job_id': op.job_id,
                        'op_id': op.id,
                        'machine': op.scheduled_machine,
                        'start': op.scheduled_start,
                        'end': op.scheduled_end,
                    })

        self._process_completed_ops()

    def _process_completed_ops(self):
        """Detect job completions and queue replacement jobs from the BO pool."""
        completed_jobs = []
        job_ops_map = {}
        for op in self.graph.operations.values():
            if op.job_id not in job_ops_map:
                job_ops_map[op.job_id] = {'total': 0, 'completed': 0, 'scheduled': 0}
            job_ops_map[op.job_id]['total'] += 1
            if op.is_scheduled:
                job_ops_map[op.job_id]['scheduled'] += 1
            if op.is_completed:
                job_ops_map[op.job_id]['completed'] += 1

        for job_id, counts in job_ops_map.items():
            if counts['total'] == counts['completed'] and counts['total'] > 0:
                if job_id not in self._completed_job_ids:
                    completed_jobs.append(job_id)
                    self._completed_job_ids.add(job_id)

        if completed_jobs:
            self._try_generate_new_jobs(count=len(completed_jobs))

    def _try_generate_new_jobs(self, count=1):
        """Generate up to `count` new jobs (one-by-one) honoring n_jobs cap."""
        for _ in range(count):
            if self.job_counter >= self.n_jobs:
                break
            job_type = 'SDL_experiment'
            job_id = f"job_{self.job_counter}"
            if job_id in self.bo_delays_map:
                bo_delay = self.bo_delays_map[job_id]
            else:
                bo_delay = float(np.random.uniform(5.0, 15.0))
            arrival_time = self.current_time + bo_delay
            self._inject_atlas_job(job_type, arrival_time)

    """
    def _inject_new_job(self):
        if self.next_arrival_idx >= len(self.arrival_queue): return
        t = self.arrival_queue[self.next_arrival_idx]
        job = Job(id=f"job_{self.next_arrival_idx}", job_type="default", operations=[], arrival_time=t)

        #  Op  Ops
        for i in range(3):
            compatible = {m.id: 5.0 for m in self.machines[:2]}  # 
            job.operations.append({'id': f"{job.id}_op{i}", 'processing_times': compatible})

        self.job_buffer.add_job(job)
        self.next_arrival_idx += 1"""

    def _add_new_jobs_to_current_graph(self, new_jobs):
        """
          Job Job
         Stage 0  Replay
        """
        #   self.graph.reset()  self.graph.operations = {}
        #  restore()  Op

        for job in new_jobs:
            for i, op_data in enumerate(job.operations):
                #  Op
                #  op_data
                op = Operation(
                    id=op_data['id'],
                    job_id=job.id,
                    op_index=i,
                    processing_times=op_data['processing_times'],
                    op_type=op_data.get('op_type', 'default'),
                    material=op_data.get('material', '')
                )

                #   (Append)
                self.graph.add_operation(op)

        #  (Precedence Constraints)
        for job in new_jobs:
            for i, op_data in enumerate(job.operations):
                if i > 0:
                    prev_op_id = job.operations[i - 1]['id']
                    curr_op_id = op_data['id']
                    self.graph.add_precedence_edge(prev_op_id, curr_op_id)

        self.graph._update_eligible_ops()

    def _build_static_subproblem(self, jobs):
        for job in jobs:
            job_arrival = job.arrival_time
            for i, op_data in enumerate(job.operations):
                if not op_data.get('processing_times'):
                    print(f"[SDLEnv] Warning: {op_data['id']} has no processing_times")

                op = Operation(
                    id=op_data['id'],
                    job_id=job.id,
                    op_index=i,
                    processing_times=op_data['processing_times'],
                    op_type=op_data.get('op_type', 'default'),  #
                    material=op_data.get('material', '')  #
                )

                #
                available_machines = [m.id for m in self.machines if m.id in op.processing_times]
                if not available_machines:
                    print(f"[SDLEnv] Warning: {op.id} has no available machines "
                          f"(needs {list(op.processing_times.keys())})")
                op.arrival_time = job_arrival
                self.graph.add_operation(op)
                if i > 0:
                    # Ajob.operationsops
                    if i - 1 < len(job.operations):
                        prev_op_id = job.operations[i - 1]['id']
                        curr_op_id = op.id

                        #
                        # print(f"  [LINK] edge: {prev_op_id} -> {curr_op_id}")
                        self.graph.add_precedence_edge(job.operations[i - 1]['id'], op.id)

                    #  Bgraph.operationsops
                    else:
                        expected_prev_id = f"{job.id}_op{i}"  #
                        if expected_prev_id in self.graph.operations:
                            self.graph.add_precedence_edge(expected_prev_id, op.id)

        print(f"\n  _build_static_subproblem done")
        print(f"  graph ops: {len(self.graph.operations)}")
        print(f"  precedence edges: {len(self.graph.precedence_edges)}")
        print(f"  Eligible ops: {len(self.graph.eligible_ops)}")
        print(f"  Eligible ops (10): {self.graph.eligible_ops[:10]}")

    """
    def all_jobs_arrived(self):
        return self.next_arrival_idx >= len(self.arrival_queue)"""

    def _check_all_jobs_completed(self):
        """All jobs generated, buffer drained, and graph complete."""
        condition1 = self.job_counter >= self.n_jobs
        condition2 = len(self.job_buffer) == 0
        condition3 = self.graph.is_complete() if self.graph else True
        return condition1 and condition2 and condition3

    """
    def _generate_arrivals(self, n, load):
        return list(np.cumsum([np.random.exponential(5.0) for _ in range(n)]))"""

    def _get_hla_state(self):
        #
        queue_features_base = torch.zeros(3)  # [n_cached, avg_slack, var_load]
        machine_status = torch.zeros(2)

        #  V31:  (+3) -
        material_counts = {'sample': 0}
        for job in self.job_buffer.buffer:
            #   operations
            if job.operations and len(job.operations) > 0:
                material = job.operations[0].get('material', '')
                if material in material_counts:
                    material_counts[material] += 1

        material_stats = torch.tensor([
            material_counts["sample"]
        ], dtype=torch.float)

        #  V32:  (+1) -
        ms_machines = [m for m in self.machines if 'ICP_MS' in m.id]
        total_queue = sum(len(m.queue) for m in ms_machines) if ms_machines else 0
        bottleneck_status = torch.tensor([min(total_queue / 10.0, 1.0)])

        #
        queue_features_full = torch.cat([
            queue_features_base,  # 3
            material_stats,  # 3
            bottleneck_status  # 1
        ])  #  7

        return {
            'graph': None,
            'queue_features': queue_features_full,  #  7
            'machine_status': machine_status  # 2
        }

    def _get_lower_state(self):
        g_data = self.graph.to_pyg_data()
        g_data.current_time = self.current_time
        return {'op_graph': g_data, 'mach_features': torch.zeros(len(self.machines), 3)}

    def _update_completed_jobs_stats(self):
        pass

    def _parse_setup_times(self, config: Dict) -> Dict[Tuple[str, str, str], float]:
        """
         YAML  setup_times
        Returns: {(from_material, to_material, machine_type): time}
        """
        result = {}
        for machine_type, setup_list in config.items():
            if isinstance(setup_list, list):
                for item in setup_list:
                    key = (item['from_material'], item['to_material'], machine_type)
                    result[key] = item['time']
        return result