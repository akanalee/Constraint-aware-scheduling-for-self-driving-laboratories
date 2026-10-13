import os
import time

import gymnasium as gym
import gurobipy as gp
import numpy as np
import yaml
from gurobipy import GRB

from configs import Config
from case_study.dataset_generator import DatasetGenerator
from .disjunctive_graph import DisjunctiveGraph, Machine, Operation
from .job_buffer import Job, JobBuffer


class SDLEnv(gym.Env):
    def __init__(self, config):
        super().__init__()
        if isinstance(config, dict) and not isinstance(config, Config):
            config = Config(config)

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
        self.buffer_size = self._get_conf('buffer_size', 100)

        self.job_counter = 0
        self._completed_job_ids = set()
        self.dataset_generator = DatasetGenerator(config_path=config_path)

        tmax_constraints_raw = self._get_conf('tmax_constraints', [])
        self.tmax_constraints = self._normalize_tmax_constraints(tmax_constraints_raw)

        self.job_buffer = JobBuffer(max_size=self.buffer_size)
        self.current_time = 0.0

        machine_list = []

        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    case_study_config = yaml.safe_load(f)

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

        self.machines = self._init_machines(machine_list)
        self.machines.sort(key=lambda m: m.id)
        self.machines_dict = {m.id: m for m in self.machines}
        self.graph = None
        self.bo_delays_map = {}

        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Dict({})

    @property
    def machine_available_time(self):
        if not hasattr(self, 'machines') or not self.machines:
            return {}
        return {m.id: m.available_time for m in self.machines}

    def _get_conf(self, key, default):
        if hasattr(self.config, 'get'): return self.config.get(key, default)
        return getattr(self.config, key, default)

    def _init_machines(self, config_list):
        machines = []
        if not config_list:
            raise RuntimeError("[SDLEnv] machine_list is empty - check sdl_config.yaml process_steps indentation")

        if isinstance(config_list, dict):
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
        if not constraints_raw:
            return []

        normalized = []
        for constraint in constraints_raw:
            if isinstance(constraint, dict):
                from_step = constraint.get('from_step') or constraint.get('from_id') or constraint.get('A')
                to_step = constraint.get('to_step') or constraint.get('to_id') or constraint.get('B')
                max_interval = constraint.get('max_interval') or constraint.get('Tmax')

                if from_step and to_step and max_interval is not None:
                    normalized.append((from_step, to_step, max_interval))
            elif isinstance(constraint, (tuple, list)) and len(constraint) == 3:
                normalized.append(tuple(constraint))

        return normalized

    def reset(self, seed=None, options=None, skip_warmstart=False):
        super().reset(seed=seed)
        self._skip_warmstart = skip_warmstart

        self.job_buffer.clear()
        self.current_time = 0.0
        self.job_counter = 0
        self._completed_job_ids = set()
        for m in self.machines:
            m.available_time = 0.0
            m.queue = []

        self.graph = DisjunctiveGraph(
            machines={m.id: m for m in self.machines},
            env=self
        )

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
            ]

            for job_type, arrival_time in warmstart_jobs:
                self._inject_atlas_job(job_type, arrival_time)

        self.machines.sort(key=lambda m: m.id)
        self.machines_dict = {m.id: m for m in self.machines}
        return {}, {}

    def _solve_subproblem_with_lower(self, rule=None, time_limit=60, mip_gap_abs=None):
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

        all_ops = [op for op in self.graph.operations.values() if not op.is_completed]
        if not all_ops:
            return {'status': 'ok', 'makespan': 0.0, 'schedule': []}

        try:
            model = gp.Model("SDL_FJSP")
            model.setParam('OutputFlag', 0)
            model.setParam('Presolve', 2)
            model.setParam('MIPFocus', 1)
            if time_limit is not None:
                model.setParam('TimeLimit', time_limit)
            if mip_gap_abs is not None:
                model.setParam('MIPGapAbs', mip_gap_abs)
        except gp.GurobiError as e:
            print(f"[MILP] Gurobi init error: {e}")
            return {'status': 'error', 'makespan': 0.0, 'schedule': []}

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

        for (from_step, to_step, max_interval) in self.tmax_constraints:
            for op in all_ops:
                if op.op_type != to_step:
                    continue
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
                elif pred.is_completed or pred.is_scheduled:
                    model.addConstr(
                        op_vars[op.id]['start'] <= pred.scheduled_end + max_interval
                    )

        for machine in self.machines:
            mid = machine.id
            pot_ops = [op for op in all_ops if mid in op_vars[op.id]['mach_vars']]
            if len(pot_ops) < 2:
                continue

            for i in range(len(pot_ops)):
                for j in range(i + 1, len(pot_ops)):
                    op_a = pot_ops[i]
                    op_b = pot_ops[j]

                    x_a = op_vars[op_a.id]['mach_vars'][mid]
                    x_b = op_vars[op_b.id]['mach_vars'][mid]
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

        makespan_var = model.addVar(lb=0.0, ub=horizon, name='makespan')
        for op in all_ops:
            model.addConstr(makespan_var >= op_vars[op.id]['end'])
        model.setObjective(makespan_var, GRB.MINIMIZE)

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

        model.update()
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
        job_id = f"job_{self.job_counter}"
        self.job_counter += 1

        job = self.dataset_generator.generate_job(
            job_id=job_id,
            job_type=job_type,
            arrival_time=arrival_time
        )

        job_obj = Job(
            id=job['id'],
            job_type=job['type'],
            operations=job['operations'],
            arrival_time=job['arrival_time'],
            priority=job.get('priority', 'normal')
        )

        self.job_buffer.add_job(job_obj)

    def _advance_time_and_inject_jobs(self):
        if not self.graph or not self.graph.operations:
            if self.job_counter < self.n_jobs:
                self._try_generate_new_jobs(count=1)
            return

        pending_end_times = []
        for op in self.graph.operations.values():
            if op.is_scheduled and not op.is_completed:
                if op.scheduled_end > self.current_time:
                    pending_end_times.append(op.scheduled_end)

        if pending_end_times:
            target_time = min(pending_end_times)
        else:
            future_idle = [m.available_time for m in self.machines
                           if m.available_time > self.current_time]
            target_time = min(future_idle) if future_idle else self.current_time + 1.0

        if target_time > self.current_time:
            self.current_time = target_time

        epsilon = 1e-4
        for op in self.graph.operations.values():
            if op.is_scheduled and not op.is_completed:
                if (op.scheduled_end <= self.current_time + epsilon and
                        op.scheduled_start <= self.current_time + epsilon):
                    op.is_completed = True

        self._process_completed_ops()

    def _process_completed_ops(self):
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
        for _ in range(count):
            if self.job_counter >= self.n_jobs:
                break
            job_type = 'SDL_experiment'
            job_id = f"job_{self.job_counter}"
            if job_id in self.bo_delays_map:
                bo_delay = self.bo_delays_map[job_id]
            else:
                bo_delay = float(np.clip(np.random.normal(10.0, 2.5), 5.0, 15.0)) / 60.0
            arrival_time = self.current_time + bo_delay
            self._inject_atlas_job(job_type, arrival_time)

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
                    op_type=op_data.get('op_type', 'default'),
                    material=op_data.get('material', '')
                )

                available_machines = [m.id for m in self.machines if m.id in op.processing_times]
                if not available_machines:
                    print(f"[SDLEnv] Warning: {op.id} has no available machines "
                          f"(needs {list(op.processing_times.keys())})")
                op.arrival_time = job_arrival
                self.graph.add_operation(op)
                if i > 0:
                    self.graph.add_precedence_edge(job.operations[i - 1]['id'], op.id)

    def _check_all_jobs_completed(self):
        condition1 = self.job_counter >= self.n_jobs
        condition2 = len(self.job_buffer) == 0
        condition3 = self.graph.is_complete() if self.graph else True
        return condition1 and condition2 and condition3
