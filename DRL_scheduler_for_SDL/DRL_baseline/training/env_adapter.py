"""SDL environment adapter for the conventional DRL (Lei Kun) baseline."""
import torch
import numpy as np
from typing import Dict, List, Tuple
from copy import deepcopy


class SDLEnvAdapter:
    """Bridges SDL's DisjunctiveGraph to the MultiPPO state representation."""
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.device = torch.device(config.device)
        self.completed_jobs_last_step = set()
        assert env.machines == sorted(env.machines, key=lambda m: m.id), \
            "Machine list is not sorted!"
        self.machine_id_to_idx = {m.id: idx for idx, m in enumerate(env.machines)}
        self.idx_to_machine_id = {idx: m.id for idx, m in enumerate(env.machines)}

        self.has_tmax_checking = hasattr(env, 'mask_manager')

    def _is_job_completed(self, job_id: str) -> bool:
        """True iff job_id just transitioned to fully completed."""
        job_ops = [op for op in self.env.graph.operations.values()
                   if op.job_id == job_id]

        if not job_ops:
            return False

        all_completed = all(op.is_completed for op in job_ops)

        if all_completed and job_id not in self.completed_jobs_last_step:
            self.completed_jobs_last_step.add(job_id)
            return True

        return False

    def _get_job_makespan(self, job_id: str) -> float:
        """Makespan of a single job (last end - first start)."""
        job_ops = [op for op in self.env.graph.operations.values()
                   if op.job_id == job_id]

        if not job_ops:
            return 0.0

        start_times = [op.scheduled_start for op in job_ops
                       if op.is_scheduled or op.is_completed]
        end_times = [op.scheduled_end for op in job_ops
                     if op.is_scheduled or op.is_completed]

        if not start_times or not end_times:
            return 0.0

        job_start = min(start_times)
        job_end = max(end_times)

        return job_end - job_start

    def get_state_for_multippo(self) -> Dict:
        """Convert the SDL graph state to MultiPPO input tensors.

        adj/fea are 2D (Lei Kun's GraphCNN uses sparse 2D); the rest carry a
        batch=1 dimension expected by the actor.
        """
        graph = self.env.graph

        eligible_ops = graph.eligible_ops
        n_candidates = len(eligible_ops)

        if n_candidates == 0:
            return None

        op_list = sorted([
            op_id for op_id, op in graph.operations.items()
            if not op.is_completed
        ])
        n_ops = len(op_list)
        op_to_idx = {op_id: idx for idx, op_id in enumerate(op_list)}

        node_features = []
        for op_id in op_list:
            op = graph.operations[op_id]
            lb = self._compute_op_lb(op_id) / 100.0
            is_scheduled = 1.0 if op.is_scheduled else 0.0
            node_features.append([lb, is_scheduled])

        fea = torch.tensor(node_features, dtype=torch.float32, device=self.device)

        edge_indices = []
        edge_values = []
        for pred_id, succ_id in graph.precedence_edges:
            if pred_id in op_to_idx and succ_id in op_to_idx:
                pred_idx = op_to_idx[pred_id]
                succ_idx = op_to_idx[succ_id]
                edge_indices.append([pred_idx, succ_idx])
                edge_values.append(1.0)

        if len(edge_indices) > 0:
            edge_indices_t = torch.tensor(edge_indices, dtype=torch.long, device=self.device).t()
            edge_values_t = torch.tensor(edge_values, dtype=torch.float32, device=self.device)
            adj = torch.sparse_coo_tensor(
                edge_indices_t,
                edge_values_t,
                size=(n_ops, n_ops),
                device=self.device
            )
        else:
            adj = torch.sparse_coo_tensor(
                torch.zeros((2, 0), dtype=torch.long, device=self.device),
                torch.zeros(0, dtype=torch.float32, device=self.device),
                size=(n_ops, n_ops),
                device=self.device
            )

        candidate_indices = [op_to_idx[op_id] for op_id in eligible_ops]

        max_candidates = self.config.n_j
        if len(candidate_indices) > 0:
            padding_value = candidate_indices[0]
        else:
            padding_value = 0
        padded_candidates = candidate_indices + [padding_value] * (max_candidates - len(candidate_indices))
        padded_candidates = padded_candidates[:max_candidates]

        candidate = torch.tensor([padded_candidates], dtype=torch.long, device=self.device)

        mask = torch.zeros((1, max_candidates), dtype=torch.bool, device=self.device)
        mask[0, n_candidates:] = True

        n_machines = len(self.env.machines)
        mask_mch = torch.ones((1, n_ops, n_machines), dtype=torch.bool, device=self.device)

        for idx, op_id in enumerate(op_list):
            op = graph.operations[op_id]
            for mach_id in op.compatible_machines:
                if mach_id in self.machine_id_to_idx:
                    mach_idx = self.machine_id_to_idx[mach_id]
                    mask_mch[0, idx, mach_idx] = False

        dur = torch.zeros((1, n_ops, n_machines), dtype=torch.float32, device=self.device)

        for idx, op_id in enumerate(op_list):
            op = graph.operations[op_id]
            for mach_id, proc_time in op.processing_times.items():
                if mach_id in self.machine_id_to_idx:
                    mach_idx = self.machine_id_to_idx[mach_id]
                    dur[0, idx, mach_idx] = proc_time

        mch_time = torch.zeros((1, n_machines), dtype=torch.float32, device=self.device)
        for mach_id, avail_time in self.env.machine_available_time.items():
            if mach_id in self.machine_id_to_idx:
                mach_idx = self.machine_id_to_idx[mach_id]
                mch_time[0, mach_idx] = avail_time

        return {
            'adj': adj,  # 2D: [n_ops, n_ops]
            'fea': fea,  # 2D: [n_ops, feature_dim]
            'candidate': candidate,
            'mask': mask,
            'mask_mch': mask_mch,
            'dur': dur,
            'mch_time': mch_time,
            'op_list': op_list,
            'eligible_ops': eligible_ops
        }

    def _compute_op_lb(self, op_id: str) -> float:
        """Operation lower bound (earliest possible end time)."""
        op = self.env.graph.operations[op_id]
        if op.is_scheduled:
            return op.scheduled_end

        pred_max = 0.0
        for pred_id in op.predecessors:
            pred_op = self.env.graph.operations[pred_id]
            if pred_op.is_scheduled:
                pred_max = max(pred_max, pred_op.scheduled_end)

        return pred_max + op.get_min_processing_time()

    def action_to_op_and_machine(self, op_action, mch_action, state):
        op_idx_in_candidate = op_action[0] if isinstance(op_action, np.ndarray) else op_action

        candidate = state['candidate']
        op_idx_in_op_list = candidate[0, op_idx_in_candidate].item()
        op_list = state['op_list']

        assert 0 <= op_idx_in_op_list < len(op_list), \
            f"Invalid op index: {op_idx_in_op_list} (op_list size: {len(op_list)})"

        op_id = op_list[op_idx_in_op_list]

        assert op_id in self.env.graph.operations, \
            f"Op {op_id} not found in graph!"

        assert op_id in state['eligible_ops'], \
            f"Op {op_id} not in eligible_ops: {state['eligible_ops']}"

        mch_idx = mch_action[0] if isinstance(mch_action, np.ndarray) else mch_action
        machine_id = self.idx_to_machine_id[mch_idx]

        op = self.env.graph.operations[op_id]
        if machine_id not in op.processing_times:
            print(f"WARNING: machine {machine_id} not in processing_times for {op_id}; "
                  f"falling back to {list(op.processing_times.keys())[0]}")
            machine_id = list(op.processing_times.keys())[0]

        return op_id, machine_id

    def compute_reward(self, prev_makespan, curr_makespan, done,
                       scheduled_op_id, scheduled_machine):
        """
        Baseline reward: makespan increment only.
        No Tmax penalty — constraint-awareness is NOT part of this baseline.
        """
        step_delta = curr_makespan - prev_makespan
        return -step_delta * 2


    def _estimate_gap_with_bfs(self, scheduled_op, to_step):
        """BFS-based gap estimator (same shape as the Forward Mask logic).
        Returns predicted gap, or None if no target reachable."""
        queue = [(next_id, []) for next_id in scheduled_op.successors]
        visited = set()
        target_op = None
        future_ops_chain = []

        while queue:
            curr_id, path = queue.pop(0)
            if curr_id not in self.env.graph.operations:
                continue

            curr_op = self.env.graph.operations[curr_id]
            new_path = path + [curr_op]

            if curr_op.op_type == to_step:
                target_op = curr_op
                future_ops_chain = new_path
                break

            if curr_id not in visited:
                visited.add(curr_id)
                for succ_id in curr_op.successors:
                    queue.append((succ_id, new_path))

        if not target_op:
            return None

        accumulated_time = scheduled_op.scheduled_end

        for chain_op in future_ops_chain:
            min_proc = chain_op.get_min_processing_time()

            min_machine_avail = float('inf')
            for m_id in chain_op.compatible_machines:
                if m_id in self.env.machines_dict:
                    m_avail = self.env.machines_dict[m_id].available_time
                    min_machine_avail = min(min_machine_avail, m_avail)

            if min_machine_avail == float('inf'):
                min_machine_avail = accumulated_time

            next_start = max(accumulated_time, min_machine_avail)

            if chain_op == target_op:
                est_target_start = next_start
            else:
                accumulated_time = next_start + min_proc

        predicted_gap = est_target_start - scheduled_op.scheduled_end
        return predicted_gap


# ============================================================
# Fast CP-SAT feasibility probe (training-only, no Gurobi)
# ============================================================

def _check_feasibility_cpsat(env, jobs_to_release, time_limit_sec=0.5):
    """
    Pure feasibility probe via OR-Tools CP-SAT — no objective, no MIP search.

    Returns True if CP-SAT finds a feasible solution OR hits the time limit
    without proving infeasibility (optimistic: treat unknown as feasible).
    Returns False only when CP-SAT definitively proves INFEASIBLE.

    Does NOT mutate env.graph.
    """
    from ortools.sat.python import cp_model as _cp

    SCALE = 10
    BIG   = 100000

    graph        = env.graph
    current_time = env.current_time
    tmax_cstrs   = env.tmax_constraints
    machines_ids = [m.id for m in env.machines]

    existing_ops = {
        op_id: op for op_id, op in graph.operations.items()
        if not op.is_completed
    }

    new_ops = {}
    new_precedences = []

    for job in jobs_to_release:
        for i, op_data in enumerate(job.operations):
            op_id = op_data['id']
            new_ops[op_id] = {
                'id':               op_id,
                'job_id':           job.id,
                'op_index':         i,
                'op_type':          op_data.get('op_type', 'default'),
                'processing_times': op_data['processing_times'],
                'arrival_time':     job.arrival_time,
                'predecessors':     [job.operations[i-1]['id']] if i > 0 else [],
            }
        for i in range(1, len(job.operations)):
            new_precedences.append((
                job.operations[i-1]['id'],
                job.operations[i]['id']
            ))

    all_op_ids = list(existing_ops.keys()) + list(new_ops.keys())

    def _get_op_field(op_id, field):
        if op_id in existing_ops:
            op = existing_ops[op_id]
            if field == 'job_id':           return op.job_id
            if field == 'op_index':         return op.op_index
            if field == 'op_type':          return op.op_type
            if field == 'processing_times': return op.processing_times
            if field == 'arrival_time':     return getattr(op, 'arrival_time', 0.0)
            if field == 'predecessors':     return op.predecessors
            if field == 'is_running':
                return (op.is_scheduled and
                        op.scheduled_start <= current_time + 1e-4)
            if field == 'scheduled_start':  return op.scheduled_start
            if field == 'scheduled_end':    return op.scheduled_end
            if field == 'scheduled_machine': return op.scheduled_machine
        else:
            d = new_ops[op_id]
            if field == 'is_running':       return False
            if field == 'scheduled_start':  return 0.0
            if field == 'scheduled_end':    return 0.0
            if field == 'scheduled_machine': return None
            return d.get(field)

    mdl    = _cp.CpModel()
    solver = _cp.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_sec
    solver.parameters.num_search_workers  = 1

    op_start    = {}
    op_end      = {}
    op_presence = {}
    machine_intervals = {mid: [] for mid in machines_ids}

    for op_id in all_op_ids:
        proc_times = _get_op_field(op_id, 'processing_times')
        arrival    = _get_op_field(op_id, 'arrival_time')
        is_running = _get_op_field(op_id, 'is_running')

        if is_running:
            s_val = int(_get_op_field(op_id, 'scheduled_start') * SCALE)
            e_val = int(_get_op_field(op_id, 'scheduled_end')   * SCALE)
            s_var = mdl.NewConstant(s_val)
            e_var = mdl.NewConstant(e_val)
            op_start[op_id]    = s_var
            op_end[op_id]      = e_var
            op_presence[op_id] = {}
            m_fixed = _get_op_field(op_id, 'scheduled_machine')
            if m_fixed and m_fixed in machine_intervals:
                dur  = e_val - s_val
                ivar = mdl.NewFixedSizeIntervalVar(s_var, dur, f'iv_{op_id}_{m_fixed}')
                machine_intervals[m_fixed].append(ivar)
            continue

        lb_scaled = int(max(current_time, arrival) * SCALE)
        s_var = mdl.NewIntVar(lb_scaled, BIG, f's_{op_id}')
        e_var = mdl.NewIntVar(lb_scaled, BIG, f'e_{op_id}')
        op_start[op_id] = s_var
        op_end[op_id]   = e_var

        compat_machines = [mid for mid in machines_ids if mid in proc_times]
        if not compat_machines:
            return False

        pres_vars    = {}
        ivars_for_op = []
        for mid in compat_machines:
            dur_scaled = max(1, int(proc_times[mid] * SCALE))
            p_var = mdl.NewBoolVar(f'p_{op_id}_{mid}')
            iv = mdl.NewOptionalIntervalVar(
                s_var, dur_scaled, e_var, p_var, f'iv_{op_id}_{mid}'
            )
            pres_vars[mid] = p_var
            ivars_for_op.append((mid, p_var, dur_scaled, iv))
            machine_intervals[mid].append(iv)

        mdl.AddExactlyOne(pres_vars.values())
        dur_expr = sum(p * d for _, p, d, _ in ivars_for_op)
        mdl.Add(e_var == s_var + dur_expr)
        op_presence[op_id] = pres_vars

    for mid, ivlist in machine_intervals.items():
        if len(ivlist) >= 2:
            mdl.AddNoOverlap(ivlist)

    def _add_prec(pred_id, succ_id):
        if pred_id not in op_end or succ_id not in op_start:
            return
        mdl.Add(op_start[succ_id] >= op_end[pred_id])

    for op_id in existing_ops:
        op = existing_ops[op_id]
        for pred_id in op.predecessors:
            _add_prec(pred_id, op_id)

    for pred_id, succ_id in new_precedences:
        _add_prec(pred_id, succ_id)

    for op_id, d in new_ops.items():
        for pred_id in d['predecessors']:
            if pred_id in existing_ops:
                _add_prec(pred_id, op_id)

    job_type_map = {}
    for op_id in all_op_ids:
        jid   = _get_op_field(op_id, 'job_id')
        otype = _get_op_field(op_id, 'op_type')
        oidx  = _get_op_field(op_id, 'op_index')
        job_type_map.setdefault(jid, {}).setdefault(otype, []).append((oidx, op_id))

    for (from_step, to_step, max_iv) in tmax_cstrs:
        max_iv_scaled = int(max_iv * SCALE)
        for jid, type_dict in job_type_map.items():
            to_list   = type_dict.get(to_step,   [])
            from_list = type_dict.get(from_step, [])
            if not to_list or not from_list:
                continue
            for to_idx, to_op_id in to_list:
                cands = [(fi, fid) for fi, fid in from_list if fi < to_idx]
                if not cands:
                    continue
                _, from_op_id = max(cands, key=lambda x: x[0])

                if from_op_id not in op_end:
                    from_op_obj = graph.operations.get(from_op_id)
                    if from_op_obj and (from_op_obj.is_completed or from_op_obj.is_scheduled):
                        deadline_scaled = int(
                            (from_op_obj.scheduled_end + max_iv) * SCALE
                        )
                        if to_op_id in op_start:
                            mdl.Add(op_start[to_op_id] <= deadline_scaled)
                    continue

                if to_op_id in op_start:
                    mdl.Add(
                        op_start[to_op_id] - op_end[from_op_id] <= max_iv_scaled
                    )

    status = solver.Solve(mdl)

    if status == _cp.INFEASIBLE:
        return False
    return True


def try_release_combination_fast(env, jobs_to_release, time_limit_sec=0.5):
    """
    Drop-in replacement for try_release_combination using CP-SAT feasibility
    probe instead of Gurobi MILP.  Does NOT mutate env.graph.

    Returns same dict shape as try_release_combination:
        {'feasible': bool, 'makespan': float, 'schedule': []}
    makespan is always 0.0 — not needed during training.
    """
    feasible = _check_feasibility_cpsat(env, jobs_to_release, time_limit_sec)
    return {
        'feasible': feasible,
        'makespan': 0.0,
        'schedule': []
    }


def progressive_release_strict_fast(env, time_limit_sec=0.5):
    """
    Fast training-time version of progressive_release_strict.

    Identical release logic (k = n down to 1, exhaustive C(n,k) search,
    pick largest feasible k) but uses CP-SAT feasibility probe instead of
    Gurobi MILP, cutting per-call time from 30-60 s to 0.1-0.5 s.

    Return format identical to progressive_release_strict:
        {'success': bool, 'released_jobs': list, 'makespan': float, 'schedule': []}
    """
    from itertools import combinations as _combinations

    buffer_jobs = list(env.job_buffer.buffer)
    n = len(buffer_jobs)

    if n == 0:
        return {'success': False, 'released_jobs': [], 'makespan': 0.0, 'schedule': []}

    for k in range(n, 0, -1):
        if k == n:
            combinations_to_try = [buffer_jobs]
        else:
            combinations_to_try = list(_combinations(buffer_jobs, k))

        feasible_combos = []
        for jobs_combo in combinations_to_try:
            result = try_release_combination_fast(env, list(jobs_combo), time_limit_sec)
            if result['feasible']:
                feasible_combos.append(list(jobs_combo))

        if feasible_combos:
            return {
                'success': True,
                'released_jobs': feasible_combos[0],
                'makespan': 0.0,
                'schedule': []
            }

    return {'success': False, 'released_jobs': [], 'makespan': 0.0, 'schedule': []}