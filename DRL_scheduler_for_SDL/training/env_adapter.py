from itertools import combinations
from typing import Dict

import numpy as np
import torch

from utils import profiler
from utils.feasibility import check_release_feasibility
from utils.profiler import perf_counter


class SDLEnvAdapter:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.device = torch.device(config.device)
        assert env.machines == sorted(env.machines, key=lambda m: m.id), \
            "Machine list is not sorted!"
        self.machine_id_to_idx = {m.id: idx for idx, m in enumerate(env.machines)}
        self.idx_to_machine_id = {idx: m.id for idx, m in enumerate(env.machines)}

        self.has_tmax_checking = True

    def get_state_for_fss(self) -> Dict:
        t_state0 = perf_counter()
        graph = self.env.graph

        eligible_ops = graph.eligible_ops
        n_candidates = len(eligible_ops)
        if n_candidates == 0:
            profiler.add_time('state_construction', perf_counter() - t_state0)
            return None

        op_list = sorted([
            op_id for op_id, op in graph.operations.items()
            if not op.is_completed
        ])
        n_ops = len(op_list)
        op_to_idx = {op_id: idx for idx, op_id in enumerate(op_list)}

        _ALPHA = self.config.et_normalize_coef

        has_tmax = self.has_tmax_checking and hasattr(self.env, 'tmax_constraints')
        if has_tmax:
            job_op_type_map = {}
            for oid, o in graph.operations.items():
                key = (o.job_id, o.op_type)
                if key not in job_op_type_map:
                    job_op_type_map[key] = (oid, o)

        # node features [f1 status, f2 earliest start, f3 min processing time,
        #                f4 constraint-induced urgency, f5 instrument flexibility]
        t_urgency = 0.0
        node_features = []
        for op_id in op_list:
            op = graph.operations[op_id]

            is_scheduled = 1.0 if op.is_scheduled else 0.0

            if op.is_scheduled:
                est_start_raw = op.scheduled_start
                min_proc_raw  = op.processing_times.get(op.scheduled_machine,
                                    op.get_min_processing_time())
            else:
                pred_end = 0.0
                for pred_id in op.predecessors:
                    pred_op = graph.operations[pred_id]
                    if pred_op.is_scheduled or pred_op.is_completed:
                        pred_end = max(pred_end, pred_op.scheduled_end)

                est_start_raw = float('inf')
                min_proc_raw  = float('inf')
                for mach_id, proc_time in op.processing_times.items():
                    if mach_id in self.env.machines_dict:
                        m_avail = self.env.machines_dict[mach_id].available_time
                        s = max(pred_end, m_avail)
                        if s < est_start_raw:
                            est_start_raw = s
                    if proc_time < min_proc_raw:
                        min_proc_raw = proc_time

                if est_start_raw == float('inf'):
                    est_start_raw = pred_end
                if min_proc_raw == float('inf'):
                    min_proc_raw = op.get_min_processing_time()

            f2 = est_start_raw / _ALPHA
            f3 = min_proc_raw  / _ALPHA

            t_u0 = perf_counter()
            urgency = 0.0
            if has_tmax:
                urgency_candidates = []

                for constraint in self.env.tmax_constraints:
                    from_step, to_step, max_interval = constraint
                    if max_interval <= 0:
                        continue

                    from_entry = job_op_type_map.get((op.job_id, from_step))
                    to_entry   = job_op_type_map.get((op.job_id, to_step))
                    if not from_entry or not to_entry:
                        continue

                    from_op_id_c, from_op = from_entry
                    to_op_id_c,   to_op   = to_entry

                    if not (from_op.is_scheduled or from_op.is_completed):
                        if op_id != from_op_id_c:
                            continue
                        remaining_window = max_interval
                    else:
                        if to_op.is_scheduled or to_op.is_completed:
                            continue
                        if not self._is_on_path(op_id, from_op_id_c, to_op_id_c):
                            continue
                        deadline = from_op.scheduled_end + max_interval
                        remaining_window = deadline - est_start_raw

                    if remaining_window <= 1e-6:
                        urgency_candidates.append(2.0)
                        continue

                    est_to_start = self._bfs_est_start(op_id, to_op_id_c, est_start_raw)
                    if est_to_start == float('inf'):
                        continue

                    d_bfs = est_to_start - est_start_raw
                    u = d_bfs / remaining_window
                    urgency_candidates.append(u)

                if urgency_candidates:
                    urgency = max(urgency_candidates)
                    if urgency == float('inf'):
                        urgency = 2.0
            t_urgency += perf_counter() - t_u0

            n_compat = sum(1 for m_id in op.processing_times
                           if m_id in self.machine_id_to_idx)
            f5 = n_compat / 10.0

            node_features.append([is_scheduled, f2, f3, urgency, f5])

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
                edge_indices_t, edge_values_t,
                size=(n_ops, n_ops), device=self.device
            )
        else:
            adj = torch.sparse_coo_tensor(
                torch.zeros((2, 0), dtype=torch.long, device=self.device),
                torch.zeros(0, dtype=torch.float32, device=self.device),
                size=(n_ops, n_ops), device=self.device
            )

        candidate_indices = [op_to_idx[op_id] for op_id in eligible_ops]
        max_candidates = self.config.n_j
        padding_value = candidate_indices[0] if candidate_indices else 0
        padded_candidates = candidate_indices + [padding_value] * (max_candidates - len(candidate_indices))
        padded_candidates = padded_candidates[:max_candidates]
        candidate = torch.tensor([padded_candidates], dtype=torch.long, device=self.device)

        mask = torch.zeros((1, max_candidates), dtype=torch.bool, device=self.device)
        mask[0, n_candidates:] = True

        # bidirectional masking of instrument assignments
        t_mask0 = perf_counter()
        t_forward = 0.0
        t_backward = 0.0
        n_machines = len(self.env.machines)
        mask_mch = torch.ones((1, n_ops, n_machines), dtype=torch.bool, device=self.device)

        for idx, op_id in enumerate(op_list):
            op = graph.operations[op_id]
            compatible_machines = op.compatible_machines

            t_f0 = perf_counter()
            _fwd_checks = []
            if self.has_tmax_checking:
                for fc in self.env.tmax_constraints:
                    fs, ts, mi = fc
                    if op.op_type == fs:
                        chain = self._find_chain_to_target(op, ts)
                        if chain:
                            _fwd_checks.append(('A', chain, mi, None))
                    elif op.op_type != ts:
                        anc = self._find_scheduled_ancestor(op, fs)
                        if anc is not None:
                            chain = self._find_chain_to_target(op, ts)
                            if chain:
                                _fwd_checks.append(('mid', chain, mi, anc.scheduled_end))
            t_forward += perf_counter() - t_f0

            for mach_id in compatible_machines:
                if mach_id not in self.machine_id_to_idx:
                    continue

                mach_idx = self.machine_id_to_idx[mach_id]
                is_compatible = True

                pred_max = max([graph.operations[p].scheduled_end
                                for p in op.predecessors], default=self.env.current_time)
                curr_start_time = max(pred_max, self.env.machine_available_time.get(mach_id, 0))
                curr_processing_time = op.processing_times[mach_id]
                curr_end_time = curr_start_time + curr_processing_time

                # forward feasibility check
                t_f1 = perf_counter()
                for mode, chain, max_interval, anc_end in _fwd_checks:
                    s_lb_B = self._propagate_lb(chain, curr_end_time)
                    s_ddl_B = (curr_end_time + max_interval if mode == 'A'
                               else anc_end + max_interval)
                    if s_lb_B > s_ddl_B + 1e-4:
                        is_compatible = False
                        break
                t_forward += perf_counter() - t_f1

                # backward deadline validation
                t_b0 = perf_counter()
                if self.has_tmax_checking and is_compatible:
                    for bc in self.env.tmax_constraints:
                        fs, ts, mi = bc
                        if op.op_type == ts:
                            candidates = [o for o in graph.operations.values()
                                          if o.job_id == op.job_id and o.op_type == fs]
                            if candidates:
                                pred = candidates[0]
                                if pred.is_completed or pred.is_scheduled:
                                    deadline = pred.scheduled_end + mi
                                    if curr_start_time > deadline + 1e-4:
                                        is_compatible = False
                                        break
                t_backward += perf_counter() - t_b0

                mask_mch[0, idx, mach_idx] = not is_compatible

            physical_mask_status = []
            for mach_id in compatible_machines:
                if mach_id in self.machine_id_to_idx:
                    mach_idx = self.machine_id_to_idx[mach_id]
                    physical_mask_status.append(mask_mch[0, idx, mach_idx].item())

            if physical_mask_status and all(physical_mask_status):
                for mach_id in compatible_machines:
                    if mach_id in self.machine_id_to_idx:
                        mach_idx = self.machine_id_to_idx[mach_id]
                        mask_mch[0, idx, mach_idx] = False
        t_mask = perf_counter() - t_mask0

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

        profiler.add_time('state_construction', perf_counter() - t_state0)
        profiler.add_time('urgency_feature', t_urgency)
        profiler.add_time('mask_total', t_mask)
        profiler.add_time('mask_forward', t_forward)
        profiler.add_time('mask_backward', t_backward)
        return {
            'adj': adj,
            'fea': fea,
            'candidate': candidate,
            'mask': mask,
            'mask_mch': mask_mch,
            'dur': dur,
            'mch_time': mch_time,
            'op_list': op_list,
            'eligible_ops': eligible_ops
        }

    def _is_on_path(self, op_id: str, from_op_id: str, to_op_id: str) -> bool:
        graph = self.env.graph
        if op_id == from_op_id or op_id == to_op_id:
            return True
        queue = [from_op_id]
        visited = set()
        while queue:
            curr_id = queue.pop(0)
            if curr_id in visited:
                continue
            visited.add(curr_id)
            if curr_id == to_op_id:
                continue
            if curr_id not in graph.operations:
                continue
            for succ_id in graph.operations[curr_id].successors:
                if succ_id == op_id:
                    return self._can_reach(op_id, to_op_id)
                if succ_id not in visited:
                    queue.append(succ_id)
        return False

    def _can_reach(self, from_id: str, to_id: str) -> bool:
        graph = self.env.graph
        queue = [from_id]
        visited = set()
        while queue:
            curr_id = queue.pop(0)
            if curr_id == to_id:
                return True
            if curr_id in visited:
                continue
            visited.add(curr_id)
            if curr_id not in graph.operations:
                continue
            for succ_id in graph.operations[curr_id].successors:
                queue.append(succ_id)
        return False

    def _bfs_est_start(self, from_op_id: str, to_op_id: str,
                       from_end_time: float) -> float:
        graph = self.env.graph
        if from_op_id not in graph.operations:
            return float('inf')

        from_op = graph.operations[from_op_id]
        queue = [(succ_id, from_end_time) for succ_id in from_op.successors]
        visited = set()

        while queue:
            curr_id, acc_time = queue.pop(0)
            if curr_id not in graph.operations or curr_id in visited:
                continue
            visited.add(curr_id)
            curr_op = graph.operations[curr_id]

            min_machine_avail = float('inf')
            for m_id in curr_op.compatible_machines:
                if m_id in self.env.machines_dict:
                    m_avail = self.env.machines_dict[m_id].available_time
                    min_machine_avail = min(min_machine_avail, m_avail)
            if min_machine_avail == float('inf'):
                min_machine_avail = acc_time

            est_start = max(acc_time, min_machine_avail)

            if curr_id == to_op_id:
                return est_start

            min_proc = curr_op.get_min_processing_time()
            next_acc = est_start + min_proc
            for succ_id in curr_op.successors:
                if succ_id not in visited:
                    queue.append((succ_id, next_acc))

        return float('inf')

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
            fallback_machine = list(op.processing_times.keys())[0]
            machine_id = fallback_machine

        return op_id, machine_id

    def _find_chain_to_target(self, start_op, to_step):
        graph = self.env.graph
        queue = [(sid, []) for sid in start_op.successors]
        visited = set()
        while queue:
            cid, path = queue.pop(0)
            if cid not in graph.operations:
                continue
            cop = graph.operations[cid]
            if cop.job_id != start_op.job_id:
                continue
            new_path = path + [cop]
            if cop.op_type == to_step:
                return new_path
            if cid not in visited:
                visited.add(cid)
                for sid in cop.successors:
                    queue.append((sid, new_path))
        return None

    def _propagate_lb(self, chain_incl_B, c_lb_prev):
        machines_dict = self.env.machines_dict
        c_lb = c_lb_prev
        n = len(chain_incl_B)
        for i, cop in enumerate(chain_incl_B):
            is_last = (i == n - 1)
            t_releas = getattr(cop, 'arrival_time', 0.0)
            best = float('inf')
            for m_id in cop.compatible_machines:
                if m_id not in machines_dict:
                    continue
                t_avail = machines_dict[m_id].available_time
                if is_last:
                    best = min(best, max(c_lb, t_avail))
                else:
                    p_km = cop.processing_times.get(m_id)
                    if p_km is not None:
                        best = min(best, max(c_lb, t_avail, t_releas) + p_km)
            c_lb = best if best < float('inf') else c_lb
        return c_lb

    def _find_scheduled_ancestor(self, op, from_step):
        graph = self.env.graph
        queue = list(op.predecessors)
        visited = set()
        while queue:
            pid = queue.pop(0)
            if pid not in graph.operations:
                continue
            pop = graph.operations[pid]
            if pop.job_id != op.job_id:
                continue
            if pop.op_type == from_step and (pop.is_scheduled or pop.is_completed):
                return pop
            if pid not in visited:
                visited.add(pid)
                queue.extend(pop.predecessors)
        return None

    def compute_reward(self, prev_makespan, curr_makespan, done,
                       scheduled_op_id, scheduled_machine):
        op = self.env.graph.operations[scheduled_op_id]

        step_delta = curr_makespan - prev_makespan
        step_reward = -step_delta * 2

        tmax_penalty = 0.0
        for constraint in self.env.tmax_constraints:
            from_step, to_step, max_interval = constraint

            if op.op_type == to_step:
                job_ops = [o for o in self.env.graph.operations.values()
                           if o.job_id == op.job_id]
                from_op = next((o for o in job_ops if o.op_type == from_step), None)

                if from_op and from_op.is_scheduled:
                    actual_gap = op.scheduled_start - from_op.scheduled_end
                    if actual_gap > max_interval + 1e-4:
                        overshoot = actual_gap - max_interval
                        tmax_penalty -= max(10, (overshoot * 2))

            _chain_r = None
            _s_ddl_r = None
            if op.op_type == from_step:
                _chain_r = self._find_chain_to_target(op, to_step)
                if _chain_r:
                    _s_ddl_r = op.scheduled_end + max_interval
            elif op.op_type != to_step:
                anc_r = self._find_scheduled_ancestor(op, from_step)
                if anc_r is not None:
                    _chain_r = self._find_chain_to_target(op, to_step)
                    if _chain_r:
                        _s_ddl_r = anc_r.scheduled_end + max_interval

            if _chain_r is not None and _s_ddl_r is not None:
                s_lb_B_r = self._propagate_lb(_chain_r, op.scheduled_end)
                if s_lb_B_r > _s_ddl_r + 1e-4:
                    overshoot_fwd = s_lb_B_r - _s_ddl_r
                    tmax_penalty -= max(10.0, overshoot_fwd * 2)

        total_reward = step_reward + tmax_penalty
        return total_reward/100

def get_job_combinations(buffer_jobs, k):
    return list(combinations(buffer_jobs, k))


def try_release_combination_fast(env, jobs_to_release, time_limit_sec=0.5):
    feasible = check_release_feasibility(env, jobs_to_release, time_limit_sec)
    return {
        'feasible': feasible,
        'makespan': 0.0,
        'schedule': []
    }


def progressive_release_strict_fast(env, time_limit_sec=0.5):
    buffer_jobs = list(env.job_buffer.buffer)
    n = len(buffer_jobs)

    if n == 0:
        return {'success': False, 'released_jobs': [], 'makespan': 0.0, 'schedule': []}

    for k in range(n, 0, -1):
        if k == n:
            combinations_to_try = [buffer_jobs]
        else:
            combinations_to_try = get_job_combinations(buffer_jobs, k)

        for jobs_combo in combinations_to_try:
            result = try_release_combination_fast(env, list(jobs_combo), time_limit_sec)
            if result['feasible']:
                return {
                    'success': True,
                    'released_jobs': list(jobs_combo),
                    'makespan': 0.0,
                    'schedule': []
                }

    return {'success': False, 'released_jobs': [], 'makespan': 0.0, 'schedule': []}
