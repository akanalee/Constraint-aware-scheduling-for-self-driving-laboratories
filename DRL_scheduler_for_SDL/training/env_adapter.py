"""
SDL environment adapter - converts SDL environment state to FSS format.

Follows the FJSP environment interface for the GIN-based encoder.
"""
import torch
import numpy as np
from typing import Dict, List, Tuple
from copy import deepcopy
from itertools import combinations


class SDLEnvAdapter:
    """
    SDL environment adapter.
    Converts SDL DisjunctiveGraph state to FSS state representation.

    Key design notes:
    1. Setup Time: handled by DisjunctiveGraph.schedule_operation() automatically
       (same material: +1.0 min, different material: +2.0 min)
    2. Atlas Dynamic Parameters: handled by AtlasSimulator at job generation time.
       FSS reads op.processing_times[machine_id] directly.
    3. Job Generation: _inject_atlas_job() -> dataset_generator.generate_job()
       New jobs carry Atlas-optimized processing_times.
    """

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
        if self.has_tmax_checking:
            print("Tmax checking enabled via MaskManager")
        else:
            print("No MaskManager found, Tmax checking disabled")

    def _is_job_completed(self, job_id: str) -> bool:
        """Check if a job just completed (returns True only once per job)."""
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
        """Compute makespan for a specific job."""
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
        return max(end_times) - min(start_times)

    def get_state_for_fss(self) -> Dict:
        """
        Convert SDL graph state to FSS format.

        Returns:
            state dict with keys: adj, fea, candidate, mask, mask_mch, dur,
            mch_time, op_list, eligible_ops. Returns None if no eligible ops.
        """
        graph = self.env.graph

        # 1. Get eligible operations
        eligible_ops = graph.eligible_ops
        n_candidates = len(eligible_ops)
        if n_candidates == 0:
            return None

        # 2. Build node features and mapping
        op_list = sorted([
            op_id for op_id, op in graph.operations.items()
            if not op.is_completed
        ])
        n_ops = len(op_list)
        op_to_idx = {op_id: idx for idx, op_id in enumerate(op_list)}

        # Node features (5-dim):
        #   f1: is_scheduled       ∈ {0,1}
        #   f2: earliest_start_lb  = min_{m∈M(o)} max(c_pred, t_avail(m)) / alpha
        #   f3: min_proc_time      = min_{m∈M(o)} p(o,m) / alpha
        #   f4: urgency            = d_BFS(o->to_op) / max(deadline - earliest_start_lb, eps)
        #                            0 if not on any Tmax path; >=1 means violation imminent
        #   f5: n_compatible / 10  normalised count of compatible machines
        #
        # f2 captures the true earliest start accounting for BOTH predecessor
        # completion AND machine availability, replacing the old lb that ignored
        # machine contention.  f3 exposes the op's own duration independently.
        # urgency denominator uses the same f2 value so the ratio is consistent.

        _ALPHA = self.config.et_normalize_coef  # 100.0

        # Build (job_id, op_type) -> (op_id, op) lookup
        has_tmax = self.has_tmax_checking and hasattr(self.env, 'tmax_constraints')
        if has_tmax:
            job_op_type_map = {}
            for oid, o in graph.operations.items():
                key = (o.job_id, o.op_type)
                if key not in job_op_type_map:
                    job_op_type_map[key] = (oid, o)

        node_features = []
        for op_id in op_list:
            op = graph.operations[op_id]

            # f1: scheduling status
            is_scheduled = 1.0 if op.is_scheduled else 0.0

            # f2: earliest start lb = min over compatible machines of
            #     max(predecessor_end, machine_avail_time)
            # f3: minimum processing time over compatible machines
            if op.is_scheduled:
                # already placed: use actual scheduled start / end
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

            # f4: urgency
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
                        # from_op not yet scheduled: only compute for from_op itself
                        if op_id != from_op_id_c:
                            continue
                        # denominator degrades to delta_max
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

            # f5: normalised count of compatible machines
            n_compat = sum(1 for m_id in op.processing_times
                           if m_id in self.machine_id_to_idx)
            f5 = n_compat / 10.0

            node_features.append([is_scheduled, f2, f3, urgency, f5])

        fea = torch.tensor(node_features, dtype=torch.float32, device=self.device)
        # fea shape: [n_ops, 5]  — (is_scheduled, est_start/α, min_proc/α, urgency, n_compat/10)

        # 3. Build adjacency matrix (sparse, 2D: [n_ops, n_ops])
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

        # 4. Build candidate list [batch=1, max_candidates]
        candidate_indices = [op_to_idx[op_id] for op_id in eligible_ops]
        max_candidates = self.config.n_j
        padding_value = candidate_indices[0] if candidate_indices else 0
        padded_candidates = candidate_indices + [padding_value] * (max_candidates - len(candidate_indices))
        padded_candidates = padded_candidates[:max_candidates]
        candidate = torch.tensor([padded_candidates], dtype=torch.long, device=self.device)

        # 5. Build operation mask [batch=1, max_candidates]
        mask = torch.zeros((1, max_candidates), dtype=torch.bool, device=self.device)
        mask[0, n_candidates:] = True  # mask padding

        # 6. Build machine mask [batch=1, n_ops, n_machines]
        # Two-layer check: (1) machine compatibility, (2) Tmax constraints
        n_machines = len(self.env.machines)
        mask_mch = torch.ones((1, n_ops, n_machines), dtype=torch.bool, device=self.device)

        for idx, op_id in enumerate(op_list):
            op = graph.operations[op_id]
            compatible_machines = op.compatible_machines

            # ── Per-op precompute (machine-independent) ───────────────────────
            # Determine which forward-check entries apply to this op.
            # _find_chain_to_target / _find_scheduled_ancestor are called once
            # per op; _propagate_lb is called once per machine (c_lb varies).
            _fwd_checks = []   # list of ('A'|'mid', chain, max_interval, anc_end)
            if self.has_tmax_checking:
                for fc in self.env.tmax_constraints:
                    fs, ts, mi = fc
                    if op.op_type == fs:
                        # Op is A (from_step): s_ddl(B) = c_lb(A, m_j) + T_max
                        chain = self._find_chain_to_target(op, ts)
                        if chain:
                            _fwd_checks.append(('A', chain, mi, None))
                    elif op.op_type != ts:
                        # Potential intermediate op: check if A (from_step) is
                        # already scheduled upstream in the same job
                        anc = self._find_scheduled_ancestor(op, fs)
                        if anc is not None:
                            chain = self._find_chain_to_target(op, ts)
                            if chain:
                                _fwd_checks.append(('mid', chain, mi, anc.scheduled_end))

            for mach_id in compatible_machines:
                if mach_id not in self.machine_id_to_idx:
                    continue

                mach_idx = self.machine_id_to_idx[mach_id]
                is_compatible = True

                # c_lb(A or k, m_j): earliest completion of op on this machine
                pred_max = max([graph.operations[p].scheduled_end
                                for p in op.predecessors], default=self.env.current_time)
                curr_start_time = max(pred_max, self.env.machine_available_time.get(mach_id, 0))
                curr_processing_time = op.processing_times[mach_id]
                curr_end_time = curr_start_time + curr_processing_time

                # ── Forward Tmax check (op A and all intermediate ops) ────────
                # Correct per-machine lower-bound: Eq. c_lb_mj / c_lb_ok / s_lb_B
                for mode, chain, max_interval, anc_end in _fwd_checks:
                    s_lb_B = self._propagate_lb(chain, curr_end_time)
                    s_ddl_B = (curr_end_time + max_interval if mode == 'A'
                               else anc_end + max_interval)
                    if s_lb_B > s_ddl_B + 1e-4:
                        is_compatible = False
                        break

                # ── Backward Tmax check (op is B = to_step) ──────────────────
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
                                    # curr_start_time already accounts for machine
                                    # availability and predecessor end time
                                    if curr_start_time > deadline + 1e-4:
                                        is_compatible = False
                                        break

                mask_mch[0, idx, mach_idx] = not is_compatible

            # Deadlock recovery: if all physical machines are masked, unlock them
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

        # 7. Build processing time matrix [batch=1, n_ops, n_machines]
        dur = torch.zeros((1, n_ops, n_machines), dtype=torch.float32, device=self.device)
        for idx, op_id in enumerate(op_list):
            op = graph.operations[op_id]
            for mach_id, proc_time in op.processing_times.items():
                if mach_id in self.machine_id_to_idx:
                    mach_idx = self.machine_id_to_idx[mach_id]
                    dur[0, idx, mach_idx] = proc_time

        # 8. Build machine available time [batch=1, n_machines]
        mch_time = torch.zeros((1, n_machines), dtype=torch.float32, device=self.device)
        for mach_id, avail_time in self.env.machine_available_time.items():
            if mach_id in self.machine_id_to_idx:
                mach_idx = self.machine_id_to_idx[mach_id]
                mch_time[0, mach_idx] = avail_time

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

    def _compute_op_lb(self, op_id: str) -> float:
        """Compute operation lower bound."""
        op = self.env.graph.operations[op_id]
        if op.is_scheduled:
            return op.scheduled_end

        pred_max = 0.0
        for pred_id in op.predecessors:
            pred_op = self.env.graph.operations[pred_id]
            if pred_op.is_scheduled:
                pred_max = max(pred_max, pred_op.scheduled_end)
        return pred_max + op.get_min_processing_time()

    def _is_on_path(self, op_id: str, from_op_id: str, to_op_id: str) -> bool:
        """
        Return True if op_id lies on at least one directed path
        from from_op_id to to_op_id (endpoints inclusive).
        """
        graph = self.env.graph
        if op_id == from_op_id or op_id == to_op_id:
            return True
        # Forward BFS from from_op_id; when we encounter op_id as a successor,
        # verify it can reach to_op_id.
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
        """BFS reachability: can from_id reach to_id following successors?"""
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
        """
        Estimate the earliest possible start time of to_op_id when the chain
        departs from from_op_id finishing at from_end_time.

        BFS walks successor edges, at each hop computing:
            est_start = max(accumulated_end_time, min_machine_available_time)

        This mirrors the forward-Tmax mask BFS exactly (machine availability
        is taken into account, not just minimum processing times).

        Returns the estimated start time of to_op_id, or float('inf') if
        to_op_id is not reachable.
        """
        graph = self.env.graph
        if from_op_id not in graph.operations:
            return float('inf')

        from_op = graph.operations[from_op_id]
        # Queue entries: (op_id, accumulated_end_time_of_predecessor)
        queue = [(succ_id, from_end_time) for succ_id in from_op.successors]
        visited = set()

        while queue:
            curr_id, acc_time = queue.pop(0)
            if curr_id not in graph.operations or curr_id in visited:
                continue
            visited.add(curr_id)
            curr_op = graph.operations[curr_id]

            # Earliest start considering machine availability
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
        """Convert action indices to operation ID and machine ID."""
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
            # Fallback to first compatible machine
            fallback_machine = list(op.processing_times.keys())[0]
            machine_id = fallback_machine

        return op_id, machine_id

    # ------------------------------------------------------------------
    # Tmax lower-bound helpers (correct per the paper formulas)
    # ------------------------------------------------------------------

    def _find_chain_to_target(self, start_op, to_step):
        """
        BFS from start_op's successors (same job only) to find the first path
        to an op with op_type == to_step within the current graph.
        Returns [op_{k+1}, ..., op_B] inclusive, or None if unreachable.
        """
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
        """
        Propagate lower-bound times through chain_incl_B and return s_lb(B).

        chain_incl_B  = [op_{k+1}, ..., op_B]  (B is the last element)
        c_lb_prev     = lower-bound completion time of the op before chain[0]

        For each intermediate op k (not B):
            c_lb(k) = min_{m_j in M(k)} [ max(c_lb_prev, t_avail(m_j), t_releas(k))
                                           + p(k, m_j) ]
        For B (last):
            s_lb(B) = min_{m_j in M(B)} max(c_lb(B-1), t_avail(m_j))

        Returns s_lb(B).
        """
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
        return c_lb  # = s_lb(B) after the final (is_last) iteration

    def _find_scheduled_ancestor(self, op, from_step):
        """
        Backward BFS (same job only) to find the nearest ancestor of op
        with op_type == from_step that is already scheduled or completed.
        Returns that op, or None.
        """
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
        """
        Compute reward with makespan increment, flow time, and Tmax penalties.
        All signals normalized to similar magnitude.
        """
        op = self.env.graph.operations[scheduled_op_id]
        machine = self.env.machines_dict[scheduled_machine]

        # 1. Makespan increment (normalized)
        step_delta = curr_makespan - prev_makespan
        step_reward = -step_delta * 2

        # Flow time penalty (normalized)
        job_arrival_time = getattr(op, 'arrival_time', 0.0)
        relative_completion = op.scheduled_end - job_arrival_time
        reward_flow = -relative_completion / 130.0

        # 2. Tmax penalty
        tmax_penalty = 0.0
        for constraint in self.env.tmax_constraints:
            from_step, to_step, max_interval = constraint

            # Part 1: endpoint penalty (op is to_step)
            if op.op_type == to_step:
                job_ops = [o for o in self.env.graph.operations.values()
                           if o.job_id == op.job_id]
                from_op = next((o for o in job_ops if o.op_type == from_step), None)

                if from_op and from_op.is_scheduled:
                    actual_gap = op.scheduled_start - from_op.scheduled_end
                    if actual_gap > max_interval + 1e-4:
                        overshoot = actual_gap - max_interval
                        tmax_penalty -= max(10, (overshoot * 2))

            # Part 2: forward penalty — op A (from_step) and all intermediate ops
            # Uses correct lower-bound propagation; same floor/coefficient as
            # Part 1 so the three terms (step, forward, backward) are 1:1:1.
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

    def _estimate_gap_with_bfs(self, scheduled_op, to_step):
        """Estimate gap using BFS (same logic as forward mask)."""
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
# Progressive release functions (from behavioral cloning)
# ============================================================

def get_job_combinations(buffer_jobs, k):
    """Generate all C(n,k) job combinations from buffer."""
    return list(combinations(buffer_jobs, k))


def try_release_combination(env, jobs_to_release):
    """
    Try releasing a given job combination using snapshot/rollback mechanism.
    Does not modify the original environment state.
    """
    graph_snapshot = env.graph.snapshot()
    env._build_static_subproblem(jobs_to_release)
    result = env._solve_subproblem_with_lower()
    env.graph.restore(graph_snapshot)

    if result['status'] == 'infeasible':
        return {
            'feasible': False,
            'makespan': float('inf'),
            'schedule': []
        }
    else:
        return {
            'feasible': True,
            'makespan': result['makespan'],
            'schedule': result['schedule']
        }


def progressive_release_strict(env):
    """
    Strict exhaustive progressive release algorithm.

    Logic:
    1. Start with all n jobs
    2. If n infeasible, try all C(n, n-1) combinations
    3. If any feasible at level k, apply the best (min makespan) and stop
    4. Continue decreasing k down to 1
    5. If even k=1 fails, return failure

    Returns:
        dict with keys: success, released_jobs, makespan, schedule
    """
    buffer_jobs = list(env.job_buffer.buffer)
    n = len(buffer_jobs)

    print(f"\n{'=' * 60}")
    print(f"[Progressive Release] Buffer size: {n}")
    print(f"  Current time: {env.current_time:.2f}")
    print(f"{'=' * 60}")

    if n == 0:
        return {'success': False, 'released_jobs': [], 'makespan': 0.0, 'schedule': []}

    for k in range(n, 0, -1):
        print(f"\n--- Trying k={k} jobs ---")

        if k == n:
            combinations_to_try = [buffer_jobs]
        else:
            combinations_to_try = get_job_combinations(buffer_jobs, k)

        total_combos = len(combinations_to_try)
        print(f"  Total combinations to try: {total_combos}")

        feasible_results = []

        for idx, jobs_combo in enumerate(combinations_to_try):
            if (idx + 1) % 50 == 0 or idx == 0:
                print(f"    Progress: {idx + 1}/{total_combos}")

            result = try_release_combination(env, list(jobs_combo))

            if result['feasible']:
                feasible_results.append({
                    'jobs': list(jobs_combo),
                    'makespan': result['makespan'],
                    'schedule': result['schedule']
                })

        if feasible_results:
            best_result = min(feasible_results, key=lambda x: x['makespan'])

            print(f"  Found {len(feasible_results)} feasible solutions at k={k}")
            print(f"  Best makespan: {best_result['makespan']:.2f}")
            print(f"  Stopping search (found solution at k={k})")

            return {
                'success': True,
                'released_jobs': best_result['jobs'],
                'makespan': best_result['makespan'],
                'schedule': best_result['schedule']
            }
        else:
            print(f"  All {total_combos} combinations infeasible at k={k}")

    print(f"\n  All combinations failed (even k=1), no jobs released")
    return {'success': False, 'released_jobs': [], 'makespan': 0.0, 'schedule': []}


# ============================================================
# Fast CP-SAT feasibility probe (training-only, no Gurobi)
# ============================================================

def _check_feasibility_cpsat(env, jobs_to_release, time_limit_sec=0.5):
    """
    Pure feasibility probe via OR-Tools CP-SAT — no objective, no MIP search.

    Builds a CP-SAT model with:
      • IntervalVar + AddNoOverlap per machine  (triggers arc-consistency
        propagation, CP-SAT's strongest infeasibility detector)
      • Precedence constraints (linear)
      • Tmax constraints (linear)
      • Machine-assignment binary vars (exactly-one per op)

    CP-SAT's constraint propagation can prove infeasibility in milliseconds
    when Tmax windows are tight or machine capacity is overloaded, because it
    performs domain-reduction without ever entering a B&B tree.

    Returns True if CP-SAT finds a feasible solution OR hits the time limit
    without proving infeasibility (optimistic: treat unknown as feasible so
    the PPO agent still gets to attempt the schedule).
    Returns False only when CP-SAT definitively proves INFEASIBLE during
    presolve / propagation.

    NOTE: this function does NOT mutate env.graph.  It reads a snapshot of
    the graph state (ops already in the graph + jobs_to_release) but all
    CP-SAT variables are local.
    """
    from ortools.sat.python import cp_model as _cp

    SCALE = 10          # 0.1-minute precision  (multiply all float times by SCALE)
    BIG   = 100000      # safe upper bound in scaled units (~10000 minutes)

    graph        = env.graph
    current_time = env.current_time
    tmax_cstrs   = env.tmax_constraints   # [(from_step, to_step, max_interval), ...]
    machines_ids = [m.id for m in env.machines]

    # ------------------------------------------------------------------
    # 1. Collect ops: existing (not completed) + new jobs
    # ------------------------------------------------------------------
    # ops already in the graph that are not yet completed
    existing_ops = {
        op_id: op for op_id, op in graph.operations.items()
        if not op.is_completed
    }

    # Build lightweight dicts for new-job ops (not yet in graph)
    new_ops = {}   # op_id -> dict
    new_precedences = []  # (pred_id, succ_id)

    for job in jobs_to_release:
        for i, op_data in enumerate(job.operations):
            op_id = op_data['id']
            new_ops[op_id] = {
                'id':              op_id,
                'job_id':          job.id,
                'op_index':        i,
                'op_type':         op_data.get('op_type', 'default'),
                'processing_times': op_data['processing_times'],
                'arrival_time':    job.arrival_time,
                'predecessors':    [job.operations[i-1]['id']] if i > 0 else [],
            }
        for i in range(1, len(job.operations)):
            new_precedences.append((
                job.operations[i-1]['id'],
                job.operations[i]['id']
            ))

    # Combined op view  (new ops shadow nothing — ids are disjoint by design)
    all_op_ids = list(existing_ops.keys()) + list(new_ops.keys())

    def _get_op_field(op_id, field):
        if op_id in existing_ops:
            op = existing_ops[op_id]
            if field == 'job_id':        return op.job_id
            if field == 'op_index':      return op.op_index
            if field == 'op_type':       return op.op_type
            if field == 'processing_times': return op.processing_times
            if field == 'arrival_time':  return getattr(op, 'arrival_time', 0.0)
            if field == 'predecessors':  return op.predecessors
            if field == 'is_running':
                return (op.is_scheduled and
                        op.scheduled_start <= current_time + 1e-4)
            if field == 'scheduled_start': return op.scheduled_start
            if field == 'scheduled_end':   return op.scheduled_end
            if field == 'scheduled_machine': return op.scheduled_machine
        else:
            d = new_ops[op_id]
            if field == 'is_running':      return False
            if field == 'scheduled_start': return 0.0
            if field == 'scheduled_end':   return 0.0
            if field == 'scheduled_machine': return None
            return d.get(field)

    # ------------------------------------------------------------------
    # 2. Build CP-SAT model
    # ------------------------------------------------------------------
    mdl     = _cp.CpModel()
    solver  = _cp.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_sec
    solver.parameters.num_search_workers  = 1   # single thread for low latency

    # op_id -> {machine_id -> interval_var}
    op_machine_intervals = {}
    # op_id -> start_var, end_var, presence_vars {mach_id: bool_var}
    op_start  = {}
    op_end    = {}
    op_presence = {}   # op_id -> {mach_id: BoolVar}

    # machine_id -> list of (interval_var, presence_var) for AddNoOverlap
    machine_intervals = {mid: [] for mid in machines_ids}

    for op_id in all_op_ids:
        proc_times  = _get_op_field(op_id, 'processing_times')
        arrival     = _get_op_field(op_id, 'arrival_time')
        is_running  = _get_op_field(op_id, 'is_running')

        if is_running:
            # Fixed: already executing — pin start/end, machine
            s_val = int(_get_op_field(op_id, 'scheduled_start') * SCALE)
            e_val = int(_get_op_field(op_id, 'scheduled_end')   * SCALE)
            s_var = mdl.NewConstant(s_val)
            e_var = mdl.NewConstant(e_val)
            op_start[op_id]   = s_var
            op_end[op_id]     = e_var
            op_presence[op_id] = {}
            m_fixed = _get_op_field(op_id, 'scheduled_machine')
            if m_fixed and m_fixed in machine_intervals:
                dur  = e_val - s_val
                ivar = mdl.NewFixedSizeIntervalVar(s_var, dur, f'iv_{op_id}_{m_fixed}')
                machine_intervals[m_fixed].append(ivar)
            continue

        # lower bound on start: max(current_time, arrival)
        lb_scaled = int(max(current_time, arrival) * SCALE)

        s_var = mdl.NewIntVar(lb_scaled, BIG, f's_{op_id}')
        e_var = mdl.NewIntVar(lb_scaled, BIG, f'e_{op_id}')
        op_start[op_id] = s_var
        op_end[op_id]   = e_var

        compat_machines = [mid for mid in machines_ids if mid in proc_times]
        if not compat_machines:
            # No compatible machine  immediately infeasible
            return False

        pres_vars = {}
        ivars_for_op = []
        for mid in compat_machines:
            dur_scaled = max(1, int(proc_times[mid] * SCALE))
            p_var  = mdl.NewBoolVar(f'p_{op_id}_{mid}')
            # optional interval: present iff p_var=1
            iv = mdl.NewOptionalIntervalVar(
                s_var, dur_scaled, e_var, p_var, f'iv_{op_id}_{mid}'
            )
            pres_vars[mid] = p_var
            ivars_for_op.append((mid, p_var, dur_scaled, iv))
            machine_intervals[mid].append(iv)

        # Exactly one machine selected
        mdl.AddExactlyOne(pres_vars.values())

        # end = start + chosen_duration
        # Linearise: e = s + sum_m(p_m * dur_m)
        dur_expr = sum(p * d for _, p, d, _ in ivars_for_op)
        mdl.Add(e_var == s_var + dur_expr)

        op_presence[op_id] = pres_vars

    # ------------------------------------------------------------------
    # 3. No-overlap per machine
    # ------------------------------------------------------------------
    for mid, ivlist in machine_intervals.items():
        if len(ivlist) >= 2:
            mdl.AddNoOverlap(ivlist)

    # ------------------------------------------------------------------
    # 4. Precedence constraints
    # ------------------------------------------------------------------
    def _add_prec(pred_id, succ_id):
        if pred_id not in op_end or succ_id not in op_start:
            return
        mdl.Add(op_start[succ_id] >= op_end[pred_id])

    # From existing graph
    for op_id in existing_ops:
        op = existing_ops[op_id]
        for pred_id in op.predecessors:
            _add_prec(pred_id, op_id)

    # From new jobs
    for pred_id, succ_id in new_precedences:
        _add_prec(pred_id, succ_id)

    # Cross-boundary: new op whose predecessor is an existing completed/scheduled op
    for op_id, d in new_ops.items():
        for pred_id in d['predecessors']:
            if pred_id in existing_ops:
                _add_prec(pred_id, op_id)

    # ------------------------------------------------------------------
    # 5. Tmax constraints
    #    start(to_op) - end(from_op) <= max_interval
    #    Only between ops where from_op end time is known or in model.
    # ------------------------------------------------------------------
    # Build job  {op_type  [(op_index, op_id)]} lookup over all ops
    job_type_map = {}   # job_id -> op_type -> [(op_index, op_id)]
    for op_id in all_op_ids:
        jid    = _get_op_field(op_id, 'job_id')
        otype  = _get_op_field(op_id, 'op_type')
        oidx   = _get_op_field(op_id, 'op_index')
        job_type_map.setdefault(jid, {}).setdefault(otype, []).append((oidx, op_id))

    for (from_step, to_step, max_iv) in tmax_cstrs:
        max_iv_scaled = int(max_iv * SCALE)
        for jid, type_dict in job_type_map.items():
            to_list   = type_dict.get(to_step,   [])
            from_list = type_dict.get(from_step, [])
            if not to_list or not from_list:
                continue
            for to_idx, to_op_id in to_list:
                # closest preceding from_step op (max op_index < to_idx)
                cands = [(fi, fid) for fi, fid in from_list if fi < to_idx]
                if not cands:
                    continue
                _, from_op_id = max(cands, key=lambda x: x[0])

                # Case A: from_op is fixed (completed / running) — pure lb check
                if from_op_id not in op_end:
                    # from_op not in model (completed before current_time)
                    from_op_obj = graph.operations.get(from_op_id)
                    if from_op_obj and (from_op_obj.is_completed or from_op_obj.is_scheduled):
                        deadline_scaled = int(
                            (from_op_obj.scheduled_end + max_iv) * SCALE
                        )
                        if to_op_id in op_start:
                            mdl.Add(op_start[to_op_id] <= deadline_scaled)
                    continue

                # Case B: both in model
                if to_op_id in op_start:
                    mdl.Add(
                        op_start[to_op_id] - op_end[from_op_id] <= max_iv_scaled
                    )

    # ------------------------------------------------------------------
    # 6. Solve (feasibility only — no objective)
    # ------------------------------------------------------------------
    status = solver.Solve(mdl)

    if status == _cp.INFEASIBLE:
        return False   # definitively infeasible

    # FEASIBLE, OPTIMAL, or UNKNOWN (time-limit hit without proof)
    #  treat as feasible so the PPO agent gets a chance to schedule
    return True


def try_release_combination_fast(env, jobs_to_release, time_limit_sec=0.5):
    """
    Drop-in replacement for try_release_combination that uses CP-SAT
    feasibility probe instead of Gurobi MILP solve.

    Does NOT mutate env.graph — reads a snapshot of existing ops plus
    the candidate jobs, runs CP-SAT propagation, and returns immediately.

    Returns same dict shape as try_release_combination:
        {'feasible': bool, 'makespan': float, 'schedule': []}
    Note: makespan is always 0.0 (not computed — training doesn't use it).
    """
    feasible = _check_feasibility_cpsat(env, jobs_to_release, time_limit_sec)
    return {
        'feasible': feasible,
        'makespan': 0.0,    # not needed during training
        'schedule': []      # not needed during training
    }


def progressive_release_strict_fast(env, time_limit_sec=0.5):
    """
    Fast training-time version of progressive_release_strict.

    Identical release logic (k = n down to 1, exhaustive C(n,k) search,
    pick largest feasible k) but replaces the Gurobi MILP solve with a
    CP-SAT feasibility probe that runs in ~0.1-0.5 s instead of 30-60 s.

    Return format is identical to progressive_release_strict:
        {'success': bool, 'released_jobs': list, 'makespan': float, 'schedule': []}

    The 'schedule' field is always [] — training does not use it (FSS
    builds its own schedule after the release gate).
    """
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
                # First feasible combo at this k — return immediately.
                # Makespan not computed; FSS re-optimises after release.
                return {
                    'success': True,
                    'released_jobs': list(jobs_combo),
                    'makespan': 0.0,
                    'schedule': []
                }

    return {'success': False, 'released_jobs': [], 'makespan': 0.0, 'schedule': []}