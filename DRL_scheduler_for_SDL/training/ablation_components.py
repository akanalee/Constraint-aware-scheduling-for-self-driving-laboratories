import torch

from env_adapter import SDLEnvAdapter


class AblationAdapterNoForward(SDLEnvAdapter):
    def _find_chain_to_target(self, start_op, to_step):
        return None


class AblationAdapterNoBackward(SDLEnvAdapter):
    def get_state_for_fss(self):
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

        _ALPHA = self.config.et_normalize_coef

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

            is_scheduled = 1.0 if op.is_scheduled else 0.0

            if op.is_scheduled:
                est_start_raw = op.scheduled_start
                min_proc_raw = op.processing_times.get(op.scheduled_machine,
                                   op.get_min_processing_time())
            else:
                pred_end = 0.0
                for pred_id in op.predecessors:
                    pred_op = graph.operations[pred_id]
                    if pred_op.is_scheduled or pred_op.is_completed:
                        pred_end = max(pred_end, pred_op.scheduled_end)

                est_start_raw = float('inf')
                min_proc_raw = float('inf')
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
            f3 = min_proc_raw / _ALPHA

            urgency = 0.0
            if has_tmax:
                urgency_candidates = []

                for constraint in self.env.tmax_constraints:
                    from_step, to_step, max_interval = constraint
                    if max_interval <= 0:
                        continue

                    from_entry = job_op_type_map.get((op.job_id, from_step))
                    to_entry = job_op_type_map.get((op.job_id, to_step))
                    if not from_entry or not to_entry:
                        continue

                    from_op_id_c, from_op = from_entry
                    to_op_id_c, to_op = to_entry

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

        n_machines = len(self.env.machines)
        mask_mch = torch.ones((1, n_ops, n_machines), dtype=torch.bool, device=self.device)

        for idx, op_id in enumerate(op_list):
            op = graph.operations[op_id]
            compatible_machines = op.compatible_machines

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

                for mode, chain, max_interval, anc_end in _fwd_checks:
                    s_lb_B = self._propagate_lb(chain, curr_end_time)
                    s_ddl_B = (curr_end_time + max_interval if mode == 'A'
                               else anc_end + max_interval)
                    if s_lb_B > s_ddl_B + 1e-4:
                        is_compatible = False
                        break


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

    def compute_reward(self, prev_makespan, curr_makespan, done,
                       scheduled_op_id, scheduled_machine):
        base = super().compute_reward(
            prev_makespan, curr_makespan, done,
            scheduled_op_id, scheduled_machine
        )

        op = self.env.graph.operations[scheduled_op_id]
        back_penalty = 0.0
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
                        back_penalty -= max(10, (overshoot * 2))

        return base - back_penalty / 100


class AblationAdapterNoUrgency(SDLEnvAdapter):
    def get_state_for_fss(self):
        state = super().get_state_for_fss()
        if state is not None:
            state['fea'][:, 3] = 0.0
        return state
