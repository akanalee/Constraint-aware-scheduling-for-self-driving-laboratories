from itertools import combinations
from typing import Dict

import numpy as np
import torch

from utils.feasibility import check_release_feasibility


class SDLEnvAdapter:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.device = torch.device(config.device)
        assert env.machines == sorted(env.machines, key=lambda m: m.id), \
            "Machine list is not sorted!"
        self.machine_id_to_idx = {m.id: idx for idx, m in enumerate(env.machines)}
        self.idx_to_machine_id = {idx: m.id for idx, m in enumerate(env.machines)}

    def get_state_for_baseline(self) -> Dict:
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
        step_delta = curr_makespan - prev_makespan
        return -step_delta * 2


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
            combinations_to_try = list(combinations(buffer_jobs, k))

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
