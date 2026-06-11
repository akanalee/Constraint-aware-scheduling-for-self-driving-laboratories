import torch
import numpy as np
from typing import List, Tuple
import time

import networkx as nx


class TmaxMaskBuilder:
    """
    Tmax
    -
    -
    -
    """

    def __init__(self, constraints, depth=2, time_budget_ms=10,
                 ):
        self.constraints = self._build_constraint_graph(constraints)
        self.depth = depth
        self.time_budget = time_budget_ms / 1000
        self.cache = {}

    def build_mask(self, op, machines, graph, current_time) -> torch.BoolTensor:
        """
        opmachine mask
        Returns: BoolTensor (n_machines,) where True=
        """
        mask = torch.ones(len(machines), dtype=torch.bool)
        start_time = time.time()

        # op
        relevant_constraints = self.constraints.get(op.id, [])
        if not relevant_constraints:
            return mask  # True

        for i, mach in enumerate(machines):
            #
            cache_key = (op.id, mach.id, graph.state_hash())
            if cache_key in self.cache:
                mask[i] = self.cache[cache_key]
                continue

            #
            if (time.time() - start_time) > self.time_budget:
                break  # True

            #
            is_feasible = self._check_feasibility(
                op, mach, graph, current_time,
                relevant_constraints, depth=self.depth
            )

            mask[i] = is_feasible
            self.cache[cache_key] = is_feasible

        return mask

    def _check_feasibility(self, op, mach, graph, current_time, constraints, depth):
        #
        pred_max = 0
        if op.predecessors:
            pred_times = []
            for pred_id in op.predecessors:
                pred_op = graph.operations.get(pred_id)
                if pred_op and pred_op.is_scheduled:
                    pred_times.append(pred_op.scheduled_end)
            pred_max = max(pred_times) if pred_times else 0

        t_start = max(mach.available_time, current_time, pred_max)

        processing_time = op.processing_times.get(mach.id, 0)
        t_end = t_start + processing_time

        # 2.
        for (A_id, B_id, Tmax) in constraints:
            if A_id != op.id:
                continue

            # B
            B = graph.get_op(B_id)

            if B.is_scheduled:
                t_B_start = B.scheduled_start
            else:
                # CPM
                t_B_start = self._estimate_earliest_start(
                    B, t_end, graph
                )

            #
            interval = t_B_start - t_end
            if interval > Tmax:
                return False  #

            # 3. B
            if depth > 1 and not B.is_scheduled:
                B_constraints = self.constraints.get(B_id, [])
                if B_constraints:
                    # B""
                    best_B_mach = self._get_heuristic_machine(B, graph)
                    if not self._check_feasibility(
                            B, best_B_mach, graph, t_B_start,
                            B_constraints, depth - 1
                    ):
                        return False

        return True

    def _estimate_earliest_start(self, op, predecessor_end, graph):
        """
        CPM
        """
        #
        pred_times = [graph.get_lb_completion(p) for p in op.predecessors]
        max_pred_time = max(pred_times) if pred_times else 0

        #
        if predecessor_end > 0:
            transfer_time = graph.get_transfer_time_estimate(op)
            return max(max_pred_time, predecessor_end + transfer_time)

        return max_pred_time

    def _get_heuristic_machine(self, op, graph):
        """
        ""

        """
        compatible_machines = [m for m in graph.machines if op.id in m.compatible_ops]
        return min(compatible_machines, key=lambda m: m.available_time)

    def _build_constraint_graph(self, constraints):
        """

        {op_A_id: [(B_id, Tmax), ...]}
        """
        graph = {}
        for (A, B, Tmax) in constraints:
            if A not in graph:
                graph[A] = []
            graph[A].append((B, Tmax))
        return graph

    def clear_cache(self):
        """episode"""
        self.cache.clear()