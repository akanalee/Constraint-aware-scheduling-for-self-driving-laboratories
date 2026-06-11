"""
Mask Manager - mask
"""

import torch
from .precedence_mask import build_precedence_mask
from .capability_mask import build_capability_mask
from .conflict_mask import build_conflict_mask
from .tmax_mask import TmaxMaskBuilder
from configs import Config

class MaskManager:
    """
    Mask
    maskprecedence + capability + conflict + Tmax
    """

    def __init__(self, config):
        from configs import Config
        if isinstance(config, dict) and not isinstance(config, Config):
            config = Config(config)
        self.config = config
        self.tmax_builder = TmaxMaskBuilder(
            constraints=config.tmax_constraints,
            depth=config.tmax_lookahead_depth,
            time_budget_ms=config.tmax_time_budget_ms,
        )

    def build_op_mask(self, graph, operations) -> torch.BoolTensor:
        """
        operationmask
        precedence
        """
        return build_precedence_mask(operations, graph.scheduled_ops)

    def build_machine_mask(
            self,
            operation,
            machines,
            graph,
            current_time,
            tmax_constraints
    ) -> torch.BoolTensor:
        """
        machinemask
        capability + conflict + Tmax
        """
        # 1. Capability mask
        cap_mask = build_capability_mask(operation, machines)

        # 2. Conflict mask
        scheduled_ops_map = self._build_scheduled_ops_map(graph)
        conf_mask = build_conflict_mask(
            operation,
            machines,
            current_time,
            scheduled_ops_map
        )

        # 3. Tmax mask
        tmax_mask = self.tmax_builder.build_mask(
            operation,
            machines,
            graph,
            current_time
        )

        # mask
        combined_mask = cap_mask & conf_mask & tmax_mask

        return combined_mask

    def build_all_masks(self, state, tmax_constraints):
        """
        mask

        Returns:
            {
                'op_mask': BoolTensor (N_ops,),
                'mach_mask': BoolTensor (M,)  # op
            }
        """
        graph = state['op_graph']
        operations = list(graph.operations.values())

        # Op mask
        op_mask = self.build_op_mask(graph, operations)

        return {
            'op_mask': op_mask,
            'mach_mask': None  # op
        }

    def _build_scheduled_ops_map(self, graph):
        """op"""
        scheduled_map = {}
        for op_id in graph.scheduled_ops:
            op = graph.operations[op_id]
            scheduled_map[op_id] = {
                'machine': op.scheduled_machine,
                'start_time': op.scheduled_start,
                'end_time': op.scheduled_end
            }
        return scheduled_map

    def clear_cache(self):
        """Tmax mask"""
        self.tmax_builder.clear_cache()