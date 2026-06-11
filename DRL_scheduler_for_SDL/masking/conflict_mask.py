"""
Conflict Mask - 
"""

import torch
from typing import List, Dict


def build_conflict_mask(
        operation,
        machines: List,
        current_time: float,
        scheduled_ops_on_machines: Dict[str, List]
) -> torch.BoolTensor:
    """
    mask

    Args:
        operation: Operation
        machines: List[Machine]
        current_time: 
        scheduled_ops_on_machines: {machine_id: [scheduled_ops]}

    Returns:
        mask: (M,) BoolTensorTrue=op
    """
    mask = torch.ones(len(machines), dtype=torch.bool)

    # op
    op_earliest_start = current_time
    if hasattr(operation, 'predecessors') and operation.predecessors:
        # 
        # 
        pred_completion_times = [
            scheduled_ops_on_machines.get(pred_id, {}).get('end_time', 0)
            for pred_id in operation.predecessors
        ]
        if pred_completion_times:
            op_earliest_start = max(op_earliest_start, max(pred_completion_times))

        # 
    for i, machine in enumerate(machines):
        # 
        machine_available = machine.available_time

        # op
        if machine_available > op_earliest_start + operation.processing_times.get(machine.id, float('inf')):
            mask[i] = False

    return mask