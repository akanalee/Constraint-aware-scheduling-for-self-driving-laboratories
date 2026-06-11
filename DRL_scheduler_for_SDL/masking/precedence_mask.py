"""
Precedence Mask - 
"""

import torch
from typing import List, Set


def build_precedence_mask(
        operations: List,
        scheduled_ops: Set[str]
) -> torch.BoolTensor:
    """
    mask

    Args:
        operations: List[Operation]
        scheduled_ops: Set[op_id]

    Returns:
        mask: (N,) BoolTensorTrue=
    """
    mask = torch.zeros(len(operations), dtype=torch.bool)

    for i, op in enumerate(operations):
        if op.id in scheduled_ops:
            continue

        # 
        if all(pred_id in scheduled_ops for pred_id in op.predecessors):
            mask[i] = True

    return mask