"""
Capability Mask - 
"""

import torch
from typing import List


def build_capability_mask(
        operation,
        machines: List
) -> torch.BoolTensor:
    """
    mask

    Args:
        operation: Operation
        machines: List[Machine]

    Returns:
        mask: (M,) BoolTensorTrue=op
    """
    mask = torch.zeros(len(machines), dtype=torch.bool)

    compatible_machine_ids = set(operation.compatible_machines)

    for i, machine in enumerate(machines):
        if machine.id in compatible_machine_ids:
            mask[i] = True

    return mask