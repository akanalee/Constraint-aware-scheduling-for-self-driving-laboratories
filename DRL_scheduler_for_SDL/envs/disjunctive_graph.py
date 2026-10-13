from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class Operation:
    id: str
    job_id: str
    op_index: int
    processing_times: Dict[str, float]
    predecessors: List[str] = field(default_factory=list)
    successors: List[str] = field(default_factory=list)

    is_scheduled: bool = False
    scheduled_machine: Optional[str] = None
    scheduled_start: float = 0.0
    scheduled_end: float = 0.0

    is_completed: bool = False

    op_type: str = "default"
    material: str = ""

    @property
    def compatible_machines(self) -> List[str]:
        return list(self.processing_times.keys())

    def get_min_processing_time(self) -> float:
        if not self.processing_times:
            return 0.0
        return min(self.processing_times.values())


@dataclass
class Machine:
    id: str
    machine_type: str
    capacity: int = 1

    available_time: float = 0.0
    queue: List[str] = field(default_factory=list)


class DisjunctiveGraph:
    def __init__(self, machines: Dict[str, Machine] = None, env=None):
        self.env = env
        self.operations: Dict[str, Operation] = {}
        self.machines: Dict[str, Machine] = machines if machines is not None else {}

        self.precedence_edges: List[Tuple[str, str]] = []

        self.scheduled_ops: Set[str] = set()
        self.eligible_ops: List[str] = []

        self.makespan: float = 0.0

    def add_operation(self, op: Operation):
        self.operations[op.id] = op
        self._update_eligible_ops()

    def add_precedence_edge(self, pred_id: str, succ_id: str):
        if pred_id in self.operations and succ_id in self.operations:
            self.precedence_edges.append((pred_id, succ_id))
            self.operations[pred_id].successors.append(succ_id)
            self.operations[succ_id].predecessors.append(pred_id)

    def schedule_operation(self, op_id: str, machine_id: str, start_time: float, end_time: float = None):
        op = self.operations[op_id]
        machine = self.machines[machine_id]

        if end_time is not None:
            actual_start = start_time
        else:
            actual_start = max(start_time, machine.available_time)
            processing_time = op.processing_times[machine_id]
            end_time = actual_start + processing_time

        op.is_scheduled = True
        op.scheduled_machine = machine_id
        op.scheduled_start = actual_start
        op.scheduled_end = end_time

        machine.available_time = end_time
        machine.queue.append(op_id)

        self.scheduled_ops.add(op_id)

        self._update_eligible_ops()

        self.makespan = max(self.makespan, end_time)

    def _update_eligible_ops(self):
        self.eligible_ops = []
        for op_id, op in self.operations.items():
            if op.is_scheduled:
                continue
            if all(pred_id in self.scheduled_ops for pred_id in op.predecessors):
                self.eligible_ops.append(op_id)
        self.eligible_ops.sort()

    def is_complete(self) -> bool:
        if not self.operations:
            return True

        completed_count = sum(1 for op in self.operations.values() if op.is_completed)
        total_count = len(self.operations)

        return completed_count == total_count

    def __repr__(self):
        return f"DisjunctiveGraph(ops={len(self.operations)}, machines={len(self.machines)}, scheduled={len(self.scheduled_ops)})"

    def snapshot(self):
        snapshot_data = {
            'ops_state': {
                op_id: {
                    'is_scheduled': op.is_scheduled,
                    'scheduled_machine': op.scheduled_machine,
                    'scheduled_start': op.scheduled_start,
                    'scheduled_end': op.scheduled_end,
                    'is_completed': op.is_completed
                }
                for op_id, op in self.operations.items()
            },
            'ops_keys': set(self.operations.keys()),

            'machines_state': {
                mach_id: {
                    'available_time': mach.available_time,
                    'queue': list(mach.queue),
                }
                for mach_id, mach in self.machines.items()
            },
            'scheduled_ops': set(self.scheduled_ops),
            'makespan': self.makespan
        }
        return snapshot_data

    def restore(self, snapshot):
        snapshot_keys = snapshot['ops_keys']
        current_keys = set(self.operations.keys())
        ops_to_delete = current_keys - snapshot_keys

        for op_id in ops_to_delete:
            op = self.operations[op_id]

            for pred_id in op.predecessors:
                if pred_id in self.operations:
                    pred_op = self.operations[pred_id]
                    if op_id in pred_op.successors:
                        pred_op.successors.remove(op_id)

            for succ_id in op.successors:
                if succ_id in self.operations:
                    succ_op = self.operations[succ_id]
                    if op_id in succ_op.predecessors:
                        succ_op.predecessors.remove(op_id)

            del self.operations[op_id]

        self.precedence_edges = [
            (pred_id, succ_id) for pred_id, succ_id in self.precedence_edges
            if pred_id not in ops_to_delete and succ_id not in ops_to_delete
        ]

        for op_id, state in snapshot['ops_state'].items():
            if op_id in self.operations:
                op = self.operations[op_id]
                op.is_scheduled = state['is_scheduled']
                op.scheduled_machine = state['scheduled_machine']
                op.scheduled_start = state['scheduled_start']
                op.scheduled_end = state['scheduled_end']
                op.is_completed = state['is_completed']

        for mach_id, state in snapshot['machines_state'].items():
            if mach_id in self.machines:
                mach = self.machines[mach_id]
                mach.available_time = state['available_time']
                mach.queue = list(state['queue'])

        self.scheduled_ops = set(snapshot['scheduled_ops'])
        self.makespan = snapshot['makespan']

        self._update_eligible_ops()
