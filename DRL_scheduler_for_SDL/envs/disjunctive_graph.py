import torch
import numpy as np
from torch_geometric.data import Data
from typing import List, Dict, Tuple, Set, Optional, Union
from dataclasses import dataclass, field
from collections import defaultdict
import hashlib


@dataclass
class Operation:
    """"""
    id: str
    job_id: str
    op_index: int
    processing_times: Dict[str, float]
    predecessors: List[str] = field(default_factory=list)
    successors: List[str] = field(default_factory=list)

    # 
    is_scheduled: bool = False
    scheduled_machine: Optional[str] = None
    scheduled_start: float = 0.0
    scheduled_end: float = 0.0

    # V41
    is_completed: bool = False

    # 
    op_type: str = "default"
    material: str = ""

    @property
    def compatible_machines(self) -> List[str]:
        return list(self.processing_times.keys())

    def get_min_processing_time(self) -> float:
        if not self.processing_times:
            return 0.0
        return min(self.processing_times.values())
    """
    def __post_init__(self):
        #   processing_times
        if not self.processing_times:
            print(f"[*] Operation {self.id}  processing_times ")
        else:
            print(f" Operation {self.id} : {len(self.processing_times)}")"""


@dataclass
class Machine:
    """"""
    id: str
    machine_type: str
    capacity: int = 1

    # 
    available_time: float = 0.0
    utilization: float = 0.0
    queue: List[str] = field(default_factory=list)  # op_ids




class DisjunctiveGraph:
    """
    FJSPDisjunctive Graph
    """

    def __init__(self, machines: Dict[str, Machine] = None, env=None):  #  env
        self.env = env  #
        self.operations: Dict[str, Operation] = {}
        self.machines: Dict[str, Machine] = machines if machines is not None else {}

        #
        self.precedence_edges: List[Tuple[str, str]] = []
        self.disjunctive_edges: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

        #
        self.scheduled_ops: Set[str] = set()
        self.eligible_ops: List[str] = []

        #
        self.current_time: float = 0.0
        self.makespan: float = 0.0

        # Dummy
        self.source_id = "SOURCE"
        self.sink_id = "SINK"

    def add_operation(self, op: Operation):
        """"""
        self.operations[op.id] = op
        self._update_eligible_ops()

    def add_machine(self, machine: Machine):
        """"""
        self.machines[machine.id] = machine

    def add_precedence_edge(self, pred_id: str, succ_id: str):
        """"""
        if pred_id in self.operations and succ_id in self.operations:
            self.precedence_edges.append((pred_id, succ_id))
            self.operations[pred_id].successors.append(succ_id)
            self.operations[succ_id].predecessors.append(pred_id)

    def schedule_operation(self, op_id: str, machine_id: str, start_time: float, end_time: float = None):
        """"""
        op = self.operations[op_id]
        machine = self.machines[machine_id]

        if end_time is not None:
            actual_start = start_time
        else:
            actual_start = max(start_time, machine.available_time)
            processing_time = op.processing_times[machine_id]
            end_time = actual_start + processing_time

        # op
        op.is_scheduled = True
        op.scheduled_machine = machine_id
        op.scheduled_start = actual_start
        op.scheduled_end = end_time

        # machine
        machine.available_time = end_time
        machine.queue.append(op_id)

        #
        self.scheduled_ops.add(op_id)

        #
        self.current_time = max(self.current_time, end_time)

        # eligible ops
        self._update_eligible_ops()

        # makespan
        #if len(op.successors) == 0:
            #self.makespan = max(self.makespan, end_time)
        self.makespan = max(self.makespan, end_time)

    def _update_eligible_ops(self):
        self.eligible_ops = []
        for op_id, op in self.operations.items():
            if op.is_scheduled:
                continue
            if all(pred_id in self.scheduled_ops for pred_id in op.predecessors):
                self.eligible_ops.append(op_id)
        self.eligible_ops.sort()

    def get_makespan_lower_bound(self) -> float:
        lb_times = {}
        for op_id in self._topological_sort():
            op = self.operations[op_id]
            if op.is_scheduled:
                lb_times[op_id] = op.scheduled_end
            else:
                pred_max = max([lb_times.get(p, 0) for p in op.predecessors], default=0)
                lb_times[op_id] = pred_max + op.get_min_processing_time()
        return max(lb_times.values()) if lb_times else 0.0

    def _topological_sort(self) -> List[str]:
        in_degree = {op_id: len(op.predecessors) for op_id, op in self.operations.items()}
        queue = [op_id for op_id, deg in in_degree.items() if deg == 0]
        result = []
        while queue:
            op_id = queue.pop(0)
            result.append(op_id)
            for succ_id in self.operations[op_id].successors:
                in_degree[succ_id] -= 1
                if in_degree[succ_id] == 0:
                    queue.append(succ_id)
        return result

    def to_pyg_data(self) -> Data:
        #
        node_features = []
        node_to_idx = {}
        for idx, (op_id, op) in enumerate(self.operations.items()):
            node_to_idx[op_id] = idx
            lb = self._compute_op_lb(op_id)
            is_scheduled = 1.0 if op.is_scheduled else 0.0
            slack = 0.5
            urgency = 0.5
            #type_embed = hash(op.op_type) % 10 / 10.0
            _OP_TYPE_MAP = {
                'S1': 0.1, 'S2': 0.2, 'S3': 0.3, 'S4': 0.4,
                'S5': 0.5, 'S6': 0.6, 'S7': 0.7, 'default': 0.0
            }
            type_embed = _OP_TYPE_MAP.get(op.op_type, 0.0)
            node_features.append([lb / 100.0, is_scheduled, slack, urgency, type_embed])

        x = torch.tensor(node_features, dtype=torch.float)

        edge_index = []
        edge_attr = []

        for pred_id, succ_id in self.precedence_edges:
            if pred_id in node_to_idx and succ_id in node_to_idx:
                edge_index.append([node_to_idx[pred_id], node_to_idx[succ_id]])
                edge_attr.append([1.0, 0.0])

        if len(edge_index) > 0:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 2), dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def _compute_op_lb(self, op_id: str) -> float:
        op = self.operations[op_id]
        if op.is_scheduled:
            return op.scheduled_end
        pred_max = max([self.operations[p].scheduled_end if self.operations[p].is_scheduled else 0
                        for p in op.predecessors], default=0)
        return pred_max + op.get_min_processing_time()

    def state_hash(self) -> str:
        state_str = f"{len(self.scheduled_ops)}_{self.current_time:.2f}"
        return hashlib.md5(state_str.encode()).hexdigest()[:8]

    # [*]  is_complete
    def is_complete(self) -> bool:
        """ """
        if not self.operations:
            return True

        #   is_completed is_scheduled
        completed_count = sum(1 for op in self.operations.values() if op.is_completed)
        total_count = len(self.operations)

        return completed_count == total_count

    def get_unscheduled_ops(self) -> List[Operation]:
        return [op for op in self.operations.values() if not op.is_scheduled]

    def reset(self):
        for op in self.operations.values():
            op.is_scheduled = False
            op.scheduled_machine = None
            op.scheduled_start = 0.0
            op.scheduled_end = 0.0

        for machine in self.machines.values():
            machine.available_time = 0.0
            machine.utilization = 0.0
            machine.queue.clear()

        self.scheduled_ops.clear()
        self.eligible_ops = []
        self.current_time = 0.0
        self.makespan = 0.0
        self._update_eligible_ops()

    def __repr__(self):
        return f"DisjunctiveGraph(ops={len(self.operations)}, machines={len(self.machines)}, scheduled={len(self.scheduled_ops)})"

    def snapshot(self):
        """
        V43
        current_timeoperationskeys
        """
        snapshot_data = {
            # Ops
            'ops_state': {
                op_id: {
                    'is_scheduled': op.is_scheduled,
                    'scheduled_machine': op.scheduled_machine,
                    'scheduled_start': op.scheduled_start,
                    'scheduled_end': op.scheduled_end,
                    'is_completed': op.is_completed  # V41
                }
                for op_id, op in self.operations.items()
            },
            # V43operationskeys
            'ops_keys': set(self.operations.keys()),

            # Machines
            'machines_state': {
                mach_id: {
                    'available_time': mach.available_time,
                    'queue': list(mach.queue),
                }
                for mach_id, mach in self.machines.items()
            },
            # GraphV43current_time
            'scheduled_ops': set(self.scheduled_ops),
            'makespan': self.makespan
        }
        return snapshot_data

    def restore(self, snapshot):
        """
        V43+Graph
        OpsDangling Pointers
        """
        # V43keysOperation
        snapshot_keys = snapshot['ops_keys']
        current_keys = set(self.operations.keys())
        ops_to_delete = current_keys - snapshot_keys

        #
        for op_id in ops_to_delete:
            op = self.operations[op_id]

            # 1. successors
            for pred_id in op.predecessors:
                if pred_id in self.operations:
                    pred_op = self.operations[pred_id]
                    if op_id in pred_op.successors:
                        pred_op.successors.remove(op_id)

            # 2. predecessors
            for succ_id in op.successors:
                if succ_id in self.operations:
                    succ_op = self.operations[succ_id]
                    if op_id in succ_op.predecessors:
                        succ_op.predecessors.remove(op_id)

            # 3.
            del self.operations[op_id]

        #  precedence_edgesDangling
        self.precedence_edges = [
            (pred_id, succ_id) for pred_id, succ_id in self.precedence_edges
            if pred_id not in ops_to_delete and succ_id not in ops_to_delete
        ]

        # Ops
        for op_id, state in snapshot['ops_state'].items():
            if op_id in self.operations:
                op = self.operations[op_id]
                op.is_scheduled = state['is_scheduled']
                op.scheduled_machine = state['scheduled_machine']
                op.scheduled_start = state['scheduled_start']
                op.scheduled_end = state['scheduled_end']
                op.is_completed = state['is_completed']

        # Machines
        for mach_id, state in snapshot['machines_state'].items():
            if mach_id in self.machines:
                mach = self.machines[mach_id]
                mach.available_time = state['available_time']
                mach.queue = list(state['queue'])

        # Graph
        self.scheduled_ops = set(snapshot['scheduled_ops'])
        self.makespan = snapshot['makespan']

        # V44eligible_ops
        self._update_eligible_ops()
        #  
        #print(f"\n [Restore] :")
        #for mach in self.machines.values():
            #if mach.queue:
                #print(f"  {mach.id}: available_time={mach.available_time:.2f}, queue={mach.queue[-3:]}")