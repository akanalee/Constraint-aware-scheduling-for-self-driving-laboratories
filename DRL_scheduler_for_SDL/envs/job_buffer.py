import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
from collections import deque

@dataclass
class Job:
    """Job"""
    id: str
    job_type: str
    operations: List[Dict]
    arrival_time: float
    priority: str = "normal"
    due_date: Optional[float] = None

    def __post_init__(self):
        if self.due_date is None:
            # due date
            total_time = self._estimate_total_processing_time()
            #  due date (arrival + 3x total processing time)
            self.due_date = self.arrival_time + total_time * 3.0

    def _estimate_total_processing_time(self) -> float:
        """
        
        
        -  'processing_time' (scalar)
        -  'processing_times' (dict)
        """
        total = 0.0
        for op in self.operations:
            if 'processing_time' in op:
                total += op['processing_time']
            elif 'processing_times' in op:
                times = list(op['processing_times'].values())
                if times:
                    total += sum(times) / len(times)
        return total

class JobBuffer:
    """
    HLAJob
    """
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.current_time = 0

    def add_job(self, job: Job):
        """job"""
        if len(self.buffer) >= self.max_size:
            # 
            self.buffer.popleft()

        self.buffer.append(job)

    def flush(self) -> List[Job]:
        """jobs"""
        jobs = list(self.buffer)
        self.buffer.clear()
        return jobs

    def peek(self) -> List[Job]:
        """"""
        return list(self.buffer)

    def clear(self):
        """"""
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)

    def is_empty(self):
        return len(self.buffer) == 0

    def update_time(self, current_time: float):
        """slack"""
        self.current_time = current_time

    # ===  ===

    def get_avg_slack(self) -> float:
        """
        
        slack = due_date - current_time - estimated_total_processing_time
        """
        if len(self.buffer) == 0:
            return 0.0

        slacks = []
        for job in self.buffer:
            #  KeyError
            total_processing = job._estimate_total_processing_time()
            slack = job.due_date - self.current_time - total_processing
            slacks.append(slack)

        return float(np.mean(slacks))

    def get_total_operations(self) -> int:
        """"""
        return sum(len(job.operations) for job in self.buffer)

    def get_priority_distribution(self) -> Dict[str, int]:
        """"""
        dist = {"high": 0, "normal": 0, "low": 0}
        for job in self.buffer:
            dist[job.priority] = dist.get(job.priority, 0) + 1
        return dist

    def get_avg_waiting_time(self) -> float:
        """"""
        if len(self.buffer) == 0:
            return 0.0

        waiting_times = [self.current_time - job.arrival_time for job in self.buffer]
        return float(np.mean(waiting_times))

    def get_urgency_score(self) -> float:
        """
        HLA
        urgency = weighted_sum(slack_urgency, waiting_urgency, priority_urgency)
        """
        if len(self.buffer) == 0:
            return 0.0

        # 1. Slack urgency (slack)
        #  get_avg_slack
        avg_slack = self.get_avg_slack()
        slack_urgency = -avg_slack if avg_slack < 0 else 0

        # 2. Waiting urgency ()
        waiting_urgency = self.get_avg_waiting_time()

        # 3. Priority urgency (job)
        priority_dist = self.get_priority_distribution()
        priority_score = priority_dist.get("high", 0) * 5.0 + priority_dist.get("normal", 0) * 1.0

        # 
        urgency = (slack_urgency * 1.0 + waiting_urgency * 0.5 + priority_score * 2.0) / 10.0

        return max(0.0, urgency)

    def should_release_heuristic(self, threshold: int = 5) -> bool:
        """
        BC
        """
        if len(self.buffer) == 0:
            return False

        # 1
        if len(self.buffer) >= threshold:
            return True

        # 2job3
        for job in self.buffer:
            if job.priority == "high" and (self.current_time - job.arrival_time) > 3.0:
                return True

        # 3slack
        if self.get_avg_slack() < 0:
            return True

        return False

    def __repr__(self):
        return f"JobBuffer(size={len(self.buffer)}/{self.max_size})"