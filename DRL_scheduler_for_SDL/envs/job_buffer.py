from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class Job:
    id: str
    job_type: str
    operations: List[Dict]
    arrival_time: float
    priority: str = "normal"
    due_date: Optional[float] = None


class JobBuffer:
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def add_job(self, job: Job):
        if len(self.buffer) >= self.max_size:
            self.buffer.popleft()

        self.buffer.append(job)

    def flush(self) -> List[Job]:
        jobs = list(self.buffer)
        self.buffer.clear()
        return jobs

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)

    def __repr__(self):
        return f"JobBuffer(size={len(self.buffer)}/{self.max_size})"
