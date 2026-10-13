import os
import threading
import time

from chemos_integration.file_protocol import (
    OpCompletionEvent,
    ScheduleCommand,
    is_episode_complete,
    mark_command_completed,
    mark_command_executing,
    read_clock_state,
    read_schedule_commands,
    write_completion_event,
)

TIME_SCALE = float(os.environ.get('CHEMOS_TIME_SCALE', '0.01'))


class ScheduleExecutor:
    def __init__(self, time_scale=TIME_SCALE):
        self.time_scale = float(time_scale)
        self.running = False
        self.executor_thread = None

        self._clock_state = None
        self._clock_epoch = None

        self.executing_ops = {}
        self.completed_ops = set()
        self.job_ops_map = {}

        self.stats = {
            "commands_executed": 0,
            "ops_completed": 0,
            "jobs_completed": 0,
        }

        print(f"[ScheduleExecutor] init (time_scale={time_scale})")

    def _refresh_clock(self):
        s = read_clock_state()
        if s is None:
            return
        try:
            ep = int(s.get("epoch", 0))
        except Exception:
            ep = 0
        if self._clock_epoch is None:
            self._clock_epoch = ep
        elif ep != self._clock_epoch:
            print(f"[ScheduleExecutor] new episode clock {self._clock_epoch} -> {ep}; "
                  f"resetting executor state")
            self.executing_ops.clear()
            self.completed_ops.clear()
            self.job_ops_map.clear()
            self._clock_epoch = ep
        self._clock_state = s

    def _t_abs_min(self):
        s = self._clock_state
        if not s:
            return None
        try:
            if s.get("paused", False):
                return float(s.get("paused_min", s["base_min"]))
            return (s["base_min"]
                    + (time.time() - s["anchor_real"]) / s["time_scale"] / 60.0)
        except Exception:
            return None

    def start(self):
        self.running = True
        self.executor_thread = threading.Thread(target=self._execution_loop)
        self.executor_thread.daemon = True
        self.executor_thread.start()
        print("[ScheduleExecutor] started")

    def stop(self):
        self.running = False
        if self.executor_thread:
            self.executor_thread.join(timeout=2)
        print("[ScheduleExecutor] stopped")

    def _execution_loop(self):
        while self.running:
            try:
                self._refresh_clock()

                commands = read_schedule_commands()
                for cmd in commands:
                    if cmd.status == "pending":
                        self._execute_command(cmd)

                self._check_op_completions()

                if is_episode_complete():
                    print("[ScheduleExecutor] episode complete")
                    break

            except Exception as e:
                print(f"[ScheduleExecutor] loop error: {e}")

            sleep_real = 0.5 * max(self.time_scale, 0.001)
            t_abs = self._t_abs_min()
            if t_abs is not None and self.executing_ops:
                next_end = min(info["end"] for info in self.executing_ops.values())
                wait_min = next_end - t_abs
                if wait_min <= 0:
                    sleep_real = 0.0
                else:
                    cand = wait_min * 60.0 * self.time_scale
                    sleep_real = min(sleep_real, cand)
            if sleep_real > 0:
                time.sleep(max(sleep_real, 0.005))

    def _execute_command(self, cmd: ScheduleCommand):
        print(f"\n[ScheduleExecutor] command {cmd.command_id}")
        print(f"  jobs: {cmd.released_jobs}")
        print(f"  ops:  {len(cmd.schedule)}")

        mark_command_executing(cmd)

        for job_id in cmd.released_jobs:
            self.job_ops_map.setdefault(job_id, set())

        for start, op_id, machine, end in cmd.schedule:
            job_id = op_id.rsplit("_op", 1)[0] if "_op" in op_id else op_id
            self.job_ops_map.setdefault(job_id, set()).add(op_id)

            self.executing_ops[op_id] = {
                "start": float(start),
                "end": float(end),
                "machine": machine,
                "job_id": job_id,
            }
            print(f"    {op_id} @ {machine} [{start:.2f}-{end:.2f}min]")

        mark_command_completed(cmd)
        self.stats["commands_executed"] += 1

    def _check_op_completions(self):
        t_abs = self._t_abs_min()
        if t_abs is None:
            return
        due = sorted(
            ((op_id, info) for op_id, info in self.executing_ops.items()
             if t_abs >= info["end"]),
            key=lambda kv: (kv[1]["end"], kv[0]),
        )
        for op_id, info in due:
            self.executing_ops.pop(op_id, None)
            if op_id in self.completed_ops:
                continue
            self.completed_ops.add(op_id)
            self.stats["ops_completed"] += 1

            event = OpCompletionEvent(
                op_id=op_id,
                job_id=info["job_id"],
                machine_id=info["machine"],
                start_time=info["start"],
                end_time=info["end"],
                event_type="op_complete",
            )
            write_completion_event(event)
            print(f"  [Complete] {op_id} @ {info['machine']} "
                  f"end={info['end']:.2f}min  t_abs={t_abs:.2f}min")
            self._check_job_completion(info["job_id"])

    def _check_job_completion(self, job_id: str):
        if job_id not in self.job_ops_map:
            return
        job_ops = self.job_ops_map[job_id]
        if job_ops and job_ops.issubset(self.completed_ops):
            self.stats["jobs_completed"] += 1
            event = OpCompletionEvent(
                op_id=f"{job_id}_complete",
                job_id=job_id,
                machine_id="",
                start_time=0,
                end_time=0,
                event_type="job_complete",
            )
            write_completion_event(event)
            print(f"  [JobComplete] {job_id}")
