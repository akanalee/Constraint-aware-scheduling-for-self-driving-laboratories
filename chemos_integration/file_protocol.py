import json
import os
import pickle
import shutil
import time
import uuid
from typing import Dict, List, Optional

COMM_BASE_DIR = os.environ.get(
    'CHEMOS_COMM_DIR',
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'chemos_comm'),
)
SCHEDULE_CMD_DIR = os.path.join(COMM_BASE_DIR, 'schedule_commands')
OP_COMPLETE_DIR = os.path.join(COMM_BASE_DIR, 'op_completions')
STATUS_DIR = os.path.join(COMM_BASE_DIR, 'status')
LOGS_DIR = os.path.join(COMM_BASE_DIR, 'logs')

DRL_READY_FLAG = os.path.join(STATUS_DIR, 'drl_ready.flag')
CHEMOS_READY_FLAG = os.path.join(STATUS_DIR, 'chemos_ready.flag')
EPISODE_COMPLETE_FLAG = os.path.join(STATUS_DIR, 'episode_complete.flag')

CLOCK_STATE_FILE = os.path.join(STATUS_DIR, 'clock_state.json')


def read_clock_state() -> Optional[Dict]:
    try:
        with open(CLOCK_STATE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def read_prev_clock_epoch() -> int:
    s = read_clock_state()
    if not s:
        return 0
    try:
        return int(s.get('epoch', 0))
    except Exception:
        return 0


def init_comm_dirs():
    dirs = [COMM_BASE_DIR, SCHEDULE_CMD_DIR, OP_COMPLETE_DIR, STATUS_DIR, LOGS_DIR]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    for d in [SCHEDULE_CMD_DIR, OP_COMPLETE_DIR]:
        for f in os.listdir(d):
            try:
                os.remove(os.path.join(d, f))
            except OSError:
                pass
    for flag in [DRL_READY_FLAG, CHEMOS_READY_FLAG, EPISODE_COMPLETE_FLAG,
                 CLOCK_STATE_FILE]:
        if os.path.exists(flag):
            os.remove(flag)
    print(f"[FileProtocol] Communication dirs initialized: {COMM_BASE_DIR}")


def cleanup_comm_dirs():
    if os.path.exists(COMM_BASE_DIR):
        shutil.rmtree(COMM_BASE_DIR)
    print("[FileProtocol] Communication dirs cleaned up")


class ScheduleCommand:
    def __init__(self, schedule, released_jobs, makespan, solve_time, command_id=None):
        self.command_id = command_id or str(uuid.uuid4())[:8]
        self.timestamp = time.time()
        self.schedule = schedule
        self.released_jobs = released_jobs
        self.makespan = makespan
        self.solve_time = solve_time
        self.status = 'pending'


def write_schedule_command(cmd: ScheduleCommand) -> str:
    filename = f"cmd_{cmd.command_id}_{int(cmd.timestamp * 1000)}.pkl"
    filepath = os.path.join(SCHEDULE_CMD_DIR, filename)
    tmp_path = filepath + '.tmp'
    with open(tmp_path, 'wb') as f:
        pickle.dump(cmd, f)
    os.replace(tmp_path, filepath)
    return filepath


def read_schedule_commands() -> List[ScheduleCommand]:
    commands = []
    if not os.path.exists(SCHEDULE_CMD_DIR):
        return commands
    for filename in sorted(os.listdir(SCHEDULE_CMD_DIR)):
        if not filename.endswith('.pkl'):
            continue
        filepath = os.path.join(SCHEDULE_CMD_DIR, filename)
        try:
            with open(filepath, 'rb') as f:
                cmd = pickle.load(f)
            if cmd.status == 'pending':
                commands.append(cmd)
        except Exception as e:
            print(f"[FileProtocol] Failed to read command {filename}: {e}")
    return commands


def mark_command_executing(cmd: ScheduleCommand):
    cmd.status = 'executing'
    _update_command_file(cmd)


def mark_command_completed(cmd: ScheduleCommand):
    cmd.status = 'completed'
    _update_command_file(cmd)


def _update_command_file(cmd: ScheduleCommand):
    for filename in os.listdir(SCHEDULE_CMD_DIR):
        if cmd.command_id in filename:
            filepath = os.path.join(SCHEDULE_CMD_DIR, filename)
            tmp_path = filepath + '.tmp'
            with open(tmp_path, 'wb') as f:
                pickle.dump(cmd, f)
            os.replace(tmp_path, filepath)
            return


class OpCompletionEvent:
    def __init__(self, op_id, job_id, machine_id, start_time, end_time,
                 event_type='op_complete'):
        self.event_id = str(uuid.uuid4())[:8]
        self.timestamp = time.time()
        self.op_id = op_id
        self.job_id = job_id
        self.machine_id = machine_id
        self.start_time = start_time
        self.end_time = end_time
        self.event_type = event_type
        self.processed = False


def write_completion_event(event: OpCompletionEvent) -> str:
    filename = f"complete_{event.event_id}_{int(event.timestamp * 1000)}.pkl"
    filepath = os.path.join(OP_COMPLETE_DIR, filename)
    tmp_path = filepath + '.tmp'
    with open(tmp_path, 'wb') as f:
        pickle.dump(event, f)
    os.replace(tmp_path, filepath)
    return filepath


def read_completion_events() -> List[OpCompletionEvent]:
    events = []
    if not os.path.exists(OP_COMPLETE_DIR):
        return events
    for filename in sorted(os.listdir(OP_COMPLETE_DIR)):
        if not filename.endswith('.pkl'):
            continue
        filepath = os.path.join(OP_COMPLETE_DIR, filename)
        try:
            with open(filepath, 'rb') as f:
                event = pickle.load(f)
            if not event.processed:
                events.append(event)
        except Exception as e:
            print(f"[FileProtocol] Failed to read event {filename}: {e}")
    return events


def mark_event_processed(event: OpCompletionEvent):
    for filename in os.listdir(OP_COMPLETE_DIR):
        if event.event_id in filename:
            filepath = os.path.join(OP_COMPLETE_DIR, filename)
            try:
                os.remove(filepath)
            except OSError:
                pass
            return


def set_drl_ready():
    with open(DRL_READY_FLAG, 'w') as f:
        f.write(str(time.time()))


def set_chemos_ready():
    with open(CHEMOS_READY_FLAG, 'w') as f:
        f.write(str(time.time()))


def is_drl_ready() -> bool:
    return os.path.exists(DRL_READY_FLAG)


def is_chemos_ready() -> bool:
    return os.path.exists(CHEMOS_READY_FLAG)


def set_episode_complete():
    with open(EPISODE_COMPLETE_FLAG, 'w') as f:
        f.write(str(time.time()))


def is_episode_complete() -> bool:
    return os.path.exists(EPISODE_COMPLETE_FLAG)
