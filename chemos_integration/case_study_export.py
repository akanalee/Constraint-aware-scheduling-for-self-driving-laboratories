import json
import os
import re
from datetime import datetime
from typing import Dict, List, Optional

_OP_RE = re.compile(r'_op(\d+)$')


def _op_index(op_id: str) -> int:
    m = _OP_RE.search(op_id)
    return int(m.group(1)) if m else 0


def _job_details_from_op_records(op_records: Dict[str, Dict]) -> Dict[str, Dict]:
    jd: Dict[str, Dict] = {}
    for op_id, rec in op_records.items():
        job_id = rec['job_id']
        if job_id not in jd:
            jd[job_id] = {'ops': [], 'completed': 0, 'scheduled': 0, 'total': 0}
        op_index = _op_index(op_id)
        jd[job_id]['ops'].append({
            'op_id': op_id,
            'op_index': op_index,
            'is_completed': True,
            'is_scheduled': True,
            'scheduled_start': float(rec['start']),
            'scheduled_end': float(rec['end']),
            'scheduled_machine': rec['machine'],
        })
        jd[job_id]['total'] += 1
        jd[job_id]['scheduled'] += 1
        jd[job_id]['completed'] += 1
    return jd


def _job_details_from_op_log(op_log: List[Dict]) -> Dict[str, Dict]:
    jd: Dict[str, Dict] = {}
    for entry in op_log:
        job_id = entry['job_id']
        if job_id not in jd:
            jd[job_id] = {'ops': [], 'completed': 0, 'scheduled': 0, 'total': 0}
        jd[job_id]['ops'].append({
            'op_id': entry['op_id'],
            'op_index': entry['op_index'],
            'is_completed': True,
            'is_scheduled': True,
            'scheduled_start': float(entry['start']),
            'scheduled_end': float(entry['end']),
            'scheduled_machine': entry['machine'],
        })
        jd[job_id]['total'] += 1
        jd[job_id]['scheduled'] += 1
        jd[job_id]['completed'] += 1
    return jd


def _schedule_sequence_from_op_records(op_records: Dict[str, Dict]) -> List[Dict]:
    seq = []
    for op_id, rec in op_records.items():
        seq.append({
            'step': 0,
            'start': float(rec['start']),
            'op_id': op_id,
            'machine': rec['machine'],
            'end': float(rec['end']),
        })
    seq.sort(key=lambda x: (x['start'], x['op_id']))
    return seq


def _schedule_sequence_from_op_log(op_log: List[Dict]) -> List[Dict]:
    seq = []
    for entry in op_log:
        seq.append({
            'step': 0,
            'start': float(entry['start']),
            'op_id': entry['op_id'],
            'machine': entry['machine'],
            'end': float(entry['end']),
        })
    seq.sort(key=lambda x: (x['start'], x['op_id']))
    return seq


def _ercfss_section(ercfss_res: Dict) -> Dict:
    op_records = ercfss_res.get('op_records') or {}
    makespan = float(ercfss_res.get('makespan_min', 0.0) or 0.0)
    return {
        'makespan': makespan,
        'job_details': _job_details_from_op_records(op_records),
        'tmax_violations': int(ercfss_res.get('tmax_violations', 0) or 0),
        'total_solve_time': float(ercfss_res.get('total_solve_time', 0.0) or 0.0),
        'total_time': float(ercfss_res.get('wall_time_s', 0.0) or 0.0),
        'time_breakdown': {
            'hla_decision': 0.0, 'scheduling': 0.0, 'time_advance': 0.0,
            'feasibility_check': float(ercfss_res.get('feasibility_check_time', 0.0) or 0.0),
        },
        'schedule_sequence': _schedule_sequence_from_op_records(op_records),
        'release_history': [],
    }


def _chemos_abort_section(dispatch_res: Dict) -> Dict:
    op_log = dispatch_res.get('op_log') or []
    abort_pairs = dispatch_res.get('abort_replace_pairs') or []
    makespan = float(dispatch_res.get('makespan_min', 0.0) or 0.0)
    return {
        'makespan': makespan,
        'job_details': _job_details_from_op_log(op_log),
        'tmax_violations': 0,
        'aborted_count': int(dispatch_res.get('aborted_count', 0) or 0),
        'total_solve_time': float(dispatch_res.get('total_solve_time', 0.0) or 0.0),
        'total_time': float(dispatch_res.get('wall_time_s', 0.0) or 0.0),
        'time_breakdown': {
            'hla_decision': 0.0, 'scheduling': 0.0, 'time_advance': 0.0,
        },
        'schedule_sequence': _schedule_sequence_from_op_log(op_log),
        'release_history': [],
        'abort_replace_pairs': abort_pairs,
    }


def write_episode_json(out_dir: str,
                       episode_idx: int,
                       seed: int,
                       ercfss_res: Optional[Dict],
                       dispatch_res: Optional[Dict]) -> str:
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        'episode_idx': int(episode_idx),
        'seed': int(seed),
        'timestamp': datetime.now().isoformat(),
    }
    if ercfss_res is not None:
        payload['ercfss'] = _ercfss_section(ercfss_res)
    if dispatch_res is not None:
        payload['chemos_abort'] = _chemos_abort_section(dispatch_res)

    fname = f'episode_{int(episode_idx):03d}_seed_{int(seed)}.json'
    fpath = os.path.join(out_dir, fname)
    with open(fpath, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[CaseStudyExport] {fpath}")
    return fpath
