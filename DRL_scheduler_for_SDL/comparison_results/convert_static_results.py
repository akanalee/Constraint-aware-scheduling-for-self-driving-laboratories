#!/usr/bin/env python
"""
convert_static_results.py - Convert e2e_milp_spt.py JSON results to validation reports + Gantt charts.

Reads the JSON output produced by ThreeWayComparison.run_comparison() and generates:
  1. Per-method validation .txt files (matching ChemOS-DRL format)
  2. Per-method Gantt charts (job-based + machine-based)

Usage:
    python convert_static_results.py                              # uses ./comparison_results/
    python convert_static_results.py path/to/comparison_results   # custom dir
    python convert_static_results.py path/to/episode_000.json     # single file
"""

import os
import sys
import json
import re
import glob
from datetime import datetime
from collections import defaultdict

# Import Gantt chart functions from plot_gantt.py
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from plot_gantt import (plot_combined_gantt,
                        plot_three_way_job_gantt,
                        plot_three_way_machine_gantt)


def load_episode_results(path):
    """Load episode JSON result files from a directory or single file."""
    if os.path.isfile(path):
        with open(path, 'r', encoding='utf-8') as f:
            return [json.load(f)]
    elif os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, 'Trial~*.json'))
                       or glob.glob(os.path.join(path, 'episode_*.json')))
        results = []
        for fp in files:
            with open(fp, 'r', encoding='utf-8') as f:
                results.append(json.load(f))
        return results
    else:
        print(f"Error: {path} not found")
        sys.exit(1)


def extract_op_num(op_id):
    """Extract op number from op_id like 'job_001_op3' -> 3."""
    m = re.search(r'_op(\d+)$', op_id)
    return int(m.group(1)) if m else 0


def extract_job_num(job_id):
    """Extract numeric part from job_id like 'job_001' -> 1."""
    m = re.search(r'(\d+)', job_id)
    return int(m.group(1)) if m else 0


def build_ops_from_job_details(job_details):
    """Build op records from job_details dict (from _collect_job_details)."""
    ops = []
    for job_id, details in job_details.items():
        for op_info in details['ops']:
            if op_info['is_scheduled'] and op_info['scheduled_start'] is not None:
                # op_index is 0-based; op names (op1..op9) are 1-based
                op_num = extract_op_num(op_info['op_id'])
                if op_num == 0:
                    op_num = op_info.get('op_index', 0) + 1
                ops.append({
                    'job_id': job_id,
                    'op_id': op_info['op_id'],
                    'op_num': op_num,
                    'machine': op_info['scheduled_machine'],
                    'start': op_info['scheduled_start'],
                    'end': op_info['scheduled_end'],
                })
    return ops


def build_ops_from_schedule_sequence(schedule_sequence):
    """Build op records from schedule_sequence list."""
    ops = []
    for entry in schedule_sequence:
        op_id = entry['op_id']
        # Extract job_id from op_id (e.g. "job_001_op3" -> "job_001")
        parts = op_id.rsplit('_op', 1)
        job_id = parts[0] if len(parts) == 2 else op_id.split('_')[0] + '_' + op_id.split('_')[1]

        ops.append({
            'job_id': job_id,
            'op_id': op_id,
            'op_num': extract_op_num(op_id),
            'machine': entry['machine'],
            'start': entry['start'],
            'end': entry['end'],
        })
    return ops


def build_fcfs_abort_ops(fcfs_data, n_original=40):
    """
    Build ops for ChemOS Built-in Dispatch by walking the original job list top-to-bottom
    and compacting replacement jobs onto the row that triggered them.

    Algorithm (matches the user's spec):
      1. List all jobs sorted by their original job index (job_0, job_1, …).
      2. Walk the list top-to-bottom.  For each unconsumed job J:
         - Start a chain  C = [J]
         - While the head of the chain was terminated and has a replacement R
           (per `abort_replace_pairs`), append R to C, mark R as CONSUMED,
           and continue (the replacement may itself be terminated).
         - Emit one row containing every op of every job in C, plus a red ×
           at the termination time of every non-final job in C.
      3. Consumed replacements never get their own row — every subsequent
         row simply shifts up to fill the gap, so the chart has no blanks.

    With 12 aborts on a 40-budget run, the system creates 52 jobs and we
    consume 12 of them as replacements  exactly 40 rows.

    Row labels are renumbered job_0 .. job_{N-1} in walked order so the
    Gantt y-axis reads E_0..E_{N-1} contiguously.
    """
    jd    = fcfs_data.get('job_details', {})
    pairs = fcfs_data.get('abort_replace_pairs', []) or []

    def _term_time(v):
        completed = [op for op in v.get('ops', [])
                     if op.get('is_completed') and op.get('scheduled_end') is not None]
        return max(op['scheduled_end'] for op in completed) if completed else None

    # Forward chain: terminated job  its replacement
    term_to_rep = {}
    for p in pairs:
        t, r = p.get('terminated_job'), p.get('replacement_job')
        if t and r:
            term_to_rep[t] = r

    all_jobs = sorted(jd.keys(), key=lambda j: int(j.split('_')[1]))

    consumed = set()
    rows = []   # list of (chain_jobs, term_times)
    for jid in all_jobs:
        if jid in consumed:
            continue
        chain = [jid]
        terms = []
        cur = jid
        seen = {jid}
        while cur in term_to_rep:
            rep = term_to_rep[cur]
            if rep in seen:
                break
            tt = _term_time(jd.get(cur, {}))
            if tt is not None:
                terms.append(tt)
            consumed.add(rep)
            chain.append(rep)
            seen.add(rep)
            cur = rep
        rows.append((chain, terms))

    # Emit ops with sequential row IDs job_0..job_{N-1}
    ops = []
    terminations = {}
    for i, (chain, terms) in enumerate(rows):
        row_id = f'job_{i}'
        if terms:
            terminations[row_id] = terms
        for jid in chain:
            for op in jd[jid].get('ops', []):
                if op.get('is_completed') and op.get('is_scheduled') and \
                   op.get('scheduled_start') is not None:
                    ops.append({
                        'job_id':  row_id,
                        'op_id':   op['op_id'],
                        'op_num':  extract_op_num(op['op_id']),
                        'machine': op['scheduled_machine'],
                        'start':   op['scheduled_start'],
                        'end':     op['scheduled_end'],
                    })

    return ops, terminations


def write_validation_report(ops, method, episode_info, out_path):
    """Write a validation report .txt in the same format as ChemOS-DRL."""
    # Group ops by job
    jobs = defaultdict(list)
    for op in ops:
        jobs[op['job_id']].append(op)

    # Sort jobs and ops
    job_ids = sorted(jobs.keys(), key=extract_job_num)
    for jid in job_ids:
        jobs[jid].sort(key=lambda o: o['op_num'])

    # Compute summary
    total_jobs = len(job_ids)
    all_ends = [op['end'] for op in ops]
    makespan = max(all_ends) if all_ends else 0.0

    seed = episode_info.get('seed', 'N/A')
    method_result = episode_info.get(method, {})
    total_time = method_result.get('total_time', 0.0)
    tmax_violations = method_result.get('tmax_violations', 0)
    release_history = method_result.get('release_history', [])
    time_breakdown = method_result.get('time_breakdown', {})
    reported_makespan = method_result.get('makespan', makespan)

    method_label = {
        'ERC-FSS': 'FSS (DRL)',
        'scip': 'SCIP (MILP)',
        'SPT': 'SPT (Heuristic)',
    }.get(method, method.upper())

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('=' * 64 + '\n')
        f.write(f'Static Test Validation Report - {method_label}\n')
        f.write(f'Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'Seed: {seed}\n')
        f.write(f'Method: {method}\n')
        f.write('=' * 64 + '\n\n')

        # Per-job sections
        for jid in job_ids:
            f.write('=' * 64 + '\n')
            f.write(f'Job: {jid}\n')
            f.write('=' * 64 + '\n')
            f.write(f'  {"Op ID":<22}| {"Machine":<22}| {"Start (min)":>11} | {"End (min)":>9}\n')

            for op in jobs[jid]:
                f.write(f'  {op["op_id"]:<22}| {op["machine"]:<22}| {op["start"]:>11.2f} | {op["end"]:>9.2f}\n')
            f.write('\n')

        # Summary section
        f.write('=' * 64 + '\n')
        f.write('Summary\n')
        f.write('=' * 64 + '\n')

        completed_jobs = sum(1 for jid in job_ids if len(jobs[jid]) == 9)
        f.write(f'  Completed jobs:           {completed_jobs}/{total_jobs}\n')
        f.write(f'  Total releases:           {len(release_history)}\n')
        f.write(f'  Makespan:                 {reported_makespan:.2f} min\n')
        f.write(f'  Actual makespan:          {makespan:.2f} min\n')
        f.write(f'  Total solve time:         {total_time:.2f} s\n')

        if time_breakdown:
            f.write(f'  Time breakdown:\n')
            for key, val in time_breakdown.items():
                f.write(f'    {key:<20}: {val:.4f} s\n')

        f.write(f'  Tmax violations:          {tmax_violations}\n')
        f.write(f'  Seed:                     {seed}\n')
        f.write('=' * 64 + '\n')

    print(f"  Validation report saved: {out_path}")


def process_episode(episode_data, output_dir):
    """Process one episode: generate reports + Gantt charts for each method."""
    seed = episode_data.get('seed', 0)
    episode_idx = episode_data.get('episode_idx', 0)

    # Standard methods (non-ChemOS Built-in Dispatch) in display order
    TARGET_METHODS = ['ERC-FSS', 'Conventional DRL', 'GA', 'MILP', 'SPT']

    # Collect ops per method for six-way comparison charts
    ops_by_method = {}
    terminations_by_method = {}

    for method in TARGET_METHODS:
        if method not in episode_data:
            continue
        method_data = episode_data[method]

        if 'error' in method_data:
            print(f"  Skipping {method} (error: {method_data['error']})")
            continue

        # Prefer job_details (final state) over schedule_sequence
        ops = []
        if method_data.get('job_details'):
            ops = build_ops_from_job_details(method_data['job_details'])
        elif method_data.get('schedule_sequence'):
            ops = build_ops_from_schedule_sequence(method_data['schedule_sequence'])

        if not ops:
            print(f"  Skipping {method} (no op data found)")
            continue

        n_jobs = len(set(op['job_id'] for op in ops))
        print(f"\n  [{method.upper()}] {len(ops)} ops across {n_jobs} jobs")

        base = os.path.join(output_dir,
                            f'static_{method}_ep{episode_idx:03d}_seed{seed}')
        write_validation_report(ops, method, episode_data, base + '.txt')
        ops_by_method[method] = ops

    # Handle ChemOS Built-in Dispatch separately (replacement jobs merged onto original rows)
    if 'ChemOS Built-in Dispatch' in episode_data and 'error' not in episode_data['ChemOS Built-in Dispatch']:
        fcfs_data = episode_data['ChemOS Built-in Dispatch']
        fcfs_ops, fcfs_terms = build_fcfs_abort_ops(fcfs_data)
        if fcfs_ops:
            n_jobs = len(set(op['job_id'] for op in fcfs_ops))
            print(f"\n  [FCFS_ABORT] {len(fcfs_ops)} ops across {n_jobs} job rows"
                  f" ({len(fcfs_terms)} terminated)")
            ops_by_method['ChemOS Built-in Dispatch'] = fcfs_ops
            if fcfs_terms:
                terminations_by_method['ChemOS Built-in Dispatch'] = fcfs_terms

    # Generate six-way comparison charts (all methods side-by-side)
    if ops_by_method:
        ep_base = os.path.join(output_dir,
                               f'static_ep{episode_idx:03d}_seed{seed}')
        plot_three_way_job_gantt(
            ops_by_method, ep_base,
            terminations_by_method=terminations_by_method or None,
        )
        plot_three_way_machine_gantt(ops_by_method, ep_base)


def main():
    # Determine input path
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        # Default: look for comparison_results next to or above this script.
        candidates = [
            os.path.join(script_dir),
            os.path.join(os.path.dirname(script_dir), 'comparison_results'),
            os.path.join(os.path.dirname(script_dir), 'testing',
                         'Training_comparison_results'),
        ]
        input_path = None
        for c in candidates:
            if os.path.exists(c):
                input_path = c
                break

        if input_path is None:
            print("Error: No comparison_results directory found.")
            print("Usage: python convert_static_results.py <path_to_results>")
            sys.exit(1)

    # Output directory
    output_dir = os.path.join(script_dir, 'static_test_results')
    os.makedirs(output_dir, exist_ok=True)

    print(f"Input:  {input_path}")
    print(f"Output: {output_dir}")

    episodes = load_episode_results(input_path)
    print(f"Loaded {len(episodes)} episode(s)")

    for ep in episodes:
        ep_idx = ep.get('episode_idx', 0)
        seed = ep.get('seed', 'N/A')
        print(f"\n{'=' * 60}")
        print(f"Processing Episode {ep_idx} (seed={seed})")
        print(f"{'=' * 60}")
        process_episode(ep, output_dir)

    print(f"\nDone. Results in: {output_dir}")


if __name__ == '__main__':
    main()