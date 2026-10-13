import argparse
import csv
import glob
import json
import os
import re
import statistics

BENCHMARK_METHODS = ['MILP', 'ERC-FSS', 'Conventional DRL', 'GA', 'SPT']
CHEMOS_DISPATCH = 'ChemOS Built-in Dispatch'
ABLATION_VARIANTS = ['erc_full', 'erc_no_backward', 'erc_no_forward', 'erc_no_urgency']
ABLATION_LABELS = {
    'erc_full': 'ERC-FSS (full)',
    'erc_no_backward': 'w/o backward validation',
    'erc_no_forward': 'w/o forward check',
    'erc_no_urgency': 'w/o urgency feature',
}
FSS_DETAIL_KEYS = [
    ('urgency_feature', 'Constraint-induced urgency feature'),
    ('other_state_construction', 'Other state construction'),
    ('policy_inference', 'GIN encoder and decoder inference'),
    ('bidirectional_masking', 'Bidirectional masking'),
    ('dynamic_mask_construction', '  Dynamic mask construction'),
    ('forward_feasibility_check', '  Forward feasibility check'),
    ('backward_deadline_validation', '  Backward deadline validation'),
    ('schedule_update', 'Schedule update'),
]


def _trial_key(path, data):
    m = re.search(r'trial_(\d+)\.json$', os.path.basename(path))
    if m:
        return (0, int(m.group(1)))
    return (1, data.get('episode_idx', 0), os.path.basename(path))


def load_results(directory):
    episodes = []
    for path in glob.glob(os.path.join(directory, '*.json')):
        if os.path.basename(path) == 'summary.json':
            continue
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        episodes.append((_trial_key(path, data), data))
    episodes.sort(key=lambda x: x[0])
    return [d for _, d in episodes]


def mean_std(values):
    values = [float(v) for v in values]
    if not values:
        return float('nan'), float('nan')
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def _fmt(values, digits=2):
    m, s = mean_std(values)
    return f"{m:.{digits}f} +/- {s:.{digits}f}"


def _ok(result):
    return isinstance(result, dict) and 'error' not in result


def release_metrics(result):
    history = result.get('release_history', [])
    return {
        'makespan': float(result['makespan']),
        'computation_time': float(result['total_solve_time']),
        'releases': len(history),
        'evaluations': int(sum(r['combinations_tried'] for r in history)),
        'tmax_violations': int(result['tmax_violations']),
    }


def _summarize_release_methods(episodes, methods):
    out = {}
    for method in methods:
        rows = [release_metrics(ep[method]) for ep in episodes if _ok(ep.get(method))]
        if not rows:
            continue
        seeds = [ep.get('seed') for ep in episodes if _ok(ep.get(method))]
        entry = {'n_trials': len(rows), 'per_trial': [dict(seed=sd, **r) for sd, r in zip(seeds, rows)]}
        for key in ('makespan', 'computation_time', 'releases', 'evaluations', 'tmax_violations'):
            m, s = mean_std([r[key] for r in rows])
            entry[key] = {'mean': m, 'std': s}
        entry['evals_to_releases_ratio'] = entry['evaluations']['mean'] / entry['releases']['mean'] \
            if entry['releases']['mean'] > 0 else float('nan')
        breakdowns = [ep[method]['runtime_breakdown'] for ep in episodes
                      if _ok(ep.get(method)) and 'runtime_breakdown' in ep[method]]
        if len(breakdowns) == len(rows):
            entry['runtime_breakdown'] = summarize_runtime(breakdowns, [r['evaluations'] for r in rows])
        out[method] = entry
    return out


def summarize_runtime(breakdowns, evaluations):
    total = [b['total'] for b in breakdowns]
    total_sum = sum(total)
    comp = {
        'feasibility_check': [b['feasibility_check'] for b in breakdowns],
        'schedule_construction': [b['schedule_construction'] for b in breakdowns],
        'tmax_verification_and_erc_operations': [b['tmax_verification'] + b['erc_operations']
                                                 for b in breakdowns],
        'total': total,
    }
    out = {}
    for key, vals in comp.items():
        m, s = mean_std(vals)
        out[key] = {'mean': m, 'std': s,
                    'share_pct': 100.0 * sum(vals) / total_sum if total_sum > 0 else float('nan')}
    n_checked = sum(b['candidates_checked'] for b in breakdowns)
    out['candidates_checked'] = n_checked
    out['candidates_infeasible'] = sum(b['candidates_infeasible'] for b in breakdowns)
    out['time_per_feasibility_check'] = (sum(comp['feasibility_check']) / n_checked
                                         if n_checked else float('nan'))
    ev_m, ev_s = mean_std(evaluations)
    out['candidate_evaluations'] = {'mean': ev_m, 'std': ev_s}
    out['scheduling_time_per_evaluation'] = (sum(comp['schedule_construction']) / sum(evaluations)
                                             if sum(evaluations) else float('nan'))
    details = [dict(b['fss_detail']) for b in breakdowns if 'fss_detail' in b]
    if len(details) == len(breakdowns):
        fss = {}
        for d in details:
            d['bidirectional_masking'] = (d['dynamic_mask_construction'] + d['forward_feasibility_check']
                                          + d['backward_deadline_validation'])
        for key, _ in FSS_DETAIL_KEYS:
            vals = [d[key] for d in details]
            m, s = mean_std(vals)
            fss[key] = {'mean': m, 'std': s,
                        'share_pct': 100.0 * sum(vals) / total_sum if total_sum > 0 else float('nan')}
        out['fss_detail'] = fss
    return out


def summarize_benchmark(episodes):
    summary = {'n_trials': len(episodes),
               'methods': _summarize_release_methods(episodes, BENCHMARK_METHODS)}
    rows = [ep[CHEMOS_DISPATCH] for ep in episodes if _ok(ep.get(CHEMOS_DISPATCH))]
    if rows:
        entry = {'n_trials': len(rows),
                 'per_trial': [{'seed': ep.get('seed'),
                                'makespan': float(ep[CHEMOS_DISPATCH]['makespan']),
                                'computation_time': float(ep[CHEMOS_DISPATCH]['total_solve_time']),
                                'tmax_violations': int(ep[CHEMOS_DISPATCH]['tmax_violations'])}
                               for ep in episodes if _ok(ep.get(CHEMOS_DISPATCH))]}
        for key in ('makespan', 'computation_time', 'tmax_violations'):
            m, s = mean_std([r[key] for r in entry['per_trial']])
            entry[key] = {'mean': m, 'std': s}
        summary['methods'][CHEMOS_DISPATCH] = entry
    return summary


def _ms(entry, key, digits=2):
    return f"{entry[key]['mean']:.{digits}f} +/- {entry[key]['std']:.{digits}f}"


def print_benchmark_summary(summary):
    methods = summary['methods']
    print(f"\nScheduling performance ({summary['n_trials']} trials, mean +/- standard deviation)")
    print(f"{'Method':26s} {'Makespan (min)':>18s} {'Computation time (s)':>22s} "
          f"{'Total releases':>16s} {'Total evals':>18s} {'Ratio':>8s} {'Tmax violations':>17s}")
    for method in BENCHMARK_METHODS:
        if method not in methods:
            continue
        e = methods[method]
        print(f"{method:26s} {_ms(e, 'makespan'):>18s} {_ms(e, 'computation_time'):>22s} "
              f"{_ms(e, 'releases'):>16s} {_ms(e, 'evaluations'):>18s} "
              f"{e['evals_to_releases_ratio']:>8.2f} {e['tmax_violations']['mean']:>17.2f}")
    if CHEMOS_DISPATCH in methods:
        e = methods[CHEMOS_DISPATCH]
        print(f"{CHEMOS_DISPATCH:26s} {_ms(e, 'makespan'):>18s} {_ms(e, 'computation_time'):>22s} "
              f"{'---':>16s} {'---':>18s} {'---':>8s} {_ms(e, 'tmax_violations'):>17s}")

    print("\nPer-trial results (makespan / computation time / releases / evals)")
    for method in BENCHMARK_METHODS:
        if method not in methods:
            continue
        print(f"  {method}")
        for k, r in enumerate(methods[method]['per_trial'], 1):
            print(f"    {k:2d}  {r['makespan']:8.2f}  {r['computation_time']:9.2f}  "
                  f"{r['releases']:4d}  {r['evaluations']:6d}")
    if CHEMOS_DISPATCH in methods:
        print(f"  {CHEMOS_DISPATCH} (makespan / computation time / Tmax violations)")
        for k, r in enumerate(methods[CHEMOS_DISPATCH]['per_trial'], 1):
            print(f"    {k:2d}  {r['makespan']:8.2f}  {r['computation_time']:9.2f}  {r['tmax_violations']:4d}")

    with_breakdown = [m for m in BENCHMARK_METHODS if m in methods and 'runtime_breakdown' in methods[m]]
    if not with_breakdown:
        return
    print("\nComputation-time breakdown (s/trial, mean +/- standard deviation)")
    print(f"{'Method':18s} {'Gurobi feasibility check':>26s} {'Schedule construction':>24s} "
          f"{'Tmax verification & ERC operations':>36s} {'Total':>20s}")
    for method in with_breakdown:
        b = methods[method]['runtime_breakdown']
        print(f"{method:18s} {_ms(b, 'feasibility_check'):>26s} {_ms(b, 'schedule_construction'):>24s} "
              f"{_ms(b, 'tmax_verification_and_erc_operations'):>36s} {_ms(b, 'total'):>20s}")
    print("\nCandidate evaluations and scheduling time per evaluation")
    print(f"{'Method':18s} {'Candidate evaluations (per trial)':>34s} {'Scheduling time per evaluation (s)':>36s}")
    for method in with_breakdown:
        b = methods[method]['runtime_breakdown']
        print(f"{method:18s} {_ms(b, 'candidate_evaluations'):>34s} "
              f"{b['scheduling_time_per_evaluation']:>36.3f}")
    if 'ERC-FSS' in with_breakdown:
        b = methods['ERC-FSS']['runtime_breakdown']
        print("\nERC-FSS computation-time breakdown (s/trial, mean +/- standard deviation, % of total)")
        for key, label in [('feasibility_check', 'Gurobi feasibility check'),
                           ('schedule_construction', 'FSS schedule construction'),
                           ('tmax_verification_and_erc_operations', 'ERC evaluations'),
                           ('total', 'Total')]:
            print(f"  {label:36s} {_ms(b, key):>20s} {b[key]['share_pct']:6.1f}")
        print(f"  Time per feasibility check: {b['time_per_feasibility_check']:.3f} s "
              f"({b['candidates_checked']} checks, {b['candidates_infeasible']} proven infeasible)")
        if 'fss_detail' in b:
            print("\nFSS schedule construction breakdown (s/trial, mean +/- standard deviation, % of total)")
            for key, label in FSS_DETAIL_KEYS:
                print(f"  {label:36s} {_ms(b['fss_detail'], key):>20s} "
                      f"{b['fss_detail'][key]['share_pct']:6.1f}")
            print(f"  {'FSS schedule construction':36s} {_ms(b, 'schedule_construction'):>20s} "
                  f"{b['schedule_construction']['share_pct']:6.1f}")


def summarize_ablation(episodes):
    return {'n_trials': len(episodes), 'variants': _summarize_release_methods(episodes, ABLATION_VARIANTS)}


def print_ablation_summary(summary):
    print(f"\nComponent ablation ({summary['n_trials']} trials)")
    print(f"{'Variant':26s} {'Makespan (min)':>18s} {'Computation time (s)':>22s} "
          f"{'Total releases':>16s} {'Total evals':>18s} {'Ratio':>8s}")
    for v in ABLATION_VARIANTS:
        if v not in summary['variants']:
            continue
        e = summary['variants'][v]
        print(f"{ABLATION_LABELS[v]:26s} {_ms(e, 'makespan'):>18s} {_ms(e, 'computation_time'):>22s} "
              f"{_ms(e, 'releases'):>16s} {_ms(e, 'evaluations'):>18s} {e['evals_to_releases_ratio']:>8.2f}")
    print("\nPer-trial results (makespan / computation time / releases / evals)")
    for v in ABLATION_VARIANTS:
        if v not in summary['variants']:
            continue
        print(f"  {ABLATION_LABELS[v]}")
        for k, r in enumerate(summary['variants'][v]['per_trial'], 1):
            print(f"    {k:2d}  {r['makespan']:8.2f}  {r['computation_time']:9.2f}  "
                  f"{r['releases']:4d}  {r['evaluations']:6d}")


def summarize_chemos_validation(episodes):
    rows = []
    for ep in episodes:
        r = ep['ERC-FSS']
        row = {'seed': ep.get('seed'), 'makespan': float(r['makespan']),
               'computation_time': float(r['total_solve_time']),
               'tmax_violations': int(r['tmax_violations'])}
        check = r.get('time_breakdown', {}).get('feasibility_check')
        if check is not None:
            row['feasibility_check'] = float(check)
        rows.append(row)
    out = {'n_trials': len(rows), 'per_trial': rows}
    for key in ('makespan', 'computation_time', 'tmax_violations'):
        vals = [r[key] for r in rows]
        m, s = mean_std(vals)
        out[key] = {'mean': m, 'std': s, 'min': min(vals), 'max': max(vals)}
    return out


def print_chemos_validation_summary(summary):
    print(f"\nERC-FSS under ChemOS-integrated execution ({summary['n_trials']} trials)")
    print(f"{'Trial':>6s} {'Makespan (min)':>16s} {'Computation time (s)':>22s} {'Tmax violations':>17s}")
    for k, r in enumerate(summary['per_trial'], 1):
        print(f"{k:6d} {r['makespan']:16.2f} {r['computation_time']:22.2f} {r['tmax_violations']:17d}")
    print(f"{'Mean':>6s} {summary['makespan']['mean']:16.2f} {summary['computation_time']['mean']:22.2f} "
          f"{summary['tmax_violations']['mean']:17.2f}")
    print(f"{'Range':>6s} {'[%.2f, %.2f]' % (summary['makespan']['min'], summary['makespan']['max']):>16s} "
          f"{'[%.2f, %.2f]' % (summary['computation_time']['min'], summary['computation_time']['max']):>22s} "
          f"{'[%d, %d]' % (summary['tmax_violations']['min'], summary['tmax_violations']['max']):>17s}")


def summarize_alkox(directory):
    paths = sorted(glob.glob(os.path.join(directory, 'episode_*.json')))
    out = []
    for path in paths:
        with open(path, 'r', encoding='utf-8') as f:
            d = json.load(f)
        out.append({
            'seed': d.get('seed'),
            'ERC-FSS': {'makespan': d['ercfss']['makespan'],
                        'computation_time': d['ercfss']['total_solve_time'],
                        'tmax_violations': d['ercfss']['tmax_violations']},
            CHEMOS_DISPATCH: {'makespan': d['chemos_abort']['makespan'],
                              'computation_time': d['chemos_abort']['total_solve_time'],
                              'tmax_violations': d['chemos_abort'].get('aborted_count',
                                                                       d['chemos_abort']['tmax_violations'])},
        })
    return out


def print_alkox_summary(rows):
    print("\nBiocatalytic benzyl alcohol oxidation case study")
    print(f"{'Method':28s} {'Makespan (min)':>16s} {'Computation time (s)':>22s} {'Tmax violations':>17s}")
    for row in rows:
        for method in ('ERC-FSS', CHEMOS_DISPATCH):
            r = row[method]
            print(f"{method:28s} {r['makespan']:16.2f} {r['computation_time']:22.3f} {r['tmax_violations']:17d}")


def summarize_suzuki(directory):
    path = os.path.join(directory, 'suzuki_9grid_schedule_summary.csv')
    with open(path, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        out.append({'alpha': r['severity'], 'method': r['policy'],
                    'makespan': float(r['makespan_min']),
                    'computation_time': float(r['total_solving_time_s']),
                    'tmax_violations': int(r['tmax_violations'])})
    best = {}
    for tag in ('ERCFSS', 'abort', 'noabort'):
        for sev in ('10', '20', '30'):
            fpath = os.path.join(directory, f'suzuki_truecurve_{tag}_noise_{sev}pct.csv')
            if not os.path.exists(fpath):
                continue
            with open(fpath, 'r', encoding='utf-8') as f:
                vals = [float(x['best_so_far']) for x in csv.DictReader(f) if x['best_so_far']]
            best[f'{tag}_{sev}pct'] = max(vals) if vals else float('nan')
    return {'schedules': out, 'best_yield': best}


def print_suzuki_summary(summary):
    print("\nPd-catalyzed Suzuki-Miyaura cross-coupling case study")
    print(f"{'alpha':>6s} {'Method':48s} {'Makespan (min)':>16s} {'Computation time (s)':>22s} {'Tmax violations':>17s}")
    for r in summary['schedules']:
        print(f"{r['alpha']:>6s} {r['method']:48s} {r['makespan']:16.2f} {r['computation_time']:22.3f} "
              f"{r['tmax_violations']:17d}")
    labels = {'ERCFSS': 'ERC-FSS', 'abort': 'ChemOS built-in dispatch with termination',
              'noabort': 'ChemOS built-in dispatch without termination'}
    print("\nBest observed coupling yield (%)")
    for tag, label in labels.items():
        vals = [summary['best_yield'].get(f'{tag}_{sev}pct') for sev in ('10', '20', '30')]
        print(f"  {label:48s} " + '  '.join(f"{v:6.1f}" if v is not None else '   ---' for v in vals))


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    results = os.path.join(repo_root, 'results')
    parser = argparse.ArgumentParser(description='Summarize result files')
    parser.add_argument('kind', choices=['benchmark', 'ablation', 'chemos', 'alkox', 'suzuki'])
    parser.add_argument('directory', nargs='?', default=None)
    parser.add_argument('--save', type=str, default=None, help='Write the summary to this JSON file')
    args = parser.parse_args()

    defaults = {
        'benchmark': os.path.join(results, 'benchmark'),
        'ablation': os.path.join(results, 'ablation'),
        'chemos': os.path.join(results, 'chemos_validation'),
        'alkox': os.path.join(results, 'case_study_alkox'),
        'suzuki': os.path.join(results, 'case_study_suzuki'),
    }
    directory = args.directory or defaults[args.kind]

    if args.kind == 'benchmark':
        summary = summarize_benchmark(load_results(directory))
        print_benchmark_summary(summary)
    elif args.kind == 'ablation':
        summary = summarize_ablation(load_results(directory))
        print_ablation_summary(summary)
    elif args.kind == 'chemos':
        summary = summarize_chemos_validation(load_results(directory))
        print_chemos_validation_summary(summary)
    elif args.kind == 'alkox':
        summary = summarize_alkox(directory)
        print_alkox_summary(summary)
    else:
        summary = summarize_suzuki(directory)
        print_suzuki_summary(summary)

    if args.save:
        with open(args.save, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()
