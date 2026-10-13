import os
import csv
import json
import glob
import re
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties
from matplotlib.transforms import offset_copy
from matplotlib import rcParams

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RESULTS_DIR = os.path.join(_REPO_ROOT, 'results', 'case_study_alkox')
_FIGURE_DIR = os.path.join(_REPO_ROOT, 'outputs', 'figures')


def _natural_sort_key(s):
    return [int(c) if c.isdigit() else c.lower()
            for c in re.split(r'(\d+)', s)]


def _to_arabic_padded(val, pad_width=2):
    if isinstance(val, str) and len(val) == 1 and val.isalpha():
        n = ord(val.upper()) - ord('A') + 1
    else:
        try:
            n = int(val)
        except (ValueError, TypeError):
            return str(val)
    return str(n).zfill(pad_width)


_MACHINE_RENAMES = {
    'pre reactor':              'Pre-reactor',
    'SPU':                      'SPU',
    'cooling chamber':          'Cooling',
    'spectrometer':             'Spectroscopy',
    'ICP MS':                   'MS',
    'robot arm':                'Robotic Arm',
    'liquid handler buffer':    'LH Buffer',
    'liquid handler solution':  'LH Solution',
}


def _format_machine_name(machine_name):
    parts = machine_name.split('_')
    identifier = parts[-1]
    base = ' '.join(parts[:-1])
    base = _MACHINE_RENAMES.get(base, base)
    if base:
        base = base[0].upper() + base[1:]
    num = _to_arabic_padded(identifier, pad_width=2)
    return f'{base} {num}'


MACHINE_STEP_PREFIXES = {
    'SPU_':                      'S1',
    'robot_arm_':                'S2',
    'pre_reactor_':              'S3',
    'reactor_':                  'S4',
    'cooling_chamber_':          'S5',
    'spectrometer_':             'S6',
    'ICP_MS_':                   'S7',
    'liquid_handler_buffer_':    'S1',
    'liquid_handler_solution_':  'S2',
    'mixer_':                    'S3',
    'incubator_':                'S5',
    'liquid_handler_':           'S6',
    'S1_':                       'S1',
    'S2_':                       'S2',
    'S3_':                       'S3',
    'S4_':                       'S4',
    'S5_':                       'S5',
    'S6_':                       'S6',
    'S7_':                       'S7',
}

CASE_STUDY_STEP_PREFIXES = {
    'alkox': {
        'robot_arm_':       'S4',
        'cooling_chamber_': 'S6',
        'hplc_':            'S7',
        'incubator_':       'S5',
        'mixer_':           'S3',
        'liquid_handler_buffer_':   'S1',
        'liquid_handler_solution_': 'S2',
    },
    'suzuki': {
        'robot_arm_':       'S3',
        'cooling_chamber_': 'S5',
        'pre_reactor_':     'S2',
        'reactor_':         'S4',
        'liquid_handler_':  'S6',
        'spectrometer_':    'S7',
        'SPU_':             'S1',
    },
}


def machine_to_step(machine_name, case_study=None):
    if case_study and case_study in CASE_STUDY_STEP_PREFIXES:
        for prefix, step in CASE_STUDY_STEP_PREFIXES[case_study].items():
            if machine_name.startswith(prefix):
                return step
    for prefix, step in MACHINE_STEP_PREFIXES.items():
        if machine_name.startswith(prefix):
            return step
    return 'S?'


OP_COLORS_DISTINCT = {
    1: "#4e79a7", 2: "#f28e2b", 3: "#59a14f", 4: "#e15759",
    5: "#b07aa1", 6: "#edc948", 7: "#76b7b2", 8: "#ff9da7",
    9: "#9c755f",
}
_OP_LABELS = {1: 'Op1', 2: 'Op2', 3: 'Op3', 4: 'Op4', 5: 'Op5',
              6: 'Op6', 7: 'Op7', 8: 'Op8', 9: 'Op9'}


def _adaptive_xlim(ops_by_method):
    max_end = 0.0
    for ops in ops_by_method.values():
        for op in ops:
            if op['end'] > max_end:
                max_end = op['end']
    if max_end <= 0:
        return 100.0
    target = max_end * 1.05
    step = 10
    return float(int(target / step + 1) * step)


def load_episode_results(path):
    if os.path.isfile(path):
        with open(path, 'r', encoding='utf-8') as f:
            return [json.load(f)]
    elif os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, 'episode_*.json')))
        results = []
        for fp in files:
            with open(fp, 'r', encoding='utf-8') as f:
                results.append(json.load(f))
        return results
    else:
        print(f"Error: {path} not found")
        return []


def extract_op_num(op_id):
    m = re.search(r'_op(\d+)$', op_id)
    if not m:
        return 0
    return int(m.group(1)) + 1


def extract_job_num(job_id):
    m = re.search(r'(\d+)\s*$', job_id)
    if m:
        return int(m.group(1))
    m = re.search(r'(\d+)', job_id)
    return int(m.group(1)) if m else 0


def build_ops_from_job_details(job_details):
    ops = []
    for job_id, details in job_details.items():
        for op_info in details['ops']:
            if op_info['is_scheduled'] and op_info['scheduled_start'] is not None:
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
    ops = []
    for entry in schedule_sequence:
        op_id = entry['op_id']
        parts = op_id.rsplit('_op', 1)
        job_id = parts[0] if len(parts) == 2 else op_id
        ops.append({
            'job_id': job_id,
            'op_id': op_id,
            'op_num': extract_op_num(op_id),
            'machine': entry['machine'],
            'start': entry['start'],
            'end': entry['end'],
        })
    return ops


def build_chemos_abort_ops(chemos_data):
    jd = chemos_data.get('job_details', {})
    pairs = chemos_data.get('abort_replace_pairs', []) or []

    def _term_time(v):
        completed = [op for op in v.get('ops', [])
                     if op.get('is_completed')
                     and op.get('scheduled_end') is not None]
        return max(op['scheduled_end'] for op in completed) if completed else None

    term_to_rep = {}
    for p in pairs:
        t, r = p.get('terminated_job'), p.get('replacement_job')
        if t and r:
            term_to_rep[t] = r

    all_jobs = sorted(jd.keys(), key=extract_job_num)

    consumed = set()
    rows = []
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
                    op_num = extract_op_num(op['op_id'])
                    if op_num == 0:
                        op_num = op.get('op_index', 0) + 1
                    ops.append({
                        'job_id':  row_id,
                        'op_id':   op['op_id'],
                        'op_num':  op_num,
                        'machine': op['scheduled_machine'],
                        'start':   op['scheduled_start'],
                        'end':     op['scheduled_end'],
                    })

    return ops, terminations


CASE_STUDY        = 'alkox'

BO_YLABEL = 'Best conversion (%)'
BO_YMAX   = 35
B_CAPTION = '(a) Instrument-level Gantt charts'
A_CAPTION = '(b) Optimization progress under ERC-FSS and ChemOS built-in dispatch'

rcParams.update({
    'font.family':       'serif',
    'font.serif':        ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.weight':       'bold',
    'axes.labelweight':  'bold',
    'axes.titleweight':  'bold',
    'mathtext.fontset':  'stix',
})

_PROP_LEGEND_GANTT = FontProperties(family='Times New Roman',
                                    weight='bold', size=30)
_PROP_LEGEND_BO    = FontProperties(family='Times New Roman',
                                    weight='bold', size=22)

STYLE = {
    'ERCFSS':    {'line': '#2a7d3f', 'fill': '#2a7d3f', 'fill_alpha': 0.13,
               'marker': 'o', 'label': 'ERC-FSS'},
    'ChemOSDispatch': {'line': '#e07b00', 'fill': '#e07b00', 'fill_alpha': 0.13,
               'marker': 's', 'label': 'ChemOS Built-in Dispatch'},
}
HIT100_COLOR = '#d62728'

_METHOD_ORDER  = ['ercfss', 'chemos_abort']
_ARM_CAPTIONS  = {
    'ercfss':   '(i) ERC-FSS (Proposed)',
    'chemos_abort': '(ii) ChemOS Built-in Dispatch',
}


def _load_curve(path):
    pts, makespan = [], None
    with open(path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            ms_raw = row.get('makespan_marker_min', '')
            if ms_raw not in (None, ''):
                try:
                    makespan = float(ms_raw)
                except ValueError:
                    pass
            vt, bs = row.get('virtual_time_min', ''), row.get('best_so_far', '')
            if vt in (None, '') or bs in (None, ''):
                continue
            try:
                pts.append((float(vt), float(bs)))
            except ValueError:
                continue
    pts.sort(key=lambda x: x[0])
    if pts and pts[0][0] > 0:
        pts = [(0.0, 0.0)] + pts
    return pts, makespan


def _first_hit_100(pts):
    for t, y in pts:
        if y >= 100.0 - 1e-6:
            return t, y
    return None, None


def _interp_y_at_x(pts, x_target):
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        if x0 <= x_target <= x1:
            if abs(x1 - x0) < 1e-12:
                return y0
            return y0 + (y1 - y0) * (x_target - x0) / (x1 - x0)
    return pts[-1][1] if pts else 0.0


def _draw_bo(ax, ercfss_pts, ercfss_ms, dispatch_pts, dispatch_ms, ymax):
    entries = {'ERCFSS': (ercfss_pts, ercfss_ms), 'ChemOSDispatch': (dispatch_pts, dispatch_ms)}

    for mode in ('ERCFSS', 'ChemOSDispatch'):
        pts, ms = entries[mode]
        if not pts:
            continue
        st = STYLE[mode]
        xs, ys = zip(*pts)
        ax.fill_between(xs, 0, ys, color=st['fill'], alpha=st['fill_alpha'])
        ms_label = f" (makespan = {ms:.1f} min)" if ms is not None else ''
        ax.plot(xs, ys,
                color=st['line'], lw=3.4,
                marker=st['marker'], ms=8,
                markerfacecolor=st['line'], markeredgecolor=st['line'],
                label=f"{st['label']}{ms_label}")
        if ms is not None:
            y_at_ms = _interp_y_at_x(pts, ms)
            ax.plot([ms, ms], [0, y_at_ms],
                    color=st['line'], lw=2.0, linestyle='--', alpha=0.75)
            ax.plot(ms, 0, marker='^', ms=14,
                    color=st['line'], zorder=5, clip_on=False)

    label_offsets = []
    for mode in ('ERCFSS', 'ChemOSDispatch'):
        pts, _ = entries[mode]
        if not pts:
            continue
        t100, y100 = _first_hit_100(pts)
        if t100 is None:
            continue
        ax.plot([t100, t100], [0, y100],
                color=HIT100_COLOR, lw=2.2, linestyle='--', alpha=0.9, zorder=4)
        ax.plot(t100, 0, marker='o', ms=11,
                color=HIT100_COLOR, zorder=6, clip_on=False)
        x_text = t100
        for prev_x in label_offsets:
            if abs(x_text - prev_x) < 12:
                x_text = prev_x + 14
        label_offsets.append(x_text)
        ax.text(x_text + 0.8, y100 * 0.45,
                f"{STYLE[mode]['label']}\n100% @ {t100:.1f} min",
                color=HIT100_COLOR,
                fontsize=18, fontweight='bold',
                fontfamily='Times New Roman',
                va='center', ha='left',
                bbox=dict(boxstyle='round,pad=0.3',
                          fc='white', ec=HIT100_COLOR,
                          alpha=0.8, lw=1.4))

    ax.set_xlabel('Time (min)', fontsize=27, fontweight='bold',
                  fontfamily='Times New Roman')
    ax.set_ylabel(BO_YLABEL, fontsize=27, fontweight='bold',
                  fontfamily='Times New Roman')
    ax.tick_params(axis='x', labelsize=25)
    ax.tick_params(axis='y', labelsize=25)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight('bold')
    for spine in ax.spines.values():
        spine.set_linewidth(2.2)
    ax.set_xlim(left=0)
    ax.set_ylim(0, ymax)
    ax.grid(alpha=0.25, linewidth=1.0)
    ax.legend(loc='upper left', prop=_PROP_LEGEND_BO, framealpha=0.9)


def _draw_gantt(gantt_axes, ops_by_method, case_study,
                unified_machine_pool=True):
    methods = [m for m in _METHOD_ORDER if m in ops_by_method]
    if not methods:
        raise RuntimeError('No method data available for Gantt panel.')

    step_order = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7']

    pooled = {}
    if unified_machine_pool:
        for method in methods:
            for op in ops_by_method[method]:
                m = op['machine']
                if m not in pooled:
                    pooled[m] = machine_to_step(m, case_study)

    per_method = {}
    for method in methods:
        ops = ops_by_method[method]
        machine_step = pooled if unified_machine_pool else {}
        if not unified_machine_pool:
            for op in ops:
                m = op['machine']
                if m not in machine_step:
                    machine_step[m] = machine_to_step(m, case_study)
        step_machines = defaultdict(set)
        for m, s in machine_step.items():
            step_machines[s].add(m)
        machine_list, group_boundaries = [], []
        for step in step_order + ['S?']:
            ms_sorted = sorted(step_machines.get(step, []),
                               key=_natural_sort_key)
            if ms_sorted:
                group_boundaries.append((len(machine_list), step))
                machine_list.extend(ms_sorted)
        per_method[method] = {
            'ops': ops,
            'machine_list': machine_list,
            'machine_y': {m: i for i, m in enumerate(machine_list)},
            'group_boundaries': group_boundaries,
            'n_machines': len(machine_list),
        }

    xlim_max = _adaptive_xlim(ops_by_method)
    bar_height = 0.85
    ytick_fontsize = 11

    for ax, method in zip(gantt_axes, methods):
        d = per_method[method]
        for op in d['ops']:
            m = op['machine']
            if m not in d['machine_y']:
                continue
            y = d['machine_y'][m]
            duration = op['end'] - op['start']
            color = OP_COLORS_DISTINCT.get(op['op_num'], '#aaaaaa')
            ax.barh(y, duration, left=op['start'], height=bar_height,
                    color=color, edgecolor='white', linewidth=0.3)

        labels = [_format_machine_name(m) for m in d['machine_list']]
        ax.set_yticks(range(d['n_machines']))
        ax.set_yticklabels(labels, fontsize=ytick_fontsize, fontweight='bold')
        ax.invert_yaxis()

        for i in range(1, len(d['group_boundaries'])):
            sep_y = d['group_boundaries'][i][0] - 0.5
            ax.axhline(y=sep_y, color='grey', linewidth=0.5,
                       linestyle='--', alpha=0.5)

        ax.set_xlim(0, xlim_max)
        ax.set_xlabel('Time (min)', fontsize=27, fontweight='bold')
        ax.tick_params(axis='x', labelsize=25)
        for lbl in ax.get_xticklabels():
            lbl.set_fontweight('bold')
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=2)

    return methods, per_method


def _build_ops_by_method(episode_data):
    ops_by_method = {}
    if 'ercfss' in episode_data and 'error' not in episode_data['ercfss']:
        m = episode_data['ercfss']
        ops = []
        if m.get('job_details'):
            ops = build_ops_from_job_details(m['job_details'])
        elif m.get('schedule_sequence'):
            ops = build_ops_from_schedule_sequence(m['schedule_sequence'])
        if ops:
            ops_by_method['ercfss'] = ops
    if 'chemos_abort' in episode_data and 'error' not in episode_data['chemos_abort']:
        ops, _terms = build_chemos_abort_ops(episode_data['chemos_abort'])
        if ops:
            ops_by_method['chemos_abort'] = ops
    return ops_by_method


def build_integrated_figure(results_dir=_RESULTS_DIR, out_prefix=None):
    episode_json = sorted(glob.glob(os.path.join(results_dir, 'episode_*.json')))[0]
    curve_ercfss_csv = os.path.join(results_dir, 'alkox_conversion_curve_e0_ERCFSS.csv')
    curve_dispatch_csv = os.path.join(results_dir, 'alkox_conversion_curve_e0_ChemOSDispatch.csv')
    if out_prefix is None:
        os.makedirs(_FIGURE_DIR, exist_ok=True)
        out_prefix = os.path.join(_FIGURE_DIR, 'alkox_case_study')

    eps = load_episode_results(episode_json)
    if not eps:
        raise FileNotFoundError(f'Could not load {episode_json}')
    ops_by_method = _build_ops_by_method(eps[0])
    if not ops_by_method:
        raise RuntimeError('No ops parsed from episode JSON.')

    ercfss_pts, ercfss_ms = _load_curve(curve_ercfss_csv)
    dispatch_pts, dispatch_ms = _load_curve(curve_dispatch_csv)

    pooled_machines = {op['machine'] for ops in ops_by_method.values() for op in ops}
    max_machines = len(pooled_machines)
    row_height   = max(max_machines * 0.25, 8)
    gantt_h_in   = row_height * 1.5 + 1.5
    gantt_axis_h_in = (0.78 - 0.20) * gantt_h_in
    bo_axis_h_in    = gantt_axis_h_in

    legend_room_in  = 1.2
    gap_in          = 0.55
    a_caption_in    = 0.9
    top_pad_in      = 0.3
    mid_in = legend_room_in + gap_in + a_caption_in

    total_h_in = (a_caption_in + bo_axis_h_in + mid_in
                  + gantt_axis_h_in + legend_room_in + top_pad_in)

    fig = plt.figure(figsize=(26, total_h_in))

    g_left, g_right, g_wspace = 0.09, 0.98, 0.18
    g_bottom = (a_caption_in + bo_axis_h_in + mid_in) / total_h_in
    g_top    = g_bottom + gantt_axis_h_in / total_h_in
    gs_b = fig.add_gridspec(1, 2,
                            left=g_left, right=g_right,
                            top=g_top, bottom=g_bottom,
                            wspace=g_wspace)
    gantt_axes = [fig.add_subplot(gs_b[0, i]) for i in range(2)]

    methods, _ = _draw_gantt(gantt_axes, ops_by_method,
                             case_study=CASE_STUDY,
                             unified_machine_pool=True)

    fig.canvas.draw()
    pos_first = gantt_axes[0].get_position()
    pos_last  = gantt_axes[-1].get_position()

    _max_op = max(
        (op['op_num'] for ops in ops_by_method.values() for op in ops
         if isinstance(op.get('op_num'), int) and op['op_num'] > 0),
        default=9,
    )
    legend_handles = [
        mpatches.Patch(color=OP_COLORS_DISTINCT[n], label=_OP_LABELS[n])
        for n in range(1, _max_op + 1)
    ]
    leg = fig.legend(
        handles=legend_handles,
        bbox_to_anchor=(pos_first.x0, pos_last.y1 + 0.01,
                        pos_last.x1 - pos_first.x0, 0.0),
        loc='lower left',
        mode='expand',
        ncol=_max_op,
        borderaxespad=0.,
        prop=_PROP_LEGEND_GANTT,
        framealpha=0.9,
    )
    leg.get_frame().set_linewidth(2)

    for ax, method in zip(gantt_axes, methods):
        cap_tf = offset_copy(ax.transAxes, fig=fig, x=0, y=-65, units='points')
        ax.text(0.5, 0, _ARM_CAPTIONS.get(method, f'({method})'),
                transform=cap_tf, ha='center', va='top',
                fontsize=30, fontweight='bold', fontfamily='Times New Roman')

    bx_center = 0.5 * (pos_first.x0 + pos_last.x1)
    by_b      = pos_first.y0 - 120.0 / 72.0 / total_h_in
    fig.text(bx_center, by_b, B_CAPTION,
             ha='center', va='top',
             fontsize=30, fontweight='bold', fontfamily='Times New Roman')

    bo_left   = pos_first.x0
    bo_right  = pos_last.x1
    bo_bottom = a_caption_in / total_h_in
    bo_top    = bo_bottom + bo_axis_h_in / total_h_in
    ax_bo = fig.add_axes([bo_left, bo_bottom,
                          bo_right - bo_left, bo_top - bo_bottom])

    _draw_bo(ax_bo, ercfss_pts, ercfss_ms, dispatch_pts, dispatch_ms, BO_YMAX)

    cap_tf_a = offset_copy(ax_bo.transAxes, fig=fig, x=0, y=-65, units='points')
    ax_bo.text(0.5, 0, A_CAPTION,
               transform=cap_tf_a, ha='center', va='top',
               fontsize=30, fontweight='bold', fontfamily='Times New Roman')

    eps_path = out_prefix + '.eps'
    png_path = out_prefix + '.png'
    plt.savefig(eps_path, format='eps', bbox_inches='tight')
    plt.savefig(png_path, format='png', dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {eps_path}')
    print(f'Saved: {png_path}')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Plot the biocatalytic benzyl alcohol oxidation case study')
    ap.add_argument('--results-dir', default=_RESULTS_DIR)
    ap.add_argument('--out', default=None, help='Output path without extension')
    args = ap.parse_args()
    build_integrated_figure(args.results_dir, args.out)
