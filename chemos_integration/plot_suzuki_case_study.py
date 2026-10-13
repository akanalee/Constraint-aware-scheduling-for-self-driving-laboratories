import os
import csv
import json
import glob
import re
import argparse
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties
from matplotlib import rcParams

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RESULTS_DIR = os.path.join(_REPO_ROOT, 'results', 'case_study_suzuki')
_FIGURE_DIR = os.path.join(_REPO_ROOT, 'outputs', 'figures')
CASE_STUDY = 'suzuki'


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


rcParams.update({
    'font.family':      'serif',
    'font.serif':       ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.weight':      'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'mathtext.fontset': 'stix',
})

FS_BO_LABEL = 30
FS_BO_SUBTITLE = FS_BO_LABEL + 4
FS_BO_TICK  = 30
FS_BO_LEG   = 18
FS_G_LEG    = 30
FS_G_XLABEL = 30
FS_G_COLTITLE = FS_G_XLABEL + 4
FS_G_XTICK  = 30
FS_G_YTICK  = 11
FS_PANEL    = 48

_ABORT_NAME   = 'ChemOS built-in dispatch\n($T_\\mathrm{max}$ violation termination)'
_NOABORT_NAME = 'ChemOS built-in dispatch\n(no $T_\\mathrm{max}$ violation termination)'

STYLE = {
    'ercfss':   {'line': '#2a7d3f', 'fill': '#2a7d3f', 'fill_alpha': 0.13,
                   'marker': 'o', 'label': 'ERC-FSS'},
    'chemos_abort': {'line': '#e07b00', 'fill': '#e07b00', 'fill_alpha': 0.13,
                   'marker': 's', 'label': _ABORT_NAME},
    'noabort':    {'line': '#8b1a1a', 'fill': '#8b1a1a', 'fill_alpha': 0.20,
                   'marker': '^', 'label': _NOABORT_NAME},
}
POLICY_ORDER = ['ercfss', 'chemos_abort', 'noabort']
ROMAN = ['i', 'ii', 'iii']
PANEL = ['(a)', '(b)', '(c)', '(d)']
LEVELS = [('noise_10pct', '10%'), ('noise_20pct', '20%'), ('noise_30pct', '30%')]

BO_YMAX = 90


def _load_clean_curve(path):
    pts, makespan = [], None
    if not path or not os.path.isfile(path):
        return None, None
    with open(path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            ms = row.get('makespan_min', row.get('makespan_marker_min', ''))
            if ms not in (None, ''):
                try:
                    makespan = float(ms)
                except ValueError:
                    pass
            vt, bs = row.get('time', ''), row.get('best_so_far', '')
            if vt in (None, '') or bs in (None, ''):
                continue
            try:
                pts.append((float(vt), float(bs)))
            except ValueError:
                continue
    pts.sort(key=lambda x: x[0])
    return pts, makespan


def _terminal_ends(ops):
    if not ops:
        return []
    mx = max((o['op_num'] for o in ops if isinstance(o.get('op_num'), int)), default=0)
    return sorted(o['end'] for o in ops if o.get('op_num') == mx)


def _finalize_curve(pts, ops):
    if not pts:
        return pts
    xs = [t for t, _ in pts]
    if max(xs) - min(xs) < 1e-6:
        ends = _terminal_ends(ops)
        if len(ends) == len(pts):
            pts = [(ends[i], pts[i][1]) for i in range(len(pts))]
    if pts and pts[0][0] > 0:
        pts = [(pts[0][0], 0.0)] + pts
    return pts


def _interp_y(pts, x):
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        if x0 <= x <= x1:
            if abs(x1 - x0) < 1e-12:
                return y0
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return pts[-1][1] if pts else 0.0


def _global_machine_pool(all_ops_lists):
    pooled = {}
    for ops in all_ops_lists:
        for op in ops or []:
            m = op['machine']
            if m not in pooled:
                pooled[m] = machine_to_step(m, CASE_STUDY)
    step_order = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S?']
    step_machines = defaultdict(list)
    for m, s in pooled.items():
        step_machines[s].append(m)
    machine_list, boundaries = [], []
    for step in step_order:
        ms = sorted(step_machines.get(step, []), key=_natural_sort_key)
        if ms:
            boundaries.append((len(machine_list), step))
            machine_list.extend(ms)
    machine_y = {m: i for i, m in enumerate(machine_list)}
    return machine_list, machine_y, boundaries


def _draw_bo(ax, curves, with_ylabel):
    for pol in ('noabort', 'chemos_abort', 'ercfss'):
        entry = curves.get(pol)
        if not entry or not entry[0]:
            continue
        pts, _ = entry
        st = STYLE[pol]
        xs, ys = zip(*pts)
        ax.fill_between(xs, 0, ys, color=st['fill'], alpha=st['fill_alpha'],
                        zorder=1)
    for z, pol in enumerate(POLICY_ORDER, start=3):
        entry = curves.get(pol)
        if not entry or not entry[0]:
            continue
        pts, ms = entry
        st = STYLE[pol]
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=st['line'], lw=3.0, zorder=z + 3,
                marker=st['marker'], ms=6, markevery=max(1, len(xs) // 14),
                markerfacecolor=st['line'], markeredgecolor=st['line'],
                label=st['label'])
        if ms is not None:
            yms = _interp_y(pts, ms)
            ax.plot([ms, ms], [0, yms], color=st['line'], lw=1.8,
                    linestyle='--', alpha=0.65, zorder=z)
            ax.plot(ms, 0, marker='^', ms=11, color=st['line'],
                    zorder=10, clip_on=False)

    ax.set_xlabel('Time (min)', fontsize=FS_BO_LABEL, fontweight='bold',
                  fontfamily='Times New Roman')
    if with_ylabel:
        ax.set_ylabel('Best yield (%)', fontsize=FS_BO_LABEL,
                      fontweight='bold', fontfamily='Times New Roman')
    ax.tick_params(axis='both', labelsize=FS_BO_TICK)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight('bold')
    ax.set_xlim(left=0)
    ax.set_ylim(0, BO_YMAX)
    for sp in ax.spines.values():
        sp.set_linewidth(2.2)
    ax.grid(alpha=0.25, linewidth=1.0)
    ax.legend(loc='upper left', framealpha=0.9,
              prop=FontProperties(family='Times New Roman', weight='bold',
                                  size=FS_BO_LEG))


def _draw_gantt_panel(ax, ops, machine_y, boundaries, n_machines, xlim_max):
    for op in ops or []:
        m = op['machine']
        if m not in machine_y:
            continue
        y = machine_y[m]
        ax.barh(y, op['end'] - op['start'], left=op['start'], height=0.85,
                color=OP_COLORS_DISTINCT.get(op['op_num'], '#aaaaaa'),
                edgecolor='white', linewidth=0.3)
    ax.set_xlim(0, xlim_max)
    ax.set_ylim(n_machines - 0.5, -0.5)
    for i in range(1, len(boundaries)):
        ax.axhline(boundaries[i][0] - 0.5, color='grey', lw=0.5,
                   linestyle='--', alpha=0.5)
    ax.set_xlabel('Time (min)', fontsize=FS_G_XLABEL, fontweight='bold')
    ax.tick_params(axis='x', labelsize=FS_G_XTICK)
    for lbl in ax.get_xticklabels():
        lbl.set_fontweight('bold')
    for sp in ax.spines.values():
        sp.set_linewidth(2.0)
    ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=1.5)


def build_figure(bo_by_level, ops_by_level, machine_pool, out_prefix):
    machine_list, machine_y, boundaries = machine_pool
    n_machines = len(machine_list)
    ylabels = [_format_machine_name(m) for m in machine_list]

    all_ops = [ops_by_level[lv][p]
               for lv in range(3) for p in POLICY_ORDER if ops_by_level[lv].get(p)]
    xlim_max = _adaptive_xlim({i: o for i, o in enumerate(all_ops)}) if all_ops else 300.0

    W = 26.0
    LEFT, RIGHT, WSP = 0.075, 0.985, 0.16

    def lh(fs):
        return fs / 72.0 * 1.3

    top_pad     = 0.55
    bo_axis     = 5.0
    bo_xlabel   = lh(FS_BO_LABEL)
    bo_sub_gap  = lh(FS_BO_LABEL)
    bo_subtitle = lh(FS_BO_SUBTITLE)
    gap_r1_leg  = -0.25 * lh(FS_BO_SUBTITLE)
    g_legend    = lh(FS_G_LEG) * 1.9
    g_axis      = max(n_machines * 0.12, 3.6)
    g_xlabel    = lh(FS_G_XLABEL)
    g_sub_gap   = lh(FS_G_XLABEL)
    g_coltitle  = lh(FS_G_COLTITLE)
    g_row_gap   = 0 * lh(FS_G_COLTITLE)
    bottom_pad  = 0.30

    H = (top_pad + bo_axis + bo_xlabel + bo_sub_gap + bo_subtitle + gap_r1_leg
         + g_legend + g_axis + g_xlabel + g_sub_gap + g_coltitle
         + g_row_gap + g_axis + g_xlabel + g_sub_gap + g_coltitle
         + g_row_gap + g_axis + g_xlabel + g_sub_gap + g_coltitle
         + bottom_pad)

    fig = plt.figure(figsize=(W, H))

    def band(top_in, h_in):
        return (H - (top_in + h_in)) / H, (H - top_in) / H

    def yfrac(top_in):
        return (H - top_in) / H

    w = (RIGHT - LEFT) / (3 + 2 * WSP)
    centers = [LEFT + j * w * (1 + WSP) + w / 2 for j in range(3)]
    tnr = dict(fontfamily='Times New Roman', fontweight='bold')

    bo_b, bo_t = band(top_pad, bo_axis)
    gs_bo = fig.add_gridspec(1, 3, left=LEFT, right=RIGHT, top=bo_t,
                             bottom=bo_b, wspace=WSP)
    sub_top_in = top_pad + bo_axis + bo_xlabel + bo_sub_gap
    for j in range(3):
        ax = fig.add_subplot(gs_bo[0, j])
        _draw_bo(ax, bo_by_level[j], with_ylabel=(j == 0))
        fig.text(centers[j], yfrac(sub_top_in),
                 f'({ROMAN[j]}) ',
                 ha='center', va='top', fontsize=FS_BO_SUBTITLE, **tnr)
    fig.text(0.012, yfrac(top_pad), PANEL[0], ha='left', va='bottom',
             fontsize=FS_PANEL, **tnr)

    cur = sub_top_in + bo_subtitle + gap_r1_leg
    for k in range(3):
        has_legend = (k == 0)
        legend_top_in = cur
        axis_top_in = cur + (g_legend if has_legend else 0.0)
        gax_b, gax_t = band(axis_top_in, g_axis)
        gs_g = fig.add_gridspec(1, 3, left=LEFT, right=RIGHT, top=gax_t,
                                bottom=gax_b, wspace=WSP)
        axes = [fig.add_subplot(gs_g[0, c]) for c in range(3)]
        for c, pol in enumerate(POLICY_ORDER):
            _draw_gantt_panel(axes[c], ops_by_level[k].get(pol), machine_y,
                              boundaries, n_machines, xlim_max)
            axes[c].set_yticks(range(n_machines))
            axes[c].set_yticklabels(ylabels if c == 0 else [],
                                    fontsize=FS_G_YTICK, fontweight='bold')

        col_sub_top_in = axis_top_in + g_axis + g_xlabel + g_sub_gap
        for c, pol in enumerate(POLICY_ORDER):
            fig.text(centers[c], yfrac(col_sub_top_in),
                     f'({ROMAN[c]}) ',
                     ha='center', va='top', fontsize=FS_G_COLTITLE, **tnr)

        fig.text(0.012, yfrac(legend_top_in), PANEL[k + 1],
                 ha='left', va='bottom', fontsize=FS_PANEL, **tnr)

        if has_legend:
            max_op = max((op['op_num'] for pol in POLICY_ORDER
                          for op in (ops_by_level[k].get(pol) or [])
                          if isinstance(op.get('op_num'), int) and op['op_num'] > 0),
                         default=9)
            handles = [mpatches.Patch(color=OP_COLORS_DISTINCT[n], label=_OP_LABELS[n])
                       for n in range(1, max_op + 1)]
            leg_y = yfrac(legend_top_in + g_legend * 0.85)
            leg = fig.legend(handles=handles,
                             bbox_to_anchor=(LEFT, leg_y, RIGHT - LEFT, 0.0),
                             loc='lower left', mode='expand', ncol=max_op,
                             borderaxespad=0.,
                             prop=FontProperties(family='Times New Roman',
                                                 weight='bold', size=FS_G_LEG),
                             framealpha=0.9)
            leg.get_frame().set_linewidth(2)

        cur = col_sub_top_in + g_coltitle + g_row_gap

    png = out_prefix + '.png'
    eps = out_prefix + '.eps'
    fig.savefig(png, dpi=1000, bbox_inches=None)
    try:
        fig.savefig(eps, format='eps', bbox_inches=None)
    except Exception as e:
        print(f"[eps skip] {e}")
    plt.close(fig)
    print(f"Saved: {png}")
    print(f"Saved: {eps}")


def main():
    ap = argparse.ArgumentParser(description='Plot the Suzuki-Miyaura case study')
    ap.add_argument('--results-dir', default=_RESULTS_DIR)
    ap.add_argument('--out', default=None, help='Output path without extension')
    args = ap.parse_args()

    nadir = args.results_dir

    POL = [('ercfss', 'ERCFSS', 'ERC-FSS'),
           ('chemos_abort', 'abort', 'ChemOS built-in dispatch with termination'),
           ('noabort', 'noabort', 'ChemOS built-in dispatch with no termination')]

    def _ops_from_json(path, human):
        if not os.path.isfile(path):
            return None
        try:
            with open(path, 'r', encoding='utf-8') as f:
                ep = json.load(f)
        except Exception:
            return None
        sec = ep.get(human)
        jd = sec.get('job_details') if isinstance(sec, dict) else None
        return build_ops_from_job_details(jd) if jd else None

    bo_by_level, ops_by_level = [], []
    for tag, _pct in LEVELS:
        bo, ops = {}, {}
        for dkey, ftag, human in POL:
            o = _ops_from_json(os.path.join(nadir, f'sched9_{human}_{tag}',
                                            'schedule.json'), human)
            if o:
                ops[dkey] = o
            curve = os.path.join(nadir, f'suzuki_truecurve_{ftag}_{tag}.csv')
            pts, ms = _load_clean_curve(curve)
            bo[dkey] = (_finalize_curve(pts, o), ms)
        bo_by_level.append(bo)
        ops_by_level.append(ops)

    pool = _global_machine_pool(
        [ops_by_level[lv].get(p) for lv in range(3) for p in POLICY_ORDER])

    out = args.out
    if out is None:
        os.makedirs(_FIGURE_DIR, exist_ok=True)
        out = os.path.join(_FIGURE_DIR, 'suzuki_case_study')
    build_figure(bo_by_level, ops_by_level, pool, out)


if __name__ == '__main__':
    main()
