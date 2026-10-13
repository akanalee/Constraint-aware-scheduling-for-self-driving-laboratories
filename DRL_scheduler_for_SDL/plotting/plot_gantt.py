import os
import re
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties
from matplotlib.transforms import offset_copy


_PROP_AXIS   = FontProperties(family='Times New Roman', weight='bold', size=30)
_PROP_LEGEND = FontProperties(family='Times New Roman', weight='bold', size=30)

import re as _re

def _natural_sort_key(s):
    return [int(c) if c.isdigit() else c.lower()
            for c in _re.split(r'(\d+)', s)]

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
    'pre reactor':      'Pre-reactor',
    'SPU':              'SPU',
    'cooling chamber':  'Cooling',
    'spectrometer':     'Spectroscopy',
    'ICP MS':           'MS',
    'robot arm':        'Robotic Arm',
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
    'SPU_':              'S1',
    'robot_arm_':        'S2',
    'pre_reactor_':      'S3',
    'reactor_':          'S4',
    'cooling_chamber_':  'S5',
    'spectrometer_':     'S6',
    'ICP_MS_':           'S7',
}

def machine_to_step(machine_name):
    for prefix, step in MACHINE_STEP_PREFIXES.items():
        if machine_name.startswith(prefix):
            return step
    return 'S?'

OP_COLORS_DISTINCT = {
    1: "#4e79a7",
    2: "#f28e2b",
    3: "#59a14f",
    4: "#e15759",
    5: "#b07aa1",
    6: "#edc948",
    7: "#76b7b2",
    8: "#ff9da7",
    9: "#9c755f",
}

_OP_LABELS = {1:'Op1', 2:'Op2', 3:'Op3', 4:'Op4', 5:'Op5',
              6:'Op6', 7:'Op7', 8:'Op8', 9:'Op9'}

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.weight': 'bold',
    'mathtext.fontset': 'stix',
})

_METHOD_ORDER_6WAY = [
    'MILP', 'ERC-FSS', 'Conventional DRL',
    'GA', 'SPT', 'ChemOS Built-in Dispatch',
]

_ROW1_METHODS = ['MILP', 'ERC-FSS', 'Conventional DRL']
_ROW2_METHODS = ['GA', 'SPT', 'ChemOS Built-in Dispatch']

_METHOD_CAPTIONS_6WAY = {
    'MILP': '(a) MILP (Gurobi$^{\mathregular{TM}}$)',
    'ERC-FSS':     '(b) ERC-FSS (Proposed)',
    'Conventional DRL': '(c) Conventional DRL',
    'GA':        '(d) GA',
    'SPT':          '(e) SPT',
    'ChemOS Built-in Dispatch':   '(f) ChemOS Built-in Dispatch',
}


def plot_experiment_gantt(ops_by_method, out_path, terminations_by_method=None):
    row1_methods = [m for m in _ROW1_METHODS if m in ops_by_method]
    row2_methods = [m for m in _ROW2_METHODS if m in ops_by_method]

    if not row1_methods and not row2_methods:
        print("  Six-way job Gantt: no method data, skipping.")
        return

    per_method = {}
    for method in row1_methods + row2_methods:
        ops = ops_by_method[method]
        jobs = defaultdict(list)
        for op in ops:
            jobs[op['job_id']].append(op)
        job_ids = sorted(jobs.keys(), key=lambda j: int(re.search(r'\d+', j).group()))
        per_method[method] = {'ops': ops, 'job_ids': job_ids}

    all_present = row1_methods + row2_methods
    max_jobs = max(len(per_method[m]['job_ids']) for m in all_present)

    bar_height = 0.85
    ytick_fontsize = 16
    row_height = max(max_jobs * 0.25, 6)

    fig = plt.figure(figsize=(26, row_height * 2.25 + 1.5))

    gs_top = fig.add_gridspec(
        1, 3,
        left=0.05, right=0.98,
        top=0.88, bottom=0.54,
        wspace=0.18
    )
    gs_bot = fig.add_gridspec(
        1, 3,
        left=0.05, right=0.98,
        top=0.46, bottom=0.10,
        wspace=0.18
    )

    n_row1 = len(row1_methods)
    n_row2 = len(row2_methods)
    axes_row1 = [fig.add_subplot(gs_top[0, i]) for i in range(n_row1)]
    axes_row2 = [fig.add_subplot(gs_bot[0, i]) for i in range(n_row2)]
    all_axes   = axes_row1 + axes_row2
    all_methods_ordered = row1_methods + row2_methods

    for ax, method in zip(all_axes, all_methods_ordered):
        d = per_method[method]
        ops = d['ops']
        job_ids = d['job_ids']
        n_jobs = len(job_ids)
        job_y = {jid: i for i, jid in enumerate(job_ids)}

        for op in ops:
            y = job_y[op['job_id']]
            duration = op['end'] - op['start']
            color = OP_COLORS_DISTINCT.get(op['op_num'], '#aaaaaa')
            ax.barh(y, duration, left=op['start'], height=bar_height,
                    color=color, edgecolor='white', linewidth=0.3)

        if terminations_by_method and method in terminations_by_method:
            for jid, times in terminations_by_method[method].items():
                if jid in job_y:
                    y = job_y[jid]
                    for t in times:
                        ax.plot(t, y, 'x', color='red', markersize=16,
                                markeredgewidth=3, zorder=5)

        display_labels = []
        for jid in job_ids:
            mm = re.search(r'(\d+)', jid)
            num = int(mm.group(1)) if mm else 0
            display_labels.append(f'$E_{{{num:01d}}}$')

        ax.set_yticks(range(n_jobs))
        ax.set_yticklabels(display_labels, fontsize=ytick_fontsize, fontweight='bold')
        ax.invert_yaxis()
        ax.set_xlim(0, 350)
        ax.set_xlabel('Time (min)', fontsize=27, fontweight='bold')
        ax.tick_params(axis='x', labelsize=25)
        for lbl in ax.get_xticklabels():
            lbl.set_fontweight('bold')
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=2)

    fig.canvas.draw()

    pos_first = axes_row1[0].get_position()
    pos_last  = axes_row1[-1].get_position()
    legend_handles = [
        mpatches.Patch(color=OP_COLORS_DISTINCT[n], label=_OP_LABELS[n])
        for n in range(1, 10)
    ]
    leg = fig.legend(
        handles=legend_handles,
        bbox_to_anchor=(pos_first.x0, pos_last.y1 + 0.01,
                        pos_last.x1 - pos_first.x0, 0.0),
        loc='lower left',
        mode='expand',
        ncol=9,
        borderaxespad=0.,
        prop=_PROP_LEGEND,
        framealpha=0.9,
    )
    leg.get_frame().set_linewidth(2)

    for ax, method in zip(all_axes, all_methods_ordered):
        cap_tf = offset_copy(ax.transAxes, fig=fig, x=0, y=-70, units='points')
        caption = _METHOD_CAPTIONS_6WAY.get(method, f'({method})')
        ax.text(0.5, 0, caption,
                transform=cap_tf, ha='center', va='top',
                fontsize=30, fontweight='bold', fontfamily='Times New Roman')

    base_path = os.path.splitext(out_path)[0]
    eps_path = base_path + '_gantt_6way_experiment.eps'
    png_path = base_path + '_gantt_6way_experiment.png'
    plt.savefig(eps_path, format='eps', bbox_inches='tight')
    plt.savefig(png_path, format='png', dpi=600, bbox_inches='tight')
    plt.close()
    print(f"  Six-way job Gantt saved: {eps_path} + {png_path}")


def plot_instrument_gantt(ops_by_method, out_path):
    row1_methods = [m for m in _ROW1_METHODS if m in ops_by_method]
    row2_methods = [m for m in _ROW2_METHODS if m in ops_by_method]

    if not row1_methods and not row2_methods:
        print("  Six-way machine Gantt: no method data, skipping.")
        return

    step_order = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7']

    per_method = {}
    max_machines = 0
    for method in row1_methods + row2_methods:
        ops = ops_by_method[method]
        machine_step = {}
        for op in ops:
            m = op['machine']
            if m not in machine_step:
                machine_step[m] = machine_to_step(m)
        step_machines = defaultdict(set)
        for m, s in machine_step.items():
            step_machines[s].add(m)
        machine_list = []
        group_boundaries = []
        for step in step_order:
            machines_sorted = sorted(step_machines.get(step, []), key=_natural_sort_key)
            if machines_sorted:
                group_boundaries.append((len(machine_list), step))
                machine_list.extend(machines_sorted)
        n_machines = len(machine_list)
        if n_machines > max_machines:
            max_machines = n_machines
        per_method[method] = {
            'ops': ops,
            'machine_list': machine_list,
            'machine_y': {m: i for i, m in enumerate(machine_list)},
            'group_boundaries': group_boundaries,
            'n_machines': n_machines,
        }

    bar_height = 0.85
    ytick_fontsize = 11
    row_height = max(max_machines * 0.25, 8)

    fig = plt.figure(figsize=(26, row_height * 2.25 + 1.5))

    gs_top = fig.add_gridspec(
        1, 3,
        left=0.09, right=0.98,
        top=0.88, bottom=0.54,
        wspace=0.18
    )
    gs_bot = fig.add_gridspec(
        1, 3,
        left=0.09, right=0.98,
        top=0.46, bottom=0.10,
        wspace=0.18
    )

    n_row1 = len(row1_methods)
    n_row2 = len(row2_methods)
    axes_row1 = [fig.add_subplot(gs_top[0, i]) for i in range(n_row1)]
    axes_row2 = [fig.add_subplot(gs_bot[0, i]) for i in range(n_row2)]
    all_axes          = axes_row1 + axes_row2
    all_methods_ordered = row1_methods + row2_methods

    for ax, method in zip(all_axes, all_methods_ordered):
        d = per_method[method]
        ops = d['ops']
        machine_list = d['machine_list']
        machine_y = d['machine_y']
        group_boundaries = d['group_boundaries']
        n_machines = d['n_machines']

        for op in ops:
            m = op['machine']
            if m not in machine_y:
                continue
            y = machine_y[m]
            duration = op['end'] - op['start']
            color = OP_COLORS_DISTINCT.get(op['op_num'], '#aaaaaa')
            ax.barh(y, duration, left=op['start'], height=bar_height,
                    color=color, edgecolor='white', linewidth=0.3)

        display_machine_labels = [_format_machine_name(m) for m in machine_list]
        ax.set_yticks(range(n_machines))
        ax.set_yticklabels(display_machine_labels, fontsize=ytick_fontsize, fontweight='bold')
        ax.invert_yaxis()

        for i in range(1, len(group_boundaries)):
            sep_y = group_boundaries[i][0] - 0.5
            ax.axhline(y=sep_y, color='grey', linewidth=0.5, linestyle='--', alpha=0.5)

        ax.set_xlim(0, 350)
        ax.set_xlabel('Time (min)', fontsize=27, fontweight='bold')
        ax.tick_params(axis='x', labelsize=25)
        for lbl in ax.get_xticklabels():
            lbl.set_fontweight('bold')
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=2)

    fig.canvas.draw()

    pos_first = axes_row1[0].get_position()
    pos_last  = axes_row1[-1].get_position()
    legend_handles = [
        mpatches.Patch(color=OP_COLORS_DISTINCT[n], label=_OP_LABELS[n])
        for n in range(1, 10)
    ]
    leg = fig.legend(
        handles=legend_handles,
        bbox_to_anchor=(pos_first.x0, pos_last.y1 + 0.01,
                        pos_last.x1 - pos_first.x0, 0.0),
        loc='lower left',
        mode='expand',
        ncol=9,
        borderaxespad=0.,
        prop=_PROP_LEGEND,
        framealpha=0.9,
    )
    leg.get_frame().set_linewidth(2)

    for ax, method in zip(all_axes, all_methods_ordered):
        cap_tf = offset_copy(ax.transAxes, fig=fig, x=0, y=-65, units='points')
        caption = _METHOD_CAPTIONS_6WAY.get(method, f'({method})')
        ax.text(0.5, 0, caption,
                transform=cap_tf, ha='center', va='top',
                fontsize=30, fontweight='bold', fontfamily='Times New Roman')

    base_path = os.path.splitext(out_path)[0]
    eps_path = base_path + '_gantt_6way_instrument.eps'
    png_path = base_path + '_gantt_6way_instrument.png'
    plt.savefig(eps_path, format='eps', bbox_inches='tight')
    plt.savefig(png_path, format='png', dpi=600, bbox_inches='tight')
    plt.close()
    print(f"  Six-way machine Gantt saved: {eps_path} + {png_path}")
