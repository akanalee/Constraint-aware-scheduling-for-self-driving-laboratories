import os
import re
import sys
import glob
import json
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties
from matplotlib.transforms import offset_copy


_PROP_LEGEND = FontProperties(family='Times New Roman', weight='bold', size=30)

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.weight': 'bold',
    'mathtext.fontset': 'stix',
})

OP_COLORS_DISTINCT = {
    1: "#4e79a7", 2: "#f28e2b", 3: "#59a14f", 4: "#e15759",
    5: "#b07aa1", 6: "#edc948", 7: "#76b7b2", 8: "#ff9da7",
    9: "#9c755f",
}
_OP_LABELS = {1: 'Op1', 2: 'Op2', 3: 'Op3', 4: 'Op4', 5: 'Op5',
              6: 'Op6', 7: 'Op7', 8: 'Op8', 9: 'Op9'}

_MACHINE_RENAMES = {
    'pre reactor':      'Pre-reactor',
    'SPU':              'SPU',
    'cooling chamber':  'Cooling',
    'spectrometer':     'Spectroscopy',
    'ICP MS':           'MS',
    'robot arm':        'Robotic Arm',
}

MACHINE_STEP_PREFIXES = {
    'SPU_':              'S1',
    'robot_arm_':        'S2',
    'pre_reactor_':      'S3',
    'reactor_':          'S4',
    'cooling_chamber_':  'S5',
    'spectrometer_':     'S6',
    'ICP_MS_':           'S7',
}


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


def _format_machine_name(machine_name):
    parts = machine_name.split('_')
    identifier = parts[-1]
    base = ' '.join(parts[:-1])
    base = _MACHINE_RENAMES.get(base, base)
    if base:
        base = base[0].upper() + base[1:]
    num = _to_arabic_padded(identifier, pad_width=2)
    return f'{base} {num}'


def _machine_to_step(machine_name):
    for prefix, step in MACHINE_STEP_PREFIXES.items():
        if machine_name.startswith(prefix):
            return step
    return 'S?'


def _op_num_from_op_id(op_id: str) -> int:
    m = re.search(r'_op(\d+)$', op_id)
    if not m:
        return 1
    return int(m.group(1))


def load_ops_from_episode(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    job_details = data.get('ERC-FSS', {}).get('job_details', {})
    ops = []
    for job_id, details in job_details.items():
        for op_info in details.get('ops', []):
            if not op_info.get('is_scheduled'):
                continue
            if op_info.get('scheduled_start') is None:
                continue
            op_id = op_info['op_id']
            ops.append({
                'job_id':  job_id,
                'op_id':   op_id,
                'op_num':  _op_num_from_op_id(op_id),
                'machine': op_info['scheduled_machine'],
                'start':   float(op_info['scheduled_start']),
                'end':     float(op_info['scheduled_end']),
            })
    return ops


def _adaptive_xlim(ops):
    max_end = max((op['end'] for op in ops), default=0.0)
    if max_end <= 0:
        return 100.0
    target = max_end * 1.05
    step = 10
    return float(int(target / step + 1) * step)


def _draw_legend(fig, axes, max_op):
    pos_first = axes[0].get_position()
    pos_last  = axes[-1].get_position()
    legend_handles = [
        mpatches.Patch(color=OP_COLORS_DISTINCT[n], label=_OP_LABELS[n])
        for n in range(1, max_op + 1)
    ]
    leg = fig.legend(
        handles=legend_handles,
        bbox_to_anchor=(pos_first.x0, pos_last.y1 + 0.01,
                        pos_last.x1 - pos_first.x0, 0.0),
        loc='lower left',
        mode='expand',
        ncol=max_op,
        borderaxespad=0.,
        prop=_PROP_LEGEND,
        framealpha=0.9,
    )
    leg.get_frame().set_linewidth(2)


def plot_chemos_validation(json_path, out_dir=None):
    ops = load_ops_from_episode(json_path)
    if not ops:
        print(f"  No ops in {json_path}, skipping")
        return
    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(json_path))
    os.makedirs(out_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(json_path))[0]
    out_path_root = os.path.join(out_dir, base)

    jobs_map = defaultdict(list)
    for op in ops:
        jobs_map[op['job_id']].append(op)
    job_ids = sorted(jobs_map.keys(),
                     key=lambda j: int(re.search(r'\d+', j).group()))

    machine_step = {op['machine']: _machine_to_step(op['machine']) for op in ops}
    step_machines = defaultdict(set)
    for m, s in machine_step.items():
        step_machines[s].add(m)
    step_order = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7']
    machine_list = []
    group_boundaries = []
    for step in step_order + ['S?']:
        ms = sorted(step_machines.get(step, []), key=_natural_sort_key)
        if ms:
            group_boundaries.append((len(machine_list), step))
            machine_list.extend(ms)
    machine_y = {m: i for i, m in enumerate(machine_list)}

    xlim_max = _adaptive_xlim(ops)
    max_op = max((op['op_num'] for op in ops if op['op_num'] > 0), default=9)

    row_height = max(len(job_ids) * 0.25, len(machine_list) * 0.25, 8)
    fig = plt.figure(figsize=(26, row_height * 1.5 + 1.5))

    gs = fig.add_gridspec(
        1, 2,
        left=0.05, right=0.98,
        top=0.78, bottom=0.20,
        wspace=0.18,
    )
    ax_exp = fig.add_subplot(gs[0, 0])
    ax_mch = fig.add_subplot(gs[0, 1])

    bar_height = 0.85

    job_y = {jid: i for i, jid in enumerate(job_ids)}
    for op in ops:
        y = job_y[op['job_id']]
        duration = op['end'] - op['start']
        color = OP_COLORS_DISTINCT.get(op['op_num'], '#aaaaaa')
        ax_exp.barh(y, duration, left=op['start'], height=bar_height,
                    color=color, edgecolor='white', linewidth=0.3)

    display_job_labels = []
    for jid in job_ids:
        mm = re.search(r'(\d+)', jid)
        num = int(mm.group(1)) if mm else 0
        display_job_labels.append(f'$E_{{{num:01d}}}$')
    ax_exp.set_yticks(range(len(job_ids)))
    ax_exp.set_yticklabels(display_job_labels, fontsize=16, fontweight='bold')
    ax_exp.invert_yaxis()
    ax_exp.set_xlim(0, xlim_max)
    ax_exp.set_xlabel('Time (min)', fontsize=27, fontweight='bold')
    ax_exp.tick_params(axis='x', labelsize=25)
    for lbl in ax_exp.get_xticklabels():
        lbl.set_fontweight('bold')
    for spine in ax_exp.spines.values():
        spine.set_linewidth(2)
    ax_exp.grid(axis='x', alpha=0.3, linestyle='--', linewidth=2)

    for op in ops:
        if op['machine'] not in machine_y:
            continue
        y = machine_y[op['machine']]
        duration = op['end'] - op['start']
        color = OP_COLORS_DISTINCT.get(op['op_num'], '#aaaaaa')
        ax_mch.barh(y, duration, left=op['start'], height=bar_height,
                    color=color, edgecolor='white', linewidth=0.3)

    ax_mch.set_yticks(range(len(machine_list)))
    ax_mch.set_yticklabels([_format_machine_name(m) for m in machine_list],
                           fontsize=11, fontweight='bold')
    ax_mch.invert_yaxis()
    for i in range(1, len(group_boundaries)):
        sep_y = group_boundaries[i][0] - 0.5
        ax_mch.axhline(y=sep_y, color='grey', linewidth=0.5,
                       linestyle='--', alpha=0.5)
    ax_mch.set_xlim(0, xlim_max)
    ax_mch.set_xlabel('Time (min)', fontsize=27, fontweight='bold')
    ax_mch.tick_params(axis='x', labelsize=25)
    for lbl in ax_mch.get_xticklabels():
        lbl.set_fontweight('bold')
    for spine in ax_mch.spines.values():
        spine.set_linewidth(2)
    ax_mch.grid(axis='x', alpha=0.3, linestyle='--', linewidth=2)

    fig.canvas.draw()

    _draw_legend(fig, [ax_exp, ax_mch], max_op)

    captions = {
        ax_exp: '(a) Experiment-Level Gantt',
        ax_mch: '(b) Instrument-Level Gantt',
    }
    for ax, caption in captions.items():
        cap_tf = offset_copy(ax.transAxes, fig=fig, x=0, y=-70, units='points')
        ax.text(0.5, 0, caption,
                transform=cap_tf, ha='center', va='top',
                fontsize=30, fontweight='bold',
                fontfamily='Times New Roman')

    eps_path = out_path_root + '_chemos_validation_gantt.eps'
    png_path = out_path_root + '_chemos_validation_gantt.png'
    plt.savefig(eps_path, format='eps', bbox_inches='tight')
    plt.savefig(png_path, format='png', dpi=600, bbox_inches='tight')
    plt.close()
    print(f"  ChemOS-validation Gantt saved:")
    print(f"    {png_path}")
    print(f"    {eps_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Plot ChemOS-validation Gantt (experiment + instrument)'
    )
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    here = os.path.join(repo_root, 'results', 'chemos_validation')
    parser.add_argument(
        '--json', type=str, default=None,
        help='Specific result JSON; if omitted, plots every trial_*.json '
             'in results/chemos_validation.',
    )
    parser.add_argument(
        '--out_dir', type=str, default=os.path.join(repo_root, 'outputs', 'figures'),
        help='Output directory for PNG/EPS (default: outputs/figures)',
    )
    args = parser.parse_args()

    if args.json:
        targets = [args.json]
    else:
        targets = sorted(glob.glob(os.path.join(here, 'trial_*.json')))

    if not targets:
        print(f"No episode JSON files found in {here}")
        sys.exit(1)

    for p in targets:
        print(f"\nPlotting: {p}")
        plot_chemos_validation(p, out_dir=args.out_dir)


if __name__ == '__main__':
    main()
