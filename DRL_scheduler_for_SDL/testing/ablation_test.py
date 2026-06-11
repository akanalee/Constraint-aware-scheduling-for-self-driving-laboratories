"""ERC component ablation test: erc_full vs erc_no_forward vs erc_no_backward vs erc_no_urgency.

Reuses the e2e_comparison_v3 pipeline verbatim; only the comparison arms are
swapped for the four ERC ablation groups. Every arm runs the identical FSS
inference + ERC progressive-release loop, differing only in (adapter variant,
trained checkpoint). AblationComparison subclasses ThreeWayComparison and
overrides only the arms.
"""
import os
import sys
import time
import copy
import torch
import numpy as np
import yaml
import json
from itertools import combinations
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'training'))

from ppo import PPO
from config import get_default_config
from env_adapter import SDLEnvAdapter
from ablation_components import (
    AblationAdapterNoForward,
    AblationAdapterNoBackward,
    AblationAdapterNoUrgency,
)
from envs.sdl_env import SDLEnv
from utils.mb_agg import g_pool_cal
from utils.agent_utils import greedy_select_action2

# ============================================================================
# DRL Baseline imports (independent from main framework)
# Uses importlib to load directly from file path, bypassing sys.modules cache.
# This ensures baseline files are used even if main framework modules with the
# same name (config, env_adapter, ppo) are already imported.
# To swap baseline: replace env_adapter.py, config.py, model files in DRL_baseline/
# ============================================================================
import importlib.util as _ilu
import sys as _sys
import types as _types

def _load_module_from_path(module_name, file_path):
    """Load a module from an absolute file path, ignoring sys.modules cache."""
    spec = _ilu.spec_from_file_location(module_name, file_path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_baseline_root     = os.path.join(project_root, 'DRL_baseline')
_baseline_training = os.path.join(_baseline_root, 'training')
_baseline_models_dir = os.path.join(_baseline_root, 'models')

# Step 1: load graphcnn and mlp under unique baseline names
_bl_graphcnn_mod = _load_module_from_path(
    'bl_graphcnn', os.path.join(_baseline_models_dir, 'graphcnn.py'))
_bl_mlp_mod = _load_module_from_path(
    'bl_mlp', os.path.join(_baseline_models_dir, 'mlp.py'))

# Step 2: inject a fake 'models' package into sys.modules pointing at baseline files
# This must happen BEFORE ppo_actor.py is exec'd, because ppo_actor.py contains
#   from models.graphcnn import GraphCNN
#   from models.mlp import MLPActor, MLPCritic
# at module level, which is resolved during exec_module.
_saved_models_pkg    = _sys.modules.get('models', None)
_saved_models_gcnn   = _sys.modules.get('models.graphcnn', None)
_saved_models_mlp    = _sys.modules.get('models.mlp', None)
_saved_models_actor  = _sys.modules.get('models.ppo_actor', None)

_fake_models_pkg = _types.ModuleType('models')
_fake_models_pkg.__path__ = [_baseline_models_dir]
_fake_models_pkg.__package__ = 'models'
_sys.modules['models']          = _fake_models_pkg
_sys.modules['models.graphcnn'] = _bl_graphcnn_mod
_sys.modules['models.mlp']      = _bl_mlp_mod

# Step 3: now load ppo_actor — its from-imports will resolve correctly
_bl_actor_mod = _load_module_from_path(
    'bl_ppo_actor', os.path.join(_baseline_models_dir, 'ppo_actor.py'))
_sys.modules['models.ppo_actor'] = _bl_actor_mod

# Step 4: load config, env_adapter, ppo from DRL_baseline/training/
# utils.* (memory, mb_agg, agent_utils) are shared with main framework — no conflict
_bl_config_mod  = _load_module_from_path(
    'bl_config',  os.path.join(_baseline_training, 'config.py'))
_bl_adapter_mod = _load_module_from_path(
    'bl_env_adapter', os.path.join(_baseline_training, 'env_adapter.py'))
_bl_ppo_mod     = _load_module_from_path(
    'bl_ppo', os.path.join(_baseline_training, 'ppo.py'))

# Step 5: restore original sys.modules entries so main framework is unaffected
def _restore(key, saved):
    if saved is None:
        _sys.modules.pop(key, None)
    else:
        _sys.modules[key] = saved

_restore('models',          _saved_models_pkg)
_restore('models.graphcnn', _saved_models_gcnn)
_restore('models.mlp',      _saved_models_mlp)
_restore('models.ppo_actor',_saved_models_actor)

get_baseline_config   = _bl_config_mod.get_default_config
BaselineSDLEnvAdapter = _bl_adapter_mod.SDLEnvAdapter
BaselinePPO           = _bl_ppo_mod.PPO


# ============================================================================
# Tmax
# ============================================================================

def _verify_tmax_feasibility(env):
    """
     Tmax
     env.graph  ops  gap

    Returns:
        True False
    """
    if not hasattr(env, 'tmax_constraints') or not env.tmax_constraints:
        return True

    all_ops = list(env.graph.operations.values())
    for fs, ts, mx in env.tmax_constraints:
        for op in all_ops:
            if op.op_type == ts and (op.is_scheduled or op.is_completed):
                # closest preceding from_step op by op_index — consistent with
                # _check_tmax_feasibility and _check_tmax_violations
                candidates = [o for o in all_ops
                              if o.job_id == op.job_id
                              and o.op_type == fs
                              and o.op_index < op.op_index]
                if candidates:
                    pred = max(candidates, key=lambda o: o.op_index)
                    if pred.is_scheduled or pred.is_completed:
                        gap = op.scheduled_start - pred.scheduled_end
                        if gap > mx + 1e-4:
                            return False
    return True


# ============================================================================
# Part 1:  ()
# ============================================================================

def global_reschedule(env, adapter, ppo, method='ERC-FSS', device='cpu',
                      ga_pop_size=20, ga_patience=20):
    """
     ()

     scheduledops

    Returns:
        dict: {
            'success': bool,
            'makespan': float,
            'schedule': list,  # [(start, op_id, machine, end)]
            'rollback_count': int,  # op
            'scheduled_count': int,  # op
            'time_breakdown': dict  #
        }
    """
    time_start = time.time()

    # ========================================
    # Step 1: ops
    # ========================================
    rollback_start = time.time()
    machines_to_update = set()
    rollback_count = 0

    for op in env.graph.operations.values():
        if not op.is_completed and op.is_scheduled:
            if op.scheduled_start > env.current_time + 1e-4:
                rollback_count += 1
                if op.scheduled_machine:
                    machines_to_update.add(op.scheduled_machine)

                # op
                op.is_scheduled = False
                op.scheduled_machine = None
                op.scheduled_start = 0.0
                op.scheduled_end = 0.0

                if op.id in env.graph.scheduled_ops:
                    env.graph.scheduled_ops.remove(op.id)

    #
    for m_id in machines_to_update:
        machine = env.graph.machines[m_id]
        running_ops = [op for op in env.graph.operations.values()
                       if (op.scheduled_machine == m_id and
                           op.is_scheduled and not op.is_completed)]
        if running_ops:
            machine.available_time = max(env.current_time,
                                         max(op.scheduled_end for op in running_ops))
        else:
            machine.available_time = env.current_time

    env.graph._update_eligible_ops()
    rollback_time = time.time() - rollback_start

    # ========================================
    # Step 2:
    # ========================================
    schedule_start = time.time()

    if method == 'ERC-FSS':
        result = _schedule_with_fss(env, adapter, ppo, device)
    elif method == 'Conventional DRL':
        result = _schedule_with_drl_baseline(env, adapter, ppo, device)
    elif method == 'ga':
        result = _schedule_with_ga(env, pop_size=ga_pop_size, patience=ga_patience)
    elif method == 'milp':
        result = _schedule_with_milp(env)
    elif method == 'MILP':
        result = _schedule_with_milp(env, time_limit=60000, mip_gap_abs=None)
    elif method == 'SPT':
        result = _schedule_with_spt(env)
    else:
        raise ValueError(f"Unknown method: {method}")

    schedule_time = time.time() - schedule_start
    total_time = time.time() - time_start

    #
    result['rollback_count'] = rollback_count
    result['scheduled_count'] = len(result.get('schedule', []))
    result['time_breakdown'] = {
        'rollback': rollback_time,
        'scheduling': schedule_time,
        'total': total_time
    }

    return result


def _schedule_with_fss(env, adapter, ppo, device):
    """FSS"""
    #
    state = adapter.get_state_for_fss()
    if state is None:
        return {
            'success': False,
            'makespan': float('inf'),
            'schedule': []
        }

    # Opmask
    for i, op_id in enumerate(state['eligible_ops']):
        op_idx_in_list = state['op_list'].index(op_id)
        op_mask = state['mask_mch'][0, op_idx_in_list, :]

        op = env.graph.operations[op_id]
        compatible_indices = [adapter.machine_id_to_idx[m_id]
                              for m_id in op.compatible_machines
                              if m_id in adapter.machine_id_to_idx]

        if compatible_indices:
            all_masked = all(op_mask[idx].item() for idx in compatible_indices)

            if all_masked:
                return {
                    'success': False,
                    'makespan': float('inf'),
                    'schedule': []
                }

    #
    schedule = []
    mch_a = None

    while len(env.graph.eligible_ops) > 0:
        state = adapter.get_state_for_fss()
        if state is None:
            break

        env_mask_mch = state['mask_mch'].to(device)
        env_dur = state['dur'].to(device)

        n_ops = state['fea'].size(0)
        g_pool_step = g_pool_cal(
            ppo.config.graph_pool_type,
            torch.Size([1, n_ops, n_ops]),
            n_ops,
            device
        )

        env_adj = state['adj'].to(device)
        env_fea = state['fea'].to(device)
        env_candidate = state['candidate'].to(device)
        env_mask = state['mask'].to(device)
        env_mch_time = state['mch_time'].to(device)

        with torch.no_grad():
            action, a_idx, _, action_node, action_feature, mask_mch_action, hx, _ = \
                ppo.policy_old_job(
                    x=env_fea,
                    graph_pool=g_pool_step,
                    padded_nei=None,
                    adj=env_adj,
                    candidate=env_candidate,
                    mask=env_mask,
                    mask_mch=env_mask_mch,
                    dur=env_dur,
                    a_index=0,
                    old_action=0,
                    old_policy=True,
                    greedy=True
                )

            pi_mch, _ = ppo.policy_old_mch(
                action_node=action_node,
                action_feature=action_feature,
                mask_mch_action=mask_mch_action,
                mch_time=env_mch_time,
                mch_a=mch_a,
                last_hh=None,
                policy=False,
                et_normalize_coef=ppo.config.et_normalize_coef
            )

            mch_a = greedy_select_action2(pi_mch)

        op_action = a_idx if isinstance(a_idx, int) else a_idx.cpu().numpy()
        op_id, machine_id = adapter.action_to_op_and_machine(
            op_action, mch_a.cpu().numpy(), state
        )

        op = env.graph.operations[op_id]
        processing_time = op.processing_times[machine_id]
        pred_max = max([env.graph.operations[p].scheduled_end
                        for p in op.predecessors], default=env.current_time)
        arrival = getattr(op, 'arrival_time', 0.0)
        start_time = max(pred_max, env.machine_available_time[machine_id], arrival)
        end_time = start_time + processing_time

        schedule.append((start_time, op_id, machine_id, end_time))
        env.graph.schedule_operation(op_id, machine_id, start_time, end_time)
        env.graph._update_eligible_ops()

    # Tmaxmilp/SPT/GA
    if not _verify_tmax_feasibility(env):
        return {'success': False, 'makespan': float('inf'), 'schedule': []}

    return {
        'success': True,
        'makespan': env.graph.makespan,
        'schedule': schedule
    }


def _schedule_with_drl_baseline(env, adapter, ppo, device):
    """
    DRL Baseline scheduler.
    Uses the conventional DRL architecture from DRL_baseline/.
    Identical scheduling loop to _schedule_with_fss but uses
    the baseline adapter (2-dim features) and baseline PPO (original
    architecture with mch_pool cross-step passing).
    To swap the baseline model: replace env_adapter.py, config.py,
    and the model checkpoint in DRL_baseline/.
    """
    state = adapter.get_state_for_multippo()
    if state is None:
        return {'success': False, 'makespan': float('inf'), 'schedule': []}

    # Check that no eligible op has all machines masked
    for op_id in state['eligible_ops']:
        op_idx = state['op_list'].index(op_id)
        op_mask = state['mask_mch'][0, op_idx, :]
        op = env.graph.operations[op_id]
        compatible_indices = [adapter.machine_id_to_idx[m_id]
                              for m_id in op.compatible_machines
                              if m_id in adapter.machine_id_to_idx]
        if compatible_indices and all(op_mask[idx].item() for idx in compatible_indices):
            return {'success': False, 'makespan': float('inf'), 'schedule': []}

    schedule = []
    mch_a = None
    pool = None  # baseline uses mch_pool cross-step passing

    while len(env.graph.eligible_ops) > 0:
        state = adapter.get_state_for_multippo()
        if state is None:
            break

        env_mask_mch = state['mask_mch'].to(device)
        env_dur = state['dur'].to(device)

        n_ops = state['fea'].size(0)
        g_pool_step = g_pool_cal(
            ppo.config.graph_pool_type,
            torch.Size([1, n_ops, n_ops]),
            n_ops,
            device
        )

        env_adj = state['adj'].to(device)
        env_fea = state['fea'].to(device)
        env_candidate = state['candidate'].to(device)
        env_mask = state['mask'].to(device)
        env_mch_time = state['mch_time'].to(device)

        with torch.no_grad():
            action, a_idx, _, action_node, action_feature, mask_mch_action, hx, _ = \
                ppo.policy_old_job(
                    x=env_fea,
                    graph_pool=g_pool_step,
                    padded_nei=None,
                    adj=env_adj,
                    candidate=env_candidate,
                    mask=env_mask,
                    mask_mch=env_mask_mch,
                    dur=env_dur,
                    a_index=0,
                    old_action=0,
                    mch_pool=pool,        # baseline uses mch_pool
                    old_policy=True,
                    greedy=True
                )

            pi_mch, pool = ppo.policy_old_mch(   # baseline returns pool for next step
                action_node=action_node,
                hx=hx,                            # baseline uses hx (global mean)
                mask_mch_action=mask_mch_action,
                mch_time=env_mch_time,
                mch_a=mch_a,
                last_hh=None,
                policy=False,
                et_normalize_coef=ppo.config.et_normalize_coef
            )

            mch_a = greedy_select_action2(pi_mch)

        op_action = a_idx if isinstance(a_idx, int) else a_idx.cpu().numpy()
        op_id, machine_id = adapter.action_to_op_and_machine(
            op_action, mch_a.cpu().numpy(), state
        )

        op = env.graph.operations[op_id]
        processing_time = op.processing_times[machine_id]
        pred_max = max([env.graph.operations[p].scheduled_end
                        for p in op.predecessors], default=env.current_time)
        arrival = getattr(op, 'arrival_time', 0.0)
        start_time = max(pred_max, env.machine_available_time[machine_id], arrival)
        end_time = start_time + processing_time

        schedule.append((start_time, op_id, machine_id, end_time))
        env.graph.schedule_operation(op_id, machine_id, start_time, end_time)
        env.graph._update_eligible_ops()

    if not _verify_tmax_feasibility(env):
        return {'success': False, 'makespan': float('inf'), 'schedule': []}

    return {
        'success': True,
        'makespan': env.graph.makespan,
        'schedule': schedule
    }


def _schedule_with_ga(env, pop_size=20, elite_size=2, patience=20):
    """
    Genetic Algorithm (GA) scheduler for FJSP.

    Seeds the initial population with an SPT-greedy solution.
    Evaluation uses a pure-function decoder (no env mutation during search).
    Locked-in (completed / currently running) operations form fixed schedule
    anchors; only the remaining eligible operations are evolved.

    Termination: convergence-based — stops after `patience` consecutive
    generations with no improvement in best makespan.

    Args:
        env        : SDLEnv instance (after rollback)
        pop_size   : GA population size (default 20)
        elite_size : number of elites preserved per generation (default 2)
        patience   : generations without improvement before stopping (default 20)

    Returns:
        dict: same format as other scheduler functions
    """
    ga_start = time.time()

    # ------------------------------------------------------------------
    # 1. Extract a read-only snapshot of the env state
    # ------------------------------------------------------------------
    ops_info = {}
    for op_id, op in env.graph.operations.items():
        is_fixed = op.is_completed or (
            op.is_scheduled and op.scheduled_start <= env.current_time + 1e-4
        )
        ops_info[op_id] = {
            'predecessors': list(op.predecessors),
            'processing_times': dict(op.processing_times),
            'compatible_machines': list(op.compatible_machines),
            'op_type': op.op_type,
            'job_id': op.job_id,
            'is_fixed': is_fixed,
            'scheduled_end': op.scheduled_end if (op.is_scheduled or op.is_completed) else None,
            'arrival_time': getattr(op, 'arrival_time', 0.0),
            'op_index': op.op_index,
        }

    # job_id -> {op_type -> [(op_index, op_id), ...]} for Tmax look-ups
    # Stored as list to handle repeated op_types (e.g. multiple S2 steps).
    job_op_lookup = {}
    for op_id, info in ops_info.items():
        jid = info['job_id']
        if jid not in job_op_lookup:
            job_op_lookup[jid] = {}
        otype = info['op_type']
        if otype not in job_op_lookup[jid]:
            job_op_lookup[jid][otype] = []
        job_op_lookup[jid][otype].append((info['op_index'], op_id))

    machine_avail_init = dict(env.machine_available_time)
    tmax_constraints = list(env.tmax_constraints)
    current_time = env.current_time

    # Ops that the GA must schedule (not yet started / not completed)
    pending_ops = [op_id for op_id, info in ops_info.items() if not info['is_fixed']]
    if not pending_ops:
        return {'success': True, 'makespan': env.graph.makespan, 'schedule': []}

    # Fixed schedule anchors (completed / running ops)
    fixed_sched_end = {
        op_id: info['scheduled_end']
        for op_id, info in ops_info.items()
        if info['is_fixed'] and info['scheduled_end'] is not None
    }

    # ------------------------------------------------------------------
    # 2. Pure-function decoder — no env mutation
    # ------------------------------------------------------------------
    def decode(priority_order, machine_assign):
        """Decode (priority_order, machine_assign)  (makespan, feasible)."""
        sched_end = dict(fixed_sched_end)
        mach_avail = dict(machine_avail_init)
        priority_idx = {oid: i for i, oid in enumerate(priority_order)}
        remaining = set(priority_order)

        while remaining:
            ready = [
                oid for oid in remaining
                if all(p in sched_end for p in ops_info[oid]['predecessors'])
            ]
            if not ready:
                return float('inf'), False

            op_id = min(ready, key=lambda x: priority_idx[x])
            info = ops_info[op_id]
            machine_id = machine_assign.get(op_id)
            if machine_id is None or machine_id not in info['compatible_machines']:
                machine_id = info['compatible_machines'][0]

            pred_end = max(
                (sched_end[p] for p in info['predecessors']),
                default=current_time
            )
            start = max(pred_end,
                        mach_avail.get(machine_id, current_time),
                        info.get('arrival_time', 0.0))

            def _tmax_ok(m_id, t_start):
                for fs, ts, mx in tmax_constraints:
                    if info['op_type'] != ts:
                        continue
                    # find closest preceding from_step op for this job
                    candidates = job_op_lookup.get(info['job_id'], {}).get(fs, [])
                    pred_candidates = [(idx, fid) for idx, fid in candidates
                                       if idx < info['op_index']]
                    if not pred_candidates:
                        continue
                    _, fid = max(pred_candidates, key=lambda x: x[0])
                    if fid in sched_end and t_start - sched_end[fid] > mx:
                        return False
                return True

            if not _tmax_ok(machine_id, start):
                alts = sorted(
                    [m for m in info['compatible_machines'] if m != machine_id],
                    key=lambda m: mach_avail.get(m, current_time)
                )
                found = False
                for alt_m in alts:
                    alt_start = max(pred_end, mach_avail.get(alt_m, current_time))
                    if _tmax_ok(alt_m, alt_start):
                        machine_id, start = alt_m, alt_start
                        found = True
                        break
                if not found:
                    return float('inf'), False

            end = start + info['processing_times'][machine_id]
            sched_end[op_id] = end
            mach_avail[machine_id] = end
            remaining.remove(op_id)

        ms = max(sched_end.values()) if sched_end else current_time
        return ms, True

    # ------------------------------------------------------------------
    # 3. Initial population generation
    # ------------------------------------------------------------------
    def make_spt_seed():
        """SPT greedy: priority by min proc-time, machine by earliest feasible."""
        sched_end = dict(fixed_sched_end)
        mach_avail = dict(machine_avail_init)
        priority, assignment = [], {}
        remaining = set(pending_ops)

        while remaining:
            ready = [
                oid for oid in remaining
                if all(p in sched_end for p in ops_info[oid]['predecessors'])
            ]
            if not ready:
                ready = list(remaining)

            op_id = min(
                ready,
                key=lambda oid: min(
                    ops_info[oid]['processing_times'].get(m, float('inf'))
                    for m in ops_info[oid]['compatible_machines']
                )
            )
            info = ops_info[op_id]
            pred_end = max(
                (sched_end[p] for p in info['predecessors']),
                default=current_time
            )
            best_m, best_start = None, float('inf')
            for m in info['compatible_machines']:
                t_start = max(pred_end,
                              mach_avail.get(m, current_time),
                              info.get('arrival_time', 0.0))
                tmax_ok = True
                for fs, ts, mx in tmax_constraints:
                    if info['op_type'] == ts:
                        # find closest preceding from_step op for this job
                        candidates = job_op_lookup.get(info['job_id'], {}).get(fs, [])
                        pred_candidates = [(idx, fid) for idx, fid in candidates
                                           if idx < info['op_index']]
                        if not pred_candidates:
                            continue
                        _, fid = max(pred_candidates, key=lambda x: x[0])
                        if fid in sched_end and t_start - sched_end[fid] > mx:
                            tmax_ok = False
                            break
                if tmax_ok and t_start < best_start:
                    best_start, best_m = t_start, m

            if best_m is None:
                best_m = info['compatible_machines'][0]
                best_start = max(pred_end, mach_avail.get(best_m, current_time))

            end = best_start + info['processing_times'][best_m]
            sched_end[op_id] = end
            mach_avail[best_m] = end
            priority.append(op_id)
            assignment[op_id] = best_m
            remaining.remove(op_id)

        return priority, assignment

    def make_random():
        prio = list(pending_ops)
        np.random.shuffle(prio)
        assign = {
            oid: np.random.choice(ops_info[oid]['compatible_machines'])
            for oid in pending_ops
        }
        return prio, assign

    # ------------------------------------------------------------------
    # 4. GA operators
    # ------------------------------------------------------------------
    def tournament(pop, fits, k=3):
        idxs = np.random.choice(len(pop), min(k, len(pop)), replace=False)
        return pop[int(min(idxs, key=lambda i: fits[i]))]

    def ox_crossover(p1, p2):
        n = len(p1)
        if n <= 2:
            return list(p1)
        a, b = sorted(np.random.choice(n, 2, replace=False).tolist())
        child = [None] * n
        child[a:b + 1] = p1[a:b + 1]
        segment = set(p1[a:b + 1])
        fill = [x for x in p2 if x not in segment]
        j = 0
        for i in range(n):
            if child[i] is None:
                child[i] = fill[j]
                j += 1
        return child

    def crossover(ind1, ind2):
        child_prio = ox_crossover(ind1[0], ind2[0])
        child_assign = {
            oid: (ind1[1][oid] if np.random.random() < 0.5 else ind2[1][oid])
            for oid in pending_ops
        }
        return child_prio, child_assign

    def mutate(ind, rate=0.15):
        prio, assign = list(ind[0]), dict(ind[1])
        n = len(prio)
        if n >= 2 and np.random.random() < rate:
            i, j = np.random.choice(n, 2, replace=False)
            prio[i], prio[j] = prio[j], prio[i]
        if np.random.random() < rate:
            oid = np.random.choice(pending_ops)
            ms = ops_info[oid]['compatible_machines']
            if len(ms) > 1:
                assign[oid] = np.random.choice(ms)
        return prio, assign

    # ------------------------------------------------------------------
    # 5. Evolution loop
    # ------------------------------------------------------------------
    seed_prio, seed_assign = make_spt_seed()
    population = [(seed_prio, seed_assign)] + [make_random() for _ in range(pop_size - 1)]

    best_ms = float('inf')
    best_ind = None
    generation = 0
    eval_count = 0
    no_improve = 0

    while no_improve < patience:
        prev_best = best_ms  # snapshot before this generation

        fits = []
        for ind in population:
            ms, _ = decode(ind[0], ind[1])
            fits.append(ms)
            eval_count += 1
            if ms < best_ms:
                best_ms = ms
                best_ind = (list(ind[0]), dict(ind[1]))

        sorted_idx = sorted(range(len(population)), key=lambda i: fits[i])
        elites = [population[i] for i in sorted_idx[:elite_size]]
        new_pop = list(elites)
        while len(new_pop) < pop_size:
            child = mutate(crossover(tournament(population, fits),
                                     tournament(population, fits)))
            new_pop.append(child)
        population = new_pop
        generation += 1

        # Stagnation: did this generation improve the global best?
        if best_ms < prev_best - 1e-6:
            no_improve = 0
        else:
            no_improve += 1

    elapsed = time.time() - ga_start
    print(f"  [GA] gen={generation}, evals={eval_count}, "
          f"best_ms={best_ms:.2f}, time={elapsed:.2f}s")

    if best_ind is None or best_ms == float('inf'):
        return {'success': False, 'makespan': float('inf'), 'schedule': []}

    # ------------------------------------------------------------------
    # 6. Apply best solution to env graph
    # ------------------------------------------------------------------
    best_prio, best_assign = best_ind
    priority_idx = {oid: i for i, oid in enumerate(best_prio)}
    sched_end = dict(fixed_sched_end)
    mach_avail = dict(machine_avail_init)
    schedule = []
    remaining = set(best_prio)

    while remaining:
        ready = [
            oid for oid in remaining
            if all(p in sched_end for p in ops_info[oid]['predecessors'])
        ]
        if not ready:
            return {'success': False, 'makespan': float('inf'), 'schedule': []}

        op_id = min(ready, key=lambda x: priority_idx[x])
        info = ops_info[op_id]
        machine_id = best_assign.get(op_id, info['compatible_machines'][0])
        if machine_id not in info['compatible_machines']:
            machine_id = info['compatible_machines'][0]

        pred_end = max(
            (sched_end[p] for p in info['predecessors']),
            default=current_time
        )
        start = max(pred_end,
                    mach_avail.get(machine_id, current_time),
                    info.get('arrival_time', 0.0))
        end = start + info['processing_times'][machine_id]

        env.graph.schedule_operation(op_id, machine_id, start, end)
        env.graph._update_eligible_ops()
        schedule.append((start, op_id, machine_id, end))
        sched_end[op_id] = end
        mach_avail[machine_id] = end
        remaining.remove(op_id)

    # TmaxhelperFSS/milp/SPT
    if not _verify_tmax_feasibility(env):
        return {'success': False, 'makespan': float('inf'), 'schedule': []}

    return {
        'success': True,
        'makespan': env.graph.makespan,
        'schedule': schedule
    }


def _schedule_with_milp(env, time_limit=30, mip_gap_abs=None):
    """milp-MILP60sNone="""
    result = env._solve_subproblem_with_lower(time_limit=time_limit, mip_gap_abs=mip_gap_abs)

    if result['status'] != 'ok':
        return {
            'success': False,
            'makespan': float('inf'),
            'schedule': []
        }

    schedule = result['schedule']
    for start_time, op_id, machine_id, end_time in schedule:
        env.graph.schedule_operation(op_id, machine_id, start_time, end_time)

    return {
        'success': True,
        'makespan': env.graph.makespan,  # use graph makespan (all ops incl. history)
        'schedule': schedule
    }


def _schedule_with_spt(env):
    """SPTTmax"""
    schedule = []

    while len(env.graph.eligible_ops) > 0:
        sorted_ops = sorted(
            env.graph.eligible_ops,
            key=lambda op_id: env.graph.operations[op_id].get_min_processing_time()
        )

        scheduled = False

        for op_id in sorted_ops:
            op = env.graph.operations[op_id]

            valid_machines = []
            for mach_id in op.compatible_machines:
                if _check_tmax_feasibility(env, op, mach_id):
                    valid_machines.append(mach_id)

            if not valid_machines:
                continue

            best_machine = min(
                valid_machines,
                key=lambda m_id: env.machine_available_time.get(m_id, 0)
            )

            processing_time = op.processing_times[best_machine]
            pred_max = max([env.graph.operations[p].scheduled_end
                            for p in op.predecessors], default=env.current_time)
            arrival = getattr(op, 'arrival_time', 0.0)
            start_time = max(pred_max, env.machine_available_time[best_machine], arrival)
            end_time = start_time + processing_time

            schedule.append((start_time, op_id, best_machine, end_time))
            env.graph.schedule_operation(op_id, best_machine, start_time, end_time)
            scheduled = True
            break

        if not scheduled:
            return {
                'success': False,
                'makespan': float('inf'),
                'schedule': schedule
            }

    return {
        'success': True,
        'makespan': env.graph.makespan,
        'schedule': schedule
    }

def _check_tmax_feasibility(env, op, machine_id):
    """Tmax"""
    for constraint in env.tmax_constraints:
        from_step, to_step, max_interval = constraint

        if op.op_type == to_step:
            # pick the closest preceding from_step op (by op_index) for this job
            candidates = [o for o in env.graph.operations.values()
                          if o.job_id == op.job_id
                          and o.op_type == from_step
                          and o.op_index < op.op_index]

            if candidates:
                pred = max(candidates, key=lambda o: o.op_index)
                if pred.is_completed or pred.is_scheduled:
                    processing_time = op.processing_times[machine_id]
                    pred_max = max([env.graph.operations[p].scheduled_end
                                    for p in op.predecessors], default=env.current_time)
                    start_time = max(pred_max, env.machine_available_time[machine_id])

                    gap = start_time - pred.scheduled_end

                    if gap > max_interval:
                        return False

    return True


# ============================================================================
# Part 2:  ()
# ============================================================================

def progressive_release(env, adapter, ppo, method='ERC-FSS', device='cpu',
                        ga_pop_size=20, ga_patience=20):
    """
     ()

    Returns:
        dict: {
            'success': bool,
            'released_jobs': list,
            'makespan': float,
            'schedule': list,
            'k_value': int,  # job
            'combinations_tried': int,  #
            'time_per_k': dict  # k
        }
    """
    buffer_jobs = list(env.job_buffer.buffer)
    n = len(buffer_jobs)

    if n == 0:
        return {
            'success': False,
            'released_jobs': [],
            'makespan': 0.0,
            'schedule': [],
            'k_value': 0,
            'combinations_tried': 0,
            'time_per_k': {}
        }

    time_per_k = {}
    total_combinations = 0

    # n1
    for k in range(n, 0, -1):
        k_start_time = time.time()

        if k == n:
            combinations_to_try = [buffer_jobs]
        else:
            combinations_to_try = list(combinations(buffer_jobs, k))

        total_combos = len(combinations_to_try)
        total_combinations += total_combos

        feasible_results = []

        for idx, jobs_combo in enumerate(combinations_to_try):
            graph_snapshot = env.graph.snapshot()
            env._build_static_subproblem(list(jobs_combo))
            result = global_reschedule(env, adapter, ppo, method=method, device=device,
                                       ga_pop_size=ga_pop_size, ga_patience=ga_patience)

            if result['success']:
                #
                scheduled_snapshot = env.graph.snapshot()
                feasible_results.append({
                    'jobs': list(jobs_combo),
                    'makespan': result['makespan'],
                    'schedule': result['schedule'],
                    'scheduled_snapshot': scheduled_snapshot
                })

            env.graph.restore(graph_snapshot)

        k_time = time.time() - k_start_time
        time_per_k[f'k={k}'] = {
            'time': k_time,
            'combinations': total_combos,
            'feasible': len(feasible_results)
        }

        if feasible_results:
            best_result = min(feasible_results, key=lambda x: x['makespan'])

            # restore(graph_snapshot)  _build_static_subproblem  ops
            # restore(scheduled_snapshot)  ops  ops
            #  combo  ops
            env._build_static_subproblem(best_result['jobs'])
            env.graph.restore(best_result['scheduled_snapshot'])

            return {
                'success': True,
                'released_jobs': best_result['jobs'],
                'makespan': best_result['makespan'],
                'schedule': best_result['schedule'],
                'k_value': k,
                'combinations_tried': total_combinations,
                'time_per_k': time_per_k
            }

    return {
        'success': False,
        'released_jobs': [],
        'makespan': 0.0,
        'schedule': [],
        'k_value': 0,
        'combinations_tried': total_combinations,
        'time_per_k': time_per_k
    }


# ============================================================================
# Part 2.5: ChemOS-faithful FCFS dispatch primitives
# ============================================================================

def _get_ready_ops(env):
    """
    Ops that ChemOS would place in its pending_requests queue:
      • All predecessors COMPLETED  (not merely scheduled)
      • Job arrived  (arrival_time ≤ current_time)
      • Op not yet scheduled or completed
    Sorted by (arrival_time, job_id, op_index) = FIFO insertion order.
    """
    ready = []
    for op_id, op in env.graph.operations.items():
        if op.is_scheduled or op.is_completed:
            continue
        if getattr(op, 'arrival_time', 0.0) > env.current_time + 1e-4:
            continue
        if all(
            env.graph.operations[p].is_completed
            for p in op.predecessors
            if p in env.graph.operations
        ):
            ready.append(op_id)
    ready.sort(key=lambda oid: (
        getattr(env.graph.operations[oid], 'arrival_time', 0.0),
        env.graph.operations[oid].job_id,
        env.graph.operations[oid].op_index,
    ))
    return ready


def _chemos_dispatch_tick(env, schedule):
    """
    One ChemOS poll tick (chemOS.py lines 124-140 + bot_manager.py lines 66-77):

      FOR op in pending_queue  (FIFO by arrival_time, job_id, op_index)
        idle_bot = first machine with avail_time ≤ current_time that is compatible
        IF found   submit immediately (machine becomes busy)
        ELSE       skip; op retried next tick

    Multiple ops dispatched per tick if they need different machines.
    Returns True if any op was dispatched.
    """
    ready_ops = _get_ready_ops(env)
    if not ready_ops:
        return False

    dispatched_any = False
    for op_id in ready_ops:
        op = env.graph.operations[op_id]
        # First currently-idle compatible machine (bot_manager.py: first match in DB)
        idle = sorted(
            m for m in op.compatible_machines
            if env.machine_available_time.get(m, 0.0) <= env.current_time + 1e-4
        )
        if not idle:
            continue
        machine_id = idle[0]
        arrival    = getattr(op, 'arrival_time', 0.0)
        start_time = max(env.current_time, arrival)
        end_time   = start_time + op.processing_times[machine_id]
        schedule.append((start_time, op_id, machine_id, end_time))
        env.graph.schedule_operation(op_id, machine_id, start_time, end_time)
        dispatched_any = True
    return dispatched_any


def _check_tmax_at_dispatch(env, op):
    """
    Before scheduling `op`, check whether any Tmax constraint that ends at this
    op's type would already be violated given the earliest possible start time.

    Returns (violated: bool, constraint_name: str, excess: float).
    """
    if not hasattr(env, 'tmax_constraints'):
        return False, None, 0.0
    start_time = max(env.current_time, getattr(op, 'arrival_time', 0.0))
    for from_step, to_step, max_interval in env.tmax_constraints:
        if op.op_type != to_step:
            continue
        candidates = [o for o in env.graph.operations.values()
                      if o.job_id == op.job_id and o.op_type == from_step
                      and o.op_index < op.op_index]
        if not candidates:
            continue
        pred = max(candidates, key=lambda o: o.op_index)
        if pred.is_completed or pred.is_scheduled:
            gap = start_time - pred.scheduled_end
            if gap > max_interval + 1e-4:
                return True, f"{from_step}{to_step}", gap - max_interval
    return False, None, 0.0


def _abort_job_tmax(env, job_id, cancelled_jobs, abort_replace_pairs=None):
    """
    Abort job_id: add to the cancelled set so future dispatch ticks skip it,
    then generate one replacement job.  Ops are left untouched so the output
    clearly shows which steps ran and which didn't (aborted op shows as [-]).
    graph.is_complete() is handled by the patched _check_all_jobs_completed
    in the episode runner which ignores cancelled jobs.

    If `abort_replace_pairs` is provided, append
        {'terminated_job': job_id, 'replacement_job': <new_id_or_None>}
    so the caller can record one-to-one terminatereplace correspondences.
    """
    cancelled_jobs.add(job_id)
    env.n_jobs += 1

    # Prefer the abort-specific replacement path (identical clone, no BO delay).
    # Fall back to generic dynamic generation only if not patched.
    replacement_id = None
    if hasattr(env, '_abort_replace') and callable(env._abort_replace):
        replacement_id = env._abort_replace(job_id)
    else:
        def _snapshot_job_ids():
            ids = set()
            if env.graph is not None:
                ids.update(op.job_id for op in env.graph.operations.values())
            try:
                ids.update(j.id for j in env.job_buffer.buffer)
            except Exception:
                pass
            return ids

        before_ids = _snapshot_job_ids()
        env._try_generate_new_jobs(count=1)
        after_ids = _snapshot_job_ids()
        new_ids = list(after_ids - before_ids)
        replacement_id = new_ids[0] if new_ids else None

    if abort_replace_pairs is not None:
        abort_replace_pairs.append({
            'terminated_job':  job_id,
            'replacement_job': replacement_id,
        })


def _chemos_dispatch_tick_tmax_abort(env, schedule, cancelled_jobs,
                                     abort_replace_pairs=None):
    """
    Like _chemos_dispatch_tick but intercepts Tmax-destination ops before
    dispatch.  If a Tmax constraint would be violated, the job is aborted and
    a replacement is generated instead of scheduling the op.
    """
    ready_ops = _get_ready_ops(env)
    ready_ops = [oid for oid in ready_ops
                 if env.graph.operations[oid].job_id not in cancelled_jobs]
    if not ready_ops:
        return False

    dispatched_any = False
    for op_id in ready_ops:
        op = env.graph.operations[op_id]

        violated, constraint_name, excess = _check_tmax_at_dispatch(env, op)
        if violated:
            print(f"    [TMAX ABORT] job={op.job_id}  op={op_id}  "
                  f"constraint={constraint_name}  excess=+{excess:.2f} min  "
                  f" aborting, generating replacement")
            _abort_job_tmax(env, op.job_id, cancelled_jobs,
                            abort_replace_pairs=abort_replace_pairs)
            dispatched_any = True
            continue

        idle = sorted(
            m for m in op.compatible_machines
            if env.machine_available_time.get(m, 0.0) <= env.current_time + 1e-4
        )
        if not idle:
            continue
        machine_id = idle[0]
        arrival    = getattr(op, 'arrival_time', 0.0)
        start_time = max(env.current_time, arrival)
        end_time   = start_time + op.processing_times[machine_id]
        schedule.append((start_time, op_id, machine_id, end_time))
        env.graph.schedule_operation(op_id, machine_id, start_time, end_time)
        dispatched_any = True
    return dispatched_any


# ============================================================================
# Part 3:  ()
# ============================================================================

class ThreeWayComparison:
    """ ()"""

    def __init__(self, env_config_path, model_path, output_dir='./comparison_results',
                 ga_pop_size=20, ga_patience=20,
                 baseline_model_path=None):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        #
        with open(env_config_path, 'r', encoding='utf-8') as f:
            self.env_config_raw = yaml.safe_load(f)

        from configs import Config
        self.env_config = Config(self.env_config_raw)

        #
        self.env = SDLEnv(self.env_config)

        # PPO (main)
        self.ppo_config = get_default_config()
        self.ppo_config.n_machines = len(self.env.machines)
        self.ppo_config.n_j = self.env_config_raw.get('env_config', {}).get('n_jobs', 80)
        self.ppo_config.device = 'cpu'
        self.device = torch.device('cpu')

        # GA parameters
        self.ga_pop_size = ga_pop_size
        self.ga_patience = ga_patience

        # Agent (main)
        self.adapter = SDLEnvAdapter(self.env, self.ppo_config)
        self.ppo = PPO(self.ppo_config)

        #
        self.model_path = model_path
        self._load_model()

        # ---- DRL Baseline ----
        # baseline_model_path: path to DRL_baseline checkpoint
        # To swap baseline: replace DRL_baseline/env_adapter.py, config.py and model file
        self.baseline_model_path = baseline_model_path
        self.ppo_baseline = None
        self.baseline_ppo_config = None
        if baseline_model_path is not None:
            self._init_baseline()

        print(f"\n{'#' * 60}")
        print(f"Three-Way Comparison Initialized (Enhanced)")
        print(f"  Machines: {len(self.env.machines)}")
        print(f"  Output Dir: {self.output_dir}")
        print(f"  DRL Baseline: {'enabled' if self.ppo_baseline is not None else 'disabled'}")
        print(f"{'#' * 60}\n")

    def _load_model(self):
        """"""
        print(f" Loading model from: {self.model_path}")

        checkpoint = torch.load(self.model_path, map_location='cpu')
        self.ppo.policy_job.load_state_dict(checkpoint['job_actor'])
        self.ppo.policy_mch.load_state_dict(checkpoint['mch_actor'])
        self.ppo.policy_old_job.load_state_dict(checkpoint['job_actor'])
        self.ppo.policy_old_mch.load_state_dict(checkpoint['mch_actor'])

        self.ppo.policy_job.eval()
        self.ppo.policy_old_job.eval()
        self.ppo.policy_mch.eval()
        self.ppo.policy_old_mch.eval()

        print(f" Model loaded successfully\n")

    def _init_baseline(self):
        """
        Initialize DRL baseline agent.
        Uses BaselineSDLEnvAdapter and BaselinePPO imported from DRL_baseline/.
        To swap the baseline model: replace env_adapter.py, config.py,
        and the checkpoint file in DRL_baseline/training/saved_models_lk/.
        """
        print(f" Loading DRL baseline from: {self.baseline_model_path}")

        # Build baseline config (from DRL_baseline/config.py)
        cfg = get_baseline_config()
        cfg.n_machines = len(self.env.machines)
        cfg.n_j = self.env_config_raw.get('env_config', {}).get('n_jobs', 80)
        cfg.device = 'cpu'
        self.baseline_ppo_config = cfg

        # Build baseline adapter (from DRL_baseline/env_adapter.py)
        self.baseline_adapter = BaselineSDLEnvAdapter(self.env, cfg)

        # Build baseline PPO (from DRL_baseline/ppo.py)
        self.ppo_baseline = BaselinePPO(cfg)

        checkpoint = torch.load(self.baseline_model_path, map_location='cpu')
        self.ppo_baseline.policy_job.load_state_dict(checkpoint['job_actor'])
        self.ppo_baseline.policy_mch.load_state_dict(checkpoint['mch_actor'])
        self.ppo_baseline.policy_old_job.load_state_dict(checkpoint['job_actor'])
        self.ppo_baseline.policy_old_mch.load_state_dict(checkpoint['mch_actor'])

        self.ppo_baseline.policy_job.eval()
        self.ppo_baseline.policy_old_job.eval()
        self.ppo_baseline.policy_mch.eval()
        self.ppo_baseline.policy_old_mch.eval()

        print(f" DRL baseline loaded successfully "
              f"(input_dim={cfg.input_dim}, n_machines={cfg.n_machines})\n")

    def _pre_generate_jobs(self, seed, n_jobs=80):
        """
        jobsopprocessing_times

        job

        Returns:
            list of dict: job:
                id, type, arrival_time, priority, operations
        """
        import os
        from case_study.dataset_generator import DatasetGenerator

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(project_root, 'configs', 'sdl_config.yaml')

        # DatasetGeneratorself.env
        gen = DatasetGenerator(config_path=config_path)

        #
        np.random.seed(seed)
        gen.atlas_simulator.rng = np.random.RandomState(seed + 3000000)

        jobs = []

        # Warmstart: 16 jobs, all arrive at t=0
        warmstart_types = ['SDL_experiment'] * 12
        for i, job_type in enumerate(warmstart_types):
            job = gen.generate_job(
                job_id=f"job_{i}",
                job_type=job_type,
                arrival_time=0.0
            )
            jobs.append(job)

        # Dynamic jobs
        for i in range(12, n_jobs):
            job_type = np.random.choice(['SDL_experiment'])
            job = gen.generate_job(
                job_id=f"job_{i}",
                job_type=job_type,
                arrival_time=0.0  # placeholder; real arrival set at replay time
            )
            jobs.append(job)

        # Pre-generate BO delays for dynamic jobs (indices 12..n_jobs-1).
        # Each is sampled once here so every method sees identical arrival offsets.
        # Delay ~ Uniform(5, 15) seconds, stored in minutes (sim runs in min).
        bo_rng = np.random.RandomState(seed + 9999999)
        bo_delays = {}
        for i in range(12, n_jobs):
            bo_delays[f"job_{i}"] = float(bo_rng.uniform(5.0, 15.0)) / 60.0

        print(f"\n[PRE-GEN]  {len(jobs)} jobs (seed={seed})")
        for i, j in enumerate(jobs):
            n_ops = len(j['operations'])
            sample_pt = '-'
            if n_ops > 0 and j['operations'][0].get('processing_times'):
                pts = list(j['operations'][0]['processing_times'].values())
                sample_pt = f"{pts[0]:.2f}" if pts else '-'
            delay_str = f" | bo_delay={bo_delays[j['id']]*60:.1f}s" if j['id'] in bo_delays else ""
            print(f"  job_{i}: {j['type']:20s} | ops={n_ops} | op1_pt_sample={sample_pt}{delay_str}")

        return jobs, bo_delays

    def run_episode_with_method(self, method='ERC-FSS',
                                seed=np.random.randint(0, 100000) + np.random.randint(0, 1000) * 100000,
                                ga_pop_size=20, ga_patience=20,
                                pregenerated_jobs=None,
                                bo_delays=None):
        """
        episode ()

        Args:
            pregenerated_jobs: jobs
                              None

        Returns:
            dict: {
                'makespan': float,
                'job_details': dict,
                'tmax_violations': int,
                'total_time': float,
                'time_breakdown': dict,
                'schedule_sequence': list,
                'release_history': list
            }
        """
        episode_start = time.time()

        # skip_warmstart=True
        env = SDLEnv(self.env_config)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if pregenerated_jobs is not None:
            # ============================================================
            # jobjob
            # ============================================================
            env.reset(seed=seed, skip_warmstart=True)

            # warmstart jobs16
            from envs.job_buffer import Job as JobClass
            for i in range(min(12, len(pregenerated_jobs))):
                job_data = pregenerated_jobs[i]
                job_obj = JobClass(
                    id=job_data['id'],
                    job_type=job_data['type'],
                    operations=job_data['operations'],
                    arrival_time=job_data['arrival_time'],
                    priority=job_data.get('priority', 'normal')
                )
                env.job_buffer.add_job(job_obj)
            env.job_counter = min(12, len(pregenerated_jobs))

            # Patch: _try_generate_new_jobs
            _next_idx = [12]  # mutable for closure
            _bo_delays = bo_delays or {}  # pregenerated BO delays

            def _replay_try_gen(count=1):
                generated = 0
                for _ in range(count):
                    if _next_idx[0] >= len(pregenerated_jobs):
                        break
                    if env.job_counter >= env.n_jobs:
                        break

                    job_data = pregenerated_jobs[_next_idx[0]]
                    # Use pregenerated BO delay so all methods see the same
                    # arrival offset; falls back to 5.0 if not provided.
                    delay = _bo_delays.get(job_data['id'], 10.0 / 60.0)
                    # Arrival = abort/trigger moment + BO delay.
                    # Physical meaning: as soon as the system knows a job needs
                    # replacing (or a new BO suggestion is requested), BO takes
                    # 5-15s to produce the next candidate.
                    arrival_time = env.current_time + delay

                    job_obj = JobClass(
                        id=job_data['id'],
                        job_type=job_data['type'],
                        operations=job_data['operations'],
                        arrival_time=arrival_time,
                        priority=job_data.get('priority', 'normal')
                    )
                    env.job_buffer.add_job(job_obj)
                    env.job_counter += 1
                    _next_idx[0] += 1
                    generated += 1

                if generated > 0:
                    print(f"    [REPLAY]  {generated} jobs "
                          f"(idx {_next_idx[0]-generated}..{_next_idx[0]-1}, "
                          f"bo_delay={delay*60:.1f}s)")
                else:
                    if env.job_counter >= env.n_jobs:
                        print(f"    [STOP] job_counter")
                    else:
                        print(f"    [REPLAY] jobs")

            env._try_generate_new_jobs = _replay_try_gen
            # Also inject into env so _try_generate_new_jobs fallback path is consistent
            env.bo_delays_map = dict(_bo_delays)

            # ---- Abort-replacement path (identical clone, no BO delay) ----
            _pregen_by_id = {jd['id']: jd for jd in pregenerated_jobs}

            def _replay_abort_replace(terminated_id):
                # Use next pool id as replacement id (preserves terminatedreplacement
                # numbering in output logs); content is a deep clone of the terminated
                # job's operations; arrival = abort moment (no delay).
                if _next_idx[0] >= len(pregenerated_jobs):
                    return None
                pool_entry = pregenerated_jobs[_next_idx[0]]
                term = _pregen_by_id.get(terminated_id)
                if term is None:
                    return None
                clone_job = JobClass(
                    id=pool_entry['id'],
                    job_type=term['type'],
                    operations=copy.deepcopy(term['operations']),
                    arrival_time=env.current_time,
                    priority=term.get('priority', 'normal'),
                )
                env.job_buffer.add_job(clone_job)
                env.job_counter += 1
                _next_idx[0] += 1
                print(f"    [REPLAY-ABORT]  replace {terminated_id}  {pool_entry['id']} "
                      f"(identical clone, arrival={env.current_time:.2f}, no BO delay)")
                return pool_entry['id']

            env._abort_replace = _replay_abort_replace
        else:
            #
            env.reset(seed=seed)

        env.machines.sort(key=lambda m: m.id)
        env.machines_dict = {m.id: m for m in env.machines}

        adapter = SDLEnvAdapter(env, self.ppo_config)

        # For drl_baseline: build a fresh baseline adapter pointing at this episode's env
        if method == 'Conventional DRL':
            if self.ppo_baseline is None:
                raise RuntimeError("DRL baseline not initialized. Pass baseline_model_path to ThreeWayComparison.")
            _adapter = BaselineSDLEnvAdapter(env, self.baseline_ppo_config)
            _ppo = self.ppo_baseline
        else:
            _adapter = adapter
            _ppo = self.ppo

        step_count = 0
        max_steps = 10000

        #
        release_history = []  #
        schedule_sequence = []  #
        time_breakdown = {
            'hla_decision': 0.0,
            'scheduling': 0.0,
            'time_advance': 0.0
        }

        print(f"\n{'' * 30}")
        print(f"Running Episode with {method.upper()}")
        print(f"{'' * 30}")

        while not env._check_all_jobs_completed() and step_count < max_steps:
            step_count += 1

            if len(env.job_buffer) == 0:
                advance_start = time.time()
                env._advance_time_and_inject_jobs()
                while len(env.job_buffer) == 0 and env.graph and any(
                    op.is_scheduled and not op.is_completed
                    for op in env.graph.operations.values()
                ):
                    env._advance_time_and_inject_jobs()
                time_breakdown['time_advance'] += time.time() - advance_start
                continue

            # HLA
            # ga100: ga100
            # milp_nolimit: milp
            _method = method
            if method == 'GA':
                _method = 'ga'
            _ga_pop = 50 if method == 'GA' else ga_pop_size
            hla_start = time.time()
            release_result = progressive_release(
                env, _adapter, _ppo, method=_method, device=self.device,
                ga_pop_size=_ga_pop, ga_patience=ga_patience
            )
            time_breakdown['hla_decision'] += time.time() - hla_start

            if not release_result['success']:
                advance_start = time.time()
                env._advance_time_and_inject_jobs()
                while len(env.job_buffer) == 0 and env.graph and any(
                    op.is_scheduled and not op.is_completed
                    for op in env.graph.operations.values()
                ):
                    env._advance_time_and_inject_jobs()
                time_breakdown['time_advance'] += time.time() - advance_start
                continue

            #
            release_info = {
                'step': step_count,
                'current_time': env.current_time,
                'released_jobs': [j.id for j in release_result['released_jobs']],
                'k_value': release_result.get('k_value', 0),
                'makespan': release_result['makespan'],
                'combinations_tried': release_result.get('combinations_tried', 0)
            }
            release_history.append(release_info)

            #
            for sched in release_result.get('schedule', []):
                schedule_sequence.append({
                    'step': step_count,
                    'start': sched[0],
                    'op_id': sched[1],
                    'machine': sched[2],
                    'end': sched[3]
                })

            # buffer
            released_jobs = release_result['released_jobs']
            for job in released_jobs:
                if job in env.job_buffer.buffer:
                    env.job_buffer.buffer.remove(job)

            #
            advance_start = time.time()
            env._advance_time_and_inject_jobs()
            while len(env.job_buffer) == 0 and env.graph and any(
                op.is_scheduled and not op.is_completed
                for op in env.graph.operations.values()
            ):
                env._advance_time_and_inject_jobs()
            time_breakdown['time_advance'] += time.time() - advance_start

        total_time = time.time() - episode_start
        # Sum of individual progressive_release() call times — the canonical
        # "solve time" metric, comparable with ChemOS validation's total_solve_time.
        total_solve_time = time_breakdown['hla_decision']
        final_makespan = env.graph.makespan if env.graph else 0.0

        # Tmax
        tmax_violations, tmax_details = self._check_tmax_violations(env)

        # job
        job_details = self._collect_job_details(env)

        print(f"\n{'=' * 60}")
        print(f"{method.upper()} Episode Completed")
        print(f"  Total steps: {step_count}")
        print(f"  Final makespan: {final_makespan:.2f}")
        print(f"  Tmax violations: {tmax_violations}")
        self._print_tmax_details(method.upper(), tmax_violations, tmax_details)
        print(f"  Solve time (accumulated): {total_solve_time:.2f}s")
        print(f"  Total wall-clock time: {total_time:.2f}s")
        print(f"{'=' * 60}\n")

        return {
            'makespan': final_makespan,
            'job_details': job_details,
            'tmax_violations': tmax_violations,
            'total_solve_time': total_solve_time,
            'total_time': total_time,
            'time_breakdown': time_breakdown,
            'schedule_sequence': schedule_sequence,
            'release_history': release_history
        }

    def run_episode_chemos_tmax_abort(self, seed, pregenerated_jobs=None, bo_delays=None):
        """
        FCFS dispatch with real-time Tmax
        enforcement: if any Tmax-destination op would be violated at dispatch
        time, that job is immediately aborted (ops fake-completed) and one
        replacement job is generated.  The episode ends when the full
        (original + replacement) n_jobs count is complete.

        Key metrics reported:
          tmax_violations  — always 0 by construction (we abort before violation)
          aborted_jobs     — number of experiments cancelled due to Tmax failure
          makespan         — wall-clock span including replacement experiments
        """
        episode_start        = time.time()
        schedule_sequence    = []
        cancelled_jobs       = set()
        abort_replace_pairs  = []   # ordered list of {terminated_job, replacement_job}
        time_breakdown       = {'dispatch': 0.0, 'time_advance': 0.0}

        env = SDLEnv(self.env_config)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if pregenerated_jobs is not None:
            env.reset(seed=seed, skip_warmstart=True)
            from envs.job_buffer import Job as JobClass

            for i in range(min(12, len(pregenerated_jobs))):
                jd = pregenerated_jobs[i]
                env.job_buffer.add_job(JobClass(
                    id=jd['id'],
                    job_type=jd['type'],
                    operations=jd['operations'],
                    arrival_time=jd['arrival_time'],
                    priority=jd.get('priority', 'normal'),
                ))
            env.job_counter = min(12, len(pregenerated_jobs))

            _next_idx  = [12]
            _bo_delays = bo_delays or {}

            def _replay_try_gen(count=1):
                generated = 0
                for _ in range(count):
                    if _next_idx[0] >= len(pregenerated_jobs):
                        break
                    if env.job_counter >= env.n_jobs:
                        break
                    jd    = pregenerated_jobs[_next_idx[0]]
                    delay        = _bo_delays.get(jd['id'], 10.0 / 60.0)
                    # Arrival = abort/trigger moment + BO delay (SDL-faithful).
                    arrival_time = env.current_time + delay
                    env.job_buffer.add_job(JobClass(
                        id=jd['id'],
                        job_type=jd['type'],
                        operations=jd['operations'],
                        arrival_time=arrival_time,
                        priority=jd.get('priority', 'normal'),
                    ))
                    env.job_counter += 1
                    _next_idx[0]   += 1
                    generated      += 1
                if generated > 0:
                    delay_used = _bo_delays.get(
                        pregenerated_jobs[_next_idx[0] - 1]['id'], 10.0 / 60.0)
                    print(f"    [REPLAY]  {generated} jobs "
                          f"(idx {_next_idx[0]-generated}..{_next_idx[0]-1}, "
                          f"bo_delay={delay_used*60:.1f}s)")

            env._try_generate_new_jobs = _replay_try_gen
            env.bo_delays_map = dict(_bo_delays)

            # ---- Abort-replacement path (identical clone, no BO delay) ----
            _pregen_by_id = {jd['id']: jd for jd in pregenerated_jobs}

            def _replay_abort_replace(terminated_id):
                # Use next pool id as replacement id (preserves terminatedreplacement
                # numbering in output). Content is a deep clone of whatever the
                # terminated job actually contained at runtime (handles cascading
                # replacements correctly via the lineage map). op ids are rewritten
                # to use the new job's prefix to avoid colliding with the terminated
                # job's already-registered ops in env.graph.
                if _next_idx[0] >= len(pregenerated_jobs):
                    return None
                pool_entry = pregenerated_jobs[_next_idx[0]]
                term = _pregen_by_id.get(terminated_id)
                if term is None:
                    return None
                new_id = pool_entry['id']
                cloned_ops = copy.deepcopy(term['operations'])
                for k, op in enumerate(cloned_ops, start=1):
                    op['id'] = f"{new_id}_op{k}"
                clone_data = {
                    'id': new_id,
                    'type': term['type'],
                    'priority': term.get('priority', 'normal'),
                    'operations': cloned_ops,
                    'arrival_time': 0.0,
                }
                # Register clone content under the new id so a future
                # termination of this clone clones the SAME content (lineage).
                _pregen_by_id[new_id] = clone_data
                clone_job = JobClass(
                    id=new_id,
                    job_type=clone_data['type'],
                    operations=copy.deepcopy(cloned_ops),
                    arrival_time=env.current_time,
                    priority=clone_data['priority'],
                )
                env.job_buffer.add_job(clone_job)
                env.job_counter += 1
                _next_idx[0] += 1
                print(f"    [REPLAY-ABORT]  replace {terminated_id}  {new_id} "
                      f"(identical clone, arrival={env.current_time:.2f}, no BO delay)")
                return new_id

            env._abort_replace = _replay_abort_replace
        else:
            env.reset(seed=seed)

        env.machines.sort(key=lambda m: m.id)
        env.machines_dict = {m.id: m for m in env.machines}

        # Patch the completion check so cancelled jobs are excluded from
        # graph.is_complete() — aborted ops are left unscheduled/uncompleted
        # so they show as [-] in the output rather than fake-completed.
        def _patched_check_done():
            if env.job_counter < env.n_jobs:
                return False
            if len(env.job_buffer) > 0:
                return False
            if env.graph:
                for op in env.graph.operations.values():
                    if op.job_id in cancelled_jobs:
                        continue
                    if not op.is_completed:
                        return False
            return True
        env._check_all_jobs_completed = _patched_check_done

        step_count = 0
        max_steps  = 100000

        print(f"\n{'=' * 60}")
        print(f"Running Episode — FCFS + Tmax-Abort  (seed={seed})")
        print(f"{'=' * 60}")

        while not env._check_all_jobs_completed() and step_count < max_steps:
            step_count += 1

            if len(env.job_buffer) > 0:
                new_jobs = list(env.job_buffer.buffer)
                env._build_static_subproblem(new_jobs)
                for job in new_jobs:
                    if job in env.job_buffer.buffer:
                        env.job_buffer.buffer.remove(job)

            t0 = time.time()
            while _chemos_dispatch_tick_tmax_abort(env, schedule_sequence,
                                                   cancelled_jobs,
                                                   abort_replace_pairs=abort_replace_pairs):
                pass
            time_breakdown['dispatch'] += time.time() - t0

            t0 = time.time()
            env._advance_time_and_inject_jobs()
            time_breakdown['time_advance'] += time.time() - t0

        total_time     = time.time() - episode_start
        final_makespan = env.graph.makespan if env.graph else 0.0

        job_details   = self._collect_job_details(env)
        aborted_count = len(cancelled_jobs)

        print(f"\n{'=' * 60}")
        print(f"FCFS+Tmax-Abort Episode Completed  (seed={seed})")
        print(f"  Total steps    : {step_count}")
        print(f"  Final makespan : {final_makespan:.2f}")
        print(f"  Aborted jobs   : {aborted_count}  (= Tmax violations; each replaced by a new job)")
        print(f"  Dispatch time  : {time_breakdown['dispatch']:.2f}s")
        print(f"  Wall-clock time: {total_time:.2f}s")
        print(f"{'=' * 60}\n")

        # Print one-to-one terminatereplace pairing for visibility
        if abort_replace_pairs:
            print("  Terminate  Replace pairs:")
            for p in abort_replace_pairs:
                print(f"    {p['terminated_job']:>10}    {p['replacement_job']}")

        return {
            'makespan':             final_makespan,
            'job_details':          job_details,
            'tmax_violations':      aborted_count,   # one violation per aborted experiment
            'aborted_jobs':         aborted_count,
            'terminated_jobs':      [p['terminated_job']  for p in abort_replace_pairs],
            'replacement_jobs':     [p['replacement_job'] for p in abort_replace_pairs],
            'abort_replace_pairs':  abort_replace_pairs,
            'total_solve_time':     time_breakdown['dispatch'],
            'total_time':           total_time,
            'time_breakdown':       time_breakdown,
            'schedule_sequence':    schedule_sequence,
            'release_history':      []
        }

    def _check_tmax_violations_excluding(self, env, excluded_job_ids):
        """Like _check_tmax_violations but skips ops belonging to excluded_job_ids."""
        violations = 0
        details = []
        for constraint in env.tmax_constraints:
            from_step, to_step, max_interval = constraint
            for op in env.graph.operations.values():
                if op.job_id in excluded_job_ids:
                    continue
                if op.op_type == to_step and op.is_scheduled:
                    candidates = [o for o in env.graph.operations.values()
                                  if o.job_id == op.job_id
                                  and o.op_type == from_step
                                  and o.op_index < op.op_index]
                    if candidates:
                        pred = max(candidates, key=lambda o: o.op_index)
                        if pred.is_scheduled or pred.is_completed:
                            interval = op.scheduled_start - pred.scheduled_end
                            if interval > max_interval + 1e-4:
                                violations += 1
                                details.append({
                                    'job_id':       op.job_id,
                                    'constraint':   f'{from_step}{to_step}',
                                    'max_interval': max_interval,
                                    'actual':       round(interval, 3),
                                    'excess':       round(interval - max_interval, 3),
                                })
        return violations, details

    def _check_tmax_violations(self, env):
        """Count Tmax violations and return (count, detail_list)."""
        violations = 0
        details = []
        for constraint in env.tmax_constraints:
            from_step, to_step, max_interval = constraint
            for op in env.graph.operations.values():
                if op.op_type == to_step and op.is_scheduled:
                    candidates = [o for o in env.graph.operations.values()
                                  if o.job_id == op.job_id
                                  and o.op_type == from_step
                                  and o.op_index < op.op_index]
                    if candidates:
                        pred = max(candidates, key=lambda o: o.op_index)
                        if pred.is_scheduled or pred.is_completed:
                            interval = op.scheduled_start - pred.scheduled_end
                            if interval > max_interval + 1e-4:
                                violations += 1
                                details.append({
                                    'job_id':      op.job_id,
                                    'constraint':  f'{from_step}{to_step}',
                                    'max_interval': max_interval,
                                    'actual':       round(interval, 3),
                                    'excess':       round(interval - max_interval, 3),
                                })
        return violations, details

    def _print_tmax_details(self, method_label, violations, details):
        """Print per-violation breakdown when violations > 0."""
        if violations == 0:
            return
        print(f"  *** {method_label} Tmax violation details ({violations} total) ***")
        # group by constraint type for readability
        from collections import defaultdict
        by_constraint = defaultdict(list)
        for d in details:
            by_constraint[d['constraint']].append(d)
        for cname, entries in sorted(by_constraint.items()):
            print(f"    [{cname}]  max={entries[0]['max_interval']} min")
            for e in sorted(entries, key=lambda x: -x['excess']):
                print(f"      {e['job_id']:12s}  actual={e['actual']:.2f}  "
                      f"excess=+{e['excess']:.2f} min")

    def _collect_job_details(self, env):
        """job"""
        job_details = {}

        for op in env.graph.operations.values():
            if op.job_id not in job_details:
                job_details[op.job_id] = {
                    'ops': [],
                    'completed': 0,
                    'scheduled': 0,
                    'total': 0
                }

            job_details[op.job_id]['ops'].append({
                'op_id': op.id,
                'op_index': op.op_index,
                'is_completed': op.is_completed,
                'is_scheduled': op.is_scheduled,
                'scheduled_start': op.scheduled_start if op.is_scheduled else None,
                'scheduled_end': op.scheduled_end if op.is_scheduled else None,
                'scheduled_machine': op.scheduled_machine if op.is_scheduled else None
            })
            job_details[op.job_id]['total'] += 1

            if op.is_scheduled:
                job_details[op.job_id]['scheduled'] += 1

            if op.is_completed:
                job_details[op.job_id]['completed'] += 1

        return job_details

    def run_comparison(self, num_episodes=1, start_seed=None,
                       ga_pop_size=20, ga_patience=20):
        """"""
        print(f"\n{'#' * 60}")
        print(f"Starting Three-Way Comparison")
        print(f"  Episodes: {num_episodes}")
        print(f"{'#' * 60}\n")

        all_results = []

        for episode_idx in range(num_episodes):
            seed = np.random.randint(0, 400000000)

            print(f"\n{'=' * 80}")
            print(f"Episode {episode_idx + 1}/{num_episodes} - Seed: {seed}")
            print(f"{'=' * 80}")

            episode_results = {
                'episode_idx': episode_idx,
                'seed': seed,
                'timestamp': datetime.now().isoformat()
            }

            # episodejobs
            pregenerated_jobs, bo_delays = self._pre_generate_jobs(seed, n_jobs=80)

            for method in ['ERC-FSS', 'Conventional DRL', 'GA', 'MILP', 'SPT']:

                try:
                    result = self.run_episode_with_method(
                        method, seed,
                        ga_pop_size=ga_pop_size, ga_patience=ga_patience,
                        pregenerated_jobs=pregenerated_jobs,
                        bo_delays=bo_delays
                    )
                    episode_results[method] = result
                except Exception as e:
                    print(f" {method.upper()} failed: {e}")
                    import traceback
                    traceback.print_exc()
                    episode_results[method] = {
                        'error': str(e),
                        'makespan': float('inf')
                    }

            # FCFS + real-time Tmax abort (abort failed jobs, generate replacements)
            try:
                result = self.run_episode_chemos_tmax_abort(
                    seed=seed,
                    pregenerated_jobs=pregenerated_jobs,
                    bo_delays=bo_delays,
                )
                episode_results['ChemOS Built-in Dispatch'] = result
            except Exception as e:
                print(f" FCFS+Tmax-Abort failed: {e}")
                import traceback
                traceback.print_exc()
                episode_results['ChemOS Built-in Dispatch'] = {
                    'error': str(e),
                    'makespan': float('inf')
                }

            all_results.append(episode_results)

            # episode
            self._save_episode_result(episode_results, episode_idx)

        #
        self._save_summary(all_results)

        #
        self._print_final_comparison(all_results)

    def _save_episode_result(self, episode_result, episode_idx):
        """episode"""
        filename = f"episode_{episode_idx:03d}_seed_{episode_result['seed']}.json"
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(episode_result, f, indent=2, ensure_ascii=False)

        print(f" Episode {episode_idx} results saved to: {filename}")

    def _save_summary(self, all_results):
        """"""
        summary = {
            'total_episodes': len(all_results),
            'timestamp': datetime.now().isoformat(),
            'statistics': {}
        }

        for method in ['ERC-FSS', 'Conventional DRL', 'GA', 'MILP', 'SPT', 'ChemOS Built-in Dispatch']:
            makespans = []
            violations = []
            times = []

            for result in all_results:
                if method in result and 'error' not in result[method]:
                    makespans.append(result[method]['makespan'])
                    violations.append(result[method]['tmax_violations'])
                    times.append(result[method]['total_solve_time'])

            if makespans:
                summary['statistics'][method] = {
                    'makespan': {
                        'mean': float(np.mean(makespans)),
                        'std': float(np.std(makespans)),
                        'min': float(np.min(makespans)),
                        'max': float(np.max(makespans))
                    },
                    'tmax_violations': {
                        'mean': float(np.mean(violations)),
                        'total': int(np.sum(violations))
                    },
                    'solve_time': {
                        'mean': float(np.mean(times)),
                        'std': float(np.std(times)),
                        'min': float(np.min(times)),
                        'max': float(np.max(times))
                    }
                }

        # JSON
        filepath = os.path.join(self.output_dir, 'summary.json')
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n Summary saved to: summary.json")

    def _print_final_comparison(self, all_results):
        """"""
        print(f"\n{'' * 40}")
        print(f"FINAL COMPARISON SUMMARY")
        print(f"{'=' * 80}")
        for method in ['ERC-FSS', 'Conventional DRL', 'GA', 'MILP', 'SPT', 'ChemOS Built-in Dispatch']:
            makespans = []
            violations = []
            times = []

            for result in all_results:
                if method in result and 'error' not in result[method]:
                    makespans.append(result[method]['makespan'])
                    violations.append(result[method]['tmax_violations'])
                    times.append(result[method]['total_solve_time'])

            if makespans:
                print(f"\n {method.upper()}:")
                print(f"  Makespan: {np.mean(makespans):.2f} ± {np.std(makespans):.2f}")
                print(f"  Tmax Violations: {np.sum(violations)} (avg: {np.mean(violations):.2f})")
                print(f"  Solve Time: {np.mean(times):.2f}s ± {np.std(times):.2f}s")
                if method == 'ChemOS Built-in Dispatch':
                    aborts = [result[method].get('aborted_jobs', 0)
                              for result in all_results
                              if method in result and 'error' not in result[method]]
                    if aborts:
                        print(f"  Aborted Jobs: {sum(aborts)} total (avg: {np.mean(aborts):.2f}/episode)")

        # gap
        fss_ms = [r['ERC-FSS']['makespan'] for r in all_results
                       if 'ERC-FSS' in r and 'error' not in r['ERC-FSS']]
        baseline_ms = [r['Conventional DRL']['makespan'] for r in all_results
                       if 'Conventional DRL' in r and 'error' not in r['Conventional DRL']]
        ga_ms = [r['ga']['makespan'] for r in all_results
                 if 'ga' in r and 'error' not in r['ga']]
        ga100_ms = [r['GA']['makespan'] for r in all_results
                    if 'GA' in r and 'error' not in r['GA']]
        milp_ms = [r['milp']['makespan'] for r in all_results
                   if 'milp' in r and 'error' not in r['milp']]
        milp_nl_ms = [r['MILP']['makespan'] for r in all_results
                      if 'MILP' in r and 'error' not in r['MILP']]

        if fss_ms and baseline_ms:
            gap = (np.mean(fss_ms) - np.mean(baseline_ms)) / np.mean(baseline_ms) * 100
            print(f"\n FSS vs DRL_Baseline Gap: {gap:+.2f}%")
        if fss_ms and ga_ms:
            gap = (np.mean(fss_ms) - np.mean(ga_ms)) / np.mean(ga_ms) * 100
            print(f" FSS vs GA Gap: {gap:+.2f}%")
        if fss_ms and ga100_ms:
            gap = (np.mean(fss_ms) - np.mean(ga100_ms)) / np.mean(ga100_ms) * 100
            print(f" FSS vs ga100 Gap: {gap:+.2f}%")
        if fss_ms and milp_ms:
            gap = (np.mean(fss_ms) - np.mean(milp_ms)) / np.mean(milp_ms) * 100
            print(f" FSS vs milp Gap: {gap:+.2f}%")
        if fss_ms and milp_nl_ms:
            gap = (np.mean(fss_ms) - np.mean(milp_nl_ms)) / np.mean(milp_nl_ms) * 100
            print(f" FSS vs milp_NOLIMIT Gap: {gap:+.2f}%")
        if baseline_ms and milp_ms:
            gap = (np.mean(baseline_ms) - np.mean(milp_ms)) / np.mean(milp_ms) * 100
            print(f" DRL_Baseline vs milp Gap: {gap:+.2f}%")
        if milp_ms and milp_nl_ms:
            gap = (np.mean(milp_ms) - np.mean(milp_nl_ms)) / np.mean(milp_nl_ms) * 100
            print(f" milp(60s) vs milp_NOLIMIT Gap: {gap:+.2f}%")
        if ga_ms and milp_ms:
            gap = (np.mean(ga_ms) - np.mean(milp_ms)) / np.mean(milp_ms) * 100
            print(f" GA vs milp Gap: {gap:+.2f}%")
        if ga100_ms and milp_ms:
            gap = (np.mean(ga100_ms) - np.mean(milp_ms)) / np.mean(milp_ms) * 100
            print(f" ga100 vs milp Gap: {gap:+.2f}%")
        if ga_ms and ga100_ms:
            gap = (np.mean(ga100_ms) - np.mean(ga_ms)) / np.mean(ga_ms) * 100
            print(f" ga100 vs GA Gap: {gap:+.2f}%")



        fcfs_abort_ms = [r['ChemOS Built-in Dispatch']['makespan'] for r in all_results
                         if 'ChemOS Built-in Dispatch' in r and 'error' not in r['ChemOS Built-in Dispatch']]
        if fss_ms and fcfs_abort_ms:
            gap = (np.mean(fss_ms) - np.mean(fcfs_abort_ms)) / np.mean(fcfs_abort_ms) * 100
            print(f"\n FSS vs FCFS+Abort Gap: {gap:+.2f}%")
        if milp_nl_ms and fcfs_abort_ms:
            gap = (np.mean(milp_nl_ms) - np.mean(fcfs_abort_ms)) / np.mean(fcfs_abort_ms) * 100
            print(f" milp_NOLIMIT vs FCFS+Abort Gap: {gap:+.2f}%")

        print(f"{'' * 40}\n")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Three-Way Comparison')
    default_config = os.path.join(project_root, 'configs', 'sdl_config.yaml')
    default_model = os.path.join(project_root, 'training', 'saved_models', 'ERC_FSS.pt')
    default_output = os.path.join(project_root, 'comparison_results')

    parser.add_argument('--config', type=str,
                        default=default_config,
                        help='Environment config path')
    parser.add_argument('--model', type=str,
                        default=default_model,
                        help='Trained FSS model path')
    parser.add_argument('--num_episodes', type=int, default=5,
                        help='Number of test episodes')
    parser.add_argument('--seed', type=int, default=np.random.randint(0, 100000) + np.random.randint(0, 1000) * 100000,
                        help='Start seed for testing')
    parser.add_argument('--output_dir', type=str, default=default_output,
                        help='Output directory for results')
    parser.add_argument('--ga_pop_size', type=int, default=20,
                        help='GA population size (default 20)')
    parser.add_argument('--ga_patience', type=int, default=20,
                        help='Generations without improvement before GA stops (default 20)')
    parser.add_argument('--baseline_model', type=str,
                        default=os.path.join(project_root, 'DRL_baseline', 'training',
                                             'saved_models_lk', 'DRL_BASELINE.pt'),
                        help='DRL baseline model path (in DRL_baseline/training/saved_models_lk/)')

    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f" Config not found: {args.config}")
        return

    if not os.path.exists(args.model):
        print(f" Model not found: {args.model}")
        return

    # baseline model is optional — skip if not found
    baseline_model_path = args.baseline_model if os.path.exists(args.baseline_model) else None
    if baseline_model_path is None:
        print(f" DRL baseline model not found at {args.baseline_model}, skipping drl_baseline method")

    runner = ThreeWayComparison(
        env_config_path=args.config,
        model_path=args.model,
        output_dir=args.output_dir,
        ga_pop_size=args.ga_pop_size,
        ga_patience=args.ga_patience,
        baseline_model_path=baseline_model_path
    )

    runner.run_comparison(
        num_episodes=args.num_episodes,
        start_seed=args.seed,
        ga_pop_size=args.ga_pop_size,
        ga_patience=args.ga_patience
    )


# ============================================================================
# ERC component ablation
#
# Every arm below is an identical FSS + ERC progressive-release run; the
# only differences are (a) which adapter variant builds the state/masks/reward
# (full vs one component ablated) and (b) which trained checkpoint drives it.
# All scheduling machinery is reused verbatim from ThreeWayComparison above.
# ============================================================================

def _resolve_checkpoint(path):
    """Return the checkpoint path if it exists, else None."""
    if path and os.path.exists(path):
        return path
    return None


class AblationComparison(ThreeWayComparison):
    """ERC ablation runner. Reuses every ThreeWayComparison method; overrides
    only the arms (model loading + per-arm adapter dispatch + reporting)."""

    GROUPS = ['erc_full', 'erc_no_forward', 'erc_no_backward', 'erc_no_urgency']

    # group -> adapter class; erc_full uses the stock adapter.
    GROUP_ADAPTER = {
        'erc_full':        SDLEnvAdapter,
        'erc_no_forward':  AblationAdapterNoForward,
        'erc_no_backward': AblationAdapterNoBackward,
        'erc_no_urgency':  AblationAdapterNoUrgency,
    }

    def __init__(self, env_config_path, model_paths, output_dir='./ablation_results',
                 ga_pop_size=20, ga_patience=20):
        # model_paths: dict {group -> checkpoint path}
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        with open(env_config_path, 'r', encoding='utf-8') as f:
            self.env_config_raw = yaml.safe_load(f)

        from configs import Config
        self.env_config = Config(self.env_config_raw)

        self.env = SDLEnv(self.env_config)

        # Single shared PPO config (identical architecture across all arms).
        self.ppo_config = get_default_config()
        self.ppo_config.n_machines = len(self.env.machines)
        self.ppo_config.n_j = self.env_config_raw.get('env_config', {}).get('n_jobs', 80)
        self.ppo_config.device = 'cpu'
        self.device = torch.device('cpu')

        self.ga_pop_size = ga_pop_size
        self.ga_patience = ga_patience

        # DRL baseline disabled for the ablation study.
        self.ppo_baseline = None
        self.baseline_ppo_config = None

        # Load one PPO per arm (same architecture, arm-specific weights).
        self.group_ppo = {}
        for g in self.GROUPS:
            self.group_ppo[g] = self._load_ppo_for_group(g, model_paths[g])

        # group -> (adapter_class, ppo)
        self.group_specs = {
            g: (self.GROUP_ADAPTER[g], self.group_ppo[g]) for g in self.GROUPS
        }

        # Keep .ppo / .adapter pointing at the full arm for inherited code.
        self.ppo = self.group_ppo['erc_full']
        self.adapter = SDLEnvAdapter(self.env, self.ppo_config)

        print(f"\n{'#' * 60}")
        print(f"ERC Ablation Comparison Initialized")
        print(f"  Machines: {len(self.env.machines)}")
        print(f"  Output Dir: {self.output_dir}")
        print(f"  Arms: {', '.join(self.GROUPS)}")
        print(f"{'#' * 60}\n")

    def _load_ppo_for_group(self, group, model_path):
        """Build a PPO and load the arm's checkpoint."""
        print(f" Loading [{group}] model from: {model_path}")
        ppo = PPO(self.ppo_config)
        checkpoint = torch.load(model_path, map_location='cpu')
        ppo.policy_job.load_state_dict(checkpoint['job_actor'])
        ppo.policy_mch.load_state_dict(checkpoint['mch_actor'])
        ppo.policy_old_job.load_state_dict(checkpoint['job_actor'])
        ppo.policy_old_mch.load_state_dict(checkpoint['mch_actor'])
        ppo.policy_job.eval()
        ppo.policy_old_job.eval()
        ppo.policy_mch.eval()
        ppo.policy_old_mch.eval()
        print(f"   [{group}] loaded.")
        return ppo

    def run_episode_with_method(self, method='erc_full', seed=None,
                                ga_pop_size=20, ga_patience=20,
                                pregenerated_jobs=None, bo_delays=None):
        """Run one ablation arm by rebinding the module's SDLEnvAdapter symbol
        to the arm's adapter class and pointing self.ppo at the arm's
        checkpoint, then delegating to the inherited 'ERC-FSS' episode runner."""
        adapter_cls, ppo = self.group_specs[method]

        global SDLEnvAdapter
        saved_adapter_cls = SDLEnvAdapter
        saved_ppo = self.ppo
        SDLEnvAdapter = adapter_cls
        self.ppo = ppo
        try:
            return super().run_episode_with_method(
                'ERC-FSS', seed,
                ga_pop_size=ga_pop_size, ga_patience=ga_patience,
                pregenerated_jobs=pregenerated_jobs, bo_delays=bo_delays
            )
        finally:
            SDLEnvAdapter = saved_adapter_cls
            self.ppo = saved_ppo

    def run_comparison(self, num_episodes=1, start_seed=None,
                       ga_pop_size=20, ga_patience=20):
        """Loop episodes; for each, run all four arms on identical
        pre-generated jobs (same seed) so the comparison is controlled."""
        print(f"\n{'#' * 60}")
        print(f"Starting ERC Ablation Comparison")
        print(f"  Episodes: {num_episodes}")
        print(f"{'#' * 60}\n")

        all_results = []

        for episode_idx in range(num_episodes):
            seed = np.random.randint(0, 400000000)

            print(f"\n{'=' * 80}")
            print(f"Episode {episode_idx + 1}/{num_episodes} - Seed: {seed}")
            print(f"{'=' * 80}")

            episode_results = {
                'episode_idx': episode_idx,
                'seed': seed,
                'timestamp': datetime.now().isoformat()
            }

            # identical pre-generated jobs for every arm (controlled comparison)
            pregenerated_jobs, bo_delays = self._pre_generate_jobs(seed, n_jobs=80)

            for group in self.GROUPS:
                print(f"\n{'-' * 60}")
                print(f"Running ablation arm: {group.upper()}  "
                      f"(adapter={self.GROUP_ADAPTER[group].__name__})")
                print(f"{'-' * 60}")
                try:
                    result = self.run_episode_with_method(
                        group, seed,
                        ga_pop_size=ga_pop_size, ga_patience=ga_patience,
                        pregenerated_jobs=pregenerated_jobs,
                        bo_delays=bo_delays
                    )
                    episode_results[group] = result
                except Exception as e:
                    print(f" {group.upper()} failed: {e}")
                    import traceback
                    traceback.print_exc()
                    episode_results[group] = {
                        'error': str(e),
                        'makespan': float('inf')
                    }

            all_results.append(episode_results)
            self._save_episode_result(episode_results, episode_idx)

        self._save_summary(all_results)
        self._print_final_comparison(all_results)

    def _save_summary(self, all_results):
        summary = {
            'total_episodes': len(all_results),
            'timestamp': datetime.now().isoformat(),
            'statistics': {}
        }

        for group in self.GROUPS:
            makespans, violations, times = [], [], []
            for result in all_results:
                if group in result and 'error' not in result[group]:
                    makespans.append(result[group]['makespan'])
                    violations.append(result[group]['tmax_violations'])
                    times.append(result[group]['total_solve_time'])

            if makespans:
                summary['statistics'][group] = {
                    'makespan': {
                        'mean': float(np.mean(makespans)),
                        'std': float(np.std(makespans)),
                        'min': float(np.min(makespans)),
                        'max': float(np.max(makespans))
                    },
                    'tmax_violations': {
                        'mean': float(np.mean(violations)),
                        'total': int(np.sum(violations))
                    },
                    'solve_time': {
                        'mean': float(np.mean(times)),
                        'std': float(np.std(times)),
                        'min': float(np.min(times)),
                        'max': float(np.max(times))
                    }
                }

        filepath = os.path.join(self.output_dir, 'summary.json')
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n Summary saved to: summary.json")

    def _print_final_comparison(self, all_results):
        print(f"\n{'=' * 80}")
        print(f"ERC ABLATION FINAL COMPARISON")
        print(f"{'=' * 80}")

        group_ms = {}
        for group in self.GROUPS:
            makespans, violations, times = [], [], []
            for result in all_results:
                if group in result and 'error' not in result[group]:
                    makespans.append(result[group]['makespan'])
                    violations.append(result[group]['tmax_violations'])
                    times.append(result[group]['total_solve_time'])
            group_ms[group] = makespans
            if makespans:
                print(f"\n {group.upper()}:")
                print(f"  Makespan: {np.mean(makespans):.2f} +/- {np.std(makespans):.2f}")
                print(f"  Tmax Violations: {np.sum(violations)} (avg: {np.mean(violations):.2f})")
                print(f"  Solve Time: {np.mean(times):.2f}s +/- {np.std(times):.2f}s")

        # Each ablation arm vs the full architecture (positive = ablation worse).
        full_ms = group_ms.get('erc_full', [])
        if full_ms:
            full_mean = np.mean(full_ms)
            for group in self.GROUPS:
                if group == 'erc_full':
                    continue
                ms = group_ms.get(group, [])
                if ms and full_mean > 0:
                    gap = (np.mean(ms) - full_mean) / full_mean * 100
                    print(f"\n {group.upper()} vs ERC_FULL makespan gap: {gap:+.2f}%")

        print(f"{'=' * 80}\n")


def _ablation_main():
    import argparse

    parser = argparse.ArgumentParser(description='ERC Component Ablation Comparison')
    default_config = os.path.join(project_root, 'configs', 'sdl_config.yaml')
    default_full   = os.path.join(project_root, 'training', 'saved_models', 'ERC_FSS.pt')
    default_nf     = os.path.join(project_root, 'saved_models_ablation_no_forward',  'noforward.pt')
    default_nb     = os.path.join(project_root, 'saved_models_ablation_no_backward', 'nobackward.pt')
    default_nu     = os.path.join(project_root, 'saved_models_ablation_no_urgency',  'nourgency.pt')
    default_output = os.path.join(project_root, 'ablation_results')

    parser.add_argument('--config', type=str, default=default_config,
                        help='Environment config path')
    parser.add_argument('--model_full', type=str, default=default_full,
                        help='ERC-full checkpoint (== the existing trained model)')
    parser.add_argument('--model_no_forward', type=str, default=default_nf,
                        help='ERC-no-forward checkpoint file')
    parser.add_argument('--model_no_backward', type=str, default=default_nb,
                        help='ERC-no-backward checkpoint file')
    parser.add_argument('--model_no_urgency', type=str, default=default_nu,
                        help='ERC-no-urgency checkpoint file')
    parser.add_argument('--num_episodes', type=int, default=1,
                        help='Number of test episodes')
    parser.add_argument('--seed', type=int,
                        default=np.random.randint(0, 100000) + np.random.randint(0, 1000) * 100000,
                        help='Start seed for testing')
    parser.add_argument('--output_dir', type=str, default=default_output,
                        help='Output directory for results')
    parser.add_argument('--ga_pop_size', type=int, default=20)
    parser.add_argument('--ga_patience', type=int, default=20)

    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f" Config not found: {args.config}")
        return

    raw_paths = {
        'erc_full':        args.model_full,
        'erc_no_forward':  args.model_no_forward,
        'erc_no_backward': args.model_no_backward,
        'erc_no_urgency':  args.model_no_urgency,
    }

    model_paths = {}
    missing = []
    for g in AblationComparison.GROUPS:
        resolved = _resolve_checkpoint(raw_paths[g])
        if resolved is None:
            missing.append((g, raw_paths[g]))
        else:
            model_paths[g] = resolved

    if missing:
        print(" Missing model checkpoint(s) -- train the ablation models first:")
        for g, p in missing:
            print(f"   {g:16s}: {p}")
        return

    print(" Resolved ablation checkpoints:")
    for g in AblationComparison.GROUPS:
        print(f"   {g:16s} -> {model_paths[g]}")

    runner = AblationComparison(
        env_config_path=args.config,
        model_paths=model_paths,
        output_dir=args.output_dir,
        ga_pop_size=args.ga_pop_size,
        ga_patience=args.ga_patience,
    )

    runner.run_comparison(
        num_episodes=args.num_episodes,
        start_seed=args.seed,
        ga_pop_size=args.ga_pop_size,
        ga_patience=args.ga_patience
    )


if __name__ == "__main__":
    _ablation_main()