import copy
import importlib.util as _ilu
import json
import os
import sys
import sys as _sys
import time
import types as _types
from datetime import datetime
from itertools import combinations

import numpy as np
import torch
import yaml

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
repo_root = os.path.dirname(project_root)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'training'))

from ppo import PPO
from config import get_default_config
from env_adapter import SDLEnvAdapter
from envs.sdl_env import SDLEnv
from utils import profiler
from utils.agent_utils import greedy_select_action2
from utils.feasibility import check_release_feasibility
from utils.mb_agg import g_pool_cal
from utils.profiler import perf_counter

METHODS = ['ERC-FSS', 'Conventional DRL', 'GA', 'MILP', 'SPT']
CHEMOS_DISPATCH = 'ChemOS Built-in Dispatch'
ALL_METHODS = METHODS + [CHEMOS_DISPATCH]


def _load_module_from_path(module_name, file_path):
    spec = _ilu.spec_from_file_location(module_name, file_path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_baseline_root     = os.path.join(project_root, 'DRL_baseline')
_baseline_training = os.path.join(_baseline_root, 'training')
_baseline_models_dir = os.path.join(_baseline_root, 'models')

_bl_graphcnn_mod = _load_module_from_path(
    'bl_graphcnn', os.path.join(_baseline_models_dir, 'graphcnn.py'))
_bl_mlp_mod = _load_module_from_path(
    'bl_mlp', os.path.join(_baseline_models_dir, 'mlp.py'))

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

_bl_actor_mod = _load_module_from_path(
    'bl_ppo_actor', os.path.join(_baseline_models_dir, 'ppo_actor.py'))
_sys.modules['models.ppo_actor'] = _bl_actor_mod

_bl_config_mod  = _load_module_from_path(
    'bl_config',  os.path.join(_baseline_training, 'config.py'))
_bl_adapter_mod = _load_module_from_path(
    'bl_env_adapter', os.path.join(_baseline_training, 'env_adapter.py'))
_bl_ppo_mod     = _load_module_from_path(
    'bl_ppo', os.path.join(_baseline_training, 'ppo.py'))

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


def _verify_tmax_feasibility(env):
    t0 = perf_counter()
    ok = _tmax_satisfied(env)
    profiler.add_time('tmax_verification', perf_counter() - t0)
    return ok


def _tmax_satisfied(env):
    if not hasattr(env, 'tmax_constraints') or not env.tmax_constraints:
        return True

    all_ops = list(env.graph.operations.values())
    for fs, ts, mx in env.tmax_constraints:
        for op in all_ops:
            if op.op_type == ts and (op.is_scheduled or op.is_completed):
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


def global_reschedule(env, adapter, ppo, method='ERC-FSS', device='cpu',
                      ga_pop_size=20, ga_patience=20):
    time_start = perf_counter()

    rollback_start = perf_counter()
    machines_to_update = set()
    rollback_count = 0

    for op in env.graph.operations.values():
        if not op.is_completed and op.is_scheduled:
            if op.scheduled_start > env.current_time + 1e-4:
                rollback_count += 1
                if op.scheduled_machine:
                    machines_to_update.add(op.scheduled_machine)

                op.is_scheduled = False
                op.scheduled_machine = None
                op.scheduled_start = 0.0
                op.scheduled_end = 0.0

                if op.id in env.graph.scheduled_ops:
                    env.graph.scheduled_ops.remove(op.id)

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
    rollback_time = perf_counter() - rollback_start

    schedule_start = perf_counter()

    if method == 'ERC-FSS':
        result = _schedule_with_fss(env, adapter, ppo, device)
    elif method == 'Conventional DRL':
        result = _schedule_with_drl_baseline(env, adapter, ppo, device)
    elif method == 'GA':
        result = _schedule_with_ga(env, pop_size=ga_pop_size, patience=ga_patience)
    elif method == 'MILP':
        result = _schedule_with_milp(env, time_limit=60000, mip_gap_abs=None)
    elif method == 'SPT':
        result = _schedule_with_spt(env)
    else:
        raise ValueError(f"Unknown method: {method}")

    schedule_time = perf_counter() - schedule_start
    total_time = perf_counter() - time_start
    profiler.add_time('scheduling', schedule_time)

    result['rollback_count'] = rollback_count
    result['scheduled_count'] = len(result.get('schedule', []))
    result['time_breakdown'] = {
        'rollback': rollback_time,
        'scheduling': schedule_time,
        'total': total_time
    }

    return result


def _schedule_with_fss(env, adapter, ppo, device):
    state = adapter.get_state_for_fss()
    if state is None:
        return {
            'success': False,
            'makespan': float('inf'),
            'schedule': []
        }

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
            t_inf0 = perf_counter()
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
            profiler.add_time('policy_inference', perf_counter() - t_inf0)

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


def _schedule_with_drl_baseline(env, adapter, ppo, device):
    state = adapter.get_state_for_baseline()
    if state is None:
        return {'success': False, 'makespan': float('inf'), 'schedule': []}

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
    pool = None

    while len(env.graph.eligible_ops) > 0:
        state = adapter.get_state_for_baseline()
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
                    mch_pool=pool,
                    old_policy=True,
                    greedy=True
                )

            pi_mch, pool = ppo.policy_old_mch(
                action_node=action_node,
                hx=hx,
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
    ga_start = time.time()

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

    pending_ops = [op_id for op_id, info in ops_info.items() if not info['is_fixed']]
    if not pending_ops:
        return {'success': True, 'makespan': env.graph.makespan, 'schedule': []}

    fixed_sched_end = {
        op_id: info['scheduled_end']
        for op_id, info in ops_info.items()
        if info['is_fixed'] and info['scheduled_end'] is not None
    }

    def decode(priority_order, machine_assign):
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

    def make_spt_seed():
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

    seed_prio, seed_assign = make_spt_seed()
    population = [(seed_prio, seed_assign)] + [make_random() for _ in range(pop_size - 1)]

    best_ms = float('inf')
    best_ind = None
    generation = 0
    eval_count = 0
    no_improve = 0

    while no_improve < patience:
        prev_best = best_ms

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

        if best_ms < prev_best - 1e-6:
            no_improve = 0
        else:
            no_improve += 1

    elapsed = time.time() - ga_start
    print(f"  [GA] gen={generation}, evals={eval_count}, "
          f"best_ms={best_ms:.2f}, time={elapsed:.2f}s")

    if best_ind is None or best_ms == float('inf'):
        return {'success': False, 'makespan': float('inf'), 'schedule': []}

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

    if not _verify_tmax_feasibility(env):
        return {'success': False, 'makespan': float('inf'), 'schedule': []}

    return {
        'success': True,
        'makespan': env.graph.makespan,
        'schedule': schedule
    }


def _schedule_with_milp(env, time_limit=30, mip_gap_abs=None):
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
        'makespan': env.graph.makespan,
        'schedule': schedule
    }


def _schedule_with_spt(env):
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
    for constraint in env.tmax_constraints:
        from_step, to_step, max_interval = constraint

        if op.op_type == to_step:
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


def progressive_release(env, adapter, ppo, method='ERC-FSS', device='cpu',
                        ga_pop_size=20, ga_patience=20):
    t0 = perf_counter()
    try:
        return _progressive_release(env, adapter, ppo, method=method, device=device,
                                    ga_pop_size=ga_pop_size, ga_patience=ga_patience)
    finally:
        profiler.add_time('release_total', perf_counter() - t0)
        profiler.count('release_calls')


def _progressive_release(env, adapter, ppo, method='ERC-FSS', device='cpu',
                         ga_pop_size=20, ga_patience=20):
    # Candidate subsets are evaluated in descending order of cardinality; the search
    # stops at the first size with a feasible subset and releases the subset with the
    # minimum makespan.
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

    for k in range(n, 0, -1):
        k_start_time = perf_counter()

        if k == n:
            combinations_to_try = [buffer_jobs]
        else:
            combinations_to_try = list(combinations(buffer_jobs, k))

        total_combos = len(combinations_to_try)
        total_combinations += total_combos

        feasible_results = []

        for idx, jobs_combo in enumerate(combinations_to_try):
            # rapid Gurobi feasibility check before the scheduling step
            t_check0 = perf_counter()
            feasible = check_release_feasibility(env, list(jobs_combo))
            profiler.add_time('feasibility_check', perf_counter() - t_check0)
            profiler.count('candidates_checked')
            if not feasible:
                profiler.count('candidates_infeasible')
                continue

            graph_snapshot = env.graph.snapshot()
            env._build_static_subproblem(list(jobs_combo))
            result = global_reschedule(env, adapter, ppo, method=method, device=device,
                                       ga_pop_size=ga_pop_size, ga_patience=ga_patience)
            profiler.count('candidates_scheduled')

            if result['success']:
                scheduled_snapshot = env.graph.snapshot()
                feasible_results.append({
                    'jobs': list(jobs_combo),
                    'makespan': result['makespan'],
                    'schedule': result['schedule'],
                    'scheduled_snapshot': scheduled_snapshot
                })

            env.graph.restore(graph_snapshot)

        k_time = perf_counter() - k_start_time
        time_per_k[f'k={k}'] = {
            'time': k_time,
            'combinations': total_combos,
            'feasible': len(feasible_results)
        }

        if feasible_results:
            best_result = min(feasible_results, key=lambda x: x['makespan'])

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


def runtime_breakdown(method):
    t = profiler.snapshot()['timers']
    c = profiler.snapshot()['counters']
    check = t.get('feasibility_check', 0.0)
    scheduling = t.get('scheduling', 0.0)
    verification = t.get('tmax_verification', 0.0)
    total = t.get('release_total', 0.0)
    out = {
        'feasibility_check': check,
        'schedule_construction': scheduling - verification,
        'tmax_verification': verification,
        'erc_operations': total - check - scheduling,
        'total': total,
        'release_calls': c.get('release_calls', 0),
        'candidates_checked': c.get('candidates_checked', 0),
        'candidates_infeasible': c.get('candidates_infeasible', 0),
        'candidates_scheduled': c.get('candidates_scheduled', 0),
    }
    if method == 'ERC-FSS':
        state = t.get('state_construction', 0.0)
        mask_total = t.get('mask_total', 0.0)
        forward = t.get('mask_forward', 0.0)
        backward = t.get('mask_backward', 0.0)
        urgency = t.get('urgency_feature', 0.0)
        inference = t.get('policy_inference', 0.0)
        out['fss_detail'] = {
            'urgency_feature': urgency,
            'other_state_construction': state - mask_total - urgency,
            'policy_inference': inference,
            'dynamic_mask_construction': mask_total - forward - backward,
            'forward_feasibility_check': forward,
            'backward_deadline_validation': backward,
            'schedule_update': out['schedule_construction'] - state - inference,
        }
    return out


def _get_ready_ops(env):
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


def _check_tmax_at_dispatch(env, op):
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
    cancelled_jobs.add(job_id)
    env.n_jobs += 1

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
                  f"aborting, generating replacement")
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


class BenchmarkComparison:
    def __init__(self, env_config_path, model_path, output_dir='./outputs/benchmark',
                 ga_pop_size=20, ga_patience=20,
                 baseline_model_path=None):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        with open(env_config_path, 'r', encoding='utf-8') as f:
            self.env_config_raw = yaml.safe_load(f)

        from configs import Config
        self.env_config = Config(self.env_config_raw)

        self.env = SDLEnv(self.env_config)

        self.ppo_config = get_default_config()
        self.ppo_config.n_machines = len(self.env.machines)
        self.ppo_config.n_j = self.env_config_raw.get('env_config', {}).get('n_jobs', 80)
        self.ppo_config.device = 'cpu'
        self.device = torch.device('cpu')

        self.ga_pop_size = ga_pop_size
        self.ga_patience = ga_patience

        self.adapter = SDLEnvAdapter(self.env, self.ppo_config)
        self.ppo = PPO(self.ppo_config)

        self.model_path = model_path
        self._load_model()

        self.baseline_model_path = baseline_model_path
        self.ppo_baseline = None
        self.baseline_ppo_config = None
        if baseline_model_path is not None:
            self._init_baseline()

        print(f"\n{'#' * 60}")
        print(f"Benchmark comparison initialized")
        print(f"  Machines: {len(self.env.machines)}")
        print(f"  Output Dir: {self.output_dir}")
        print(f"  DRL Baseline: {'enabled' if self.ppo_baseline is not None else 'disabled'}")
        print(f"{'#' * 60}\n")

    def _load_model(self):
        print(f"Loading model from: {self.model_path}")

        checkpoint = torch.load(self.model_path, map_location='cpu')
        self.ppo.policy_job.load_state_dict(checkpoint['job_actor'])
        self.ppo.policy_mch.load_state_dict(checkpoint['mch_actor'])
        self.ppo.policy_old_job.load_state_dict(checkpoint['job_actor'])
        self.ppo.policy_old_mch.load_state_dict(checkpoint['mch_actor'])

        self.ppo.policy_job.eval()
        self.ppo.policy_old_job.eval()
        self.ppo.policy_mch.eval()
        self.ppo.policy_old_mch.eval()

        print(f"Model loaded successfully\n")

    def _init_baseline(self):
        print(f"Loading DRL baseline from: {self.baseline_model_path}")

        cfg = get_baseline_config()
        cfg.n_machines = len(self.env.machines)
        cfg.n_j = self.env_config_raw.get('env_config', {}).get('n_jobs', 80)
        cfg.device = 'cpu'
        self.baseline_ppo_config = cfg

        self.baseline_adapter = BaselineSDLEnvAdapter(self.env, cfg)

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

        print(f"DRL baseline loaded successfully "
              f"(input_dim={cfg.input_dim}, n_machines={cfg.n_machines})\n")

    def _pre_generate_jobs(self, seed, n_jobs=80):
        import os
        from case_study.dataset_generator import DatasetGenerator

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(project_root, 'configs', 'sdl_config.yaml')

        gen = DatasetGenerator(config_path=config_path)

        np.random.seed(seed)
        gen.atlas_simulator.rng = np.random.RandomState(seed + 3000000)

        jobs = []

        warmstart_types = ['SDL_experiment'] * 12
        for i, job_type in enumerate(warmstart_types):
            job = gen.generate_job(
                job_id=f"job_{i}",
                job_type=job_type,
                arrival_time=0.0
            )
            jobs.append(job)

        for i in range(12, n_jobs):
            job_type = np.random.choice(['SDL_experiment'])
            job = gen.generate_job(
                job_id=f"job_{i}",
                job_type=job_type,
                arrival_time=0.0
            )
            jobs.append(job)

        bo_rng = np.random.RandomState(seed + 9999999)
        bo_delays = {}
        for i in range(12, n_jobs):
            bo_delays[f"job_{i}"] = float(np.clip(bo_rng.normal(10.0, 2.5), 5.0, 15.0)) / 60.0

        print(f"\n[PRE-GEN] {len(jobs)} jobs (seed={seed})")

        return jobs, bo_delays

    def run_episode_with_method(self, method='ERC-FSS', seed=None,
                                ga_pop_size=20, ga_patience=20,
                                pregenerated_jobs=None,
                                bo_delays=None):
        if seed is None:
            seed = int(np.random.randint(0, 400000000))
        profiler.reset()
        episode_start = time.time()

        env = SDLEnv(self.env_config)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if pregenerated_jobs is not None:
            env.reset(seed=seed, skip_warmstart=True)

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

            _next_idx = [12]
            _bo_delays = bo_delays or {}

            def _replay_try_gen(count=1):
                generated = 0
                for _ in range(count):
                    if _next_idx[0] >= len(pregenerated_jobs):
                        break
                    if env.job_counter >= env.n_jobs:
                        break

                    job_data = pregenerated_jobs[_next_idx[0]]
                    delay = _bo_delays.get(job_data['id'], 10.0 / 60.0)
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
                    print(f"    [REPLAY] {generated} jobs "
                          f"(idx {_next_idx[0]-generated}..{_next_idx[0]-1}, "
                          f"bo_delay={delay*60:.1f}s)")
                else:
                    if env.job_counter >= env.n_jobs:
                        print(f"    [STOP] all {env.n_jobs} experiments generated")
                    else:
                        print(f"    [REPLAY] pre-generated experiments exhausted")

            env._try_generate_new_jobs = _replay_try_gen
            env.bo_delays_map = dict(_bo_delays)

            _pregen_by_id = {jd['id']: jd for jd in pregenerated_jobs}

            def _replay_abort_replace(terminated_id):
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
                print(f"    [REPLAY-ABORT] replace {terminated_id} with {pool_entry['id']} "
                      f"(identical clone, arrival={env.current_time:.2f}, no BO delay)")
                return pool_entry['id']

            env._abort_replace = _replay_abort_replace
        else:
            env.reset(seed=seed)

        env.machines.sort(key=lambda m: m.id)
        env.machines_dict = {m.id: m for m in env.machines}

        adapter = SDLEnvAdapter(env, self.ppo_config)

        if method == 'Conventional DRL':
            if self.ppo_baseline is None:
                raise RuntimeError("DRL baseline not initialized. Pass baseline_model_path to BenchmarkComparison.")
            _adapter = BaselineSDLEnvAdapter(env, self.baseline_ppo_config)
            _ppo = self.ppo_baseline
        else:
            _adapter = adapter
            _ppo = self.ppo

        step_count = 0
        max_steps = 10000

        release_history = []
        schedule_sequence = []
        time_breakdown = {
            'hla_decision': 0.0,
            'scheduling': 0.0,
            'time_advance': 0.0
        }

        print(f"\n{'=' * 60}")
        print(f"Running episode with {method}")
        print(f"{'=' * 60}")

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

            _ga_pop = 50 if method == 'GA' else ga_pop_size
            release_start = perf_counter()
            release_result = progressive_release(
                env, _adapter, _ppo, method=method, device=self.device,
                ga_pop_size=_ga_pop, ga_patience=ga_patience
            )
            time_breakdown['hla_decision'] += perf_counter() - release_start

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

            release_info = {
                'step': step_count,
                'current_time': env.current_time,
                'released_jobs': [j.id for j in release_result['released_jobs']],
                'k_value': release_result.get('k_value', 0),
                'makespan': release_result['makespan'],
                'combinations_tried': release_result.get('combinations_tried', 0)
            }
            release_history.append(release_info)

            for sched in release_result.get('schedule', []):
                schedule_sequence.append({
                    'step': step_count,
                    'start': sched[0],
                    'op_id': sched[1],
                    'machine': sched[2],
                    'end': sched[3]
                })

            released_jobs = release_result['released_jobs']
            for job in released_jobs:
                if job in env.job_buffer.buffer:
                    env.job_buffer.buffer.remove(job)

            advance_start = time.time()
            env._advance_time_and_inject_jobs()
            while len(env.job_buffer) == 0 and env.graph and any(
                op.is_scheduled and not op.is_completed
                for op in env.graph.operations.values()
            ):
                env._advance_time_and_inject_jobs()
            time_breakdown['time_advance'] += time.time() - advance_start

        total_time = time.time() - episode_start
        total_solve_time = time_breakdown['hla_decision']
        final_makespan = env.graph.makespan if env.graph else 0.0

        tmax_violations, tmax_details = self._check_tmax_violations(env)

        job_details = self._collect_job_details(env)
        breakdown = runtime_breakdown(method)

        print(f"\n{'=' * 60}")
        print(f"{method} episode completed")
        print(f"  Total steps: {step_count}")
        print(f"  Final makespan: {final_makespan:.2f}")
        print(f"  Tmax violations: {tmax_violations}")
        self._print_tmax_details(method, tmax_violations, tmax_details)
        print(f"  Computation time: {total_solve_time:.2f}s "
              f"(feasibility check {breakdown['feasibility_check']:.2f}s, "
              f"schedule construction {breakdown['schedule_construction']:.2f}s)")
        print(f"  Total wall-clock time: {total_time:.2f}s")
        print(f"{'=' * 60}\n")

        return {
            'makespan': final_makespan,
            'job_details': job_details,
            'tmax_violations': tmax_violations,
            'total_solve_time': total_solve_time,
            'total_time': total_time,
            'time_breakdown': time_breakdown,
            'runtime_breakdown': breakdown,
            'schedule_sequence': schedule_sequence,
            'release_history': release_history
        }

    def run_episode_chemos_tmax_abort(self, seed, pregenerated_jobs=None, bo_delays=None):
        episode_start        = time.time()
        schedule_sequence    = []
        cancelled_jobs       = set()
        abort_replace_pairs  = []
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
                    print(f"    [REPLAY] {generated} jobs "
                          f"(idx {_next_idx[0]-generated}..{_next_idx[0]-1}, "
                          f"bo_delay={delay_used*60:.1f}s)")

            env._try_generate_new_jobs = _replay_try_gen
            env.bo_delays_map = dict(_bo_delays)

            _pregen_by_id = {jd['id']: jd for jd in pregenerated_jobs}

            def _replay_abort_replace(terminated_id):
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
                print(f"    [REPLAY-ABORT] replace {terminated_id} with {new_id} "
                      f"(identical clone, arrival={env.current_time:.2f}, no BO delay)")
                return new_id

            env._abort_replace = _replay_abort_replace
        else:
            env.reset(seed=seed)

        env.machines.sort(key=lambda m: m.id)
        env.machines_dict = {m.id: m for m in env.machines}

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
        print(f"Running episode with {CHEMOS_DISPATCH}  (seed={seed})")
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
        print(f"{CHEMOS_DISPATCH} episode completed  (seed={seed})")
        print(f"  Total steps    : {step_count}")
        print(f"  Final makespan : {final_makespan:.2f}")
        print(f"  Aborted jobs   : {aborted_count}  (= Tmax violations; each replaced by a new job)")
        print(f"  Dispatch time  : {time_breakdown['dispatch']:.2f}s")
        print(f"  Wall-clock time: {total_time:.2f}s")
        print(f"{'=' * 60}\n")

        if abort_replace_pairs:
            print("  Terminate  Replace pairs:")
            for p in abort_replace_pairs:
                print(f"    {p['terminated_job']:>10}    {p['replacement_job']}")

        return {
            'makespan':             final_makespan,
            'job_details':          job_details,
            'tmax_violations':      aborted_count,
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

    def _check_tmax_violations(self, env):
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
        if violations == 0:
            return
        print(f"  *** {method_label} Tmax violation details ({violations} total) ***")
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

    def run_comparison(self, num_episodes=1, seeds=None, methods=None,
                       ga_pop_size=20, ga_patience=20):
        methods = list(methods) if methods else list(ALL_METHODS)
        if seeds is None:
            seeds = [int(np.random.randint(0, 400000000)) for _ in range(num_episodes)]

        print(f"\n{'#' * 60}")
        print(f"Starting benchmark comparison")
        print(f"  Episodes: {len(seeds)}")
        print(f"  Methods:  {', '.join(methods)}")
        print(f"{'#' * 60}\n")

        all_results = []

        for episode_idx, seed in enumerate(seeds):
            print(f"\n{'=' * 80}")
            print(f"Episode {episode_idx + 1}/{len(seeds)} - Seed: {seed}")
            print(f"{'=' * 80}")

            episode_results = {
                'episode_idx': episode_idx,
                'seed': seed,
                'timestamp': datetime.now().isoformat()
            }

            pregenerated_jobs, bo_delays = self._pre_generate_jobs(seed, n_jobs=80)

            for method in METHODS:
                if method not in methods:
                    continue
                try:
                    result = self.run_episode_with_method(
                        method, seed,
                        ga_pop_size=ga_pop_size, ga_patience=ga_patience,
                        pregenerated_jobs=pregenerated_jobs,
                        bo_delays=bo_delays
                    )
                    episode_results[method] = result
                except Exception as e:
                    print(f"{method} failed: {e}")
                    import traceback
                    traceback.print_exc()
                    episode_results[method] = {
                        'error': str(e),
                        'makespan': float('inf')
                    }

            if CHEMOS_DISPATCH in methods:
                try:
                    result = self.run_episode_chemos_tmax_abort(
                        seed=seed,
                        pregenerated_jobs=pregenerated_jobs,
                        bo_delays=bo_delays,
                    )
                    episode_results[CHEMOS_DISPATCH] = result
                except Exception as e:
                    print(f"{CHEMOS_DISPATCH} failed: {e}")
                    import traceback
                    traceback.print_exc()
                    episode_results[CHEMOS_DISPATCH] = {
                        'error': str(e),
                        'makespan': float('inf')
                    }

            all_results.append(episode_results)
            self._save_episode_result(episode_results, episode_idx)

        self._save_summary(all_results)

    def _save_episode_result(self, episode_result, episode_idx):
        filename = f"episode_{episode_idx:03d}_seed_{episode_result['seed']}.json"
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(episode_result, f, indent=2, ensure_ascii=False)

        print(f"Episode {episode_idx} results saved to: {filename}")

    def _save_summary(self, all_results):
        from testing.summarize_results import summarize_benchmark, print_benchmark_summary
        summary = summarize_benchmark(all_results)
        filepath = os.path.join(self.output_dir, 'summary.json')
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print_benchmark_summary(summary)
        print(f"\n Summary saved to: {filepath}")


def paper_trial_seeds():
    seeds = []
    for k in range(1, 11):
        path = os.path.join(repo_root, 'results', 'benchmark', f'trial_{k}.json')
        with open(path, 'r', encoding='utf-8') as f:
            seeds.append(int(json.load(f)['seed']))
    return seeds


def main():
    import argparse

    parser = argparse.ArgumentParser(description='SDL benchmark: ERC-FSS and baseline schedulers')
    parser.add_argument('--config', type=str,
                        default=os.path.join(project_root, 'configs', 'sdl_config.yaml'),
                        help='Environment config path')
    parser.add_argument('--model', type=str,
                        default=os.path.join(project_root, 'training', 'saved_models', 'ERC_FSS.pt'),
                        help='Trained FSS model path')
    parser.add_argument('--baseline_model', type=str,
                        default=os.path.join(project_root, 'DRL_baseline', 'training',
                                             'saved_models', 'DRL_BASELINE.pt'),
                        help='Trained conventional DRL baseline model path')
    parser.add_argument('--num_episodes', type=int, default=1,
                        help='Number of scheduling trials with random seeds')
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='Run the trials with these seeds')
    parser.add_argument('--paper_seeds', action='store_true',
                        help='Run the ten trials reported in the paper '
                             '(seeds read from results/benchmark)')
    parser.add_argument('--methods', type=str, nargs='+', default=ALL_METHODS,
                        choices=ALL_METHODS, help='Methods to run')
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(repo_root, 'outputs', 'benchmark'),
                        help='Output directory for results')
    parser.add_argument('--ga_patience', type=int, default=20,
                        help='GA generations without improvement before stopping')

    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Config not found: {args.config}")
        return
    if not os.path.exists(args.model):
        print(f"Model not found: {args.model}")
        return

    baseline_model_path = None
    if 'Conventional DRL' in args.methods:
        if not os.path.exists(args.baseline_model):
            print(f"DRL baseline model not found: {args.baseline_model}")
            return
        baseline_model_path = args.baseline_model

    seeds = args.seeds
    if args.paper_seeds:
        seeds = paper_trial_seeds()

    runner = BenchmarkComparison(
        env_config_path=args.config,
        model_path=args.model,
        output_dir=args.output_dir,
        ga_patience=args.ga_patience,
        baseline_model_path=baseline_model_path
    )

    runner.run_comparison(
        num_episodes=args.num_episodes,
        seeds=seeds,
        methods=args.methods,
        ga_patience=args.ga_patience
    )


if __name__ == "__main__":
    main()
