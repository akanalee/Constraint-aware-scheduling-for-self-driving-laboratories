import os
import sys
import time
from itertools import combinations
from typing import Dict, List, Optional

import torch
import yaml

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
scheduler_path = os.path.join(project_root, 'DRL_scheduler_for_SDL')
training_path = os.path.join(scheduler_path, 'training')
sys.path.insert(0, scheduler_path)
sys.path.insert(0, training_path)

from configs import Config
from envs.disjunctive_graph import Machine
from envs.job_buffer import Job
from envs.sdl_env import SDLEnv
from training.config import get_default_config
from training.env_adapter import SDLEnvAdapter
from training.ppo import PPO
from utils.agent_utils import greedy_select_action2
from utils.feasibility import check_release_feasibility
from utils.mb_agg import g_pool_cal


class DRLSchedulerWrapper:
    def __init__(self, model_path: str, config_path: str,
                 current_time_getter: Optional[callable] = None):
        self.model_path = model_path
        self.config_path = config_path
        self._raw_time_getter = current_time_getter or (lambda: 0.0)
        self.current_time_getter = lambda: self._raw_time_getter() / 60.0

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            self.env_config_raw = yaml.safe_load(f)
        self.env_config = Config(self.env_config_raw)

        self.env = SDLEnv(self.env_config)
        self._maybe_override_env_machines()
        self.env.reset(skip_warmstart=True)

        self.ppo_config = get_default_config()
        self.ppo_config.n_machines = len(self.env.machines)
        self.ppo_config.n_j = self.env_config_raw.get('env_config', {}).get('n_jobs', 50)
        self.ppo_config.device = 'cpu'
        self.device = torch.device('cpu')

        self.adapter = SDLEnvAdapter(self.env, self.ppo_config)
        self.ppo = PPO(self.ppo_config)
        self._load_model()

        self._pending_jobs = []
        self.stats = {
            'total_releases': 0,
            'total_solve_time': 0.0,
            'avg_solve_time': 0.0,
            'total_ops_scheduled': 0
        }
        print(f"[ERC-FSS Scheduler] Initialized (model={os.path.basename(model_path)}, "
              f"machines={len(self.env.machines)})")

    def _maybe_override_env_machines(self):
        override_cfg = self.env_config_raw.get('drl_runtime_override', {}) or {}
        if not override_cfg.get('use_config_machines', False):
            return
        seen_ids = set()
        new_machines = []
        for step_info in (self.env_config_raw.get('process_steps', {}) or {}).values():
            if not isinstance(step_info, dict):
                continue
            for m in (step_info.get('machines', []) or []):
                if not isinstance(m, dict):
                    continue
                mid = m.get('id')
                mtype = m.get('type', 'default')
                if mid is None or mid in seen_ids:
                    continue
                seen_ids.add(mid)
                new_machines.append(Machine(id=mid, machine_type=mtype))
        if not new_machines:
            print("[ERC-FSS Scheduler] no machines found in process_steps; "
                  "keeping the default machine pool")
            return
        new_machines.sort(key=lambda mm: mm.id)
        self.env.machines = new_machines
        self.env.machines_dict = {mm.id: mm for mm in new_machines}
        print(f"[ERC-FSS Scheduler] machine pool from {os.path.basename(self.config_path)}: "
              f"{len(new_machines)} machines")

    def _load_model(self):
        checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
        self.ppo.policy_old_job.load_state_dict(checkpoint['job_actor'])
        self.ppo.policy_old_mch.load_state_dict(checkpoint['mch_actor'])
        self.ppo.policy_old_job.eval()
        self.ppo.policy_old_mch.eval()

    def add_job_to_buffer(self, job_dict: Dict):
        self._pending_jobs.append(job_dict)

    def clear_pending_jobs(self):
        self._pending_jobs.clear()

    def execute_progressive_release(self) -> Dict:
        print(f"\n[ERC-FSS Scheduler] Starting Progressive Release (pending={len(self._pending_jobs)})")

        solve_start = time.time()
        n = len(self._pending_jobs)

        if n == 0:
            return {'success': False, 'released_jobs': [], 'makespan': 0.0,
                    'schedule': [], 'solve_time': 0.0, 'k_value': 0,
                    'combinations_tried': 0}

        total_combinations = 0
        total_ppo_inference_time = 0.0
        check_time = 0.0
        n_checked = 0
        n_infeasible = 0

        for k in range(n, 0, -1):
            print(f"  Trying k={k}...")
            combos = [self._pending_jobs] if k == n else list(combinations(self._pending_jobs, k))
            total_combinations += len(combos)
            feasible_results = []

            for idx, jobs_combo in enumerate(combos):
                # rapid Gurobi feasibility check before the scheduling step
                t_check0 = time.perf_counter()
                feasible = check_release_feasibility(
                    self.env, [self._make_job(j) for j in jobs_combo],
                    current_time=self.current_time_getter())
                check_time += time.perf_counter() - t_check0
                n_checked += 1
                if not feasible:
                    n_infeasible += 1
                    continue

                graph_snapshot = self.env.graph.snapshot()
                for job_dict in jobs_combo:
                    self._add_job_to_env(job_dict)

                result = self._global_reschedule()
                total_ppo_inference_time += result.get('ppo_inference_time', 0.0)

                if result['success']:
                    scheduled_snapshot = self.env.graph.snapshot()
                    feasible_results.append({
                        'jobs': list(jobs_combo), 'makespan': result['makespan'],
                        'schedule': result['schedule'],
                        'scheduled_snapshot': scheduled_snapshot
                    })

                self.env.graph.restore(graph_snapshot)
                self.env.job_buffer.buffer.clear()

            if feasible_results:
                best = min(feasible_results, key=lambda x: x['makespan'])
                for job_dict in best['jobs']:
                    self._add_job_to_env(job_dict)
                buffer_jobs = list(self.env.job_buffer.buffer)
                if buffer_jobs:
                    self.env._build_static_subproblem(buffer_jobs)
                self.env.job_buffer.buffer.clear()
                self.env.graph.restore(best['scheduled_snapshot'])

                solve_time = time.time() - solve_start
                self.stats['total_releases'] += 1
                self.stats['total_solve_time'] += solve_time
                self.stats['avg_solve_time'] = self.stats['total_solve_time'] / self.stats['total_releases']
                self.stats['total_ops_scheduled'] += len(best['schedule'])
                self.stats.setdefault('total_ppo_inference_time', 0.0)
                self.stats['total_ppo_inference_time'] += total_ppo_inference_time

                released_ids = set()
                for job in best['jobs']:
                    released_ids.add(job['id'] if isinstance(job, dict) else job.id)
                self._pending_jobs = [
                    j for j in self._pending_jobs
                    if (j['id'] if isinstance(j, dict) else j.id) not in released_ids
                ]

                print(f"  Progressive Release succeeded! (k={k}, makespan={best['makespan']:.2f}min, "
                      f"solve={solve_time:.2f}s)")
                return {
                    'success': True, 'released_jobs': best['jobs'],
                    'makespan': best['makespan'], 'schedule': best['schedule'],
                    'solve_time': solve_time, 'ppo_inference_time': total_ppo_inference_time,
                    'k_value': k, 'combinations_tried': total_combinations,
                    'feasibility_check_time': check_time,
                    'candidates_checked': n_checked,
                    'candidates_infeasible': n_infeasible,
                }

        solve_time = time.time() - solve_start
        print(f"  No feasible solution (solve={solve_time:.2f}s)")
        return {'success': False, 'released_jobs': [], 'makespan': 0.0,
                'schedule': [], 'solve_time': solve_time, 'k_value': 0,
                'combinations_tried': total_combinations,
                'feasibility_check_time': check_time,
                'candidates_checked': n_checked,
                'candidates_infeasible': n_infeasible}

    @staticmethod
    def _make_job(job_dict: Dict) -> Job:
        return Job(
            id=job_dict['id'], job_type=job_dict.get('type', 'default'),
            operations=job_dict.get('operations', []),
            arrival_time=job_dict.get('arrival_time', 0.0),
            priority=job_dict.get('priority', 'normal')
        )

    def _add_job_to_env(self, job_dict: Dict):
        try:
            self.env.job_buffer.add_job(self._make_job(job_dict))
        except Exception as e:
            print(f"[ERC-FSS Scheduler] Failed to add job: {e}")

    def _global_reschedule(self) -> Dict:
        current_time = self.current_time_getter()

        buffer_jobs = list(self.env.job_buffer.buffer)
        if buffer_jobs:
            self.env._build_static_subproblem(buffer_jobs)

        machines_to_update = set()

        for op in self.env.graph.operations.values():
            if not op.is_completed and op.is_scheduled:
                if op.scheduled_start > current_time + 1e-4:
                    if op.scheduled_machine:
                        machines_to_update.add(op.scheduled_machine)
                    op.is_scheduled = False
                    op.scheduled_machine = None
                    op.scheduled_start = 0.0
                    op.scheduled_end = 0.0
                    if op.id in self.env.graph.scheduled_ops:
                        self.env.graph.scheduled_ops.remove(op.id)

        for m_id in machines_to_update:
            machine = self.env.graph.machines[m_id]
            running_ops = [op for op in self.env.graph.operations.values()
                          if (op.scheduled_machine == m_id and
                              op.is_scheduled and not op.is_completed)]
            if running_ops:
                machine.available_time = max(current_time,
                                            max(op.scheduled_end for op in running_ops))
            else:
                machine.available_time = current_time

        self.env.graph._update_eligible_ops()

        state = self.adapter.get_state_for_fss()
        if state is None:
            return {'success': False, 'makespan': float('inf'),
                    'schedule': [], 'ppo_inference_time': 0.0}

        for i, op_id in enumerate(state['eligible_ops']):
            op_idx_in_list = state['op_list'].index(op_id)
            op_mask = state['mask_mch'][0, op_idx_in_list, :]
            op = self.env.graph.operations[op_id]
            compatible_indices = [self.adapter.machine_id_to_idx[m_id]
                                  for m_id in op.compatible_machines
                                  if m_id in self.adapter.machine_id_to_idx]
            if compatible_indices:
                if all(op_mask[idx].item() for idx in compatible_indices):
                    return {'success': False, 'makespan': float('inf'),
                            'schedule': [], 'ppo_inference_time': 0.0}

        schedule = []
        mch_a = None
        ppo_inference_time = 0.0

        for step in range(1000):
            if len(self.env.graph.eligible_ops) == 0:
                break

            state = self.adapter.get_state_for_fss()
            if state is None:
                break

            env_mask_mch = state['mask_mch'].to(self.device)
            env_dur = state['dur'].to(self.device)
            n_ops = state['fea'].size(0)
            g_pool_step = g_pool_cal(
                self.ppo_config.graph_pool_type,
                torch.Size([1, n_ops, n_ops]), n_ops, self.device
            )
            env_adj = state['adj'].to(self.device)
            env_fea = state['fea'].to(self.device)
            env_candidate = state['candidate'].to(self.device)
            env_mask = state['mask'].to(self.device)
            env_mch_time = state['mch_time'].to(self.device)

            t0 = time.time()
            with torch.no_grad():
                action, a_idx, _, action_node, action_feature, mask_mch_action, hx, _ = \
                    self.ppo.policy_old_job(
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
                pi_mch, _ = self.ppo.policy_old_mch(
                    action_node=action_node,
                    action_feature=action_feature,
                    mask_mch_action=mask_mch_action,
                    mch_time=env_mch_time,
                    mch_a=mch_a,
                    last_hh=None,
                    policy=False,
                    et_normalize_coef=self.ppo.config.et_normalize_coef
                )
                mch_a = greedy_select_action2(pi_mch)
                p_flat = pi_mch.detach().cpu().reshape(-1)
                if p_flat.numel() > 1 and self.env.machines:
                    p_max = float(p_flat.max())
                    near = (p_flat >= p_max - 1e-3).nonzero(as_tuple=False).reshape(-1).tolist()
                    if len(near) > 1:
                        avail = self.env.machine_available_time
                        ms = self.env.machines
                        best = min(near, key=lambda i: (avail.get(ms[i].id, 0.0), i))
                        if best != int(mch_a.reshape(-1)[0].item()):
                            mch_a = torch.tensor([best], dtype=mch_a.dtype, device=mch_a.device)
            ppo_inference_time += time.time() - t0

            op_action = a_idx if isinstance(a_idx, int) else a_idx.cpu().numpy()
            op_id, machine_id = self.adapter.action_to_op_and_machine(
                op_action, mch_a.cpu().numpy(), state
            )

            op = self.env.graph.operations[op_id]
            processing_time = op.processing_times[machine_id]
            pred_max = max([self.env.graph.operations[p].scheduled_end
                           for p in op.predecessors], default=current_time)
            start_time = max(pred_max,
                            self.env.machine_available_time.get(machine_id, current_time),
                            getattr(op, 'arrival_time', 0.0))
            end_time = start_time + processing_time

            schedule.append((start_time, op_id, machine_id, end_time))
            self.env.graph.schedule_operation(op_id, machine_id, start_time, end_time)
            self.env.graph._update_eligible_ops()

        new_op_ids = {op_id for _, op_id, _, _ in schedule}
        if not self._verify_tmax_feasibility(restrict_to=new_op_ids):
            return {'success': False, 'makespan': float('inf'),
                    'schedule': [], 'ppo_inference_time': ppo_inference_time}

        return {
            'success': True, 'makespan': self.env.graph.makespan,
            'schedule': schedule, 'ppo_inference_time': ppo_inference_time
        }

    def _verify_tmax_feasibility(self, restrict_to=None) -> bool:
        if not hasattr(self.env, 'tmax_constraints') or not self.env.tmax_constraints:
            return True
        all_ops = list(self.env.graph.operations.values())
        for fs, ts, mx in self.env.tmax_constraints:
            for op in all_ops:
                if op.op_type == ts and (op.is_scheduled or op.is_completed):
                    if restrict_to is not None and op.id not in restrict_to:
                        continue
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

    def _check_tmax_violations(self) -> List[Dict]:
        violations = []
        tmax_constraints = getattr(self.env, 'tmax_constraints', [])
        if not tmax_constraints:
            return violations

        all_ops = list(self.env.graph.operations.values())

        for from_step, to_step, max_interval in tmax_constraints:
            for to_op in all_ops:
                if to_op.op_type != to_step:
                    continue
                if not (to_op.is_scheduled or to_op.is_completed):
                    continue
                candidates = [o for o in all_ops
                              if o.job_id == to_op.job_id
                              and o.op_type == from_step
                              and o.op_index < to_op.op_index]
                if not candidates:
                    continue
                from_op = max(candidates, key=lambda o: o.op_index)
                if from_op.is_scheduled or from_op.is_completed:
                    actual = to_op.scheduled_start - from_op.scheduled_end
                    if actual > max_interval + 1e-4:
                        violations.append({
                            'job_id': to_op.job_id, 'from_op': from_op.id,
                            'to_op': to_op.id, 'actual_interval': actual,
                            'max_interval': max_interval,
                            'violation': actual - max_interval
                        })
        return violations
