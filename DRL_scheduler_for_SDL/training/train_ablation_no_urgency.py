"""
FSS training -- ablation: urgency node feature (f4) zeroed.

train.py with the adapter swapped for AblationAdapterNoUrgency and an isolated
output directory; all training logic is identical.
"""
import os
import sys
import time
import torch
import numpy as np
from collections import defaultdict
from copy import deepcopy

from torch.distributions import Categorical
from tqdm import tqdm

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from ppo import PPO
from config import FSSConfig, get_default_config
from env_adapter import SDLEnvAdapter, progressive_release_strict_fast as progressive_release_strict
from ablation_components import AblationAdapterNoUrgency
from utils.memory import Memory
from utils.mb_agg import aggr_obs, g_pool_cal
from utils.agent_utils import select_action2

from envs.sdl_env import SDLEnv
import yaml


class _NullWriter:
    """Null writer to suppress stdout from external functions."""
    def write(self, *args, **kwargs): pass
    def flush(self, *args, **kwargs): pass


class Config:
    """Simple config class to convert dicts to attribute access."""
    def __init__(self, config_dict):
        self._raw = config_dict
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)
    def get(self, key, default=None):
        return getattr(self, key, default)
    def items(self):
        return self._raw.items()
    def keys(self):
        return self._raw.keys()
    def values(self):
        return self._raw.values()


class FSSTrainer:
    """FSS trainer adapted for SDL environments."""

    def __init__(self, env_config_path, ppo_config=None):
        """
        Args:
            env_config_path: path to SDL environment config (sdl_config.yaml)
            ppo_config: FSS configuration object
        """
        with open(env_config_path, 'r', encoding='utf-8') as f:
            self.env_config_raw = yaml.safe_load(f)

        self.env_config = Config(self.env_config_raw)

        self.env = SDLEnv(self.env_config)

        actual_n_machines = len(self.env.machines)
        print(f"\nSDL System Configuration:")
        print(f"  Total machines: {actual_n_machines}")
        print(f"  Machine types: {set(m.machine_type for m in self.env.machines)}")

        if ppo_config is None:
            ppo_config = get_default_config()

        ppo_config.n_machines = actual_n_machines
        ppo_config.n_j = self.env_config_raw.get('env_config', {}).get('n_jobs', 30)
        self.ppo_config = ppo_config

        self.adapter = AblationAdapterNoUrgency(self.env, ppo_config)
        self.ppo = PPO(ppo_config)

        self.episode_rewards = []
        self.episode_makespans = []
        self.device = torch.device(ppo_config.device)

        print(f"\nValidating environment configuration...")
        self._validate_machine_mapping()

    def _validate_machine_mapping(self):
        """Validate machine mapping completeness."""
        print(f"  Total machines: {len(self.env.machines)}")
        assert len(self.adapter.machine_id_to_idx) == len(self.env.machines), \
            "Machine mapping size mismatch!"

        for idx, machine in enumerate(self.env.machines):
            assert self.adapter.machine_id_to_idx[machine.id] == idx
            assert self.adapter.idx_to_machine_id[idx] == machine.id

        print(f"  Machine mapping verified")

    def _validate_operations(self):
        """Validate operations completeness (call after reset)."""
        if not self.env.graph or not self.env.graph.operations:
            return

        for op in self.env.graph.operations.values():
            if not op.processing_times:
                raise ValueError(f"Op {op.id} has empty processing_times!")
            for mach_id in op.processing_times.keys():
                if mach_id not in self.adapter.machine_id_to_idx:
                    raise ValueError(
                        f"Unknown machine {mach_id} in op {op.id}. "
                        f"Available: {list(self.adapter.machine_id_to_idx.keys())}"
                    )

        print(f"  Operations verified ({len(self.env.graph.operations)} total)")

    def collect_episode(self, memory: Memory, episode_idx: int):
        """
        Collect one episode of experience.

        Only triggers decision epochs when job_buffer grows (new job arrives).
        Otherwise advances time.

        Returns:
            ep_reward, ep_makespan, avg_job_entropy, avg_mch_entropy
        """
        _stdout, _quiet = sys.stdout, _NullWriter()

        seed = np.random.randint(0, 100000) + episode_idx * 100000
        sys.stdout = _quiet
        self.env.reset(seed=seed)
        sys.stdout = _stdout

        if episode_idx == 0:
            self._validate_operations()

        job_log_prob = []
        mch_log_prob = []
        r_mb = []
        done_mb = []
        job_entropy_list = []
        mch_entropy_list = []
        ep_rewards = -self.env.graph.makespan if self.env.graph else 0.0
        prev_makespan = self.env.graph.makespan if self.env.graph else 0.0

        mch_a = None

        step_count = 0
        max_steps = 10000

        n_total_jobs = self.ppo_config.n_j

        def _count_completed_jobs():
            if not self.env.graph or not self.env.graph.operations:
                return 0
            job_ops = defaultdict(list)
            for op in self.env.graph.operations.values():
                job_ops[op.job_id].append(op.is_completed)
            return sum(1 for ops in job_ops.values() if ops and all(ops))

        pbar = tqdm(
            total=n_total_jobs,
            desc=f"Ep {episode_idx:3d}",
            unit="job",
            ncols=100,
            leave=False
        )
        last_completed = 0
        buffer_watermark = 0

        sys.stdout = _quiet

        while not self.env._check_all_jobs_completed() and step_count < max_steps:
            step_count += 1
            current_buffer = len(self.env.job_buffer)

            # Sparse decision: skip if buffer hasn't grown
            if current_buffer <= buffer_watermark:
                self.env._advance_time_and_inject_jobs()
                curr_done = _count_completed_jobs()
                if curr_done > last_completed:
                    pbar.update(curr_done - last_completed)
                    last_completed = curr_done
                continue

            buffer_watermark = current_buffer

            # Rule-based release
            release_result = progressive_release_strict(self.env)

            if not release_result['success']:
                self.env._advance_time_and_inject_jobs()
                curr_done = _count_completed_jobs()
                if curr_done > last_completed:
                    pbar.update(curr_done - last_completed)
                    last_completed = curr_done
                continue

            released_jobs = release_result['released_jobs']

            for job in released_jobs:
                if job in self.env.job_buffer.buffer:
                    self.env.job_buffer.buffer.remove(job)

            self.env._build_static_subproblem(released_jobs)

            # Machine state rollback (fix ghost blocking)
            machines_to_update = set()
            for op in self.env.graph.operations.values():
                if not op.is_completed and op.is_scheduled:
                    if op.scheduled_start > self.env.current_time:
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
                running_ops = [
                    op for op in self.env.graph.operations.values()
                    if (op.scheduled_machine == m_id and
                        op.is_scheduled and
                        not op.is_completed)
                ]
                if running_ops:
                    max_end = max(op.scheduled_end for op in running_ops)
                    machine.available_time = max(self.env.current_time, max_end)
                else:
                    machine.available_time = self.env.current_time

            self.env.graph._update_eligible_ops()

            # FSS scheduling loop
            _init_state = self.adapter.get_state_for_fss()
            if _init_state is not None:
                _n_ops = _init_state['fea'].size(0)
                g_pool_step = g_pool_cal(
                    graph_pool_type=self.ppo_config.graph_pool_type,
                    batch_size=torch.Size([1, _n_ops, _n_ops]),
                    n_nodes=_n_ops,
                    device=self.device
                )
            else:
                g_pool_step = None

            while len(self.env.graph.eligible_ops) > 0:
                state = self.adapter.get_state_for_fss()
                if state is None or g_pool_step is None:
                    break

                env_mask_mch = state['mask_mch'].to(self.device)
                env_dur = state['dur'].to(self.device)
                env_adj = state['adj'].to(self.device)
                env_fea = state['fea'].to(self.device)
                env_candidate = state['candidate'].to(self.device)
                env_mask = state['mask'].to(self.device)
                env_mch_time = state['mch_time'].to(self.device)

                # Job Actor
                action, a_idx, log_a, action_node, action_feature, \
                    mask_mch_action, hx, job_entropy = \
                    self.ppo.policy_old_job(
                        x=env_fea, graph_pool=g_pool_step,
                        padded_nei=None, adj=env_adj,
                        candidate=env_candidate, mask=env_mask,
                        mask_mch=env_mask_mch, dur=env_dur,
                        a_index=0, old_action=0,
                        old_policy=True, greedy=False
                    )

                job_entropy_list.append(job_entropy.cpu())

                # Machine Actor
                pi_mch, _ = self.ppo.policy_old_mch(
                    action_node=action_node, action_feature=action_feature,
                    mask_mch_action=mask_mch_action,
                    mch_time=env_mch_time, mch_a=mch_a,
                    last_hh=None, policy=False,
                    et_normalize_coef=self.ppo_config.et_normalize_coef
                )

                mch_a, log_mch = select_action2(pi_mch)

                with torch.no_grad():
                    dist_mch = Categorical(pi_mch)
                    mch_entropy = dist_mch.entropy().mean()
                    mch_entropy_list.append(mch_entropy.cpu())

                job_log_prob.append(log_a)
                mch_log_prob.append(log_mch)

                # Save to memory
                memory.adj_mb.append(env_adj)
                memory.fea_mb.append(env_fea)
                memory.candidate_mb.append(env_candidate)
                memory.mask_mb.append(env_mask)
                memory.a_mb.append(a_idx)
                memory.mch_time.append(env_mch_time)
                memory.action.append(action.clone())
                memory.mch.append(mch_a)
                memory.mask_mch.append(env_mask_mch)
                memory.dur.append(env_dur)

                # Convert action and execute
                op_id, machine_id = self.adapter.action_to_op_and_machine(
                    a_idx.cpu().numpy(), mch_a.cpu().numpy(), state
                )

                op = self.env.graph.operations[op_id]
                processing_time = op.processing_times[machine_id]

                pred_max = max([self.env.graph.operations[p].scheduled_end
                               for p in op.predecessors], default=self.env.current_time)
                start_time = max(pred_max, self.env.machine_available_time[machine_id])
                end_time = start_time + processing_time

                self.env.graph.schedule_operation(op_id, machine_id, start_time, end_time)
                self.env.graph._update_eligible_ops()

                # Compute reward
                curr_makespan = self.env.graph.makespan
                reward = self.adapter.compute_reward(
                    prev_makespan, curr_makespan, False,
                    scheduled_op_id=op_id, scheduled_machine=machine_id
                )
                prev_makespan = curr_makespan

                ep_rewards += reward
                r_mb.append(reward)
                done_mb.append(0)

            buffer_watermark = len(self.env.job_buffer)

            self.env._advance_time_and_inject_jobs()
            curr_done = _count_completed_jobs()
            if curr_done > last_completed:
                pbar.update(curr_done - last_completed)
                last_completed = curr_done

        sys.stdout = _stdout
        pbar.close()

        if len(done_mb) > 0:
            done_mb[-1] = 1

        memory.job_logprobs.append(job_log_prob)
        memory.mch_logprobs.append(mch_log_prob)
        memory.r_mb.append(torch.tensor([r_mb], dtype=torch.float32).to(self.device))
        memory.done_mb.append(torch.tensor([done_mb], dtype=torch.float32).to(self.device))

        # Compute true episode makespan
        if self.env.graph and self.env.graph.operations:
            scheduled_ends = [
                op.scheduled_end
                for op in self.env.graph.operations.values()
                if (op.is_scheduled or op.is_completed) and op.scheduled_end > 0
            ]
            final_makespan = max(scheduled_ends) if scheduled_ends else 0.0
        else:
            final_makespan = 0.0

        avg_job_entropy = float(torch.stack(job_entropy_list).mean()) if job_entropy_list else 0.0
        avg_mch_entropy = float(torch.stack(mch_entropy_list).mean()) if mch_entropy_list else 0.0

        return ep_rewards, final_makespan, avg_job_entropy, avg_mch_entropy

    def _check_tmax_violations(self):
        """Count Tmax constraint violations."""
        violations = 0
        for constraint in self.env.tmax_constraints:
            from_step, to_step, max_interval = constraint

            for op in self.env.graph.operations.values():
                if op.op_type == to_step and op.is_scheduled:
                    candidates = [o for o in self.env.graph.operations.values()
                                  if o.job_id == op.job_id and o.op_type == from_step]
                    if candidates:
                        pred = candidates[0]
                        if pred.is_scheduled:
                            interval = op.scheduled_start - pred.scheduled_end
                            if interval > max_interval:
                                violations += 1
        return violations

    def train(self, num_episodes=1000, save_interval=1):
        """
        Training loop.

        Args:
            num_episodes: number of training episodes
            save_interval: model checkpoint interval
        """
        print(f"\n{'#'*130}")
        print(f"Training Start")
        print(f"  Num episodes: {num_episodes}")
        print(f"  Device: {self.device}")
        print(f"  Save interval: {save_interval}")
        print(f"{'#'*130}\n")

        _stdout_train = sys.stdout
        _quiet_train = _NullWriter()

        for episode in range(num_episodes):
            memory = Memory()

            t0 = time.time()
            ep_reward, ep_makespan, policy_entropy, mch_entropy = \
                self.collect_episode(memory, episode)
            t_collect = time.time() - t0

            self.episode_rewards.append(ep_reward)
            self.episode_makespans.append(ep_makespan)

            tmax_violations = self._check_tmax_violations()

            t1 = time.time()
            job_loss, mch_loss, kl_div = self.ppo.update(memory, episode)
            t_train = time.time() - t1

            print(f"Ep {episode:3d} | makespan={ep_makespan:.2f} | reward={ep_reward:.2f} | "
                  f"tmax={tmax_violations} | jloss={job_loss:.4f} | mloss={mch_loss:.4f} | "
                  f"kl={kl_div:.4f} | collect={t_collect:.1f}s | train={t_train:.1f}s")

            memory.clear_memory()

            if (episode + 1) % 10 == 0:
                recent_rewards = self.episode_rewards[-10:]
                recent_makespans = self.episode_makespans[-10:]

                print(f"\n{'*'*60}")
                print(f"Episode {episode + 1}/{num_episodes}")
                print(f"  Avg Reward (last 10): {np.mean(recent_rewards):.2f}")
                print(f"  Avg Makespan (last 10): {np.mean(recent_makespans):.2f}")
                print(f"  Job Loss: {job_loss:.4f}")
                print(f"  Mch Loss: {mch_loss:.4f}")
                print(f"{'*'*60}\n")

            if (episode + 1) % save_interval == 0:
                save_dir = os.path.join(project_root, 'saved_models_ablation_no_urgency')
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"model_ep{episode+1}.pt")
                sys.stdout = _quiet_train
                self.ppo.save(save_path)
                sys.stdout = _stdout_train

        print(f"\n{'#'*130}")
        print(f"Training Completed")
        print(f"  Total episodes: {num_episodes}")
        print(f"  Final avg makespan (last 50): {np.mean(self.episode_makespans[-30:]):.2f}")
        print(f"{'#'*130}\n")


def main():
    env_config_path = os.path.join(project_root, 'configs', 'sdl_config.yaml')

    if not os.path.exists(env_config_path):
        print(f"Config file not found: {env_config_path}")
        return

    ppo_config = get_default_config()
    ppo_config.num_episodes = 1000
    ppo_config.save_interval = 1
    ppo_config.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\n{'#'*130}")
    print(f"Initializing FSS Trainer for SDL System")
    print(f"{'#'*130}\n")

    trainer = FSSTrainer(env_config_path, ppo_config)

    print(f"\n{'#'*130}")
    print(f"Configuration Summary:")
    print(f"  n_jobs: {trainer.ppo_config.n_j}")
    print(f"  n_operations: {trainer.ppo_config.n_m}")
    print(f"  n_machines: {trainer.ppo_config.n_machines}")
    print(f"  device: {trainer.ppo_config.device}")
    print(f"{'#'*130}\n")

    trainer.train(num_episodes=ppo_config.num_episodes, save_interval=ppo_config.save_interval)


if __name__ == "__main__":
    main()