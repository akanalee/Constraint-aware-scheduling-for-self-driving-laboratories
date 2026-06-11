"""PPO agent for the Lei Kun baseline (Job Actor + Machine Actor)."""
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR, LambdaLR
from copy import deepcopy
import numpy as np

from models.ppo_actor import JobActor, MachineActor
from utils.memory import Memory, adv_normalize
from utils.mb_agg import g_pool_cal
from utils.agent_utils import select_action2
from config import MultiPPOConfig


class PPO:
    """PPO agent (Lei Kun): two-level Job Actor + Machine Actor."""
    def __init__(self, config: MultiPPOConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        self.lr = config.lr
        self.gamma = config.gamma
        self.eps_clip = config.eps_clip
        self.k_epochs = config.k_epochs
        
        # Job Actor
        self.policy_job = JobActor(
            n_j=config.n_j,
            n_m=config.n_m,
            num_layers=config.num_layers,
            learn_eps=config.learn_eps,
            neighbor_pooling_type=config.neighbor_pooling_type,
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            num_mlp_layers_feature_extract=config.num_mlp_layers_feature_extract,
            num_mlp_layers_critic=config.num_mlp_layers_critic,
            hidden_dim_critic=config.hidden_dim_critic,
            device=self.device
        ).to(self.device)
        
        # Machine Actor
        self.policy_mch = MachineActor(
            n_j=config.n_j,
            n_m=config.n_m,
            num_layers=config.num_layers,
            learn_eps=config.learn_eps,
            neighbor_pooling_type=config.neighbor_pooling_type,
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            num_mlp_layers_feature_extract=config.num_mlp_layers_feature_extract,
            device=self.device
        ).to(self.device)
        
        # Old policies (for PPO)
        self.policy_old_job = deepcopy(self.policy_job)
        self.policy_old_mch = deepcopy(self.policy_mch)
        
        self.policy_old_job.load_state_dict(self.policy_job.state_dict())
        self.policy_old_mch.load_state_dict(self.policy_mch.state_dict())
        
        # Optimizers
        self.job_optimizer = torch.optim.Adam(self.policy_job.parameters(), lr=self.lr)
        self.mch_optimizer = torch.optim.Adam(self.policy_mch.parameters(), lr=self.lr)
        
        # Schedulers
        self.job_scheduler = StepLR(
            self.job_optimizer,
            step_size=config.decay_step_size,
            gamma=config.decay_ratio
        )
        self.mch_scheduler = StepLR(
            self.mch_optimizer,
            step_size=config.decay_step_size,
            gamma=config.decay_ratio
        )
        
        # Loss function
        self.MSE = nn.MSELoss()
        
        self.vloss_coef = config.vloss_coef
        self.ploss_coef = config.ploss_coef
        self.entloss_coef = config.entloss_coef
    
    def select_action(self, state, greedy=False):
        """Sample (or greedy-pick) an op and a machine from the current policy."""
        with torch.no_grad():
            adj = state['adj'].to(self.device)
            fea = state['fea'].to(self.device)
            candidate = state['candidate'].to(self.device)
            mask = state['mask'].to(self.device)
            mask_mch = state['mask_mch'].to(self.device)
            dur = state['dur'].to(self.device)
            mch_time = state['mch_time'].to(self.device)
            
            # Graph pooling
            batch_size = fea.size(0) if len(fea.shape) > 1 else 1
            n_nodes = self.config.n_j * self.config.n_m
            g_pool = g_pool_cal(
                graph_pool_type=self.config.graph_pool_type,
                batch_size=torch.Size([batch_size, n_nodes, n_nodes]),
                n_nodes=n_nodes,
                device=self.device
            )
            
            # Job Actor forward
            action, index, log_a, action_node, action_feature, mask_mch_action, hx = self.policy_old_job(
                x=fea,
                graph_pool=g_pool,
                padded_nei=None,
                adj=adj,
                candidate=candidate,
                mask=mask,
                mask_mch=mask_mch,
                dur=dur,
                a_index=0,
                old_action=0,
                mch_pool=None,
                old_policy=True,
                greedy=greedy
            )
            
            # Machine Actor forward
            pi_mch, pool = self.policy_old_mch(
                action_node=action_node,
                hx=hx,
                mask_mch_action=mask_mch_action,
                mch_time=mch_time,
                mch_a=None,
                last_hh=None,
                policy=False,
                et_normalize_coef=self.config.et_normalize_coef
            )
            
            # Select machine
            mch_a, log_mch = select_action2(pi_mch)
            
            return action.cpu().numpy(), mch_a.cpu().numpy()

    def update(self, memories, epoch):
        """PPO update; returns (job_loss, mch_loss, kl_approx)."""
        rewards_all_env = []

        for i in range(self.config.batch_size):
            rewards = []
            discounted_reward = 0

            for reward, is_terminal in zip(
                reversed((memories.r_mb[0][i]).tolist()),
                reversed(memories.done_mb[0][i].tolist())
            ):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                rewards.insert(0, discounted_reward)

            rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            rewards_all_env.append(rewards)

        rewards_all_env = torch.stack(rewards_all_env, 0)

        for _ in range(self.k_epochs):
            actual_n_ops = memories.fea_mb[0].size(0)

            g_pool_step = g_pool_cal(
                graph_pool_type=self.config.graph_pool_type,
                batch_size=torch.Size([self.config.batch_size, actual_n_ops, actual_n_ops]),
                n_nodes=actual_n_ops,
                device=self.device
            )

            job_log_prob = []
            mch_log_prob = []
            val = []
            job_entropy = []
            mch_entropies = []

            mch_a = None
            last_hh = None
            pool = None

            job_log_old_prob = memories.job_logprobs[0]
            mch_log_old_prob = memories.mch_logprobs[0]
            #env_mask_mch = memories.mask_mch[0]
            #env_dur = memories.dur[0]

            for i in range(len(memories.fea_mb)):
                env_fea = memories.fea_mb[i]
                env_adj = memories.adj_mb[i]
                env_candidate = memories.candidate_mb[i]
                env_mask = memories.mask_mb[i]
                a_index = memories.a_mb[i]
                env_mch_time = memories.mch_time[i]
                old_action = memories.action[i]
                old_mch = memories.mch[i]

                env_mask_mch = memories.mask_mch[i]
                env_dur = memories.dur[i]

                actual_n_ops_step = env_fea.size(0)
                g_pool_step = g_pool_cal(
                    graph_pool_type=self.config.graph_pool_type,
                    batch_size=torch.Size([self.config.batch_size, actual_n_ops_step, actual_n_ops_step]),
                    n_nodes=actual_n_ops_step,
                    device=self.device
                )

                # Job Actor (training mode)
                a_entropy, v, log_a, action_node, _, mask_mch_action, hx = self.policy_job(
                    x=env_fea,
                    graph_pool=g_pool_step,
                    padded_nei=None,
                    adj=env_adj,
                    candidate=env_candidate,
                    mask=env_mask,
                    mask_mch=env_mask_mch,
                    dur=env_dur,
                    a_index=a_index,
                    old_action=old_action,
                    mch_pool=pool,
                    old_policy=False
                )

                # Machine Actor
                pi_mch, pool = self.policy_mch(
                    action_node=action_node,
                    hx=hx,
                    mask_mch_action=mask_mch_action,
                    mch_time=env_mch_time,
                    mch_a=mch_a,
                    last_hh=last_hh,
                    policy=True,
                    et_normalize_coef=self.config.et_normalize_coef
                )

                from torch.distributions.categorical import Categorical
                dist = Categorical(pi_mch)
                log_mch = dist.log_prob(old_mch)
                mch_entropy = dist.entropy()

                val.append(v)
                job_entropy.append(a_entropy)
                mch_entropies.append(mch_entropy)
                job_log_prob.append(log_a)
                mch_log_prob.append(log_mch)

            # Stack tensors
            job_log_prob = torch.stack(job_log_prob, 0).permute(1, 0)
            job_log_old_prob = torch.stack(job_log_old_prob, 0).permute(1, 0)
            mch_log_prob = torch.stack(mch_log_prob, 0).permute(1, 0)
            mch_log_old_prob = torch.stack(mch_log_old_prob, 0).permute(1, 0)
            val = torch.stack(val, 0).squeeze(-1).permute(1, 0)
            job_entropy = torch.stack(job_entropy, 0).permute(1, 0)
            mch_entropies = torch.stack(mch_entropies, 0).permute(1, 0)

            job_loss_sum = 0
            mch_loss_sum = 0

            for j in range(self.config.batch_size):
                # Job loss
                job_ratios = torch.exp(job_log_prob[j] - job_log_old_prob[j].detach())
                advantages = rewards_all_env[j] - val[j].detach()
                advantages = adv_normalize(advantages)

                job_surr1 = job_ratios * advantages
                job_surr2 = torch.clamp(job_ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                job_v_loss = self.MSE(val[j], rewards_all_env[j])
                job_loss = -1 * torch.min(job_surr1, job_surr2) + 0.5 * job_v_loss - 0.01 * job_entropy[j]
                job_loss_sum += job_loss

                # Machine loss
                mch_ratios = torch.exp(mch_log_prob[j] - mch_log_old_prob[j].detach())
                mch_surr1 = mch_ratios * advantages
                mch_surr2 = torch.clamp(mch_ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                mch_loss = -1 * torch.min(mch_surr1, mch_surr2) - 0.01 * mch_entropies[j]
                mch_loss_sum += mch_loss

            self.job_optimizer.zero_grad()
            job_loss_sum.mean().backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.policy_job.parameters(), max_norm=1.0)
            self.job_optimizer.step()

            self.mch_optimizer.zero_grad()
            mch_loss_sum.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.policy_mch.parameters(), max_norm=1.0)
            self.mch_optimizer.step()

            self.policy_old_job.load_state_dict(self.policy_job.state_dict())
            self.policy_old_mch.load_state_dict(self.policy_mch.state_dict())

        # KL(π_old || π_new) ≈ E[log π_old - log π_new] = E[-log ratio]
        with torch.no_grad():
            kl_approx = 0.0
            for j in range(self.config.batch_size):
                log_ratio = job_log_old_prob[j].detach() - job_log_prob[j].detach()
                kl_approx += log_ratio.mean().item()
            kl_approx /= max(1, self.config.batch_size)

        if self.config.decayflag:
            self.job_scheduler.step()
            self.mch_scheduler.step()

        return job_loss_sum.mean().item(), mch_loss_sum.mean().item(), kl_approx

    def save(self, path):
        """Save actor/critic weights and optimizer state to path."""
        torch.save({
            'job_actor': self.policy_job.state_dict(),
            'mch_actor': self.policy_mch.state_dict(),
            'job_optimizer': self.job_optimizer.state_dict(),
            'mch_optimizer': self.mch_optimizer.state_dict()
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path):
        """Load actor weights from checkpoint at path."""
        checkpoint = torch.load(path)
        self.policy_job.load_state_dict(checkpoint['job_actor'])
        self.policy_mch.load_state_dict(checkpoint['mch_actor'])
        self.policy_old_job.load_state_dict(checkpoint['job_actor'])
        self.policy_old_mch.load_state_dict(checkpoint['mch_actor'])
        print(f"Model loaded from {path}")