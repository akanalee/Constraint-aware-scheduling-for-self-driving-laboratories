"""Job Actor and Machine Actor"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions import Categorical
from models.graphcnn import GraphCNN
from models.mlp import MLPActor, MLPCritic
from utils.agent_utils import select_action1, greedy_select_action, select_action2


class Encoder(nn.Module):
    """Graph encoder (Lei Kun)."""
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim,
                 learn_eps, neighbor_pooling_type, device):
        super(Encoder, self).__init__()
        self.feature_extract = GraphCNN(
            num_layers=num_layers,
            num_mlp_layers=num_mlp_layers,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            learn_eps=learn_eps,
            neighbor_pooling_type=neighbor_pooling_type,
            device=device
        ).to(device)

    def forward(self, x, graph_pool, padded_nei, adj):
        h_pooled, h_nodes = self.feature_extract(
            x=x,
            graph_pool=graph_pool,
            padded_nei=padded_nei,
            adj=adj
        )
        return h_pooled, h_nodes


class JobActor(nn.Module):
    """Job actor (Lei Kun); selects the next op to schedule."""
    def __init__(self, n_j, n_m, num_layers, learn_eps, neighbor_pooling_type,
                 input_dim, hidden_dim, num_mlp_layers_feature_extract,
                 num_mlp_layers_critic, hidden_dim_critic, device):
        super(JobActor, self).__init__()

        self.n_j = n_j
        self.n_m = n_m
        self.n_ops_perjob = n_m
        self.device = device

        # Batch normalization
        self.bn = torch.nn.BatchNorm1d(
            input_dim,
            track_running_stats=False
        ).to(device)

        # Encoder
        self.encoder = Encoder(
            num_layers=num_layers,
            num_mlp_layers=num_mlp_layers_feature_extract,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            learn_eps=learn_eps,
            neighbor_pooling_type=neighbor_pooling_type,
            device=device
        ).to(device)

        self._input = nn.Parameter(torch.Tensor(hidden_dim))
        self._input.data.uniform_(-1, 1).to(device)

        self.actor1 = MLPActor(3, hidden_dim * 3, hidden_dim, 1).to(device)

        # Critic network
        self.critic = MLPCritic(num_mlp_layers_critic, hidden_dim, hidden_dim_critic, 1).to(device)

        for name, p in self.named_parameters():
            if 'weight' in name:
                if len(p.size()) >= 2:
                    nn.init.orthogonal_(p, gain=1)
            elif 'bias' in name:
                nn.init.constant_(p, 0)

    def forward(self, x, graph_pool, padded_nei, adj, candidate, mask,
                mask_mch, dur, a_index, old_action, mch_pool,
                old_policy=True, T=1, greedy=False):
        """Forward pass (Lei Kun). old_policy=True samples actions;
        old_policy=False computes log-probs / value for training."""
        # Encode graph
        h_pooled, h_nodes = self.encoder(
            x=x,
            graph_pool=graph_pool,
            padded_nei=padded_nei,
            adj=adj
        )

        if old_policy:
            dummy = candidate.unsqueeze(-1).expand(-1, self.n_j, h_nodes.size(-1))
            batch_node = h_nodes.reshape(dummy.size(0), -1, dummy.size(-1)).to(self.device)
            candidate_feature = torch.gather(
                h_nodes.reshape(dummy.size(0), -1, dummy.size(-1)),
                1,
                dummy
            )

            h_pooled_repeated = h_pooled.unsqueeze(-2).expand_as(candidate_feature)

            if mch_pool is None:
                mch_pooled_repeated = self._input[None, None, :].expand_as(candidate_feature).to(self.device)
            else:
                mch_pooled_repeated = mch_pool.unsqueeze(-2).expand_as(candidate_feature).to(self.device)

            # Concatenate features
            concate_fea = torch.cat((candidate_feature, h_pooled_repeated, mch_pooled_repeated), dim=-1)

            # Compute scores
            candidate_scores = self.actor1(concate_fea)
            candidate_scores = candidate_scores * 10

            # Apply mask
            mask_reshape = mask.reshape(candidate_scores.size())

            candidate_scores[mask_reshape] = -1e10

            pi = F.softmax(candidate_scores, dim=1)


            pi = pi * (~mask_reshape).float()

            pi_sum = pi.sum(dim=1, keepdim=True)
            pi_sum = torch.clamp(pi_sum, min=1e-10)
            pi = pi / pi_sum

            assert (pi.sum(dim=1) - 1.0).abs().max() < 1e-4, "Probabilities do not sum to 1"

            pi_for_entropy = pi.squeeze(-1)
            dist = Categorical(pi_for_entropy)
            entropy = dist.entropy().mean()  # [batch] -> scalar

            # Select action
            if greedy:
                action, index = greedy_select_action(pi, candidate)
                log_a = 0
            else:
                action, index, log_a = select_action1(pi, candidate)

            action1 = action.type(torch.long).to(self.device)

            batch_size = dummy.size(0)
            n_ops_actual = dur.size(1)
            n_machines = dur.size(2)

            batch_x = dur.to(self.device)

            mask_mch = mask_mch.to(self.device)

            if action1[0] >= n_ops_actual:
                print(f"WARNING: action1={action1[0]} >= n_ops={n_ops_actual}; "
                      f"candidate.size={candidate.size()}, index={index}")

            # Get machine mask for selected operation
            mask_mch_action = torch.gather(
                mask_mch,
                1,
                action1.unsqueeze(-1).unsqueeze(-1).expand(mask_mch.size(0), -1, mask_mch.size(2))
            )

            # Get selected operation features
            action_feature = torch.gather(
                batch_node,
                1,
                action1.unsqueeze(-1).unsqueeze(-1).expand(batch_node.size(0), -1, batch_node.size(2))
            ).squeeze(1)

            # Get selected operation processing times
            action_node = torch.gather(
                batch_x,
                1,
                action1.unsqueeze(-1).unsqueeze(-1).expand(batch_x.size(0), -1, batch_x.size(2))
            ).squeeze(1)

            return action, index, log_a, action_node.detach(), \
                action_feature.detach(), mask_mch_action.detach(), \
                h_pooled.detach(), entropy.detach()

        else:
            dummy = candidate.unsqueeze(-1).expand(-1, self.n_j, h_nodes.size(-1))
            batch_node = h_nodes.reshape(dummy.size(0), -1, dummy.size(-1)).to(self.device)
            candidate_feature = torch.gather(
                h_nodes.reshape(dummy.size(0), -1, dummy.size(-1)),
                1,
                dummy
            )

            h_pooled_repeated = h_pooled.unsqueeze(-2).expand_as(candidate_feature)

            if mch_pool is None:
                mch_pooled_repeated = self._input[None, None, :].expand_as(candidate_feature).to(self.device)
            else:
                mch_pooled_repeated = mch_pool.unsqueeze(-2).expand_as(candidate_feature).to(self.device)

            concate_fea = torch.cat((candidate_feature, h_pooled_repeated, mch_pooled_repeated), dim=-1)
            candidate_scores = self.actor1(concate_fea)
            candidate_scores = candidate_scores.squeeze(-1) * 10

            mask_reshape = mask.reshape(candidate_scores.size())
            candidate_scores[mask_reshape] = float('-inf')

            pi = F.softmax(candidate_scores, dim=1)
            dist = Categorical(pi)

            log_a = dist.log_prob(a_index.to(self.device))
            entropy = dist.entropy()

            action1 = old_action.type(torch.long).to(self.device)

            batch_x = dur.to(self.device)
            mask_mch = mask_mch.to(self.device)

            mask_mch_action = torch.gather(
                mask_mch,
                1,
                action1.unsqueeze(-1).unsqueeze(-1).expand(mask_mch.size(0), -1, mask_mch.size(2))
            )

            action_feature = torch.gather(
                batch_node,
                1,
                action1.unsqueeze(-1).unsqueeze(-1).expand(batch_node.size(0), -1, batch_node.size(2))
            ).squeeze(1)

            action_node = torch.gather(
                batch_x,
                1,
                action1.unsqueeze(-1).unsqueeze(-1).expand(batch_x.size(0), -1, batch_x.size(2))
            ).squeeze(1)

            v = self.critic(h_pooled)

            return entropy, v, log_a, action_node.detach(), action_feature.detach(), mask_mch_action.detach(), h_pooled.detach()


class MachineActor(nn.Module):
    """Machine actor (Lei Kun); selects a machine for the chosen op."""
    def __init__(self, n_j, n_m, num_layers, learn_eps, neighbor_pooling_type,
                 input_dim, hidden_dim, num_mlp_layers_feature_extract, device):
        super(MachineActor, self).__init__()

        self.n_j = n_j
        self.n_m = n_m
        self.n_ops_perjob = n_m
        self.device = device
        self.hidden_size = hidden_dim

        # Batch normalization
        self.bn = torch.nn.BatchNorm1d(
            hidden_dim,
            track_running_stats=False
        ).to(device)

        self.bn1 = torch.nn.BatchNorm1d(
            hidden_dim,
            track_running_stats=False
        ).to(device)

        self.fc2 = nn.Linear(2, hidden_dim, bias=False).to(device)

        self.actor = MLPActor(3, hidden_dim * 3, hidden_dim, 1).to(device)

        for name, p in self.named_parameters():
            if 'weight' in name:
                if len(p.size()) >= 2:
                    nn.init.orthogonal_(p, gain=1)
            elif 'bias' in name:
                nn.init.constant_(p, 0)

    def forward(self, action_node, hx, mask_mch_action, mch_time,
                mch_a=None, last_hh=None, policy=False, et_normalize_coef=100.0):
        """Forward pass (Lei Kun). Returns (pi_mch, machine_pool_embedding)."""
        mch_time = mch_time / et_normalize_coef
        action_node = action_node / et_normalize_coef

        feature = torch.cat([mch_time.unsqueeze(-1), action_node.unsqueeze(-1)], -1)

        batch_size = feature.size(0)
        n_machines_actual = feature.size(1)

        action_node = self.bn(
            self.fc2(feature).reshape(-1, self.hidden_size)
        ).reshape(batch_size, n_machines_actual, self.hidden_size)

        pool = action_node.mean(dim=1)

        h_pooled_repeated = pool.unsqueeze(1).expand_as(action_node)
        pooled_repeated = hx.unsqueeze(1).expand_as(action_node)
        concate_fea = torch.cat((action_node, h_pooled_repeated, pooled_repeated), dim=-1)

        mch_scores = self.actor(concate_fea)
        mch_scores = mch_scores.squeeze(-1) * 10

        # Apply mask
        mch_scores = mch_scores.masked_fill(mask_mch_action.squeeze(1).bool(), float("-inf"))

        # Softmax
        pi_mch = F.softmax(mch_scores, dim=1)

        return pi_mch, pool