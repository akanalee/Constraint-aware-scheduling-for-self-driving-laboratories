"""
Merged neural network architecture for FSS scheduling.
Contains: MLP, GraphCNN (GIN encoder), MLPActor, MLPCritic, Encoder, JobActor, MachineActor.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from utils.agent_utils import select_action1, greedy_select_action, select_action2


# ============================================================
# Basic MLP
# ============================================================

class MLP(nn.Module):
    """Multi-layer perceptron with batch normalization and ReLU."""

    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.linear_or_not = True
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))
            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim, track_running_stats=False))

    def forward(self, x):
        if self.linear_or_not:
            return self.linear(x)
        else:
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
            return self.linears[self.num_layers - 1](h)


# ============================================================
# Actor/Critic MLPs (tanh activation)
# ============================================================

class MLPActor(nn.Module):
    """Actor MLP with tanh activation."""

    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super(MLPActor, self).__init__()
        self.linear_or_not = True
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        if self.linear_or_not:
            return self.linear(x)
        else:
            h = x
            for layer in range(self.num_layers - 1):
                h = torch.tanh(self.linears[layer](h))
            return self.linears[self.num_layers - 1](h)


class MLPCritic(nn.Module):
    """Critic MLP with tanh activation."""

    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super(MLPCritic, self).__init__()
        self.linear_or_not = True
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        if self.linear_or_not:
            return self.linear(x)
        else:
            h = x
            for layer in range(self.num_layers - 1):
                h = torch.tanh(self.linears[layer](h))
            return self.linears[self.num_layers - 1](h)


# ============================================================
# GIN (Graph Isomorphism Network) Encoder
# ============================================================

class GraphCNN(nn.Module):
    """GIN encoder for disjunctive graph node embeddings."""

    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim,
                 learn_eps, neighbor_pooling_type, device):
        super(GraphCNN, self).__init__()
        self.num_layers = num_layers
        self.learn_eps = learn_eps
        self.neighbor_pooling_type = neighbor_pooling_type
        self.device = device

        if self.learn_eps:
            self.eps = nn.Parameter(torch.zeros(self.num_layers))

        self.mlps = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers):
            if layer == 0:
                self.mlps.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
            else:
                self.mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim, track_running_stats=False))

    def next_layer(self, h, layer, padded_neighbor_list=None, adj_matrix=None):
        """Single GIN layer propagation."""
        if self.neighbor_pooling_type == "sum":
            if padded_neighbor_list is None:
                pooled = torch.spmm(adj_matrix, h)
            else:
                pooled = torch.sum(h[padded_neighbor_list], dim=1)
        elif self.neighbor_pooling_type == "average":
            if padded_neighbor_list is None:
                pooled = torch.spmm(adj_matrix, h)
                degree = torch.spmm(adj_matrix, torch.ones((adj_matrix.shape[0], 1)).to(self.device))
                pooled = pooled / (degree + 1e-6)
            else:
                pooled = torch.mean(h[padded_neighbor_list], dim=1)
        elif self.neighbor_pooling_type == "max":
            if padded_neighbor_list is None:
                pooled = torch.spmm(adj_matrix, h)
            else:
                pooled = torch.max(h[padded_neighbor_list], dim=1)[0]
        else:
            raise ValueError("Unsupported pooling type")

        if self.learn_eps:
            pooled_rep = self.mlps[layer]((1 + self.eps[layer]) * h + pooled)
        else:
            pooled_rep = self.mlps[layer](h + pooled)

        h = self.batch_norms[layer](pooled_rep)
        h = F.relu(h)
        return h

    def forward(self, x, graph_pool, padded_nei, adj):
        """
        Forward pass.
        Args:
            x: Node features [n_nodes, input_dim]
            graph_pool: Graph pooling matrix [batch, n_nodes] (sparse)
            padded_nei: Padded neighbor list (optional)
            adj: Adjacency matrix [n_nodes, n_nodes] (sparse)
        Returns:
            h_pooled: Graph-level representation [batch, hidden_dim]
            h_nodes: Node-level representations [n_nodes, hidden_dim]
        """
        h = x
        for layer in range(self.num_layers):
            h = self.next_layer(h, layer, padded_neighbor_list=padded_nei, adj_matrix=adj)
        h_pooled = torch.spmm(graph_pool, h)
        return h_pooled, h


# ============================================================
# Encoder wrapper
# ============================================================

class Encoder(nn.Module):
    """Graph encoder wrapper around GraphCNN."""
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim,
                 learn_eps, neighbor_pooling_type, device):
        super(Encoder, self).__init__()
        self.feature_extract = GraphCNN(
            num_layers=num_layers, num_mlp_layers=num_mlp_layers,
            input_dim=input_dim, hidden_dim=hidden_dim,
            learn_eps=learn_eps, neighbor_pooling_type=neighbor_pooling_type,
            device=device
        ).to(device)

    def forward(self, x, graph_pool, padded_nei, adj):
        return self.feature_extract(x=x, graph_pool=graph_pool, padded_nei=padded_nei, adj=adj)


# ============================================================
# Job Actor (operation selection)
# ============================================================

class JobActor(nn.Module):
    """Selects the next operation to schedule."""

    def __init__(self, n_j, n_m, num_layers, learn_eps, neighbor_pooling_type,
                 input_dim, hidden_dim, num_mlp_layers_feature_extract,
                 num_mlp_layers_critic, hidden_dim_critic, device):
        super(JobActor, self).__init__()
        self.n_j = n_j
        self.n_m = n_m
        self.n_ops_perjob = n_m
        self.device = device

        self.bn = torch.nn.BatchNorm1d(input_dim, track_running_stats=False).to(device)
        self.encoder = Encoder(
            num_layers=num_layers, num_mlp_layers=num_mlp_layers_feature_extract,
            input_dim=input_dim, hidden_dim=hidden_dim,
            learn_eps=learn_eps, neighbor_pooling_type=neighbor_pooling_type,
            device=device
        ).to(device)

        self.actor1 = MLPActor(3, hidden_dim * 2, hidden_dim, 1).to(device)
        self.critic = MLPCritic(num_mlp_layers_critic, hidden_dim * 2, hidden_dim_critic, 1).to(device)

        for name, p in self.named_parameters():
            if 'weight' in name:
                if len(p.size()) >= 2:
                    nn.init.orthogonal_(p, gain=1)
            elif 'bias' in name:
                nn.init.constant_(p, 0)

    def forward(self, x, graph_pool, padded_nei, adj, candidate, mask,
                mask_mch, dur, a_index, old_action,
                old_policy=True, T=1, greedy=False):
        """
        Forward pass for operation selection.
        When old_policy=True (sampling): returns action, index, log_a, action_node, action_feature, mask_mch_action, h_pooled, entropy
        When old_policy=False (training): returns entropy, v, log_a, action_node, action_feature, mask_mch_action, h_pooled
        """
        h_pooled, h_nodes = self.encoder(x=x, graph_pool=graph_pool, padded_nei=padded_nei, adj=adj)

        if old_policy:
            # Sampling phase
            dummy = candidate.unsqueeze(-1).expand(-1, self.n_j, h_nodes.size(-1))
            batch_node = h_nodes.reshape(dummy.size(0), -1, dummy.size(-1)).to(self.device)
            candidate_feature = torch.gather(
                h_nodes.reshape(dummy.size(0), -1, dummy.size(-1)), 1, dummy)

            h_pooled_repeated = h_pooled.unsqueeze(-2).expand_as(candidate_feature)

            concate_fea = torch.cat((candidate_feature, h_pooled_repeated), dim=-1)
            candidate_scores = self.actor1(concate_fea) * 10

            mask_reshape = mask.reshape(candidate_scores.size())
            candidate_scores[mask_reshape] = -1e10

            pi = F.softmax(candidate_scores, dim=1)
            pi = pi * (~mask_reshape).float()
            pi_sum = torch.clamp(pi.sum(dim=1, keepdim=True), min=1e-10)
            pi = pi / pi_sum

            pi_for_entropy = pi.squeeze(-1)
            dist = Categorical(pi_for_entropy)
            entropy = dist.entropy().mean()

            if greedy:
                action, index = greedy_select_action(pi, candidate)
                log_a = 0
            else:
                action, index, log_a = select_action1(pi, candidate)

            action1 = action.type(torch.long).to(self.device)
            n_ops_actual = dur.size(1)
            batch_x = dur.to(self.device)
            mask_mch = mask_mch.to(self.device)

            mask_mch_action = torch.gather(
                mask_mch, 1,
                action1.unsqueeze(-1).unsqueeze(-1).expand(mask_mch.size(0), -1, mask_mch.size(2)))
            action_feature = torch.gather(
                batch_node, 1,
                action1.unsqueeze(-1).unsqueeze(-1).expand(batch_node.size(0), -1, batch_node.size(2))).squeeze(1)
            action_node = torch.gather(
                batch_x, 1,
                action1.unsqueeze(-1).unsqueeze(-1).expand(batch_x.size(0), -1, batch_x.size(2))).squeeze(1)

            return action, index, log_a, action_node.detach(), \
                action_feature.detach(), mask_mch_action.detach(), \
                h_pooled.detach(), entropy.detach()
        else:
            # Training phase
            dummy = candidate.unsqueeze(-1).expand(-1, self.n_j, h_nodes.size(-1))
            batch_node = h_nodes.reshape(dummy.size(0), -1, dummy.size(-1)).to(self.device)
            candidate_feature = torch.gather(
                h_nodes.reshape(dummy.size(0), -1, dummy.size(-1)), 1, dummy)

            h_pooled_repeated = h_pooled.unsqueeze(-2).expand_as(candidate_feature)

            concate_fea = torch.cat((candidate_feature, h_pooled_repeated), dim=-1)
            candidate_scores = self.actor1(concate_fea).squeeze(-1) * 10

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
                mask_mch, 1,
                action1.unsqueeze(-1).unsqueeze(-1).expand(mask_mch.size(0), -1, mask_mch.size(2)))
            action_feature = torch.gather(
                batch_node, 1,
                action1.unsqueeze(-1).unsqueeze(-1).expand(batch_node.size(0), -1, batch_node.size(2))).squeeze(1)
            action_node = torch.gather(
                batch_x, 1,
                action1.unsqueeze(-1).unsqueeze(-1).expand(batch_x.size(0), -1, batch_x.size(2))).squeeze(1)

            v = self.critic(
                torch.cat([h_pooled, candidate_feature.mean(dim=1)], dim=-1)
            )
            return entropy, v, log_a, action_node.detach(), action_feature.detach(), \
                mask_mch_action.detach(), h_pooled.detach()


# ============================================================
# Machine Actor (machine selection)
# ============================================================

class MachineActor(nn.Module):
    """Selects the machine for a given operation."""

    def __init__(self, n_j, n_m, num_layers, learn_eps, neighbor_pooling_type,
                 input_dim, hidden_dim, num_mlp_layers_feature_extract, device):
        super(MachineActor, self).__init__()
        self.n_j = n_j
        self.n_m = n_m
        self.n_ops_perjob = n_m
        self.device = device
        self.hidden_size = hidden_dim

        self.bn = torch.nn.BatchNorm1d(hidden_dim, track_running_stats=False).to(device)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim, track_running_stats=False).to(device)
        self.fc2 = nn.Linear(2, hidden_dim, bias=False).to(device)
        self.actor = MLPActor(3, hidden_dim * 3, hidden_dim, 1).to(device)

        for name, p in self.named_parameters():
            if 'weight' in name:
                if len(p.size()) >= 2:
                    nn.init.orthogonal_(p, gain=1)
            elif 'bias' in name:
                nn.init.constant_(p, 0)

    def forward(self, action_node, action_feature, mask_mch_action, mch_time,
                mch_a=None, last_hh=None, policy=False, et_normalize_coef=100.0):
        """
        Forward pass for machine selection.
        Args:
            action_node: Processing times for selected op [batch, n_m]
            action_feature: GIN embedding of the selected op [batch, hidden_dim]
            mask_mch_action: Machine mask [batch, 1, n_m]
            mch_time: Machine available times [batch, n_m]
            et_normalize_coef: Normalization coefficient
        Returns:
            pi_mch: Machine probability [batch, n_m]
            pool: Machine pool embedding [batch, hidden_dim]
        """
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
        op_feature_repeated = action_feature.unsqueeze(1).expand_as(action_node)
        concate_fea = torch.cat((action_node, h_pooled_repeated, op_feature_repeated), dim=-1)

        mch_scores = self.actor(concate_fea).squeeze(-1) * 10
        mch_scores = mch_scores.masked_fill(mask_mch_action.squeeze(1).bool(), float("-inf"))
        pi_mch = F.softmax(mch_scores, dim=1)

        return pi_mch, pool