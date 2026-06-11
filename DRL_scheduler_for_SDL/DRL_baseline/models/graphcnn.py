"""GIN (Graph Isomorphism Network) Encoder"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Basic multi-layer MLP (Lei Kun)."""

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


class GraphCNN(nn.Module):
    """GIN encoder (Lei Kun)."""

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
        """One GIN propagation step (Lei Kun)."""
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
                # max pooling for sparse matrix (approximation)
                pooled = torch.spmm(adj_matrix, h)
            else:
                pooled = torch.max(h[padded_neighbor_list], dim=1)[0]
        else:
            raise ValueError("Unsupported pooling type")

        if self.learn_eps:
            pooled_rep = self.mlps[layer]((1 + self.eps[layer]) * h + pooled)
        else:
            pooled_rep = self.mlps[layer](h + pooled)

        # Batch normalization
        h = self.batch_norms[layer](pooled_rep)

        # Non-linearity
        h = F.relu(h)

        return h

    def forward(self, x, graph_pool, padded_nei, adj):
        """GIN forward; returns (graph-level pooled, per-node embeddings)."""
        h = x

        for layer in range(self.num_layers):
            h = self.next_layer(h, layer, padded_neighbor_list=padded_nei, adj_matrix=adj)

        h_pooled = torch.spmm(graph_pool, h)

        return h_pooled, h