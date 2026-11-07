import torch
import torch.nn as nn
import torch.nn.functional as F


def scatter_mean(values, index, num_nodes):
    total = torch.zeros(num_nodes, values.size(1), device=values.device, dtype=values.dtype)
    total.index_add_(0, index, values)
    count = torch.zeros(num_nodes, device=values.device, dtype=values.dtype)
    count.index_add_(0, index, torch.ones_like(index, dtype=values.dtype))
    return total / count.clamp(min=1).unsqueeze(1)


class EdgeSAGELayer(nn.Module):
    def __init__(self, node_dim, edge_dim, out_dim):
        super().__init__()
        self.message = nn.Linear(node_dim + edge_dim, out_dim)
        self.update = nn.Linear(node_dim + out_dim, out_dim)

    def forward(self, node_state, src, dst, edge_attr):
        both_src = torch.cat([src, dst])
        both_dst = torch.cat([dst, src])
        both_edge = torch.cat([edge_attr, edge_attr])
        messages = F.relu(self.message(torch.cat([node_state[both_src], both_edge], dim=1)))
        neighborhood = scatter_mean(messages, both_dst, node_state.size(0))
        return F.relu(self.update(torch.cat([node_state, neighborhood], dim=1)))


class StreamingEGraphSAGE(nn.Module):
    def __init__(self, edge_dim, hidden_dim, num_classes, num_layers=2, dropout=0.2):
        super().__init__()
        self.edge_dim = edge_dim
        dims = [edge_dim] + [hidden_dim] * num_layers
        self.layers = nn.ModuleList(
            EdgeSAGELayer(dims[i], edge_dim, dims[i + 1]) for i in range(num_layers)
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, graph):
        src, dst, edge_attr = graph["src"], graph["dst"], graph["edge_attr"]
        node_state = torch.ones(graph["num_nodes"], self.edge_dim, device=edge_attr.device)
        for layer in self.layers:
            node_state = self.dropout(layer(node_state, src, dst, edge_attr))
        edge_repr = torch.cat([node_state[src], node_state[dst], edge_attr], dim=1)
        return self.classifier(edge_repr)


class FlowMLP(nn.Module):
    def __init__(self, edge_dim, hidden_dim, num_classes, dropout=0.2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, graph):
        return self.network(graph["edge_attr"])
