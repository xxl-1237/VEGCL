import numpy as np
import networkx as nx
import torch
from typing import Sequence
from cdlib import algorithms
from cdlib.utils import convert_graph_formats
from torch_geometric.utils import  subgraph

def ced(edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        p: float,
        threshold: float = 1.) -> torch.Tensor:
    edge_weight = abs(edge_weight)  
    edge_weight = edge_weight / edge_weight.mean() * (1. - p) 
    edge_weight = edge_weight.where(edge_weight > (1. - threshold), torch.ones_like(edge_weight) * (1. - threshold))
    edge_weight = edge_weight.where(edge_weight < 1, torch.ones_like(edge_weight) * 1)
    sel_mask = torch.bernoulli(edge_weight).to(torch.bool)
    return edge_index[:, sel_mask]

def cnd(edge_index: torch.Tensor,
        node_weights: torch.Tensor,
        p: float,
        threshold: float = 1.) -> tuple[torch.Tensor, torch.Tensor]:
    num_nodes = edge_index.max().item() + 1
    node_weights = node_weights * (-1)  # Invert weights
    node_weights = node_weights / node_weights.mean() * (1. - p)  # Normalize
    node_weights = torch.where(node_weights > (1. - threshold), node_weights,
                               torch.ones_like(node_weights) * (1. - threshold))
    node_weights = torch.where(node_weights < 1, node_weights, torch.ones_like(node_weights) * 1)
    subset = torch.bernoulli(node_weights).to(torch.bool).to(edge_index.device)  # Boolean mask for kept nodes
    edge_index_relabeled, _ = subgraph(subset, edge_index, relabel_nodes=True)  # Relabel to [0, sum(subset)-1]
    return edge_index_relabeled, subset  # Return relabeled edges and original subset mask


def cav_dense(feature: torch.Tensor,
              node_cs: np.ndarray,
              p: float,
              max_threshold: float = 0.7) -> torch.Tensor:
    x = feature.abs()
    w = x.t() @ torch.tensor(node_cs).to(feature.device)
    w = w.log()
    w = (w.max() - w) / (w.max() - w.min())
    w = w / w.mean() * p
    w = w.where(w < max_threshold, torch.ones_like(w) * max_threshold)
    drop_mask = torch.bernoulli(w).to(torch.bool)
    feature = feature.clone()
    feature[:, drop_mask] = 0.
    return feature


def cav(feature: torch.Tensor,
        node_cs: np.ndarray,
        p: float,
        max_threshold: float = 0.7) -> torch.Tensor:
    x = feature.abs()
    device = feature.device
    w = x.t() @ torch.tensor(node_cs).to(device)
    w[torch.nonzero(w == 0)] = w.max()  # for redundant attributes of Cora
    w = w.log()
    w = (w.max() - w) / (w.max() - w.min())
    w = w / w.mean() * p
    w = w.where(w < max_threshold, max_threshold * torch.ones(1).to(device))
    w = w.where(w > 0, torch.zeros(1).to(device))
    drop_mask = torch.bernoulli(w).to(torch.bool)
    feature = feature.clone()
    feature[:, drop_mask] = 0.
    return feature


def transition(communities: Sequence[Sequence[int]],
               num_nodes: int) -> np.ndarray:
    classes = np.full(num_nodes, -1)
    for i, node_list in enumerate(communities):
        classes[np.asarray(node_list)] = i
    return classes
