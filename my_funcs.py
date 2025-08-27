import torch
import os
from torch import nn
from torch.nn import functional as F
import numpy as np
import json
from distri_diff_measure import *



def get_filter_adj(adjA, adjB):
    # [batch, 1, token]
    diff_adj = torch.abs(adjA - adjB)
    # epsilon = 1e-6
    # diff_adj = torch.sqrt((adjA - adjB) ** 2 + epsilon)

    # [batch, token]
    diff_adj = diff_adj.squeeze(1).contiguous()
    return diff_adj



def get_soft_topk(adj_diffs, top_k, weight=50.0):
    # [batch, k]
    values, indices = torch.topk(adj_diffs, k=top_k+1, dim=-1, largest=True, sorted=True)
    # [batch, 1]
    threshold = values[:, -1].unsqueeze(1)
    # [batch, token]
    adj_act = torch.sigmoid(weight * (adj_diffs - threshold))
    return adj_act



def get_full_edge(batch_size, node_num, device):
    # all_num = node_num * batch_size
    edge = []
    for k in range(batch_size):
        for i in range(node_num):
            p = k * node_num + i
            for j in range(node_num):
                if i == j:
                    continue
                q = k * node_num + j
                edge.append([p, q])
    edge_tensor = (torch.from_numpy((np.array(edge)).T)).contiguous().to(device)
    return edge_tensor

