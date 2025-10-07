# 2022.06.17-Changed for building ViG model
# Modified for GraphLeap: CPU-based batched graph construction
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


def pairwise_distance_cpu(x):
    """
    Compute pairwise distance of a point cloud on CPU.
    Args:
        x: numpy array (batch_size, num_points, num_dims)
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    x_inner = -2 * np.matmul(x, x.transpose(0, 2, 1))
    x_square = np.sum(x * x, axis=-1, keepdims=True)
    return x_square + x_inner + x_square.transpose(0, 2, 1)


def part_pairwise_distance_cpu(x, start_idx=0, end_idx=1):
    """
    Compute pairwise distance of a point cloud on CPU (memory efficient).
    Args:
        x: numpy array (batch_size, num_points, num_dims)
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    x_part = x[:, start_idx:end_idx]
    x_square_part = np.sum(x_part * x_part, axis=-1, keepdims=True)
    x_inner = -2 * np.matmul(x_part, x.transpose(0, 2, 1))
    x_square = np.sum(x * x, axis=-1, keepdims=True)
    return x_square_part + x_inner + x_square.transpose(0, 2, 1)


def xy_pairwise_distance_cpu(x, y):
    """
    Compute pairwise distance between two point clouds on CPU.
    Args:
        x: numpy array (batch_size, num_points, num_dims)
        y: numpy array (batch_size, num_points_y, num_dims)
    Returns:
        pairwise distance: (batch_size, num_points, num_points_y)
    """
    xy_inner = -2 * np.matmul(x, y.transpose(0, 2, 1))
    x_square = np.sum(x * x, axis=-1, keepdims=True)
    y_square = np.sum(y * y, axis=-1, keepdims=True)
    return x_square + xy_inner + y_square.transpose(0, 2, 1)


def dense_knn_matrix_cpu(x, k=16, relative_pos=None):
    """Get KNN based on the pairwise distance on CPU (batched).
    Args:
        x: numpy array (batch_size, num_dims, num_points, 1)
        k: int
        relative_pos: numpy array or None
    Returns:
        nearest neighbors: (2, batch_size, num_points, k)
    """
    x = x.transpose(0, 2, 1, 3).squeeze(-1)  # (batch_size, num_points, num_dims)
    batch_size, n_points, n_dims = x.shape
    
    ### memory efficient implementation ###
    n_part = 10000
    if n_points > n_part:
        nn_idx_list = []
        groups = math.ceil(n_points / n_part)
        for i in range(groups):
            start_idx = n_part * i
            end_idx = min(n_points, n_part * (i + 1))
            dist = part_pairwise_distance_cpu(x, start_idx, end_idx)
            if relative_pos is not None:
                dist = dist + relative_pos[:, start_idx:end_idx]
            nn_idx_part = np.argpartition(-dist, k, axis=-1)[:, :, :k]
            # Sort the k nearest neighbors
            for b in range(batch_size):
                for p in range(end_idx - start_idx):
                    nn_idx_part[b, p] = nn_idx_part[b, p][np.argsort(-dist[b, p, nn_idx_part[b, p]])]
            nn_idx_list.append(nn_idx_part)
        nn_idx = np.concatenate(nn_idx_list, axis=1)
    else:
        dist = pairwise_distance_cpu(x)
        if relative_pos is not None:
            dist = dist + relative_pos
        nn_idx = np.argpartition(-dist, k, axis=-1)[:, :, :k]
        # Sort the k nearest neighbors
        for b in range(batch_size):
            for p in range(n_points):
                nn_idx[b, p] = nn_idx[b, p][np.argsort(-dist[b, p, nn_idx[b, p]])]
    
    center_idx = np.arange(0, n_points).reshape(1, n_points, 1)
    center_idx = np.repeat(center_idx, batch_size, axis=0)
    center_idx = np.repeat(center_idx, k, axis=2)
    
    return np.stack((nn_idx, center_idx), axis=0)


def xy_dense_knn_matrix_cpu(x, y, k=16, relative_pos=None):
    """Get KNN based on the pairwise distance between x and y on CPU (batched).
    Args:
        x: numpy array (batch_size, num_dims, num_points, 1)
        y: numpy array (batch_size, num_dims, num_points_y, 1)
        k: int
        relative_pos: numpy array or None
    Returns:
        nearest neighbors: (2, batch_size, num_points, k)
    """
    x = x.transpose(0, 2, 1, 3).squeeze(-1)  # (batch_size, num_points, num_dims)
    y = y.transpose(0, 2, 1, 3).squeeze(-1)  # (batch_size, num_points_y, num_dims)
    batch_size, n_points, n_dims = x.shape
    
    dist = xy_pairwise_distance_cpu(x, y)
    if relative_pos is not None:
        dist = dist + relative_pos
    
    nn_idx = np.argpartition(-dist, k, axis=-1)[:, :, :k]
    # Sort the k nearest neighbors
    for b in range(batch_size):
        for p in range(n_points):
            nn_idx[b, p] = nn_idx[b, p][np.argsort(-dist[b, p, nn_idx[b, p]])]
    
    center_idx = np.arange(0, n_points).reshape(1, n_points, 1)
    center_idx = np.repeat(center_idx, batch_size, axis=0)
    center_idx = np.repeat(center_idx, k, axis=2)
    
    return np.stack((nn_idx, center_idx), axis=0)


class DenseDilated(nn.Module):
    """
    Find dilated neighbor from neighbor list
    edge_index: (2, batch_size, num_points, k)
    """
    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(DenseDilated, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k

    def forward(self, edge_index):
        if self.stochastic:
            if torch.rand(1) < self.epsilon and self.training:
                num = self.k * self.dilation
                randnum = torch.randperm(num)[:self.k]
                edge_index = edge_index[:, :, :, randnum]
            else:
                edge_index = edge_index[:, :, :, ::self.dilation]
        else:
            edge_index = edge_index[:, :, :, ::self.dilation]
        return edge_index


class DenseDilatedKnnGraph(nn.Module):
    """
    Find the neighbors' indices based on dilated knn
    CPU-based batched graph construction for GraphLeap
    """
    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(DenseDilatedKnnGraph, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k
        self._dilated = DenseDilated(k, dilation, stochastic, epsilon)

    def forward(self, x, y=None, relative_pos=None):
        """
        Performs graph construction on CPU for batched inputs
        Args:
            x: tensor on CPU (batch_size, num_dims, num_points, 1)
            y: tensor on CPU or None
            relative_pos: tensor on CPU or None
        Returns:
            edge_index on CPU ready to be moved to GPU
        """
        device = x.device
        
        # Convert to numpy for CPU computation
        x_np = x.cpu().numpy()
        
        # Normalize
        x_norm = x_np / (np.linalg.norm(x_np, axis=1, keepdims=True) + 1e-8)
        
        if y is not None:
            y_np = y.cpu().numpy()
            y_norm = y_np / (np.linalg.norm(y_np, axis=1, keepdims=True) + 1e-8)
            rel_pos_np = relative_pos.cpu().numpy() if relative_pos is not None else None
            edge_index_np = xy_dense_knn_matrix_cpu(x_norm, y_norm, self.k * self.dilation, rel_pos_np)
        else:
            rel_pos_np = relative_pos.cpu().numpy() if relative_pos is not None else None
            edge_index_np = dense_knn_matrix_cpu(x_norm, self.k * self.dilation, rel_pos_np)
        
        # Convert back to tensor on original device
        edge_index = torch.from_numpy(edge_index_np).to(device)
        
        return self._dilated(edge_index)
