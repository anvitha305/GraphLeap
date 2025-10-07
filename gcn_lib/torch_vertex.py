# 2022.06.17-Changed for building ViG model with GraphLeap support
# Modified for batched CPU-GPU GraphLeap execution
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
import numpy as np
import torch
from torch import nn
from .torch_nn import BasicConv, batched_index_select, act_layer
from .torch_edge import DenseDilatedKnnGraph
from .pos_embed import get_2d_relative_pos_embed
import torch.nn.functional as F
from timm.models.layers import DropPath
import threading
import queue


class MRConv2d(nn.Module):
    """
    Max-Relative Graph Convolution (Paper: https://arxiv.org/abs/1904.03751) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(MRConv2d, self).__init__()
        self.nn = BasicConv([in_channels*2, out_channels], act, norm, bias)

    def forward(self, features, edge_index, y=None):
        x_i = batched_index_select(features, edge_index[1])
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(features, edge_index[0])
        x_j, _ = torch.max(x_j - x_i, -1, keepdim=True)
        b, c, n, _ = features.shape
        x = torch.cat([features.unsqueeze(2), x_j.unsqueeze(2)], dim=2).reshape(b, 2 * c, n, _)
        return self.nn(x)


class EdgeConv2d(nn.Module):
    """
    Edge convolution layer (with activation, batch normalization) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(EdgeConv2d, self).__init__()
        self.nn = BasicConv([in_channels * 2, out_channels], act, norm, bias)

    def forward(self, features, edge_index, y=None):
        x_i = batched_index_select(features, edge_index[1])
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(features, edge_index[0])
        max_value, _ = torch.max(self.nn(torch.cat([x_i, x_j - x_i], dim=1)), -1, keepdim=True)
        return max_value


class GraphSAGE(nn.Module):
    """
    GraphSAGE Graph Convolution (Paper: https://arxiv.org/abs/1706.02216) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(GraphSAGE, self).__init__()
        self.nn1 = BasicConv([in_channels, in_channels], act, norm, bias)
        self.nn2 = BasicConv([in_channels*2, out_channels], act, norm, bias)

    def forward(self, features, edge_index, y=None):
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(features, edge_index[0])
        x_j, _ = torch.max(self.nn1(x_j), -1, keepdim=True)
        return self.nn2(torch.cat([features, x_j], dim=1))


class GINConv2d(nn.Module):
    """
    GIN Graph Convolution (Paper: https://arxiv.org/abs/1810.00826) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(GINConv2d, self).__init__()
        self.nn = BasicConv([in_channels, out_channels], act, norm, bias)
        eps_init = 0.0
        self.eps = nn.Parameter(torch.Tensor([eps_init]))

    def forward(self, features, edge_index, y=None):
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(features, edge_index[0])
        x_j = torch.sum(x_j, -1, keepdim=True)
        return self.nn((1 + self.eps) * features + x_j)


class GraphConv2d(nn.Module):
    """
    Static graph convolution layer
    """
    def __init__(self, in_channels, out_channels, conv='edge', act='relu', norm=None, bias=True):
        super(GraphConv2d, self).__init__()
        if conv == 'edge':
            self.gconv = EdgeConv2d(in_channels, out_channels, act, norm, bias)
        elif conv == 'mr':
            self.gconv = MRConv2d(in_channels, out_channels, act, norm, bias)
        elif conv == 'sage':
            self.gconv = GraphSAGE(in_channels, out_channels, act, norm, bias)
        elif conv == 'gin':
            self.gconv = GINConv2d(in_channels, out_channels, act, norm, bias)
        else:
            raise NotImplementedError(f'conv:{conv} is not supported')

    def forward(self, features, edge_index, y=None):
        return self.gconv(features, edge_index, y)


class DyGraphConv2d(GraphConv2d):
    """
    Dynamic graph convolution layer with GraphLeap CPU-based graph construction
    """
    def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1, conv='edge', act='relu',
                 norm=None, bias=True, stochastic=False, epsilon=0.0, r=1):
        super(DyGraphConv2d, self).__init__(in_channels, out_channels, conv, act, norm, bias)
        self.k = kernel_size
        self.d = dilation
        self.r = r
        self.dilated_knn_graph = DenseDilatedKnnGraph(kernel_size, dilation, stochastic, epsilon)

    def forward(self, features, edge_index_precomputed=None, features_for_graph=None):
        """
        features: The current feature tensor to be convolved (FU) - on GPU
        edge_index_precomputed: Pre-computed edge index from CPU - on GPU
        features_for_graph: Not used when edge_index is precomputed
        """
        B, C, H, W = features.shape
        
        if edge_index_precomputed is not None:
            # Use precomputed edge index from CPU
            edge_index = edge_index_precomputed
            y = None
            if self.r > 1:
                # Recompute y for pooling if needed
                features_graph = features  # Use current features
                y = F.avg_pool2d(features_graph, self.r, self.r)
                y = y.reshape(B, C, -1, 1).contiguous()
        else:
            # Fallback: compute graph on GPU (original behavior)
            features_graph = features_for_graph if features_for_graph is not None else features
            y = None
            if self.r > 1:
                y = F.avg_pool2d(features_graph, self.r, self.r)
                y = y.reshape(B, C, -1, 1).contiguous()
            
            features_graph = features_graph.reshape(B, C, -1, 1).contiguous()
            edge_index = self.dilated_knn_graph(features_graph, y, None)
        
        features_reshape = features.reshape(B, C, -1, 1).contiguous()
        x = super(DyGraphConv2d, self).forward(features_reshape, edge_index, y)
        return x.reshape(B, -1, H, W).contiguous()


class Grapher(nn.Module):
    """
    Grapher module with CPU-GPU GraphLeap support for batched inference
    """
    def __init__(self, in_channels, kernel_size=9, dilation=1, conv='edge', act='relu', norm=None,
                 bias=True, stochastic=False, epsilon=0.0, r=1, n=196, drop_path=0.0, relative_pos=False):
        super(Grapher, self).__init__()
        self.channels = in_channels
        self.n = n
        self.r = r
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.graph_conv = DyGraphConv2d(in_channels, in_channels * 2, kernel_size, dilation, conv,
                                        act, norm, bias, stochastic, epsilon, r)
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.relative_pos = None
        if relative_pos:
            print('using relative_pos')
            relative_pos_tensor = torch.from_numpy(np.float32(
                get_2d_relative_pos_embed(in_channels, int(n**0.5)))
            ).unsqueeze(0).unsqueeze(1)
            relative_pos_tensor = F.interpolate(
                relative_pos_tensor, size=(n, n//(r*r)), mode='bicubic', align_corners=False)
            self.relative_pos = nn.Parameter(-relative_pos_tensor.squeeze(1), requires_grad=False)

    def _get_relative_pos(self, relative_pos, H, W):
        if relative_pos is None or H * W == self.n:
            return relative_pos
        else:
            N = H * W
            N_reduced = N // (self.r * self.r)
            return F.interpolate(relative_pos.unsqueeze(0), size=(N, N_reduced), mode="bicubic").squeeze(0)

    def forward(self, features, edge_index=None, features_for_graph=None):
        """
        features: current input features for forward (FU) - on GPU
        edge_index: precomputed edge index from CPU - on GPU (for GraphLeap)
        features_for_graph: features used for graph construction (GC) - only used if edge_index is None
        """
        _tmp = features
        features_after_fc1 = self.fc1(features)

        B, C, H, W = features_after_fc1.shape

        # Pass precomputed edge_index to graph convolution
        features_after_gc = self.graph_conv(features_after_fc1, edge_index, features_for_graph)
        features_after_fc2 = self.fc2(features_after_gc)
        output = self.drop_path(features_after_fc2) + _tmp

        # Return output and the FC1 output for use in next layer's graph construction
        return output, features_after_fc1


class CPUGraphConstructor:
    """
    Manages CPU-based graph construction in a separate thread for GraphLeap
    """
    def __init__(self, grapher_module, device='cuda'):
        self.grapher = grapher_module
        self.device = device
        self.graph_queue = queue.Queue(maxsize=2)
        self.feature_queue = queue.Queue(maxsize=2)
        self.stop_flag = threading.Event()
        self.worker_thread = None
        
    def start(self):
        """Start the CPU graph construction thread"""
        self.stop_flag.clear()
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        
    def stop(self):
        """Stop the CPU graph construction thread"""
        self.stop_flag.set()
        if self.worker_thread is not None:
            self.worker_thread.join()
            
    def _worker(self):
        """Worker thread for CPU graph construction"""
        while not self.stop_flag.is_set():
            try:
                features_cpu = self.feature_queue.get(timeout=0.1)
                if features_cpu is None:
                    break
                    
                # Perform graph construction on CPU
                B, C, H, W = features_cpu.shape
                relative_pos = self.grapher._get_relative_pos(self.grapher.relative_pos, H, W)
                
                y = None
                if self.grapher.r > 1:
                    y = F.avg_pool2d(features_cpu, self.grapher.r, self.grapher.r)
                    y = y.reshape(B, C, -1, 1).contiguous()
                
                features_cpu_reshaped = features_cpu.reshape(B, C, -1, 1).contiguous()
                
                # Graph construction on CPU
                edge_index = self.grapher.graph_conv.dilated_knn_graph(
                    features_cpu_reshaped, y, relative_pos
                )
                
                # Move to GPU and put in queue
                edge_index_gpu = edge_index.to(self.device, non_blocking=True)
                self.graph_queue.put(edge_index_gpu)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in CPU graph constructor: {e}")
                break
                
    def submit_features(self, features):
        """Submit features for graph construction (non-blocking)"""
        # Move to CPU for graph construction
        features_cpu = features.detach().cpu()
        self.feature_queue.put(features_cpu)
        
    def get_graph(self, timeout=None):
        """Get the computed graph (blocking)"""
        return self.graph_queue.get(timeout=timeout)
