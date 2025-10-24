# 2022.10.31-Changed for building ViG model with GraphLeap support
# Modified for batched CPU-GPU GraphLeap execution
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq
from gcn_lib import Grapher, act_layer
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
import threading
import queue
import numpy as np

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    'gnn_patch16_224': _cfg(
        crop_pct=0.9, input_size=(3, 224, 224),
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
}

class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='relu', drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_features),
        )
        self.act = act_layer(act)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_features),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x

class Stem(nn.Module):
    """ Image to Visual Word Embedding
    Overlap: https://arxiv.org/pdf/2106.13797.pdf
    """
    def __init__(self, img_size=224, in_dim=3, out_dim=768, act='relu'):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_dim, out_dim//8, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim//8),
            act_layer(act),
            nn.Conv2d(out_dim//8, out_dim//4, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim//4),
            act_layer(act),
            nn.Conv2d(out_dim//4, out_dim//2, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim//2),
            act_layer(act),
            nn.Conv2d(out_dim//2, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            act_layer(act),
            nn.Conv2d(out_dim, out_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.convs(x)
        return x


class CPUGraphConstructorThread:
    """
    Manages CPU-based graph construction in a separate thread pool for batched GraphLeap
    """
    def __init__(self, grapher_list, device='cuda', n_blocks=12):
        self.grapher_list = grapher_list
        self.device = device
        self.n_blocks = n_blocks
        self.graph_queues = [queue.Queue(maxsize=1) for _ in range(n_blocks)]
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
        self.feature_queue.put(None)  # Signal to stop
        if self.worker_thread is not None:
            self.worker_thread.join(timeout=2.0)
            
    def _construct_graph_cpu(self, features_cpu, layer_idx):
        """Construct graph on CPU for a specific layer"""
        grapher = self.grapher_list[layer_idx]
        B, C, H, W = features_cpu.shape
        
        # Get relative position if needed
        relative_pos = grapher._get_relative_pos(grapher.relative_pos, H, W)
        if relative_pos is not None:
            relative_pos = relative_pos.cpu()
        
        # Handle pooling if r > 1
        y = None
        if grapher.r > 1:
            y = F.avg_pool2d(features_cpu, grapher.r, grapher.r)
            y = y.reshape(B, C, -1, 1).contiguous()
        
        features_cpu_reshaped = features_cpu.reshape(B, C, -1, 1).contiguous()
        
        # Graph construction on CPU using numpy-based kNN
        edge_index = grapher.graph_conv.dilated_knn_graph(
            features_cpu_reshaped, y, relative_pos
        )
        
        return edge_index
        
    def _worker(self):
        """Worker thread for CPU graph construction"""
        while not self.stop_flag.is_set():
            try:
                item = self.feature_queue.get(timeout=0.1)
                if item is None:
                    break
                    
                layer_idx, features_cpu = item
                
                # Construct graph for the specified layer
                edge_index = self._construct_graph_cpu(features_cpu, layer_idx)
                
                # Move to GPU and put in corresponding queue
                edge_index_gpu = edge_index.to(self.device, non_blocking=True)
                self.graph_queues[layer_idx].put(edge_index_gpu)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in CPU graph constructor worker: {e}")
                import traceback
                traceback.print_exc()
                
    def submit_features(self, layer_idx, features):
        """Submit features for graph construction (non-blocking)"""
        if layer_idx >= self.n_blocks:
            return  # Don't construct graph beyond last layer
        # Move to CPU for graph construction
        features_cpu = features.detach().cpu()
        try:
            self.feature_queue.put_nowait((layer_idx, features_cpu))
        except queue.Full:
            # If queue is full, do synchronous construction to avoid blocking
            edge_index = self._construct_graph_cpu(features_cpu, layer_idx)
            edge_index_gpu = edge_index.to(self.device, non_blocking=True)
            self.graph_queues[layer_idx].put(edge_index_gpu)
        
    def get_graph(self, layer_idx, timeout=10.0):
        """Get the computed graph for a specific layer (blocking)"""
        try:
            return self.graph_queues[layer_idx].get(timeout=timeout)
        except queue.Empty:
            raise RuntimeError(f"Timeout waiting for graph construction for layer {layer_idx}")


class DeepGCN(torch.nn.Module):
    def __init__(self, opt):
        super(DeepGCN, self).__init__()
        channels = opt.n_filters
        k = opt.k
        act = opt.act
        norm = opt.norm
        bias = opt.bias
        epsilon = opt.epsilon
        stochastic = opt.use_stochastic
        conv = opt.conv
        self.n_blocks = opt.n_blocks
        drop_path = opt.drop_path
        self.use_graphleap = opt.use_graphleap if hasattr(opt, 'use_graphleap') else False
        
        self.stem = Stem(out_dim=channels, act=act)
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.n_blocks)]  # stochastic depth decay rule 
        num_knn = [int(x.item()) for x in torch.linspace(k, 2*k, self.n_blocks)]  # number of knn's k
        max_dilation = 196 // max(num_knn)
        
        self.pos_embed = nn.Parameter(torch.zeros(1, channels, 14, 14))
        
        # Build blocks
        self.grapher_list = nn.ModuleList()
        self.ffn_list = nn.ModuleList()
        
        if opt.use_dilation:
            for i in range(self.n_blocks):
                grapher = Grapher(channels, num_knn[i], min(i // 4 + 1, max_dilation), conv, act, norm,
                                  bias, stochastic, epsilon, 1, drop_path=dpr[i])
                ffn = FFN(channels, channels * 4, act=act, drop_path=dpr[i])
                self.grapher_list.append(grapher)
                self.ffn_list.append(ffn)
        else:
            for i in range(self.n_blocks):
                grapher = Grapher(channels, num_knn[i], 1, conv, act, norm,
                                  bias, stochastic, epsilon, 1, drop_path=dpr[i])
                ffn = FFN(channels, channels * 4, act=act, drop_path=dpr[i])
                self.grapher_list.append(grapher)
                self.ffn_list.append(ffn)

        self.prediction = Seq(
            nn.Conv2d(channels, 1024, 1, bias=True),
            nn.BatchNorm2d(1024),
            act_layer(act),
            nn.Dropout(opt.dropout),
            nn.Conv2d(1024, opt.n_classes, 1, bias=True)
        )
        
        # Initialize CPU graph constructor for GraphLeap
        self.cpu_graph_constructor = None
        if self.use_graphleap:
            self.cpu_graph_constructor = CPUGraphConstructorThread(
                self.grapher_list, 
                device='cuda' if torch.cuda.is_available() else 'cpu',
                n_blocks=self.n_blocks
            )
        
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, inputs):
        x = self.stem(inputs) + self.pos_embed
        B, C, H, W = x.shape

        if self.use_graphleap and not self.training:
            # GraphLeap inference mode
            return self._forward_graphleap(x)
        else:
            # Standard training mode
            return self._forward_standard(x)
    
    def _forward_standard(self, x):
        """Standard forward pass without GraphLeap"""
        features_for_graph = None
        
        for i in range(self.n_blocks):
            grapher = self.grapher_list[i]
            ffn = self.ffn_list[i]
            
            if i == 0:
                # First block uses initial embedding for graph construction
                x, features_for_graph = grapher(x, edge_index=None, features_for_graph=None)
            else:
                # Subsequent blocks use previous FC1 output for graph construction
                x, features_for_graph = grapher(x, edge_index=None, features_for_graph=features_for_graph)
            
            x = ffn(x)
        
        x = F.adaptive_avg_pool2d(x, 1)
        return self.prediction(x).squeeze(-1).squeeze(-1)
    
    def _forward_graphleap(self, x):
        """
        GraphLeap forward pass with CPU-GPU overlap
        Algorithm 2 from GraphLeap paper
        """
        if self.cpu_graph_constructor is None:
            raise RuntimeError("CPU graph constructor not initialized for GraphLeap")
        
        # Start the CPU worker thread
        if self.cpu_graph_constructor.worker_thread is None or not self.cpu_graph_constructor.worker_thread.is_alive():
            self.cpu_graph_constructor.start()
        
        # Layer 0: Warm-up
        grapher_0 = self.grapher_list[0]
        ffn_0 = self.ffn_list[0]
        
        # FC1 for layer 0
        _tmp = x
        features_0_fc1 = grapher_0.fc1(x)
        
        # Submit for GC_0 and GC_1 construction on CPU (warm-up)
        self.cpu_graph_constructor.submit_features(0, features_0_fc1)
        self.cpu_graph_constructor.submit_features(1, features_0_fc1)
        
        # Wait for GC_0
        edge_index_0 = self.cpu_graph_constructor.get_graph(0)
        
        # Complete layer 0 on GPU
        B, C, H, W = features_0_fc1.shape
        features_0_gc = grapher_0.graph_conv(features_0_fc1, edge_index_0, None)
        features_0_fc2 = grapher_0.fc2(features_0_gc)
        x = grapher_0.drop_path(features_0_fc2) + _tmp
        x = ffn_0(x)
        
        # FC1 for layer 1
        grapher_1 = self.grapher_list[1]
        _tmp = x
        features_1_fc1 = grapher_1.fc1(x)
        
        # Main loop: layers 1 to L-1
        for l in range(1, self.n_blocks):
            grapher_l = self.grapher_list[l]
            ffn_l = self.ffn_list[l]
            
            # Parallel execution:
            # CPU: GC_{l+1} using features from FC1_l
            # GPU: Complete layer l
            
            if l < self.n_blocks - 1:
                # Submit GC_{l+1} to CPU (using current FC1 output)
                self.cpu_graph_constructor.submit_features(l + 1, features_1_fc1 if l == 1 else features_fc1)
            
            # Wait for GC_l (should be ready from previous iteration)
            edge_index_l = self.cpu_graph_constructor.get_graph(l)
            
            # GPU: Feature Update for layer l
            if l == 1:
                # Use features_1_fc1 computed earlier
                features_fc1 = features_1_fc1
                _tmp_l = _tmp
            else:
                # FC1 was computed in previous iteration
                _tmp_l = x
                features_fc1 = grapher_l.fc1(x)
            
            # GNN_l, FC2_l on GPU
            B, C, H, W = features_fc1.shape
            features_gc = grapher_l.graph_conv(features_fc1, edge_index_l, None)
            features_fc2 = grapher_l.fc2(features_gc)
            x = grapher_l.drop_path(features_fc2) + _tmp_l
            
            # FFN_l on GPU
            x = ffn_l(x)
        
        # Final prediction
        x = F.adaptive_avg_pool2d(x, 1)
        return self.prediction(x).squeeze(-1).squeeze(-1)
    
    def enable_graphleap(self):
        """Enable GraphLeap mode"""
        self.use_graphleap = True
        if self.cpu_graph_constructor is None:
            self.cpu_graph_constructor = CPUGraphConstructorThread(
                self.grapher_list,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                n_blocks=self.n_blocks
            )
    
    def disable_graphleap(self):
        """Disable GraphLeap mode"""
        self.use_graphleap = False
        if self.cpu_graph_constructor is not None:
            self.cpu_graph_constructor.stop()


@register_model
def vig_ti_224_gelu(pretrained=False, **kwargs):
    class OptInit:
        def __init__(self, num_classes=1000, drop_path_rate=0.0, drop_rate=0.0, num_knn=9, use_graphleap=False, **kwargs):
            self.k = num_knn
            self.conv = 'mr'
            self.act = 'gelu'
            self.norm = 'batch'
            self.bias = True
            self.n_blocks = 12
            self.n_filters = 192
            self.n_classes = num_classes
            self.dropout = drop_rate
            self.use_dilation = True
            self.epsilon = 0.2
            self.use_stochastic = False
            self.drop_path = drop_path_rate
            self.use_graphleap = use_graphleap
    opt = OptInit(**kwargs)
    model = DeepGCN(opt)
    model.default_cfg = default_cfgs['gnn_patch16_224']
    return model

@register_model
def vig_s_224_gelu(pretrained=False, **kwargs):
    class OptInit:
        def __init__(self, num_classes=1000, drop_path_rate=0.0, drop_rate=0.0, num_knn=9, use_graphleap=False, **kwargs):
            self.k = num_knn
            self.conv = 'mr'
            self.act = 'gelu'
            self.norm = 'batch'
            self.bias = True
            self.n_blocks = 16
            self.n_filters = 320
            self.n_classes = num_classes
            self.dropout = drop_rate
            self.use_dilation = True
            self.epsilon = 0.2
            self.use_stochastic = False
            self.drop_path = drop_path_rate
            self.use_graphleap = use_graphleap
    opt = OptInit(**kwargs)
    model = DeepGCN(opt)
    model.default_cfg = default_cfgs['gnn_patch16_224']
    return model

@register_model
def vig_b_224_gelu(pretrained=False, **kwargs):
    class OptInit:
        def __init__(self, num_classes=1000, drop_path_rate=0.0, drop_rate=0.0, num_knn=9, use_graphleap=False, **kwargs):
            self.k = num_knn
            self.conv = 'mr'
            self.act = 'gelu'
            self.norm = 'batch'
            self.bias = True
            self.n_blocks = 16
            self.n_filters = 640
            self.n_classes = num_classes
            self.dropout = drop_rate
            self.use_dilation = True
            self.epsilon = 0.2
            self.use_stochastic = False
            self.drop_path = drop_path_rate
            self.use_graphleap = use_graphleap
    opt = OptInit(**kwargs)
    model = DeepGCN(opt)
    model.default_cfg = default_cfgs['gnn_patch16_224']
    return model
