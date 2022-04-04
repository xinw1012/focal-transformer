# --------------------------------------------------------
# Focal Transformer with MoE
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Jianwei Yang (jianwyan@microsoft.com) 
# Originally written by Zhe Liu in Swin Transformer
# --------------------------------------------------------


# changelog compared with v5
# duplciate num_heads times at the end for saving time cost

import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from PIL import Image
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
from timm.data.transforms import _pil_interp

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_partition_noreshape(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (B, num_windows_h, num_windows_w, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    # pad feature maps to multiples of window size
    pad_l = pad_t = 0
    pad_r = (window_size - W % window_size) % window_size
    pad_b = (window_size - H % window_size) % window_size
    x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

def get_roll_masks(H, W, window_size, shift_size):
    #####################################
    # move to top-left
    img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
    h_slices = (slice(0, H-window_size),
                slice(H-window_size, H-shift_size),
                slice(H-shift_size, H))
    w_slices = (slice(0, W-window_size),
                slice(W-window_size, W-shift_size),
                slice(W-shift_size, W))
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1

    mask_windows = window_partition(img_mask, window_size)  # nW, window_size, window_size, 1
    mask_windows = mask_windows.view(-1, window_size * window_size)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask_tl = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

    ####################################
    # move to top right
    img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
    h_slices = (slice(0, H-window_size),
                slice(H-window_size, H-shift_size),
                slice(H-shift_size, H))
    w_slices = (slice(0, shift_size),
                slice(shift_size, window_size),
                slice(window_size, W))
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1

    mask_windows = window_partition(img_mask, window_size)  # nW, window_size, window_size, 1
    mask_windows = mask_windows.view(-1, window_size * window_size)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask_tr = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

    ####################################
    # move to bottom left
    img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
    h_slices = (slice(0, shift_size),
                slice(shift_size, window_size),
                slice(window_size, H))
    w_slices = (slice(0, W-window_size),
                slice(W-window_size, W-shift_size),
                slice(W-shift_size, W))
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1

    mask_windows = window_partition(img_mask, window_size)  # nW, window_size, window_size, 1
    mask_windows = mask_windows.view(-1, window_size * window_size)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask_bl = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

    ####################################
    # move to bottom right
    img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
    h_slices = (slice(0, shift_size),
                slice(shift_size, window_size),
                slice(window_size, H))
    w_slices = (slice(0, shift_size),
                slice(shift_size, window_size),
                slice(window_size, W))
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1

    mask_windows = window_partition(img_mask, window_size)  # nW, window_size, window_size, 1
    mask_windows = mask_windows.view(-1, window_size * window_size)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask_br = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

    # append all
    attn_mask_all = torch.cat((attn_mask_tl, attn_mask_tr, attn_mask_bl, attn_mask_br), -1)
    return attn_mask_all

def get_relative_position_index(q_windows, k_windows):
    # get pair-wise relative position index for each token inside the window
    coords_h_q = torch.arange(q_windows[0])
    coords_w_q = torch.arange(q_windows[1])
    coords_q = torch.stack(torch.meshgrid([coords_h_q, coords_w_q]))  # 2, Wh_q, Ww_q

    coords_h_k = torch.arange(k_windows[0])
    coords_w_k = torch.arange(k_windows[1])
    coords_k = torch.stack(torch.meshgrid([coords_h_k, coords_w_k]))  # 2, Wh, Ww

    coords_flatten_q = torch.flatten(coords_q, 1)  # 2, Wh_q*Ww_q
    coords_flatten_k = torch.flatten(coords_k, 1)  # 2, Wh_k*Ww_k

    relative_coords = coords_flatten_q[:, :, None] - coords_flatten_k[:, None, :]  # 2, Wh_q*Ww_q, Wh_k*Ww_k
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh_q*Ww_q, Wh_k*Ww_k, 2
    relative_coords[:, :, 0] += k_windows[0] - 1  # shift to start from 0
    relative_coords[:, :, 1] += k_windows[1] - 1
    relative_coords[:, :, 0] *= (q_windows[1] + k_windows[1]) - 1
    relative_position_index = relative_coords.sum(-1)  #  Wh_q*Ww_q, Wh_k*Ww_k
    return relative_position_index

# def get_topk_closest_indice(q_windows, k_windows, topk=1):
def get_topk_closest_indice(input_resolution, window_size_q, window_size_k, topk=1, dilation=1, offset=0, dist_norm="l2"):
    # get pair-wise relative position index for each token inside the window
    q_windows = (
        input_resolution[0] // window_size_q[0], 
        input_resolution[1] // window_size_q[1]
    )
    k_windows = (
        input_resolution[0] // window_size_k[0], 
        input_resolution[1] // window_size_k[1], 
    )
    
    coords_h_q = torch.arange(q_windows[0])
    coords_w_q = torch.arange(q_windows[1])
    # convert to feature map coordinates
    coords_h_q = coords_h_q * window_size_q[0] + window_size_q[0] // 2
    coords_w_q = coords_w_q * window_size_q[1] + window_size_q[1] // 2

    
    coords_h_k = torch.arange(k_windows[0])
    coords_w_k = torch.arange(k_windows[1])
    # convert to feature map coordinates
    coords_h_k = coords_h_k * window_size_k[0] + window_size_k[0] // 2
    coords_w_k = coords_w_k * window_size_k[1] + window_size_k[1] // 2

    # convert q and k to mesh
    coords_q = torch.stack(torch.meshgrid([coords_h_q, coords_w_q]))  # 2, Wh_q, Ww_q
    coords_k = torch.stack(torch.meshgrid([coords_h_k, coords_w_k]))  # 2, Wh_k, Ww_k

    coords_flatten_q = torch.flatten(coords_q, 1)  # 2, Wh_q*Ww_q
    coords_flatten_k = torch.flatten(coords_k, 1)  # 2, Wh_k*Ww_k

    relative_coords = coords_flatten_q[:, :, None] - coords_flatten_k[:, None, :]  # 2, Wh_q*Ww_q, Wh_k*Ww_k
    
    if dist_norm == "l2":
        relative_position_dists = torch.sqrt(relative_coords[0].float()**2 + relative_coords[1].float()**2)
    elif dist_norm == "l1":
        relative_position_dists = torch.abs(relative_coords[0].float()) + torch.abs(relative_coords[1].float())

    topk = min(topk, relative_position_dists.shape[1])
    topk_score_k, topk_index_k = torch.topk(-relative_position_dists, topk, dim=1) # B, topK, nHeads
    indice_topk = topk_index_k[:, min(offset, topk_index_k.shape[1]-1)::dilation]
    relative_coord_topk = torch.gather(relative_coords, 2, indice_topk.unsqueeze(0).repeat(2, 1, 1))
    return indice_topk, relative_coord_topk.permute(1, 2, 0).contiguous().float()

class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, input_resolution, expand_size, shift_size, window_size, focal_window, 
                    focal_level, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., pool_method="none", 
                    use_route=False, topK=64, routing_topK=64, dist_norm="l2", dilation=1, offset=0):

        super().__init__()
        self.dim = dim
        self.shift_size = shift_size
        self.expand_size = expand_size
        self.window_size = window_size  # Wh, Ww
        self.pool_method = pool_method
        self.input_resolution = input_resolution # NWh, NWw
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.focal_level = focal_level
        self.focal_window = focal_window
        self.use_route = use_route
        self.nWh, self.nWw = self.input_resolution[0] // self.window_size[0], self.input_resolution[1] // self.window_size[1]
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        self.offset = offset
        self.topK = topK
        self.routing_topK = routing_topK

        coords_h_window = torch.arange(self.window_size[0]) - self.window_size[0] // 2
        coords_w_window = torch.arange(self.window_size[1]) - self.window_size[1] // 2        
        coords_window = torch.stack(torch.meshgrid([coords_h_window, coords_w_window]), dim=-1)  # 2, Wh_q, Ww_q
        self.register_buffer("window_coords", coords_window)

        self.coord2rpb_all = nn.ModuleList()
        self.routers = nn.ModuleList()

        for k in range(self.focal_level):
            # build relative position range            
            topk_closest_indice, topk_closest_coord = get_topk_closest_indice(
                input_resolution, self.window_size, (1, 1) if k == 0 else (self.focal_window, self.focal_window), 
                self.topK, # // (2**k), 
                dist_norm=dist_norm, 
                dilation=dilation, 
                offset=offset)
            
            # add offset for tokens in a window
            topk_closest_coord_window = topk_closest_coord.unsqueeze(1) + coords_window.view(-1, 2)[None, :, None, :]

            # normalize coordinates
            topk_closest_coord_window[:,:,:,0] = topk_closest_coord_window[:,:,:,0] / self.input_resolution[0]
            topk_closest_coord_window[:,:,:,1] = topk_closest_coord_window[:,:,:,1] / self.input_resolution[1]

            self.register_buffer("topk_cloest_indice_{}".format(k), topk_closest_indice)
            self.register_buffer("topk_cloest_coords_{}".format(k), topk_closest_coord_window)

            coord2rpb = nn.Sequential(
                nn.Linear(2, head_dim), 
                nn.ReLU(inplace=True),
                nn.Linear(head_dim, self.num_heads)
            )
            self.coord2rpb_all.append(coord2rpb)

            if self.use_route:
                router = nn.Sequential(
                    nn.Linear(dim, head_dim), 
                    nn.ReLU(inplace=True), 
                    nn.Linear(head_dim, self.num_heads)                                
                )                
                self.routers.append(router)

    def forward(self, x_all, mask_all=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        x = x_all[0] # 

        B, nH, nW, C = x.shape
        qkv = self.qkv(x).reshape(B, nH, nW, 3, C).permute(3, 0, 1, 2, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, nH, nW, C

        # partition q map
        q_windows = window_partition(q, self.window_size[0]).view(
            -1, self.window_size[0] * self.window_size[0], self.num_heads, C // self.num_heads
            ).transpose(1, 2)

        k_all = []; v_all = []; topKs = []; topk_rpbs = []; rte_scores_all = []
        for l_k in range(self.focal_level):        
            topk_closest_indice = getattr(self, "topk_cloest_indice_{}".format(l_k))
            topk_indice_k = topk_closest_indice.view(1, -1).repeat(B, 1)
            topk_coords_k = getattr(self, "topk_cloest_coords_{}".format(l_k))
            topk_rpb_k = self.coord2rpb_all[l_k](topk_coords_k)

            x_k = x_all[l_k]
            if l_k == 0:
                k_k = k.view(B, -1, self.num_heads, C // self.num_heads)
                v_k = v.view(B, -1, self.num_heads, C // self.num_heads)
            else:
                qkv_k = self.qkv(x_k).view(B, -1, 3, self.num_heads, C // self.num_heads)
                k_k, v_k = qkv_k[:,:,1], qkv_k[:,:,2]

            if self.use_route:
                # computer routing scores
                # import pdb; pdb.set_trace()
                
                rte_score_k = self.routers[l_k](x_k).view(B, -1, self.num_heads)
                # rte_score_k_selected = torch.gather(rte_score_k, 1, topk_indice_k.view(B, -1, 1).repeat(1, 1, self.num_heads)).view(
                #     B, *topk_closest_indice.shape, self.num_heads)
                rte_score_k_selected = rte_score_k[:, topk_closest_indice.view(-1)].view(
                    B, *topk_closest_indice.shape, self.num_heads)
                                
                # select a subset of tokens according to routing scores
                topk_rte_score_k, topk_rte_indice_k = torch.topk(
                    rte_score_k_selected, min(self.routing_topK, rte_score_k_selected.shape[2]), dim=2)
                rte_scores_all.append(topk_rte_score_k)

                topk_indice_k_routed = torch.gather(
                    topk_closest_indice.unsqueeze(0).unsqueeze(-1).repeat(B, 1, 1, self.num_heads), 2,
                    topk_rte_indice_k
                )
                topk_rpb_k = torch.gather(topk_rpb_k.unsqueeze(0).repeat(B, 1, 1, 1, 1), 3, 
                    topk_rte_indice_k.unsqueeze(2).repeat(1, 1, topk_rpb_k.shape[1], 1, 1))
                topk_rpb_k = topk_rpb_k.view(-1, *topk_rpb_k.shape[2:])

            else:
                topk_indice_k_routed = topk_closest_indice.unsqueeze(0).unsqueeze(-1).repeat(B, 1, 1, self.num_heads)
                topk_rpb_k = topk_rpb_k.unsqueeze(0).repeat(B, 1, 1, 1, 1)
                topk_rpb_k = topk_rpb_k.view(-1, *topk_rpb_k.shape[2:])

            topk_rpbs.append(topk_rpb_k)

            k_k_selected = torch.gather(k_k, 1, topk_indice_k_routed.view(B, -1, self.num_heads).unsqueeze(-1).repeat(1, 1, 1, C // self.num_heads)).view(
                B, *topk_indice_k_routed.shape[1:3], self.num_heads, C // self.num_heads).transpose(2, 3)
            v_k_selected = torch.gather(v_k, 1, topk_indice_k_routed.view(B, -1, self.num_heads).unsqueeze(-1).repeat(1, 1, 1, C // self.num_heads)).view(
                B, *topk_indice_k_routed.shape[1:3], self.num_heads, C // self.num_heads).transpose(2, 3)
            k_all.append(k_k_selected.view(-1, self.num_heads, topk_indice_k_routed.shape[2], C // self.num_heads))
            v_all.append(v_k_selected.view(-1, self.num_heads, topk_indice_k_routed.shape[2], C // self.num_heads))

        (k_all, v_all, topk_rpbs_all) = map(
            lambda t: torch.cat(t, 2), (k_all, v_all, topk_rpbs)
        )

        if self.use_route:
            rte_scores_all = torch.cat(rte_scores_all, 2)
            rte_scores_all = rte_scores_all.view(-1, *rte_scores_all.shape[2:]).transpose(1, 2)
            topk_scores_routed = rte_scores_all.unsqueeze(-2)        

        k_all_routed = k_all
        v_all_routed = v_all
        topk_rpbs_routed = topk_rpbs_all  
        
        N = k_all_routed.shape[-2]
        q_windows = q_windows * self.scale
        attn = (q_windows @ k_all_routed.transpose(-2, -1))  # B*nW, nHead, window_size*window_size, focal_window_size*focal_window_size
        attn = attn + topk_rpbs_routed.permute(0, 3, 1, 2)

        if self.use_route:
            attn = attn + (topk_scores_routed - topk_scores_routed.mean(dim=-1, keepdim=True))

        # attn_self = (q_windows * k_windows).sum(-1, keepdim=True)
        attn_w_self = attn # torch.cat((attn, attn_self), -1)

        attn_w_self = self.softmax(attn_w_self)
        attn_w_self = self.attn_drop(attn_w_self)

        x = (attn_w_self @ v_all_routed).transpose(1, 2).flatten(2)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N, window_size, unfold_size):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N        
        if self.pool_method != "none" and self.focal_level > 1:
            flops += self.num_heads * N * (self.dim // self.num_heads) * (unfold_size * unfold_size)          
        if self.expand_size > 0 and self.focal_level > 0:
            flops += self.num_heads * N * (self.dim // self.num_heads) * ((window_size + 2*self.expand_size)**2-window_size**2)          

        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        if self.pool_method != "none" and self.focal_level > 1:
            flops += self.num_heads * N * (self.dim // self.num_heads) * (unfold_size * unfold_size)          
        if self.expand_size > 0 and self.focal_level > 0:
            flops += self.num_heads * N * (self.dim // self.num_heads) * ((window_size + 2*self.expand_size)**2-window_size**2)          

        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

class FocalTransformerBlock(nn.Module):
    r""" Focal Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, expand_size=0, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, pool_method="none", use_route=False, 
                 focal_level=1, focal_window=1, topK=64, routing_topK=64, dist_norm="l2", dilation=1, offset=0,
                 use_layerscale=False, 
                 layerscale_value=1e-4):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.expand_size = expand_size
        self.mlp_ratio = mlp_ratio
        self.pool_method = pool_method
        self.focal_level = focal_level
        self.focal_window = focal_window
        self.use_layerscale = use_layerscale

        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.expand_size = 0
            # self.focal_level = 0
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.window_pool_size = focal_window

        self.pool_layers = nn.ModuleList()
        if self.pool_method != "none":
            for k in range(self.focal_level-1):
                window_pool_size = math.floor(self.window_pool_size / (2 ** k))
                if self.pool_method == "fc":
                    self.pool_layers.append(nn.Linear(window_pool_size * window_pool_size, 1))
                    self.pool_layers[-1].weight.data.fill_(1./(window_pool_size * window_pool_size))
                    self.pool_layers[-1].bias.data.fill_(0)
                elif self.pool_method == "conv":
                    self.pool_layers.append(nn.Conv2d(dim, dim, kernel_size=window_pool_size, stride=window_pool_size, groups=dim))

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, input_resolution=input_resolution, expand_size=self.expand_size, shift_size=self.shift_size, window_size=to_2tuple(self.window_size), 
            focal_window=focal_window, 
            focal_level=self.focal_level, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, 
            pool_method=pool_method, use_route=use_route, topK=topK, routing_topK=routing_topK, 
            dist_norm=dist_norm, dilation=dilation, offset=offset)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

        if self.use_layerscale:
            self.gamma_1 = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        
        x_windows_all = [shifted_x]
        x_window_masks_all = [self.attn_mask]

        if self.focal_level > 1 and self.pool_method != "none": 
            # if we add coarser granularity and the pool method is not none
            for k in range(self.focal_level-1):     
                window_pool_size = math.floor(self.window_pool_size / (2 ** k))
                pooled_h = math.ceil(H / self.window_pool_size) * (2 ** k)
                pooled_w = math.ceil(W / self.window_pool_size) * (2 ** k)
                H_pool = pooled_h * window_pool_size
                W_pool = pooled_w * window_pool_size

                x_level_k = shifted_x
                # trim or pad shifted_x depending on the required size
                if H > H_pool:
                    trim_t = (H - H_pool) // 2
                    trim_b = H - H_pool - trim_t
                    x_level_k = x_level_k[:, trim_t:-trim_b]
                elif H < H_pool:
                    pad_t = (H_pool - H) // 2
                    pad_b = H_pool - H - pad_t
                    x_level_k = F.pad(x_level_k, (0,0,0,0,pad_t,pad_b))
                
                if W > W_pool:
                    trim_l = (W - W_pool) // 2
                    trim_r = W - W_pool - trim_l
                    x_level_k = x_level_k[:, :, trim_l:-trim_r]
                elif W < W_pool:
                    pad_l = (W_pool - W) // 2
                    pad_r = W_pool - W - pad_l
                    x_level_k = F.pad(x_level_k, (0,0,pad_l,pad_r))

                x_windows_noreshape = window_partition_noreshape(x_level_k.contiguous(), window_pool_size) # B, nw, nw, window_size, window_size, C    
                nWh, nWw = x_windows_noreshape.shape[1:3]
                if self.pool_method == "mean":
                    x_windows_pooled = x_windows_noreshape.mean([3, 4]) # B, nWh, nWw, C
                elif self.pool_method == "max":
                    x_windows_pooled = x_windows_noreshape.max(-2)[0].max(-2)[0].view(B, nWh, nWw, C) # B, nWh, nWw, C                    
                elif self.pool_method == "fc":
                    x_windows_noreshape = x_windows_noreshape.view(B, nWh, nWw, window_pool_size*window_pool_size, C).transpose(3, 4) # B, nWh, nWw, C, wsize**2
                    x_windows_pooled = self.pool_layers[k](x_windows_noreshape).flatten(-2) # B, nWh, nWw, C                      
                elif self.pool_method == "conv":
                    x_windows_noreshape = x_windows_noreshape.view(-1, window_pool_size, window_pool_size, C).permute(0, 3, 1, 2).contiguous() # B * nw * nw, C, wsize, wsize
                    x_windows_pooled = self.pool_layers[k](x_windows_noreshape).view(B, nWh, nWw, C) # B, nWh, nWw, C           

                x_windows_all += [x_windows_pooled]
                x_window_masks_all += [None]
        
        attn_windows = self.attn(x_windows_all, mask_all=x_window_masks_all)  # nW*B, window_size*window_size, C

        attn_windows = attn_windows[:, :self.window_size ** 2]
        
        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x if (not self.use_layerscale) else (self.gamma_1 * x))
        x = x + self.drop_path(self.mlp(self.norm2(x)) if (not self.use_layerscale) else (self.gamma_2 * self.mlp(self.norm2(x))))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size, self.window_size, self.focal_window)

        if self.pool_method != "none" and self.focal_level > 1:
            for k in range(self.focal_level-1):
                window_pool_size = math.floor(self.window_pool_size / (2 ** k))
                nW_glo = nW * (2**k)
                # (sub)-window pooling
                flops += nW_glo * self.dim * window_pool_size * window_pool_size         
                # qkv for global levels
                # NOTE: in our implementation, we pass the pooled window embedding to qkv embedding layer, 
                # but theoritically, we only need to compute k and v.
                flops += nW_glo * self.dim * 3 * self.dim       

        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, img_size, patch_size=4, in_chans=3, embed_dim=96, use_conv_embed=False, norm_layer=nn.LayerNorm, use_pre_norm=False, is_stem=False):
        super().__init__()
        self.input_resolution = img_size
        self.dim = in_chans
        self.reduction = nn.Linear(4 * in_chans, 2 * in_chans, bias=False)
        self.norm = norm_layer(4 * in_chans)

    # def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
    #     super().__init__()
    #     self.input_resolution = input_resolution
    #     self.dim = dim
    #     self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
    #     self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops

# class PatchMerging(nn.Module):
#     r""" Patch Merging Layer.

#     Args:
#         input_resolution (tuple[int]): Resolution of input feature.
#         dim (int): Number of input channels.
#         norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
#     """

#     def __init__(self, img_size, patch_size=4, in_chans=3, embed_dim=96, use_conv_embed=False, norm_layer=nn.LayerNorm, use_pre_norm=False, is_stem=False):
#         super().__init__()
#         self.input_resolution = img_size
#         self.dim = in_chans
#         self.reduction = nn.Linear(4 * in_chans, 2 * in_chans, bias=False)
#         self.norm = norm_layer(4 * in_chans)

#     def forward(self, x):
#         """
#         x: B, C, H, W
#         """       
#         B, C, H, W = x.shape 

#         x = x.permute(0, 2, 3, 1).contiguous()

#         x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
#         x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
#         x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
#         x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
#         x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
#         x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

#         x = self.norm(x)
#         x = self.reduction(x)

#         return x

#     def extra_repr(self) -> str:
#         return f"input_resolution={self.input_resolution}, dim={self.dim}"

#     def flops(self):
#         H, W = self.input_resolution
#         flops = H * W * self.dim
#         flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
#         return flops

class BasicLayer(nn.Module):
    """ A basic Focal Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, expand_size, expand_layer,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, pool_method="none", use_route=False, 
                 focal_level=1, focal_window=1, topK=64, routing_topK=64, use_conv_embed=False, 
                 use_shift=False, use_pre_norm=False, dilation=1, 
                 downsample=None, use_checkpoint=False, use_layerscale=False, layerscale_value=1e-4):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        if expand_layer == "even":
            expand_factor = 0
        elif expand_layer == "odd":
            expand_factor = 1
        elif expand_layer == "all":
            expand_factor = -1
        
        # build blocks
        self.blocks = nn.ModuleList([
            FocalTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=(0 if (i % 2 == 0) else window_size // 2) if use_shift else 0,
                                 expand_size=0 if (i % 2 == expand_factor) else expand_size, 
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, 
                                 attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 pool_method=pool_method, 
                                 focal_level=focal_level, # if (i % 2 == 0) else 1, 
                                 focal_window=focal_window, 
                                 topK=topK, 
                                 routing_topK=routing_topK, 
                                 dist_norm="l2", #  if (i % 2 == 0) else "l1", 
                                 dilation=dilation, 
                                 offset=0, # (i % 2), 
                                 use_route=use_route, 
                                 use_layerscale=use_layerscale, 
                                 layerscale_value=layerscale_value)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                img_size=input_resolution, patch_size=2, in_chans=dim, embed_dim=2*dim, 
                use_conv_embed=use_conv_embed, norm_layer=norm_layer, use_pre_norm=use_pre_norm, 
                is_stem=False
            )
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        if self.downsample is not None:
            # x = x.view(x.shape[0], self.input_resolution[0], self.input_resolution[1], -1).permute(0, 3, 1, 2).contiguous()
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=(224, 224), patch_size=4, in_chans=3, embed_dim=96, use_conv_embed=False, norm_layer=None, use_pre_norm=False, is_stem=False):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.use_pre_norm = use_pre_norm

        if use_conv_embed:
            # if we choose to use conv embedding, then we treat the stem and non-stem differently
            if is_stem:
                kernel_size = 7; padding = 2; stride = 4
            else:
                kernel_size = 3; padding = 1; stride = 2
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        

        if self.use_pre_norm:
            if norm_layer is not None:
                self.pre_norm = nn.GroupNorm(1, in_chans)
            else:
                self.pre_norm = None

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        
        if self.use_pre_norm:
            x = self.pre_norm(x)

        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class FocalTransformer(nn.Module):
    r""" Focal Transformer: Focal Self-attention for Local-Global Interactions in Vision Transformer

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Focal Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False 
        use_shift (bool): Whether to use window shift proposed by Swin Transformer. We observe that using shift or not does not make difference to our Focal Transformer. Default: False
        focal_stages (list): Which stages to perform focal attention. Default: [0, 1, 2, 3], means all stages 
        focal_levels (list): How many focal levels at all stages. Note that this excludes the finest-grain level. Default: [1, 1, 1, 1] 
        focal_windows (list): The focal window size at all stages. Default: [7, 5, 3, 1] 
        expand_stages (list): Which stages to expand the finest grain window. Default: [0, 1, 2, 3], means all stages 
        expand_sizes (list): The expand size for the finest grain level. Default: [3, 3, 3, 3] 
        expand_layer (str): Which layers we want to expand the window for the finest grain leve. This can save computational and memory cost without the loss of performance. Default: "all" 
        use_conv_embed (bool): Whether use convolutional embedding. We noted that using convolutional embedding usually improve the performance, but we do not use it by default. Default: False 
        use_layerscale (bool): Whether use layerscale proposed in CaiT. Default: False 
        layerscale_value (float): Value for layer scale. Default: 1e-4 
        use_pre_norm (bool): Whether use pre-norm in patch merging/embedding layer to control the feature magtigute. Default: False
    """
    def __init__(self, 
                img_size=224, 
                patch_size=4, 
                in_chans=3, 
                num_classes=1000,
                embed_dim=96, 
                depths=[2, 2, 6, 2], 
                num_heads=[3, 6, 12, 24],
                window_size=7, 
                mlp_ratio=4., 
                qkv_bias=True, 
                qk_scale=None,
                drop_rate=0., 
                attn_drop_rate=0., 
                drop_path_rate=0.1,
                norm_layer=nn.LayerNorm, 
                ape=False, 
                patch_norm=True,
                use_checkpoint=False,                 
                use_shift=False, 
                focal_stages=[0, 1, 2, 3], 
                focal_levels=[1, 1, 1, 1], 
                focal_windows=[7, 5, 3, 1], 
                focal_topK=64, 
                focal_dilation=1, 
                focal_routing_topK=64, 
                focal_pool="fc", 
                expand_stages=[0, 1, 2, 3], 
                expand_sizes=[3, 3, 3, 3],
                expand_layer="all", 
                use_conv_embed=False, 
                use_layerscale=False, 
                use_route=False, 
                layerscale_value=1e-4, 
                use_pre_norm=False, 
                **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        
        # split image into patches using either non-overlapped embedding or overlapped embedding
        self.patch_embed = PatchEmbed(
            img_size=to_2tuple(img_size), patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, use_conv_embed=use_conv_embed, is_stem=True, 
            norm_layer=norm_layer if self.patch_norm else None)

        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, 
                               qk_scale=qk_scale,
                               drop=drop_rate, 
                               attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer, 
                               pool_method=focal_pool if i_layer in focal_stages else "none",
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               focal_level=focal_levels[i_layer], 
                               focal_window=focal_windows[i_layer], 
                               topK=focal_topK, 
                               dilation=focal_dilation, 
                               routing_topK=focal_routing_topK, 
                               expand_size=expand_sizes[i_layer], 
                               expand_layer=expand_layer,                           
                               use_conv_embed=use_conv_embed,
                               use_shift=use_shift, 
                               use_pre_norm=use_pre_norm, 
                               use_checkpoint=use_checkpoint, 
                               use_layerscale=use_layerscale, 
                               use_route=use_route, 
                               layerscale_value=layerscale_value)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table', 'relative_position_bias_table_xwin'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops


def profile(model, inputs):
    from torch.profiler import profile, record_function, ProfilerActivity
    with profile(activities=[
            ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True, record_shapes=True) as prof:
        with record_function("model_inference"):
            model(inputs)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

def build_transforms(img_size):
    t = []
    t.append(
        transforms.Resize((img_size, img_size), interpolation=_pil_interp('bicubic'))
    )
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)

if __name__ == '__main__':
    img_size = 224
    x = torch.rand(16, 3, img_size, img_size).cuda()
    model = FocalTransformer(
        img_size=img_size, 
        embed_dim=96, 
        depths=[2,2,6,2], drop_path_rate=0.2, 
        focal_levels=[2,2,2,2], 
        focal_windows=[7,7,7,7], 
        dilation=1, 
        # num_heads=[2,4,8,16],
        # num_heads=[4,8,16,32],
        # num_heads=[6,12,24,48],
        focal_topK=128, 
        focal_routing_topK=64, 
        use_conv_embed=False, 
        use_shift=False, 
        use_route=False, 
    ).cuda()
    
    # ckpt_path = "/home/jwyang/azureblobs/vlpresults/projects/focal-transformer/focal_tiny_v2_global_softmax_localglobal2/pt-results/focal-transformer-33e2a88e-focal_tiny_v2_global_softmax_localglobal2-train_focal_tiny_transformer_rpe_moe__2_2_6_2___2_2_2_2_-f4f86e2b/ckpt_epoch_91.pth"
    # ckpt = torch.load(ckpt_path)
    # model.load_state_dict(ckpt['model'])

    eval_transforms = build_transforms(img_size)
    
    img = Image.open("/extra_disk/Datasets/imagenet/n01440764/ILSVRC2012_val_00017472.JPEG")

    img_t = eval_transforms(img)

    model(img_t.unsqueeze(0).cuda())

    flops = model.flops()
    print(f"number of GFLOPs: {flops / 1e9}")

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of params: {n_parameters}")

    profile(model, x)

    model.eval()
    tic = time.time()
    for t in range(10):
        out = model(x)
    #     label = torch.zeros(out.shape[0]).to(out.device).long()
    #     with torch.autograd.set_detect_anomaly(True):
    #         loss = nn.CrossEntropyLoss()(out, label)
    #         loss.backward()
    #         memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
    #         for name, param in model.named_parameters():
    #             if param.grad is None:
    #                 print(name)
    # print("memory cost: ", memory_used)
    print("time cost: ", time.time()-tic)
