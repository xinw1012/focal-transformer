# --------------------------------------------------------
# Focal Transformer with MoE
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Jianwei Yang (jianwyan@microsoft.com) 
# Originally written by Zhe Liu in Swin Transformer
# --------------------------------------------------------


# changelog compared with v7
# instead of adding global feature to each local feature, we still use the global feature as extra tokens

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

class GlobalAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, attn_drop=0.0, qk_scale=None):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.attn_drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v):
        r"""
        q, k, v -- BxHxWxC
        """
        B, H, W, C = q.shape

        q = q.view(
            -1, self.window_size[0] * self.window_size[0], self.num_heads, C // self.num_heads
            ).transpose(1, 2)
        k = k.view(
            -1, self.window_size[0] * self.window_size[0], self.num_heads, C // self.num_heads
            ).transpose(1, 2)
        v = v.view(
            -1, self.window_size[0] * self.window_size[0], self.num_heads, C // self.num_heads
            ).transpose(1, 2)            
        attn = q @ k.transpose(-2, -1)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

        attn = attn + relative_position_bias.unsqueeze(0)

        x = (attn @ v).transpose(1, 2).flatten(2).view(B, H, W, C)

        return x

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
                    use_route=False, topK=64, routing_topK=64, shape="circle", dilation=1, offset=0, mute_fine_grain=False):

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

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        self.gsas = nn.ModuleList()
        for i in range(1, self.focal_level):
            self.gsas.append(
                GlobalAttention(dim, (self.nWh, self.nWw), num_heads)
            )
    def forward(self, x_all, mask_all=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        x = x_all[0] # 

        B, nH, nW, C = x.shape
        qkv = self.qkv(x).reshape(B, nH, nW, 3, C).permute(3, 0, 1, 2, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, nH, nW, C

        # partition q map
        q_windows = window_partition(q, self.window_size[0]).view(
            -1, self.window_size[0] * self.window_size[0], self.num_heads, C // self.num_heads
            ).transpose(1, 2)

        k_windows = window_partition(k, self.window_size[0]).view(
            -1, self.window_size[0] * self.window_size[0], self.num_heads, C // self.num_heads
            ).transpose(1, 2)

        v_windows = window_partition(v, self.window_size[0]).view(
            -1, self.window_size[0] * self.window_size[0], self.num_heads, C // self.num_heads
            ).transpose(1, 2)

        k_all = [k_windows]; v_all = [v_windows]

        for l_k in range(1, self.focal_level):
            x_lk = x_all[l_k]
            qkv_lk = self.qkv(x_lk).reshape(B, *x_lk.shape[1:3], 3, C).permute(3, 0, 1, 2, 4).contiguous()
            q_lk, k_lk, v_lk = qkv_lk[0], qkv_lk[1], qkv_lk[2]  # B, nH, nW, C
            # global self-attention inside window attention
            v_lk = v_lk + self.gsas[l_k-1](q_lk, k_lk, v_lk)
            k_lk = k_lk.view(-1, self.num_heads, 1, C // self.num_heads)
            v_lk = v_lk.view(-1, self.num_heads, 1, C // self.num_heads)
            k_all.append(k_lk)
            v_all.append(v_lk)

        k_all_routed = torch.cat(k_all, 2)
        v_all_routed = torch.cat(v_all, 2)

        N = k_all_routed.shape[-2]
        q_windows = q_windows * self.scale
        attn = (q_windows @ k_all_routed.transpose(-2, -1))  # B*nW, nHead, window_size*window_size, focal_window_size*focal_window_size

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

        if self.focal_level > 1:
            attn[:,:,:,:-(self.focal_level-1)] = attn[:,:,:,:-(self.focal_level-1)] + relative_position_bias.unsqueeze(0)
        else:
            attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v_all_routed).transpose(1, 2).flatten(2)

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
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, pool_method="none", use_route=False, mute_fine_grain=False, 
                 focal_level=1, focal_window=1, focal_kernel=3, topK=64, routing_topK=64, shape="circle", 
                 dilation=1, offset=0, use_layerscale=False, layerscale_value=1e-4):
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
                window_pool_size = min(self.window_pool_size * (2**k), min(self.input_resolution))                
                if self.pool_method == "conv":
                    self.pool_layers.append(nn.Conv2d(dim, dim, kernel_size=window_pool_size+4, stride=window_pool_size, groups=dim, padding=2))
                elif self.pool_method == "conv_share":
                    # share parameters across different channels     
                    if k == 0:
                        self.pool_layers.append(
                            nn.Sequential(
                                nn.Conv2d(1, 1, kernel_size=window_pool_size, stride=window_pool_size, groups=1, padding=0),
                                # nn.GELU(),
                            )
                        )                             
                    else:
                        self.pool_layers.append(
                            nn.Sequential(
                                nn.Conv2d(1, 1, kernel_size=focal_kernel, stride=1, groups=1, padding=focal_kernel//2),
                                # nn.GELU(),
                            )
                        )   

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, input_resolution=input_resolution, expand_size=self.expand_size, shift_size=self.shift_size, window_size=to_2tuple(self.window_size), 
            focal_window=focal_window, 
            focal_level=self.focal_level, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, 
            pool_method=pool_method, use_route=use_route, topK=topK, routing_topK=routing_topK, 
            shape=shape, dilation=dilation, offset=offset, mute_fine_grain=mute_fine_grain)

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
        shifted_x = x
        
        x_windows_all = [shifted_x]
        x_window_masks_all = [self.attn_mask]
        
        x_p = x.permute(0, 3, 1, 2).contiguous()
        x_p = x_p.view(B * C, 1, H, W)

        if self.focal_level > 1 and self.pool_method != "none": 
            # if we add coarser granularity and the pool method is not none
            for k in range(self.focal_level-1):         
                if self.pool_method == "conv":
                    x_windows_pooled = self.pool_layers[k](x_p).permute(0, 2, 3, 1).contiguous() # B, nWh, nWw, C           
                elif self.pool_method == "conv_share":
                    x_p = self.pool_layers[k](x_p)                    
                    x_windows_pooled = x_p.view(B, C, *x_p.shape[2:]).permute(0, 2, 3, 1).contiguous() # B, nWh, nWw, C           
                    x_windows_all.append(x_windows_pooled)
        
        attn_windows = self.attn(x_windows_all, mask_all=x_window_masks_all)  # nW*B, window_size*window_size, C
        
        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
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
        self.use_conv_embed = use_conv_embed

        if self.use_conv_embed:
            kernel_size = 3; padding = 1; stride = 2
            self.proj = nn.Conv2d(in_chans, 2*in_chans, kernel_size=kernel_size, stride=stride, padding=padding)
            self.norm = norm_layer(2 * in_chans)
        else:
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

        if self.use_conv_embed:
            x = x.transpose(1, 2).view(B, C, H, W)
            x = self.proj(x)
            x = x.view(B, 2*C, -1).transpose(1, 2)
            x = self.norm(x)
        else:
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
                 focal_level=1, focal_window=1, focal_kernel=3, topK=64, routing_topK=64, use_conv_embed=False, 
                 use_shift=False, use_pre_norm=False, dilation=1, shape="circle", mute_fine_grain=False, 
                 downsample=None, use_checkpoint=False, use_layerscale=False, layerscale_value=1e-4):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            FocalTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=(0 if (i % 2 == 0) else window_size // 2) if use_shift else 0,
                                 expand_size=0,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, 
                                 attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 pool_method=pool_method, 
                                 focal_level=focal_level, # if (i % 2 == 0) else 1, 
                                 focal_window=focal_window, 
                                 focal_kernel=focal_kernel, 
                                 topK=topK, 
                                 routing_topK=routing_topK, 
                                 shape=shape, #  if (i % 2 == 0) else "l1", 
                                 dilation=dilation, 
                                 offset=0, # (i % 2), 
                                 use_route=use_route, 
                                 mute_fine_grain=mute_fine_grain, 
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
            x = self.downsample(x.contiguous())
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
                kernel_size = 7; padding = 3; stride = 4
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
                focal_kernels=[3, 3, 3, 3], 
                focal_windows=[7, 5, 3, 1], 
                focal_topK=64, 
                focal_dilation=1, 
                focal_routing_topK=64, 
                focal_pool="conv", 
                focal_shape="circle", 
                expand_stages=[0, 1, 2, 3], 
                expand_sizes=[3, 3, 3, 3],
                expand_layer="all", 
                use_conv_embed=False, 
                mute_fine_grain=False, 
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
                               focal_kernel=focal_kernels[i_layer], 
                               topK=focal_topK, 
                               dilation=focal_dilation, 
                               shape=focal_shape, 
                               routing_topK=focal_routing_topK, 
                               expand_size=expand_sizes[i_layer], 
                               expand_layer=expand_layer,                           
                               use_conv_embed=use_conv_embed,
                               use_shift=use_shift, 
                               mute_fine_grain=mute_fine_grain, 
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
        focal_levels=[3,3,3,2], 
        focal_kernels=[3,3,3,1], 
        focal_windows=[7,7,7,7], 
        dilation=1, 
        # num_heads=[2,4,8,16],
        # num_heads=[4,8,16,32],
        # num_heads=[6,12,24,48],
        focal_topK=49, 
        focal_shape="square", 
        focal_routing_topK=128, 
        use_conv_embed=False, 
        mute_fine_grain=False,
        use_shift=False, 
        use_route=False, 
        focal_pool="conv_share"
    ).cuda()
    
    # ckpt_path = "/home/jwyang/azureblobs/vlpresults/projects/focal-transformer/focal_tiny_v2_global_softmax_localglobal2/pt-results/focal-transformer-33e2a88e-focal_tiny_v2_global_softmax_localglobal2-train_focal_tiny_transformer_rpe_moe__2_2_6_2___2_2_2_2_-f4f86e2b/ckpt_epoch_91.pth"
    # ckpt = torch.load(ckpt_path)
    # model.load_state_dict(ckpt['model'])

    # eval_transforms = build_transforms(img_size)
    
    # img = Image.open("/extra_disk/Datasets/imagenet/n01440764/ILSVRC2012_val_00017472.JPEG")

    # img_t = eval_transforms(img)

    # model(img_t.unsqueeze(0).cuda())

    # print(model)

    flops = model.flops()
    print(f"number of GFLOPs: {flops / 1e9}")

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of params: {n_parameters}")

    profile(model, x)

    model.train()
    tic = time.time()
    for t in range(10):
        out = model(x)
        label = torch.zeros(out.shape[0]).to(out.device).long()
        with torch.autograd.set_detect_anomaly(True):
            loss = nn.CrossEntropyLoss()(out, label)
            loss.backward()
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            for name, param in model.named_parameters():
                if param.grad is None:
                    print(name)
    print("memory cost: ", memory_used)
    print("time cost: ", time.time()-tic)
