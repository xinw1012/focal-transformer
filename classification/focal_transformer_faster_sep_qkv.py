# --------------------------------------------------------
# Focal Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Jianwei Yang (jianwyan@microsoft.com)
# Written originally by Ze Liu (Author of Swin Transformer)
# --------------------------------------------------------

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

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

    def __init__(self, dim, input_resolution, expand_size, shift_size, window_size, window_size_glo, focal_window, 
                    focal_level, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., pool_method="none"):

        super().__init__()
        self.dim = dim
        self.shift_size = shift_size
        self.expand_size = expand_size
        self.window_size = window_size  # Wh, Ww
        self.window_size_glo = window_size_glo
        self.pool_method = pool_method
        self.input_resolution = input_resolution # NWh, NWw
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.focal_level = focal_level

        self.window_size_of_key = self.window_size[0] * self.window_size[1] if self.expand_size == 0 else \
            (5 * self.window_size[0] * self.window_size[1] - 4 * (self.window_size[0] -  self.expand_size) * (self.window_size[0] -  self.expand_size))

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(1, num_heads, window_size[0] * window_size[1], self.window_size_of_key))  # Wh*Ww, nH, Wh_unfold*Ww_unfold
        trunc_normal_(self.relative_position_bias_table, std=.02)

        pooled_windows = 0
        if pool_method != "none" and focal_level > 1:
            self.max_nwh_glo = []
            self.max_nww_glo = []
            self.relative_position_bias_table_patch2wins = nn.ParameterList()

            # build relative position bias between local patch and pooled windows
            for k in range(focal_level-1):
                focal_window_k = focal_window
                relative_position_bias_table_xwin = nn.Parameter(
                    torch.zeros(
                        self.num_heads,
                        window_size[0]*window_size[1], 
                        focal_window_k*focal_window_k, 
                        )
                ) # 2*Wh-1 * 2*Ww-1, nH
                trunc_normal_(relative_position_bias_table_xwin, std=.02)
                self.relative_position_bias_table_patch2wins.append(relative_position_bias_table_xwin)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        # get mask for rolled k and rolled v
        mask_tl = torch.ones(self.window_size[0], self.window_size[1]); mask_tl[:-self.expand_size, :-self.expand_size] = 0
        mask_tr = torch.ones(self.window_size[0], self.window_size[1]); mask_tr[:-self.expand_size, self.expand_size:] = 0
        mask_bl = torch.ones(self.window_size[0], self.window_size[1]); mask_bl[self.expand_size:, :-self.expand_size] = 0
        mask_br = torch.ones(self.window_size[0], self.window_size[1]); mask_br[self.expand_size:, self.expand_size:] = 0
        mask_rolled = torch.stack((mask_tl, mask_tr, mask_bl, mask_br), 0).flatten(0)
        self.register_buffer("valid_ind_rolled", (mask_rolled == 1).nonzero().flatten(0))

        if self.expand_size > 0 and focal_level > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            attn_mask = get_roll_masks(H, W, self.window_size[0], self.expand_size)
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x_all, mask_all=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        x = x_all[0] # 
        tic = time.time()
        B, nH, nW, C = x.shape

        (q, k, v) = map(
            lambda t: t(x), (self.q, self.k, self.v)
        ) # B, nH, nW, C

        # partition q map
        (q_windows, k_windows, v_windows) = map(
            lambda t: window_partition(t, self.window_size[0]).view(
            -1, self.window_size[0] * self.window_size[0], self.num_heads, C // self.num_heads
            ).permute(0, 2, 1, 3).contiguous(), 
            (q, k, v)
        )
        # print("time cost for qkv partition: ", time.time() - tic)
        # tic = time.time()

        if self.expand_size > 0 and x_all[1] is not None:
            # mask out tokens in current window
            x_expanded = x_all[1]
            x_expanded = x_expanded[:, self.valid_ind_rolled]

            (k_expanded, v_expanded) = map(
                lambda t: t(x_expanded).reshape(x_expanded.shape[0], -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous(),
                (self.k, self.v)
            )
            # k_expanded = self.k(x_expanded).reshape(x_expanded.shape[0], -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
            # v_expanded = self.v(x_expanded).reshape(x_expanded.shape[0], -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
            k_rolled = torch.cat((k_windows, k_expanded), 2)
            v_rolled = torch.cat((v_windows, v_expanded), 2)
        else:
            k_rolled = k_windows; v_rolled = v_windows; 

        # print("time cost for kv rolling: ", time.time() - tic)
        # tic = time.time()
        if self.pool_method != "none" and len(x_all) > 2:
            x_pooled = torch.cat(x_all[2:], 1)
            B_p, nW_p, C_p = x_pooled.shape
            (k_pooled, v_pooled) = map(
                lambda t: t(x_pooled).reshape(B_p, nW_p, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous(), 
                (self.k, self.v)
            )
            # k_pooled = self.k(x_pooled).reshape(B_p, nW_p, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
            # v_pooled = self.v(x_pooled).reshape(B_p, nW_p, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()            
            k_all = torch.cat((k_rolled, k_pooled), 2)
            v_all = torch.cat((v_rolled, v_pooled), 2)
        else:
            k_all = k_rolled
            v_all = v_rolled
        
        N = k_all.shape[-2]
        q_windows = q_windows * self.scale
        attn = (q_windows @ k_all.transpose(-2, -1))  # B*nW, nHead, window_size*window_size, focal_window_size*focal_window_size

        # print("time cost for computing attn: ", time.time() - tic)
        # tic = time.time() 
        # add relative position bias for patches inside a window
        window_area = self.window_size[0] * self.window_size[1]        
        window_area_rolled = k_rolled.shape[2]
        attn[:, :, :window_area, :window_area_rolled] = attn[:, :, :window_area, :window_area_rolled] + self.relative_position_bias_table
        
        # NOTE: add mask between window to unfolded window
        # if mask_rolled is not None:
        #     attn[:, :, :window_area, :window_area_rolled] = attn[:, :, :window_area, :window_area_rolled] + mask_rolled[None, None, None, :]

        if self.pool_method != "none" and self.focal_level > 1:
            # add relative position bias for different windows in an image        
            offset = window_area_rolled
            for k in range(self.focal_level-1):
                relative_position_bias_xwin = self.relative_position_bias_table_patch2wins[k] # NWh*NWw,NWh*NWw,nH
                attn[:, :, :window_area, offset:(offset + x_all[k+2].shape[1])] = \
                    attn[:, :, :window_area, offset:(offset + x_all[k+2].shape[1])] + relative_position_bias_xwin.unsqueeze(0)
                # add mask
                attn[:, :, :window_area, offset:(offset + x_all[k+2].shape[1])] = \
                    attn[:, :, :window_area, offset:(offset + x_all[k+2].shape[1])] + \
                        mask_all[k+2][:, :, None, None, :].repeat(attn.shape[0] // mask_all[k+2].shape[1], 1, 1, 1, 1).view(-1, 1, 1, mask_all[k+2].shape[-1])
                offset += x_all[k+2].shape[1]
        # print("time cost for adding masks: ", time.time() - tic)
        # tic = time.time()
        if mask_all[0] is not None:
            nW = mask_all[0].shape[0]
            attn = attn.view(attn.shape[0] // nW, nW, self.num_heads, window_area, N)
            attn[:, :, :, :, :window_area] = attn[:, :, :, :, :window_area] + mask_all[0][None, :, None, :, :]
            attn = attn.view(-1, self.num_heads, window_area, N)
            attn = self.softmax(attn)
        else:          
            attn = self.softmax(attn)
        
        attn = self.attn_drop(attn)

        # get attn statistics during evaluation
        if not self.training:
            attn_m = attn.mean(0)
            attn_window = attn_m[:,:,:49].sum(2).mean(1).mean()
            attn_surrounding = attn_m[:,:,49:window_area_rolled].sum(2).mean(1).mean()
            attn_global = attn_m[:,:,window_area_rolled:].sum(2).mean(1).mean()
            self.attn_dist = torch.stack((attn_window, attn_surrounding, attn_global), 0)

            attn_global =  attn_m[:,:,window_area_rolled:].mean(0)

        x = (attn @ v_all).transpose(1, 2).reshape(attn.shape[0], window_area, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        # print("time cost for compute v: ", time.time() - tic)
        # tic = time.time()
        # import pdb; pdb.set_trace()
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
        if self.pool_method != "none":
            flops += self.num_heads * N * (self.dim // self.num_heads) * (unfold_size * unfold_size)          
        if self.expand_size > 0:
            flops += self.num_heads * N * (self.dim // self.num_heads) * ((window_size + 2*self.expand_size)**2-window_size**2)          

        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        if self.pool_method != "none":
            flops += self.num_heads * N * (self.dim // self.num_heads) * (unfold_size * unfold_size)          
        if self.expand_size > 0:
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
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, pool_method="none",  
                 focal_level=1, focal_window=1, use_layerscale=False, layerscale_value=1e-4):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.pool_method = pool_method
        self.focal_level = focal_level
        self.focal_window = focal_window
        self.expand_size = expand_size 
        self.use_layerscale = use_layerscale

        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.window_size_glo = self.window_size

        self.pool_layers = nn.ModuleList()
        self.unfolds = nn.ModuleList()
        for k in range(self.focal_level-1):

            self.unfolds += [nn.Unfold(
                kernel_size=(self.focal_window, self.focal_window), 
                stride=1, padding=self.focal_window // 2)
            ]

            window_size_glo = math.floor(self.window_size_glo * (2 ** k))

            if self.pool_method == "fc":
                self.pool_layers.append(nn.Linear(window_size_glo * window_size_glo, 1))
                self.pool_layers[-1].weight.data.fill_(1./(window_size_glo * window_size_glo))
                self.pool_layers[-1].bias.data.fill_(0)
            elif self.pool_method == "fp1":
                H, W = self.input_resolution
                self.num_windows = int(H * W / (window_size_glo ** 2))
                self.pool_layers.append(nn.Linear(window_size_glo * window_size_glo,  self.num_windows))
            elif self.pool_method == "fp2":
                H, W = self.input_resolution
                self.num_windows = int(H * W / (window_size_glo ** 2))
                self.pool_layers.append(nn.Conv2d(self.num_windows, self.num_windows, kernel_size=window_size_glo, stride=window_size_glo, groups=self.num_windows))
            elif self.pool_method == "conv":
                self.pool_layers.append(nn.Conv2d(dim, dim, kernel_size=window_size_glo, stride=window_size_glo, groups=dim))

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, input_resolution=input_resolution, expand_size=expand_size, shift_size=self.shift_size, window_size=to_2tuple(self.window_size), 
            window_size_glo=to_2tuple(self.window_size_glo), focal_window=focal_window, 
            focal_level=focal_level, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, pool_method=pool_method)

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

        if self.focal_level > 0 and self.expand_size > 0:
            # if we expand the finest granularity 
            x_tl = torch.roll(shifted_x, shifts=(-self.expand_size, -self.expand_size), dims=(1, 2))
            x_tr = torch.roll(shifted_x, shifts=(-self.expand_size, self.expand_size), dims=(1, 2))
            x_bl = torch.roll(shifted_x, shifts=(self.expand_size, -self.expand_size), dims=(1, 2))
            x_br = torch.roll(shifted_x, shifts=(self.expand_size, self.expand_size), dims=(1, 2))  

            (x_tl_windows, x_tr_windows, x_bl_windows, x_br_windows) = map(
                lambda t: window_partition(t, self.window_size).view(-1, self.window_size * self.window_size, C), 
                (x_tl, x_tr, x_bl, x_br)
            )            
            # NOTE: if expand_size is not equal to half of window_size, there will be either 
            # overlapped region (expand_size < window_size / 2) or missed tokens (expand_size > window_size / 2)
            x_rolled = torch.cat((x_tl_windows, x_tr_windows, x_bl_windows, x_br_windows), 1)

            x_windows_all += [x_rolled]
            x_window_masks_all += [None]
        else:
            x_windows_all += [None]
            x_window_masks_all += [None]
        
        if self.focal_level > 1 and self.pool_method != "none": 
            # if we add coarser granularity and the pool method is not none          
            for k in range(self.focal_level-1):                
                window_size_glo = math.floor(self.window_size_glo * (2 ** k))
                x_windows_noreshape = window_partition_noreshape(shifted_x, window_size_glo) # B, nw, nw, window_size, window_size, C    
                nWh, nWw = x_windows_noreshape.shape[1:3]
                if self.pool_method == "mean":
                    x_windows_pooled = x_windows_noreshape.mean([3, 4]).view(B, -1, C) # B, nw * nw, C
                elif self.pool_method == "fc":
                    x_windows_noreshape = x_windows_noreshape.view(B, -1, window_size_glo*window_size_glo, C).permute(0, 1, 3, 2).contiguous() # B, nw * nw, C, wsize**2
                    x_windows_pooled = self.pool_layers[k](x_windows_noreshape).view(B, -1, C)           
                elif self.pool_method == "fp1":
                    x_windows_noreshape = x_windows_noreshape.view(B, -1, window_size_glo*window_size_glo, C).permute(0, 1, 3, 2).contiguous() # B, nw * nw, C, wsize**2
                    x_windows_pooled = self.pool_layers[k](x_windows_noreshape).permute(0, 1, 3, 2).contiguous()  # B, nw * nw, nw * nw, C
                    x_windows_pooled = x_windows_pooled.view(B * self.num_windows, -1, C) # nW*B, nw*nw, C                
                elif self.pool_method == "fp2":
                    x_windows_noreshape = x_windows_noreshape.view(B, -1, window_size_glo, window_size_glo, C).permute(0, 4, 1, 2, 3).contiguous() # B, C,  nw * nw, wsize, wsize
                    x_windows_noreshape = x_windows_noreshape.view(-1, self.num_windows, window_size_glo, window_size_glo).contiguous() # B * C,  nw * nw, wsize, wsize    
                    x_windows_pooled = self.pool_layers[k](x_windows_noreshape).view(B, C, -1).permute(0, 2, 1).contiguous() # B, nw * nw, C                
                elif self.pool_method == "max":
                    x_windows_pooled = x_windows_noreshape.max(-2)[0].max(-2)[0].view(B, -1, C) # B, nw * nw, C
                elif self.pool_method == "conv":
                    x_windows_noreshape = x_windows_noreshape.view(-1, window_size_glo, window_size_glo, C).permute(0, 3, 1, 2).contiguous() # B * nw * nw, C, wsize, wsize
                    x_windows_pooled = self.pool_layers[k](x_windows_noreshape).view(B, -1, C) # B, nw * nw, C           

                x_windows_pooled = x_windows_pooled.permute(0, 2, 1).contiguous().view(B, C, nWh, nWw)
                unfolded_x = self.unfolds[k](x_windows_pooled).view(
                    B, C, self.focal_window, self.focal_window, -1).permute(0, 4, 2, 3, 1).contiguous().\
                    view(-1, self.focal_window, self.focal_window, C)  # (B x (nH*nW)) x unfold_wsize x unfold_wsize x C
                x_windows_all += [unfolded_x.view(unfolded_x.shape[0], -1, unfolded_x.shape[-1])]

                mask = x_windows_pooled.new(x_windows_pooled.shape[2], x_windows_pooled.shape[3]).fill_(1)
                unfolded_mask = self.unfolds[k](mask.unsqueeze(0).unsqueeze(1)).view(
                    1, 1, self.focal_window, self.focal_window, -1).contiguous().permute(0, 4, 2, 3, 1).contiguous().\
                    view(-1, self.focal_window, self.focal_window, 1)
                x_window_masks = unfolded_mask.flatten(1).unsqueeze(0)
                x_window_masks = x_window_masks.masked_fill(x_window_masks == 0, float(-100.0)).masked_fill(x_window_masks > 0, float(0.0))            
                x_window_masks_all += [x_window_masks]
        else:
            x_windows_all += [None]
            x_window_masks_all += [None]

        attn_windows = self.attn(x_windows_all, mask_all=x_window_masks_all)  # nW*B, window_size*window_size, C

        if not self.training:
            self.attn_dist = self.attn.attn_dist
        
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

        if self.pool_method != "none":
            # window squeeze
            flops += nW * self.dim * self.window_size * self.window_size           
            # qkv for global
            flops += nW * self.dim * 2 * self.dim

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

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

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
                 drop_path=0., norm_layer=nn.LayerNorm, pool_method="none", 
                 focal_level=1, focal_window=1, use_conv_embed=False, use_shift=False, use_pre_norm=False, 
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
                                 focal_level=focal_level, 
                                 focal_window=focal_window, 
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
        self.attn_dist = []
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
            if not self.training:
                self.attn_dist.append(blk.attn_dist)

        if self.downsample is not None:
            x = x.view(x.shape[0], self.input_resolution[0], self.input_resolution[1], -1).permute(0, 3, 1, 2).contiguous()
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
                focal_pool="fc", 
                expand_stages=[0, 1, 2, 3], 
                expand_sizes=[3, 3, 3, 3],
                expand_layer="all", 
                use_conv_embed=False, 
                use_layerscale=False, 
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
                               downsample=PatchEmbed if (i_layer < self.num_layers - 1) else None,
                               focal_level=focal_levels[i_layer], 
                               focal_window=focal_windows[i_layer], 
                               expand_size=expand_sizes[i_layer], 
                               expand_layer=expand_layer,                           
                               use_conv_embed=use_conv_embed,
                               use_shift=use_shift, 
                               use_pre_norm=use_pre_norm, 
                               use_checkpoint=use_checkpoint, 
                               use_layerscale=use_layerscale, 
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

        self.attn_dist = []
        for layer in self.layers:
            x = layer(x)
            if not self.training:
                self.attn_dist += layer.attn_dist
        if not self.training:
            self.attn_dist = torch.stack(self.attn_dist)
        
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