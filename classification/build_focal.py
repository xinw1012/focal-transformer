# --------------------------------------------------------
# Focal Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Jianwei Yang (jianwyan@microsoft.com)
# --------------------------------------------------------

from .focal_transformer import FocalTransformer as focal
from .focal_transformer_moe import FocalTransformer as focal_moe
from .focal_transformer_moe_local import FocalTransformer as focal_moe_local
from .focal_transformer_moe_global import FocalTransformer as focal_moe_global

# from .focal_transformer_rpe import FocalTransformer as focal_rpe
# from .focal_transformer_rpe_sep_qkv import FocalTransformer as focal_rpe_sep_qkv

def build_model(config):
    model_type = config.MODEL.TYPE
    print(f"Creating model: {model_type}")
    model = eval(model_type)(
        img_size=config.DATA.IMG_SIZE,
        patch_size=config.MODEL.FOCAL.PATCH_SIZE,
        in_chans=config.MODEL.FOCAL.IN_CHANS,
        num_classes=config.MODEL.NUM_CLASSES,
        embed_dim=config.MODEL.FOCAL.EMBED_DIM,
        depths=config.MODEL.FOCAL.DEPTHS,
        num_heads=config.MODEL.FOCAL.NUM_HEADS,
        window_size=config.MODEL.FOCAL.WINDOW_SIZE,
        mlp_ratio=config.MODEL.FOCAL.MLP_RATIO,
        qkv_bias=config.MODEL.FOCAL.QKV_BIAS,
        qk_scale=config.MODEL.FOCAL.QK_SCALE,
        drop_rate=config.MODEL.DROP_RATE,
        drop_path_rate=config.MODEL.DROP_PATH_RATE,
        ape=config.MODEL.FOCAL.APE,
        patch_norm=config.MODEL.FOCAL.PATCH_NORM,
        use_shift=config.MODEL.FOCAL.USE_SHIFT, 
        expand_stages=config.MODEL.FOCAL.EXPAND_STAGES,
        expand_sizes=config.MODEL.FOCAL.EXPAND_SIZES, 
        expand_layer=config.MODEL.FOCAL.EXPAND_LAYER,         
        focal_pool=config.MODEL.FOCAL.FOCAL_POOL,     
        focal_stages=config.MODEL.FOCAL.FOCAL_STAGES, 
        focal_windows=config.MODEL.FOCAL.FOCAL_WINDOWS,                                                   
        focal_levels=config.MODEL.FOCAL.FOCAL_LEVELS,    
        use_conv_embed=config.MODEL.FOCAL.USE_CONV_EMBED, 
        use_conv_route=config.MODEL.FOCAL.USE_CONV_ROUTE, 
        use_layerscale=config.MODEL.FOCAL.USE_LAYERSCALE, 
        use_pre_norm=config.MODEL.FOCAL.USE_PRE_NORM, 
        use_checkpoint=config.TRAIN.USE_CHECKPOINT
    )
    return model
