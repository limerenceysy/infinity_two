#!/usr/bin/env python3
"""
查看 Infinity 模型中 CrossAttention 的 mat_kv 和 proj 参数权重工具

功能：
1. 列出所有 CrossAttention 层的 mat_kv 和/或 proj 参数信息
2. 支持选择只查看 mat_kv 或只查看 proj
3. 使用简化加载（不需要 VAE 等参数）
"""

import os
import sys
import torch
import torch.nn as nn
import argparse
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))


def load_model_simple(model_path, model_type='infinity_2b'):
    """简化加载模型（仅用于查看层信息，不需要VAE等参数）"""
    from infinity.models.infinity import Infinity
    
    # 根据模型类型设置参数
    model_configs = {
        "infinity_2b": dict(depth=32, embed_dim=2048, num_heads=2048 // 128, drop_path_rate=0.1, mlp_ratio=4, block_chunks=8),
        "infinity_layer12": dict(depth=12, embed_dim=768, num_heads=8, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4),
        "infinity_layer16": dict(depth=16, embed_dim=1152, num_heads=12, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4),
        "infinity_layer24": dict(depth=24, embed_dim=1536, num_heads=16, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4),
        "infinity_layer32": dict(depth=32, embed_dim=2080, num_heads=20, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4),
        "infinity_layer40": dict(depth=40, embed_dim=2688, num_heads=24, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4),
        "infinity_layer48": dict(depth=48, embed_dim=3360, num_heads=28, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4),
    }
    
    kwargs = model_configs.get(model_type, model_configs["infinity_2b"])
    
    # 先读取权重，推断输出头大小 V（来自 head.weight 的第一维），以匹配 checkpoint
    state_dict = torch.load(model_path, map_location='cpu')
    inferred_V = None
    for k, v in state_dict.items():
        if k.endswith('head.weight') and v.ndim == 2:
            inferred_V = v.shape[0]
            break
    if inferred_V is None:
        inferred_V = 1024
    
    # 提供一个最小 VAE stub（仅需 embed_dim 与 vocab_size 字段）
    class _DummyVAE:
        def __init__(self, vocab_size):
            self.embed_dim = 32
            self.vocab_size = vocab_size
    
    model = Infinity(
        vae_local=_DummyVAE(inferred_V),
        text_channels=2048,
        text_maxlen=512,
        shared_aln=True,
        checkpointing='full-block',
        customized_flash_attn=False,
        fused_norm=True,
        pad_to_multiplier=128,
        use_flex_attn=False,
        add_lvl_embeding_only_first_block=1,
        use_bit_label=0,  # 设为0以避免访问 vae_local.quantizer.lfq.mask
        rope2d_each_sa_layer=1,
        rope2d_normalized_by_hw=2,
        pn="1M",
        apply_spatial_patchify=0,
        inference_mode=True,
        train_h_div_w_list=[1.0],
        **kwargs,
    )
    
    # 加载权重（宽松匹配）
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    return model


def get_parameter_tensor(module, param_name):
    """获取参数张量，支持 Linear 的 weight 和 Parameter"""
    param_obj = getattr(module, param_name, None)
    if param_obj is None:
        return None, None
    
    if isinstance(param_obj, nn.Linear):
        return param_obj.weight, 'linear_weight'
    elif isinstance(param_obj, torch.nn.Parameter):
        return param_obj, 'parameter'
    else:
        return None, None


def list_crossattention_params(model, show_mat_kv=True, show_proj=True):
    """列出所有 CrossAttention 层的指定参数信息"""
    layers = []
    
    for name, module in model.named_modules():
        if module.__class__.__name__ == "CrossAttention":
            layer_info = {'name': name, 'module': module}
            
            # 获取 mat_kv
            if show_mat_kv:
                mat_kv_weight, mat_kv_kind = get_parameter_tensor(module, 'mat_kv')
                if mat_kv_weight is not None:
                    layer_info['mat_kv'] = {
                        'weight': mat_kv_weight,
                        'kind': mat_kv_kind,
                        'shape': tuple(mat_kv_weight.shape),
                        'mean': mat_kv_weight.mean().item(),
                        'std': mat_kv_weight.std().item(),
                        'min': mat_kv_weight.min().item(),
                        'max': mat_kv_weight.max().item(),
                    }
                else:
                    layer_info['mat_kv'] = None
            
            # 获取 proj
            if show_proj:
                proj_weight, proj_kind = get_parameter_tensor(module, 'proj')
                if proj_weight is not None:
                    layer_info['proj'] = {
                        'weight': proj_weight,
                        'kind': proj_kind,
                        'shape': tuple(proj_weight.shape),
                        'mean': proj_weight.mean().item(),
                        'std': proj_weight.std().item(),
                        'min': proj_weight.min().item(),
                        'max': proj_weight.max().item(),
                    }
                else:
                    layer_info['proj'] = None
            
            # 获取 v_bias（mat_kv 相关的 bias）
            if show_mat_kv:
                v_bias = getattr(module, 'v_bias', None)
                if isinstance(v_bias, torch.nn.Parameter):
                    layer_info['v_bias'] = {
                        'weight': v_bias,
                        'kind': 'parameter',
                        'shape': tuple(v_bias.shape),
                        'mean': v_bias.mean().item(),
                        'std': v_bias.std().item(),
                        'min': v_bias.min().item(),
                        'max': v_bias.max().item(),
                    }
                else:
                    layer_info['v_bias'] = None
            
            layers.append(layer_info)
    
    return layers


def print_layer_info(layers, show_mat_kv=True, show_proj=True):
    """打印层信息"""
    param_names = []
    if show_mat_kv:
        param_names.append('mat_kv')
    if show_proj:
        param_names.append('proj')
    
    title = f"CrossAttention 层的 {', '.join(param_names)} 参数信息"
    
    print("\n" + "="*80)
    print(title)
    print("="*80)
    
    for idx, layer in enumerate(layers):
        print(f"\n层 {idx}: {layer['name']}")
        
        if show_mat_kv and layer.get('mat_kv') is not None:
            mk = layer['mat_kv']
            print(f"  mat_kv ({mk['kind']}):")
            print(f"    形状: {mk['shape']}")
            print(f"    统计: mean={mk['mean']:.6f}, std={mk['std']:.6f}, "
                  f"min={mk['min']:.6f}, max={mk['max']:.6f}")
        
        if show_mat_kv and layer.get('v_bias') is not None:
            vb = layer['v_bias']
            print(f"  v_bias ({vb['kind']}):")
            print(f"    形状: {vb['shape']}")
            print(f"    统计: mean={vb['mean']:.6f}, std={vb['std']:.6f}, "
                  f"min={vb['min']:.6f}, max={vb['max']:.6f}")
        
        if show_proj and layer.get('proj') is not None:
            pj = layer['proj']
            print(f"  proj ({pj['kind']}):")
            print(f"    形状: {pj['shape']}")
            print(f"    统计: mean={pj['mean']:.6f}, std={pj['std']:.6f}, "
                  f"min={pj['min']:.6f}, max={pj['max']:.6f}")
    
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='查看 Infinity 模型 CrossAttention 的 mat_kv 和 proj 参数')
    
    parser.add_argument('--model_path', type=str, required=True, help='模型权重文件路径')
    parser.add_argument('--model_type', type=str, default='infinity_2b',
                        choices=['infinity_2b', 'infinity_layer12', 'infinity_layer16', 
                                'infinity_layer24', 'infinity_layer32', 'infinity_layer40', 'infinity_layer48'],
                        help='模型类型')
    
    parser.add_argument('--mat_kv', action='store_true', default=False, 
                        help='查看 mat_kv 参数（包括 v_bias）')
    parser.add_argument('--proj', action='store_true', default=False,
                        help='查看 proj 参数')
    parser.add_argument('--all', action='store_true', default=False,
                        help='查看所有参数（mat_kv 和 proj）')
    
    args = parser.parse_args()
    
    # 确定要查看的参数
    if args.all:
        show_mat_kv = True
        show_proj = True
    elif args.mat_kv or args.proj:
        show_mat_kv = args.mat_kv
        show_proj = args.proj
    else:
        # 默认查看所有
        show_mat_kv = True
        show_proj = True
    
    if not os.path.exists(args.model_path):
        print(f"错误: 权重文件不存在: {args.model_path}")
        return
    
    print("加载模型（简化模式，仅用于查看参数信息）...")
    model = load_model_simple(args.model_path, args.model_type)
    
    # 列出所有 CrossAttention 层的参数
    layers = list_crossattention_params(model, show_mat_kv=show_mat_kv, show_proj=show_proj)
    
    if not layers:
        print("错误: 未找到任何 CrossAttention 层")
        return
    
    # 打印信息
    print_layer_info(layers, show_mat_kv=show_mat_kv, show_proj=show_proj)
    
    print(f"共找到 {len(layers)} 个 CrossAttention 层")
    print("完成!")


if __name__ == "__main__":
    main()

