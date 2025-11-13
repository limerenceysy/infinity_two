#!/usr/bin/env python3
"""
手动修改 Infinity 模型中 CrossAttention 的 mat_q 权重工具

功能：
1. 列出所有 CrossAttention 层
2. 修改指定层的 mat_q 权重（--layer_idx）
3. 修改所有层的 mat_q 权重（--all_layers）
4. 保存为新权重文件（不影响原权重）
5. 支持临时修改（用于快速测试）

使用方法：
  # 列出所有层
  python manual_modify_mat_q.py --model_path weights/infinity_2b_reg.pth --list_layers
  
  # 修改单层（例如第0层）
  python manual_modify_mat_q.py --model_path weights/infinity_2b_reg.pth --layer_idx 0 --method scale --scale 0.8
  
  # 修改所有层
  python manual_modify_mat_q.py --model_path weights/infinity_2b_reg.pth --all_layers --method scale --scale 0.8
"""

import os
import sys
import torch
import torch.nn as nn
import argparse
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from tools.run_infinity import (
    load_tokenizer,
    load_visual_tokenizer,
    load_transformer,
    add_common_arguments,
)


def _get_mat_q_tensor(module):
    """返回 (tensor, kind)；kind ∈ {"linear", "parameter"} 用于区分 mat_q 类型。"""
    mq = getattr(module, 'mat_q', None)
    if isinstance(mq, nn.Linear):
        return mq.weight, 'linear'
    if isinstance(mq, torch.nn.Parameter):
        return mq, 'parameter'
    raise AttributeError('Unsupported mat_q type')


def list_all_crossattention_layers(model):
    """列出所有 CrossAttention 层及其 mat_q 权重信息"""
    layers = []
    for name, module in model.named_modules():
        if module.__class__.__name__ == "CrossAttention":
            weight, kind = _get_mat_q_tensor(module)
            layers.append({
                'name': name,
                'shape': tuple(weight.shape),
                'mean': weight.mean().item(),
                'std': weight.std().item(),
                'min': weight.min().item(),
                'max': weight.max().item(),
                'module': module,
                'kind': kind,
            })
    return layers


def print_layer_info(layers):
    """打印层信息"""
    print("\n" + "="*80)
    print("所有 CrossAttention 层信息:")
    print("="*80)
    for idx, layer in enumerate(layers):
        print(f"\n层 {idx}: {layer['name']} (mat_q={layer.get('kind', 'linear')})")
        print(f"  形状: {layer['shape']}")
        print(f"  统计: mean={layer['mean']:.6f}, std={layer['std']:.6f}, "
              f"min={layer['min']:.6f}, max={layer['max']:.6f}")
    print("="*80 + "\n")


def modify_mat_q_weight(module, method, scale=1.0, noise_std=0.1, target_value=0.0):
    """
    修改 mat_q 权重
    
    Args:
        module: CrossAttention 模块
        method: 修改方法
            - 'zero': 置零
            - 'scale': 缩放 (scale倍)
            - 'noise': 添加噪声 (噪声标准差=noise_std)
            - 'set': 设置为指定值 (target_value)
            - 'replace': 用新张量替换（需要传入 target_value 作为新张量）
        scale: 缩放倍数
        noise_std: 噪声标准差
        target_value: 目标值或新张量
    """
    weight_tensor, kind = _get_mat_q_tensor(module)
    original_weight = weight_tensor.clone()
    
    with torch.no_grad():
        if method == 'zero':
            weight_tensor.zero_()
            print(f"  已将权重置零")
            
        elif method == 'scale':
            weight_tensor.mul_(scale)
            print(f"  已将权重缩放 {scale} 倍")
            print(f"  原均值: {original_weight.mean().item():.6f}, "
                  f"现均值: {weight_tensor.mean().item():.6f}")
            
        elif method == 'noise':
            noise = torch.randn_like(weight_tensor) * noise_std
            weight_tensor.add_(noise)
            print(f"  已添加噪声 (std={noise_std})")
            print(f"  原均值: {original_weight.mean().item():.6f}, "
                  f"现均值: {weight_tensor.mean().item():.6f}")
            
        elif method == 'set':
            weight_tensor.fill_(target_value)
            print(f"  已将权重设置为 {target_value}")
            
        elif method == 'replace':
            if isinstance(target_value, torch.Tensor):
                if target_value.shape != original_weight.shape:
                    raise ValueError(f"形状不匹配: 期望 {original_weight.shape}, 得到 {target_value.shape}")
                weight_tensor.copy_(target_value)
                print(f"  已用新张量替换")
            else:
                raise ValueError("replace 方法需要传入 torch.Tensor")
        
        else:
            raise ValueError(f"未知修改方法: {method}")


def save_modified_weights(model, original_path, output_path, layer_indices=None):
    """
    保存修改后的权重到新文件
    
    Args:
        model: 修改后的模型
        original_path: 原始权重文件路径
        output_path: 输出权重文件路径
        layer_indices: 被修改的层索引列表（用于记录）
    """
    print(f"\n正在保存修改后的权重到: {output_path}")
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # 保存完整的 state_dict
    state_dict = model.state_dict()
    torch.save(state_dict, output_path)
    
    print(f"✓ 权重已保存")
    if layer_indices:
        print(f"  修改的层索引: {layer_indices}")
    print(f"  原权重文件: {original_path} (未修改)")
    print(f"  新权重文件: {output_path}")


def load_model_simple(model_path, model_type='infinity_2b'):
    """简化加载模型（仅用于查看/临时修改，不依赖 VAE/文本编码器）。"""
    from infinity.models.infinity import Infinity
    # 与 predict.py/tools.run_infinity.py 保持一致的档位配置
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
        # 回退一个安全默认值
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

def main():
    # 第一阶段：最小参数解析（不引入 add_common_arguments，避免 --pn 约束）
    parser = argparse.ArgumentParser(description='手动修改 Infinity 模型 CrossAttention mat_q 权重')
    parser.add_argument('--model_path', type=str, required=True, help='模型权重文件路径')
    parser.add_argument('--model_type', type=str, default='infinity_2b',
                        choices=['infinity_2b', 'infinity_layer12', 'infinity_layer16', 'infinity_layer24', 'infinity_layer32', 'infinity_layer40', 'infinity_layer48'])
    parser.add_argument('--list_layers', action='store_true', help='列出所有 CrossAttention 层信息')
    parser.add_argument('--layer_idx', type=int, default=None, help='要修改的层索引（从0开始）。如果使用 --all_layers，则忽略此参数')
    parser.add_argument('--all_layers', action='store_true', help='修改所有 CrossAttention 层的 mat_q 权重')
    parser.add_argument('--method', type=str, choices=['zero', 'scale', 'noise', 'set', 'replace'], help='修改方法')
    parser.add_argument('--scale', type=float, default=1.0, help='缩放倍数（用于 scale 方法）')
    parser.add_argument('--noise_std', type=float, default=0.1, help='噪声标准差（用于 noise 方法）')
    parser.add_argument('--target_value', type=float, default=0.0, help='目标值（用于 set 方法）')
    parser.add_argument('--save_path', type=str, default=None, help='保存修改后权重的路径')
    parser.add_argument('--temporary', action='store_true', help='临时修改（不保存），用于快速测试')
    args, _ = parser.parse_known_args()
    
    # 必须提供的参数
    if not args.model_path:
        print("错误: 必须提供 --model_path")
        parser.print_help()
        return
    
    if not os.path.exists(args.model_path):
        print(f"错误: 权重文件不存在: {args.model_path}")
        return
    
    # 当仅列出/临时修改时，走简化加载（不依赖 VAE/文本编码器）
    if args.list_layers or args.temporary:
        print("加载模型（简化模式，仅查看/临时修改）...")
        model = load_model_simple(args.model_path, getattr(args, 'model_type', 'infinity_2b'))
    else:
        # 第二阶段：需要完整参数时，再引入 add_common_arguments（此时用户应显式提供 --pn 等）
        parser_full = argparse.ArgumentParser(description='手动修改 Infinity 模型 CrossAttention mat_q 权重（完整加载）')
        add_common_arguments(parser_full)
        # 复用第一阶段自定义参数
        parser_full.add_argument('--list_layers', action='store_true')
        parser_full.add_argument('--layer_idx', type=int, default=None)
        parser_full.add_argument('--all_layers', action='store_true', help='修改所有 CrossAttention 层的 mat_q 权重')
        parser_full.add_argument('--method', type=str, choices=['zero', 'scale', 'noise', 'set', 'replace'])
        parser_full.add_argument('--scale', type=float, default=1.0)
        parser_full.add_argument('--noise_std', type=float, default=0.1)
        parser_full.add_argument('--target_value', type=float, default=0.0)
        parser_full.add_argument('--save_path', type=str, default=None)
        parser_full.add_argument('--temporary', action='store_true')
        # 保留合理默认
        parser_full.set_defaults(model_type=args.model_type, pn='1M')
        args = parser_full.parse_args()

        # 加载 VAE/模型
        print("加载 VAE...")
        try:
            vae = load_visual_tokenizer(args)
        except Exception as e:
            print(f"警告: VAE 加载失败，使用 None: {e}")
            vae = None
        print("加载 Infinity 模型...")
        model = load_transformer(vae, args)
    
    # 列出所有 CrossAttention 层
    layers = list_all_crossattention_layers(model)
    
    if not layers:
        print("错误: 未找到任何 CrossAttention 层")
        return
    
    print_layer_info(layers)
    
    # 如果只是列出层信息，退出
    if args.list_layers:
        return
    
    # 检查是否提供了修改方法
    if args.method is None:
        print("错误: 必须提供 --method 来指定修改方法")
        return
    
    # 确定要修改的层索引列表
    if args.all_layers:
        # 修改所有层
        layer_indices = list(range(len(layers)))
        print(f"\n将修改所有 {len(layers)} 个 CrossAttention 层")
    else:
        # 修改指定层
        if args.layer_idx is None:
            print("错误: 必须提供 --layer_idx 来指定要修改的层，或使用 --all_layers 修改所有层")
            print(f"可用层索引: 0 到 {len(layers)-1}")
            return
        
        if args.layer_idx < 0 or args.layer_idx >= len(layers):
            print(f"错误: 层索引 {args.layer_idx} 超出范围 [0, {len(layers)-1}]")
            return
        
        layer_indices = [args.layer_idx]
    
    # 执行修改
    modified_indices = []
    print(f"\n开始修改 {len(layer_indices)} 个层...")
    print("=" * 80)
    
    for idx in layer_indices:
        target_layer = layers[idx]
        print(f"\n层 {idx}/{len(layers)-1}: {target_layer['name']}")
        print(f"  形状: {target_layer['shape']}")
        print(f"  原始统计: mean={target_layer['mean']:.6f}, std={target_layer['std']:.6f}")
        
        try:
            modify_mat_q_weight(
                target_layer['module'],
                method=args.method,
                scale=args.scale,
                noise_std=args.noise_std,
                target_value=args.target_value,
            )
            
            # 获取修改后的权重（处理不同类型的mat_q）
            weight_tensor, _ = _get_mat_q_tensor(target_layer['module'])
            print(f"  修改后统计: mean={weight_tensor.mean().item():.6f}, std={weight_tensor.std().item():.6f}")
            
            modified_indices.append(idx)
            
        except Exception as e:
            print(f"  ✗ 错误: 修改层 {idx} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "=" * 80)
    print(f"完成修改: {len(modified_indices)}/{len(layer_indices)} 个层成功修改")
    
    if not modified_indices:
        print("错误: 没有成功修改任何层")
        return
    
    # 保存或临时修改
    if args.temporary:
        print("\n✓ 临时修改完成（未保存）")
        print("  可以在推理脚本中使用此修改后的模型")
        print("  注意: 如果重新加载模型，修改会丢失")
    else:
        if args.save_path:
            save_modified_weights(model, args.model_path, args.save_path, modified_indices)
        else:
            # 自动生成保存路径
            base_path = Path(args.model_path)
            if args.all_layers:
                save_path = base_path.parent / f"{base_path.stem}_all_layers_{args.method}"
                if args.method == 'scale':
                    save_path = base_path.parent / f"{base_path.stem}_all_layers_{args.method}{args.scale}"
                elif args.method == 'noise':
                    save_path = base_path.parent / f"{base_path.stem}_all_layers_{args.method}{args.noise_std}"
                save_path = save_path.with_suffix(base_path.suffix)
            else:
                save_path = base_path.parent / f"{base_path.stem}_modified_layer{args.layer_idx}_{args.method}{base_path.suffix}"
            save_modified_weights(model, args.model_path, str(save_path), modified_indices)
    
    print("\n完成!")


if __name__ == "__main__":
    main()


