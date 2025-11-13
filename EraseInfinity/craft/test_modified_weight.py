#!/usr/bin/env python3
"""
快速测试修改后的权重文件

用法：
python test_modified_weight.py --model_path weights/infinity_2b_reg_layer0_zero.pth --prompt "a cat"
"""

import os
import sys
import argparse

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from tools.run_infinity import (
    load_tokenizer,
    load_visual_tokenizer,
    load_transformer,
    gen_one_img,
    add_common_arguments,
)
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w

def main():
    parser = argparse.ArgumentParser(description='测试修改后的 Infinity 模型权重')
    add_common_arguments(parser)
    
    parser.add_argument('--prompt', type=str, required=True, help='测试提示词')
    parser.add_argument('--save_file', type=str, default='./test_output.jpg', help='输出图像路径')
    
    # 设置默认参数
    parser.set_defaults(
        model_type='infinity_2b',
        vae_type=32,
        add_lvl_embeding_only_first_block=1,
        use_bit_label=1,
        rope2d_each_sa_layer=1,
        rope2d_normalized_by_hw=2,
        use_scale_schedule_embedding=0,
        apply_spatial_patchify=0,
        use_flex_attn=0,
        bf16=1,
        checkpoint_type='torch',
        text_channels=2048,
        cfg='3',
        tau=1.0,
        seed=0,
    )
    
    args = parser.parse_args()
    
    if not args.model_path:
        print("错误: 必须提供 --model_path")
        return
    
    if not os.path.exists(args.model_path):
        print(f"错误: 权重文件不存在: {args.model_path}")
        return
    
    print(f"加载模型: {args.model_path}")
    print(f"提示词: {args.prompt}")
    
    # 加载文本编码器
    if args.text_encoder_ckpt:
        print("加载文本编码器...")
        text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    else:
        print("警告: 未提供 --text_encoder_ckpt，跳过文本编码")
        text_tokenizer, text_encoder = None, None
    
    # 加载 VAE
    print("加载 VAE...")
    vae = load_visual_tokenizer(args)
    
    # 加载 Infinity 模型（使用修改后的权重）
    print("加载 Infinity 模型...")
    infinity = load_transformer(vae, args)
    
    # 准备 scale_schedule
    if args.pn:
        h_div_w = 1.0
        scale_schedule = dynamic_resolution_h_w[h_div_w][args.pn]['scales']
        scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
    else:
        scale_schedule = None
    
    # 生成图像
    if text_tokenizer and text_encoder:
        print("生成图像...")
        import cv2
        
        generated_image = gen_one_img(
            infinity,
            vae,
            text_tokenizer,
            text_encoder,
            args.prompt,
            g_seed=args.seed if args.seed else None,
            gt_leak=0,
            gt_ls_Bl=None,
            cfg_list=float(args.cfg) if isinstance(args.cfg, str) else args.cfg,
            tau_list=args.tau,
            scale_schedule=scale_schedule,
            cfg_insertion_layer=[getattr(args, 'cfg_insertion_layer', 0)],
            vae_type=args.vae_type,
            sampling_per_bits=getattr(args, 'sampling_per_bits', 1),
            enable_positive_prompt=0,
        )
        
        # 保存图像
        os.makedirs(os.path.dirname(args.save_file) if os.path.dirname(args.save_file) else '.', exist_ok=True)
        cv2.imwrite(args.save_file, generated_image.cpu().numpy())
        print(f"✓ 图像已保存: {args.save_file}")
    else:
        print("警告: 无法生成图像（缺少文本编码器）")
    
    print("完成!")

if __name__ == "__main__":
    main()



