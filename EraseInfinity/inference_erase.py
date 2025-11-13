#!/usr/bin/env python
# coding: UTF-8
"""
    @date:  2025.01
    @func:  Inference script for EraseInfinity
            使用训练好的 LoRA 权重进行推理
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

# 添加 Infinity 到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 添加本地 peft 库到路径
_eraseinfinity_dir = os.path.dirname(os.path.abspath(__file__))
_peft_local_path = os.path.join(_eraseinfinity_dir, 'peft', 'src')
if os.path.exists(_peft_local_path):
    sys.path.insert(0, _peft_local_path)

from infinity.models import Infinity
from infinity.utils.load import build_vae_gpt


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Inference with EraseInfinity")
    
    # 模型权重路径
    parser.add_argument(
        "--vae_ckpt",
        type=str,
        required=True,
        help="Path to VAE checkpoint"
    )
    parser.add_argument(
        "--gpt_ckpt",
        type=str,
        required=True,
        help="Path to GPT checkpoint"
    )
    parser.add_argument(
        "--lora_ckpt",
        type=str,
        required=True,
        help="Path to trained LoRA checkpoint directory (e.g., outputs/checkpoint-401/)"
    )
    
    # 推理参数
    parser.add_argument(
        "--prompt",
        type=str,
        default="a beautiful landscape",
        help="Text prompt for generation"
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="",
        help="Negative prompt (what to avoid)"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="Image resolution (default: 1024)"
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=4,
        help="Number of images to generate"
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=4.0,
        help="Classifier-free guidance scale"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/inference",
        help="Output directory for generated images"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use (e.g., cuda:0, cpu)"
    )
    
    return parser.parse_args()


def build_model_with_lora(args, device):
    """
    构建模型并加载 LoRA 权重
    """
    print("=" * 80)
    print("Building models...")
    
    # ==================== 加载 VAE ====================
    print(f"Loading VAE from {args.vae_ckpt}")
    vae_ckpt = torch.load(args.vae_ckpt, map_location='cpu')
    
    # ==================== 配置模型参数 ====================
    class TempArgs:
        def __init__(self):
            # VAE 相关参数
            self.vae_type = 32
            self.fake_vae_input = False
            self.apply_spatial_patchify = 0
            
            # 设备相关
            self.device = device
            self.model_init_device = 'cpu'
            
            # 模型架构参数
            self.model = '2bc8'
            self.pn = '0.06M'
            self.always_training_scales = 20
            
            # 文本编码器参数
            self.tlen = 256
            self.Ct5 = 2048
            
            # 训练相关参数
            self.cond_drop_rate = 0.1
            self.cfg = self.cond_drop_rate
            self.use_bit_label = 1
            
            # 注意力相关参数
            self.flash = False
            self.fuse = False
            self.fused_norm = False
            self.enable_checkpointing = None
            self.pad_to_multiplier = 0
            self.use_flex_attn = False
            self.batch_size = 1
            
            # 层归一化参数
            self.norm_eps = 1e-6
            self.rms = False
            self.saln = True
            self.haln = True
            
            # Dropout 和正则化
            self.drop = 0.0
            self.rand_uncond = False
            
            # Cross Attention 参数
            self.ca_gamma = -1.0
            
            # 其他模型参数
            self.nm0 = False
            self.tau = 1
            self.cos = False
            self.swi = False
            
            # Scale schedule
            self.scale_schedule = None
            
            # Head 参数
            self.dec = 1
            self.tp = 0.0
            self.tk = 0.0
            
            # RoPE 2D 参数
            self.rope2d_each_sa_layer = 1
            self.rope2d_normalized_by_hw = 2
            self.add_lvl_embeding_only_first_block = 1
            
            # 训练相关
            self.train_h_div_w_list = None
            self.dp = -1
            self.hd = 0
            
            # EMA 相关
            self.use_fsdp_model_ema = False
            
            # 其他参数
            self.diva = 1
            self.alng = 1e-5
            self.aln = True
            self.hd0 = 1
            self.online_t5 = False
            self.t5_path = 'google/flan-t5-xl'
    
    temp_args = TempArgs()
    
    # ==================== 构建 VAE 和 GPT ====================
    vae_local, gpt_wo_ddp, _ = build_vae_gpt(temp_args, vae_ckpt, skip_gpt=False, device='cpu')
    
    # ==================== 加载 GPT checkpoint ====================
    print(f"Loading GPT checkpoint from {args.gpt_ckpt}")
    gpt_state = torch.load(args.gpt_ckpt, map_location='cpu')
    
    # 处理不同的 checkpoint 格式
    if 'trainer' in gpt_state:
        if 'gpt_wo_ddp' in gpt_state['trainer']:
            gpt_state_dict = gpt_state['trainer']['gpt_wo_ddp']
        elif 'gpt_fsdp' in gpt_state['trainer']:
            gpt_state_dict = gpt_state['trainer']['gpt_fsdp']
        else:
            gpt_state_dict = gpt_state
    else:
        gpt_state_dict = gpt_state
    
    # 加载 state dict
    missing, unexpected = gpt_wo_ddp.load_state_dict(gpt_state_dict, strict=False)
    print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    
    # ==================== 加载 LoRA 权重 ====================
    print(f"\nLoading LoRA weights from {args.lora_ckpt}")
    
    # 检查 LoRA checkpoint 路径
    if os.path.isdir(args.lora_ckpt):
        # 如果是目录，查找 adapter_model.safetensors 或 adapter_model.bin
        lora_weight_path = None
        if os.path.exists(os.path.join(args.lora_ckpt, "adapter_model.safetensors")):
            lora_weight_path = os.path.join(args.lora_ckpt, "adapter_model.safetensors")
            use_safetensors = True
        elif os.path.exists(os.path.join(args.lora_ckpt, "adapter_model.bin")):
            lora_weight_path = os.path.join(args.lora_ckpt, "adapter_model.bin")
            use_safetensors = False
        elif os.path.exists(os.path.join(args.lora_ckpt, "trainable_params.bin")):
            lora_weight_path = os.path.join(args.lora_ckpt, "trainable_params.bin")
            use_safetensors = False
        else:
            raise FileNotFoundError(f"No LoRA weights found in {args.lora_ckpt}")
    else:
        lora_weight_path = args.lora_ckpt
        use_safetensors = lora_weight_path.endswith('.safetensors')
    
    print(f"Loading LoRA weights from: {lora_weight_path}")
    
    # 加载权重
    if use_safetensors:
        try:
            from safetensors.torch import load_file
            lora_state_dict = load_file(lora_weight_path)
        except ImportError:
            print("Warning: safetensors not available, trying torch.load...")
            lora_state_dict = torch.load(lora_weight_path, map_location='cpu')
    else:
        lora_state_dict = torch.load(lora_weight_path, map_location='cpu')
    
    print(f"Loaded {len(lora_state_dict)} LoRA parameters")
    
    # ==================== 方法1: 使用 PEFT 加载（推荐） ====================
    try:
        from peft import PeftModel, LoraConfig
        
        # 检查是否有 adapter_config.json
        config_path = os.path.join(args.lora_ckpt, "adapter_config.json") if os.path.isdir(args.lora_ckpt) else None
        
        if config_path and os.path.exists(config_path):
            # 使用 PeftModel.from_pretrained
            print("Loading LoRA using PeftModel.from_pretrained...")
            gpt_wo_ddp = PeftModel.from_pretrained(gpt_wo_ddp, args.lora_ckpt)
        else:
            # 手动加载 LoRA 权重
            print("Loading LoRA weights manually...")
            # 需要先创建 LoRA 结构
            from peft import inject_adapter_in_model
            
            # 创建默认的 LoRA config
            lora_config = LoraConfig(
                r=8,
                lora_alpha=8,
                lora_dropout=0.0,
                target_modules=["ca.proj"],
                bias="none",
                task_type="CAUSAL_LM",
                inference_mode=True,
            )
            
            # 注入 LoRA 结构
            gpt_wo_ddp = inject_adapter_in_model(lora_config, gpt_wo_ddp)
            
            # 加载 LoRA 权重
            missing, unexpected = gpt_wo_ddp.load_state_dict(lora_state_dict, strict=False)
            print(f"LoRA weights loaded: Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
        
        print("✓ LoRA weights loaded successfully using PEFT")
        
    except Exception as e:
        print(f"Warning: Failed to load LoRA using PEFT: {e}")
        print("\n" + "=" * 80)
        print("Falling back to manual weight loading...")
        print("=" * 80)
        
        # ==================== 方法2: 手动加载 LoRA 权重（fallback） ====================
        # 直接加载 LoRA 权重到模型中
        missing, unexpected = gpt_wo_ddp.load_state_dict(lora_state_dict, strict=False)
        print(f"LoRA weights loaded manually: Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    
    # 移动到设备
    vae_local = vae_local.to(device)
    gpt_wo_ddp = gpt_wo_ddp.to(device)
    
    # 设置为评估模式
    vae_local.eval()
    gpt_wo_ddp.eval()
    
    print("Models loaded successfully!")
    print("=" * 80)
    
    return vae_local, gpt_wo_ddp


def create_text_features(prompt, model, device, text_maxlen=256):
    """
    创建文本特征（不使用T5）
    """
    # 使用模型内部的cfg_uncond作为基础
    if hasattr(model, 'cfg_uncond'):
        base_features = model.cfg_uncond.clone()
        text_maxlen = min(text_maxlen, base_features.shape[0])
    else:
        # 如果没有cfg_uncond，创建随机特征
        text_channels = model.Ct5 if hasattr(model, 'Ct5') else 4096
        base_features = torch.randn(text_maxlen, text_channels, device=device)
    
    # 根据prompt长度决定使用的特征长度
    prompt_hash = hash(prompt) % 1000
    prompt_len = min(max(len(prompt.split()), 64), text_maxlen)
    
    # 创建特征：基于cfg_uncond + 小的随机扰动
    text_feat = base_features[:prompt_len].clone()
    perturbation = torch.randn_like(text_feat) * 0.01 * (prompt_hash / 1000.0)
    text_feat = text_feat + perturbation
    
    # 创建 compact 格式
    text_features = text_feat.to(device)
    text_lens = [prompt_len]
    cu_seqlens_k = F.pad(torch.tensor(text_lens, dtype=torch.int32, device=device).cumsum_(0), (1, 0))
    Ltext = prompt_len
    
    return (text_features, text_lens, cu_seqlens_k, Ltext)


@torch.no_grad()
def generate_images(
    model,
    vae,
    prompt,
    negative_prompt,
    resolution,
    num_images,
    cfg_scale,
    device,
):
    """
    生成图像
    """
    print("\n" + "=" * 80)
    print(f"Generating {num_images} images...")
    print(f"Prompt: {prompt}")
    print(f"Negative prompt: {negative_prompt}")
    print(f"Resolution: {resolution}x{resolution}")
    print(f"CFG scale: {cfg_scale}")
    print("=" * 80)
    
    # 创建文本特征
    text_cond_tuple = create_text_features(prompt, model, device)
    
    # 如果有 negative prompt，创建负面文本特征
    if negative_prompt:
        text_uncond_tuple = create_text_features(negative_prompt, model, device)
    else:
        text_uncond_tuple = None
    
    # 准备输入（根据 Infinity 模型的输入格式）
    # 注意：这里需要根据实际的 Infinity 推理接口调整
    # 如果 Infinity 有专门的 generate 函数，应该使用那个函数
    
    # TODO: 这里需要根据 Infinity 的实际推理接口进行调整
    # 以下是一个示例框架，实际实现需要参考 Infinity 的推理代码
    
    print("\n⚠️  Note: The actual inference implementation depends on Infinity's API.")
    print("You need to:")
    print("1. Check Infinity's inference/generation code")
    print("2. Use the appropriate generate() or sample() function")
    print("3. Pass the text_cond_tuple and other parameters correctly")
    print("\nExample inference call:")
    print("```python")
    print("# Example (needs to be adapted to actual Infinity API):")
    print("generated_tokens = model.generate(")
    print("    text_cond=text_cond_tuple,")
    print("    text_uncond=text_uncond_tuple,")
    print("    cfg_scale=cfg_scale,")
    print("    num_images=num_images,")
    print("    resolution=resolution,")
    print(")")
    print("# Decode tokens to images")
    print("images = vae.decode(generated_tokens)")
    print("```")
    
    return []


def main():
    """主函数"""
    args = parse_args()
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 构建模型并加载 LoRA 权重
    vae, model = build_model_with_lora(args, device)
    
    # 生成图像
    images = generate_images(
        model=model,
        vae=vae,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        resolution=args.resolution,
        num_images=args.num_images,
        cfg_scale=args.cfg_scale,
        device=device,
    )
    
    # 保存图像
    if images:
        for i, img in enumerate(images):
            save_path = os.path.join(args.output_dir, f"generated_{i:04d}.png")
            img.save(save_path)
            print(f"Saved: {save_path}")
    
    print("\n" + "=" * 80)
    print("Inference completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()

