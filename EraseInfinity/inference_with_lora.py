#!/usr/bin/env python
# coding: UTF-8
"""
    @date:  2025.01
    @func:  Complete inference script for EraseInfinity with LoRA
            完整的推理脚本，集成了 LoRA 权重和 Infinity 推理功能
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import List

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

# 添加 Infinity 到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 添加本地 peft 库到路径
_eraseinfinity_dir = os.path.dirname(os.path.abspath(__file__))
_peft_local_path = os.path.join(_eraseinfinity_dir, 'peft', 'src')
if os.path.exists(_peft_local_path):
    sys.path.insert(0, _peft_local_path)

from infinity.models import Infinity
from infinity.utils.load import build_vae_gpt
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Inference with EraseInfinity + LoRA")
    
    # 模型权重路径
    parser.add_argument("--vae_ckpt", type=str, required=True, help="Path to VAE checkpoint")
    parser.add_argument("--gpt_ckpt", type=str, required=True, help="Path to GPT checkpoint")
    parser.add_argument("--lora_ckpt", type=str, required=True, help="Path to trained LoRA checkpoint directory")
    # 推理参数（T5 已禁用，不需要 t5_path）
    
    parser.add_argument("--prompt", type=str, default="a beautiful landscape", help="Text prompt")
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt")
    parser.add_argument("--pn", type=str, default="0.06M", help="Point number (resolution preset)")
    parser.add_argument("--h_div_w_template", type=float, default=1.0, help="Height/Width ratio")
    parser.add_argument("--cfg", type=float, default=4.0, help="CFG scale")
    parser.add_argument("--tau", type=float, default=1.0, help="Temperature")
    parser.add_argument("--top_k", type=int, default=900, help="Top-k sampling")
    parser.add_argument("--top_p", type=float, default=0.97, help="Top-p sampling")
    parser.add_argument("--cfg_insertion_layer", type=int, default=-5, help="CFG insertion layer")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--vae_type", type=int, default=32, help="VAE type")
    parser.add_argument("--sampling_per_bits", type=int, default=1, help="Sampling per bits")
    
    # 输出参数
    parser.add_argument("--output_dir", type=str, default="./outputs/inference_lora", help="Output directory")
    parser.add_argument("--save_file", type=str, default=None, help="Specific save file path")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device (cuda:0, cpu, etc.)")
    
    # LoRA 相关
    parser.add_argument("--use_lora", action="store_true", default=True, help="Use LoRA weights")
    parser.add_argument("--no_lora", action="store_true", help="Disable LoRA (compare with original model)")
    parser.add_argument("--lora_target_modules", type=str, nargs="+", default=None, 
                        help="Target modules for LoRA (e.g., ca.proj ca.mat_q ca.mat_kv). If not specified, will auto-detect from adapter_config.json")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank (default: 8)")
    parser.add_argument("--lora_alpha", type=int, default=8, help="LoRA alpha (default: 8)")
    
    return parser.parse_args()


def create_text_features_from_prompts(
    prompts: list,
    model,
    device: torch.device,
    text_maxlen: int = 256,
    text_channels: int = 2048,
):
    """
    不使用T5，直接从prompt创建文本特征
    基于模型内部的cfg_uncond创建文本特征（与训练脚本相同）
    """
    B = len(prompts)
    
    # 使用模型内部的cfg_uncond作为基础
    if hasattr(model, 'cfg_uncond'):
        base_features = model.cfg_uncond.clone()  # [text_maxlen, text_channels]
        text_maxlen = min(text_maxlen, base_features.shape[0])
    else:
        # 如果没有cfg_uncond，创建随机特征
        base_features = torch.randn(text_maxlen, text_channels, device=device)
        text_maxlen = text_maxlen
    
    # 为每个prompt创建文本特征
    text_features_list = []
    text_lens_list = []
    
    for prompt in prompts:
        # 根据prompt长度决定使用的特征长度
        prompt_hash = hash(prompt) % 1000
        # 使用prompt长度（最小64，最大text_maxlen）
        prompt_len = min(max(len(prompt.split()), 64), text_maxlen)
        
        # 创建特征：基于cfg_uncond + 小的随机扰动（基于prompt的hash）
        text_feat = base_features[:prompt_len].clone()
        # 添加基于prompt hash的小扰动，使不同prompt有不同特征
        perturbation = torch.randn_like(text_feat) * 0.01 * (prompt_hash / 1000.0)
        text_feat = text_feat + perturbation
        
        text_features_list.append(text_feat)
        text_lens_list.append(prompt_len)
    
    # Infinity模型期望的格式是compact格式：将所有文本特征concatenate成[total_len, Ct5]
    text_features_compact = []
    for feat in text_features_list:
        text_features_compact.append(feat)
    
    # Concatenate所有特征
    text_features = torch.cat(text_features_compact, dim=0).to(device)  # [total_len, Ct5]
    
    # 创建cu_seqlens_k（累积序列长度）
    cu_seqlens_k = F.pad(torch.tensor(text_lens_list, dtype=torch.int32, device=device).cumsum_(0), (1, 0))
    Ltext = max(text_lens_list)
    
    return (text_features, text_lens_list, cu_seqlens_k, Ltext)


def detect_lora_target_modules(lora_ckpt_dir, lora_state_dict=None, fallback_modules=None):
    """
    自动检测 LoRA target_modules
    
    优先级:
    1. 从 adapter_config.json 读取
    2. 从权重文件的 key 推断
    3. 使用 fallback_modules 或默认值
    """
    import json
    
    # 方法 1: 尝试从 adapter_config.json 读取
    if os.path.isdir(lora_ckpt_dir):
        config_path = os.path.join(lora_ckpt_dir, "adapter_config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    if 'target_modules' in config:
                        target_modules = config['target_modules']
                        print(f"✓ Detected target_modules from adapter_config.json: {target_modules}")
                        return config.get('r', 8), config.get('lora_alpha', 8), target_modules
            except Exception as e:
                print(f"Warning: Failed to read adapter_config.json: {e}")
    
    # 方法 2: 从权重文件的 key 推断
    if lora_state_dict is not None:
        detected_modules = set()
        for key in lora_state_dict.keys():
            # LoRA 权重格式: base_model.model.xxx.lora_A.default.weight
            # 或: blocks.0.ca.proj.lora_A.default.weight
            if 'lora_A' in key or 'lora_B' in key:
                # 提取模块名称
                parts = key.split('.')
                # 寻找 ca.proj, ca.mat_q, ca.mat_kv 等模式
                for i in range(len(parts) - 1):
                    if parts[i] == 'ca' and parts[i+1] in ['proj', 'mat_q', 'mat_kv']:
                        detected_modules.add(f"ca.{parts[i+1]}")
                    elif parts[i] == 'sa' and parts[i+1] in ['proj', 'mat_qkv']:
                        detected_modules.add(f"sa.{parts[i+1]}")
                    elif parts[i] == 'ffn' and parts[i+1] in ['fc1', 'fc2']:
                        detected_modules.add(f"ffn.{parts[i+1]}")
        
        if detected_modules:
            target_modules = sorted(list(detected_modules))
            print(f"✓ Inferred target_modules from weight keys: {target_modules}")
            # 假设默认 rank=8, alpha=8
            return 8, 8, target_modules
    
    # 方法 3: 使用 fallback
    if fallback_modules:
        print(f"⚠ Using fallback target_modules: {fallback_modules}")
        return 8, 8, fallback_modules
    
    # 默认值
    default_modules = ["ca.proj"]
    print(f"⚠ WARNING: Could not detect target_modules, using default: {default_modules}")
    print(f"   If your checkpoint uses different modules, please specify --lora_target_modules")
    return 8, 8, default_modules


def build_model_with_lora(args, device):
    """构建模型并加载 LoRA 权重"""
    print("=" * 80)
    print("Building models...")
    
    # ==================== 加载 VAE ====================
    print(f"Loading VAE from {args.vae_ckpt}")
    vae_ckpt = torch.load(args.vae_ckpt, map_location='cpu')
    
    # ==================== 配置模型参数 ====================
    class TempArgs:
        def __init__(self, args_in):
            self.vae_type = args_in.vae_type
            self.fake_vae_input = False
            self.apply_spatial_patchify = 0
            self.device = device
            self.model_init_device = 'cpu'
            # 从 checkpoint 文件名推断模型名称
            gpt_ckpt_name = os.path.basename(args_in.gpt_ckpt)  # infinity_2b_reg.pth
            # 移除扩展名和常见后缀
            model_name = gpt_ckpt_name.replace('.pth', '').replace('_reg', '')  # infinity_2b
            self.model = model_name
            self.pn = args_in.pn
            self.always_training_scales = 20
            self.tlen = 256
            self.Ct5 = 2048
            self.cond_drop_rate = 0.1
            self.cfg = self.cond_drop_rate
            self.use_bit_label = 1
            self.flash = False
            self.fuse = False
            self.fused_norm = False
            self.enable_checkpointing = None
            self.pad_to_multiplier = 0
            self.use_flex_attn = False
            self.batch_size = 1
            self.norm_eps = 1e-6
            self.rms = False
            self.saln = True
            self.haln = True
            self.drop = 0.0
            self.rand_uncond = False
            self.ca_gamma = -1.0
            self.nm0 = False
            self.tau = 1
            self.cos = False
            self.swi = False
            self.scale_schedule = None
            self.dec = 1
            self.tp = 0.0
            self.tk = 0.0
            self.rope2d_each_sa_layer = 1
            self.rope2d_normalized_by_hw = 2
            self.add_lvl_embeding_only_first_block = 1
            self.train_h_div_w_list = None
            self.dp = -1
            self.hd = 0
            self.use_fsdp_model_ema = False
            self.diva = 1
            self.alng = 1e-5
            self.aln = True
            self.hd0 = 1
            self.online_t5 = False
            self.t5_path = 'google/flan-t5-xl'  # 已禁用，但保留参数以防需要
    
    temp_args = TempArgs(args)
    
    # ==================== 构建 VAE 和 GPT ====================
    vae_local, gpt_wo_ddp, _ = build_vae_gpt(temp_args, vae_ckpt, skip_gpt=False, device='cpu')
    
    # ==================== 加载 GPT checkpoint ====================
    print(f"Loading GPT checkpoint from {args.gpt_ckpt}")
    gpt_state = torch.load(args.gpt_ckpt, map_location='cpu')
    
    if 'trainer' in gpt_state:
        if 'gpt_wo_ddp' in gpt_state['trainer']:
            gpt_state_dict = gpt_state['trainer']['gpt_wo_ddp']
        elif 'gpt_fsdp' in gpt_state['trainer']:
            gpt_state_dict = gpt_state['trainer']['gpt_fsdp']
        else:
            gpt_state_dict = gpt_state
    else:
        gpt_state_dict = gpt_state
    
    missing, unexpected = gpt_wo_ddp.load_state_dict(gpt_state_dict, strict=False)
    print(f"GPT loaded: Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    
    # ==================== 加载 LoRA 权重 ====================
    if not args.no_lora and args.use_lora:
        print(f"\n{'='*80}")
        print(f"Loading LoRA weights from {args.lora_ckpt}")
        print(f"{'='*80}")
        
        #检查路径是否为目录，如果是目录，则查找adapter_model.safetensors或adapter_model.bin或trainable_params.bin
        if os.path.isdir(args.lora_ckpt):
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
            #如果是文件，直接使用该文件路径，根据扩展名判断是否为safetensors格式
            lora_weight_path = args.lora_ckpt
            use_safetensors = lora_weight_path.endswith('.safetensors')
        
        print(f"Loading from: {lora_weight_path}")
        
        if use_safetensors:
            try:
                from safetensors.torch import load_file
                lora_state_dict = load_file(lora_weight_path)
            except ImportError:
                print("Warning: safetensors not available, using torch.load...")
                lora_state_dict = torch.load(lora_weight_path, map_location='cpu')
        else:
            lora_state_dict = torch.load(lora_weight_path, map_location='cpu')
        
        print(f"Loaded {len(lora_state_dict)} LoRA parameters")
        
        # ==================== 自动检测 LoRA 配置 ====================
        # 从命令行参数、配置文件或权重文件自动检测 target_modules
        lora_rank, lora_alpha, target_modules = detect_lora_target_modules(
            lora_ckpt_dir=args.lora_ckpt,
            lora_state_dict=lora_state_dict,
            fallback_modules=args.lora_target_modules
        )
        
        # 优先使用命令行参数
        if args.lora_target_modules is not None:
            target_modules = args.lora_target_modules
            print(f"✓ Using target_modules from command line: {target_modules}")
        
        # 使用命令行参数覆盖 rank 和 alpha（如果指定）
        lora_rank = args.lora_rank if hasattr(args, 'lora_rank') else lora_rank
        lora_alpha = args.lora_alpha if hasattr(args, 'lora_alpha') else lora_alpha
        
        print(f"\nLoRA Configuration:")
        print(f"  - rank: {lora_rank}")
        print(f"  - alpha: {lora_alpha}")
        print(f"  - target_modules: {target_modules}")
        print()
        
        # 尝试使用 PEFT 加载
        try:
            from peft import PeftModel, LoraConfig, inject_adapter_in_model
            import json
            
            # 添加假方法避免 PEFT 检查错误（在所有加载方式前都需要）
            def dummy_prepare_inputs_for_generation(self, *args, **kwargs):
                return kwargs if kwargs else {}
            
            if not hasattr(gpt_wo_ddp, 'prepare_inputs_for_generation'):
                gpt_wo_ddp.prepare_inputs_for_generation = dummy_prepare_inputs_for_generation.__get__(gpt_wo_ddp, type(gpt_wo_ddp))
                print("✓ Added prepare_inputs_for_generation method for PEFT compatibility")
            
            config_path = os.path.join(args.lora_ckpt, "adapter_config.json") if os.path.isdir(args.lora_ckpt) else None
            
            if config_path and os.path.exists(config_path):
                # 修复配置文件格式（如果配置嵌套在 "default" 键下）
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                
                # 检查是否需要修复配置格式
                if "default" in config_data and "peft_type" not in config_data:
                    print("⚠ Detected nested config format, fixing adapter_config.json...")
                    # 提取 "default" 下的配置到顶层
                    fixed_config = config_data["default"]
                    # 保存修复后的配置
                    with open(config_path, 'w') as f:
                        json.dump(fixed_config, f, indent=2)
                    print("✓ Fixed adapter_config.json format")
                
                print("Loading LoRA using PeftModel.from_pretrained...")
                gpt_wo_ddp = PeftModel.from_pretrained(gpt_wo_ddp, args.lora_ckpt)
            else:
                print("Loading LoRA weights manually with PEFT...")
                lora_config = LoraConfig(
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=0.0,
                    target_modules=target_modules,
                    bias="none",
                    task_type="CAUSAL_LM",
                    inference_mode=True,
                )
                
                gpt_wo_ddp = inject_adapter_in_model(lora_config, gpt_wo_ddp)
                missing, unexpected = gpt_wo_ddp.load_state_dict(lora_state_dict, strict=False)
                print(f"LoRA weights loaded: Missing={len(missing)}, Unexpected={len(unexpected)}")
            
            # 验证 LoRA 参数
            lora_params = [name for name, param in gpt_wo_ddp.named_parameters() if 'lora' in name.lower()]
            print(f"✓ Found {len(lora_params)} LoRA parameters in model")
            if len(lora_params) > 0:
                print(f"  Example: {lora_params[0]}")
            
        except Exception as e:
            print(f"Warning: Failed to load LoRA using PEFT: {e}")
            print("Falling back to direct weight loading...")
            missing, unexpected = gpt_wo_ddp.load_state_dict(lora_state_dict, strict=False)
            print(f"LoRA weights loaded directly: Missing={len(missing)}, Unexpected={len(unexpected)}")
    else:
        print("\n⚠️  LoRA disabled - using original model weights only")
    
    # 移动到设备
    vae_local = vae_local.to(device)
    gpt_wo_ddp = gpt_wo_ddp.to(device)
    
    # 修复：如果 block_chunks=1，模型使用 self.blocks 而不是 self.block_chunks
    # 但 autoregressive_infer_cfg 需要 self.block_chunks，所以需要创建兼容的 block_chunks
    # 注意：PEFT 包装后，原始模型在 base_model.model 或通过 get_base_model() 访问
    
    # 获取实际的 Infinity 模型（可能被 PEFT 包装）
    actual_model = gpt_wo_ddp
    is_peft_wrapped = False
    
    if hasattr(gpt_wo_ddp, 'get_base_model'):
        # PEFT 提供的标准方法
        actual_model = gpt_wo_ddp.get_base_model()
        is_peft_wrapped = True
        print("✓ Detected PEFT-wrapped model, accessing base model via get_base_model()")
    elif hasattr(gpt_wo_ddp, 'base_model') and hasattr(gpt_wo_ddp.base_model, 'model'):
        # 备选方法：直接访问
        actual_model = gpt_wo_ddp.base_model.model
        is_peft_wrapped = True
        print("✓ Detected PEFT-wrapped model, accessing base_model.model")
    
    # 在实际模型上添加 block_chunks（如果需要）
    if not hasattr(actual_model, 'block_chunks') and hasattr(actual_model, 'blocks'):
        # 当 num_block_chunks == 1 时，创建一个兼容的 block_chunks
        # 将 self.blocks 包装成 MultipleLayers 格式
        from infinity.models.infinity import MultipleLayers
        actual_model.block_chunks = nn.ModuleList([
            MultipleLayers(actual_model.blocks, len(actual_model.blocks), 0)
        ])
        print("✓ Created compatible block_chunks for block_chunks=1 case")
    
    # 如果是 PEFT 包装的模型，确保包装器能访问到所有必要的属性
    # PEFT 的 __getattr__ 会自动转发，但我们可以显式添加关键属性以确保兼容性
    if is_peft_wrapped:
        # 关键：确保 PEFT 包装器能访问 block_chunks
        critical_attrs = ['block_chunks', 'num_block_chunks', 'num_blocks_in_a_chunk']
        for attr in critical_attrs:
            if hasattr(actual_model, attr) and not hasattr(gpt_wo_ddp, attr):
                setattr(gpt_wo_ddp, attr, getattr(actual_model, attr))
        print("✓ Ensured PEFT wrapper has access to critical attributes")
    
    # 设置为评估模式
    vae_local.eval()
    gpt_wo_ddp.eval()
    
    print("✓ Models loaded successfully!")
    print("=" * 80)
    
    return vae_local, gpt_wo_ddp


@torch.no_grad()
def gen_one_img(
    infinity_model,
    vae,
    prompt,
    cfg_list=4.0,
    tau_list=1.0,
    negative_prompt='',
    scale_schedule=None,
    top_k=900,
    top_p=0.97,
    cfg_sc=3,
    cfg_insertion_layer=-5,
    vae_type=32,
    g_seed=None,
    sampling_per_bits=1,
    device=None,
    text_maxlen=256,
    text_channels=2048,
):
    """生成单张图像（不使用T5）"""
    print(f"\n{'='*80}")
    print(f"Generating image...")
    print(f"Prompt: {prompt}")
    if negative_prompt:
        print(f"Negative prompt: {negative_prompt}")
    print(f"CFG: {cfg_list}, Tau: {tau_list}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    if not isinstance(cfg_list, list):
        cfg_list = [cfg_list] * len(scale_schedule)
    if not isinstance(tau_list, list):
        tau_list = [tau_list] * len(scale_schedule)
    
    # 编码文本（不使用T5，使用与训练脚本相同的方法）
    text_cond_tuple = create_text_features_from_prompts(
        prompts=[prompt],
        model=infinity_model,
        device=device,
        text_maxlen=text_maxlen,
        text_channels=text_channels,
    )
    
    if negative_prompt:
        negative_label_tuple = create_text_features_from_prompts(
            prompts=[negative_prompt],
            model=infinity_model,
            device=device,
            text_maxlen=text_maxlen,
            text_channels=text_channels,
        )
    else:
        negative_label_tuple = None
    
    # 推理
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True):
        infer_start = time.time()
        _, _, img_list = infinity_model.autoregressive_infer_cfg(
            vae=vae,
            scale_schedule=scale_schedule,
            label_B_or_BLT=text_cond_tuple,
            g_seed=g_seed,
            B=1,
            negative_label_B_or_BLT=negative_label_tuple,
            force_gt_Bhw=None,
            cfg_sc=cfg_sc,
            cfg_list=cfg_list,
            tau_list=tau_list,
            top_k=top_k,
            top_p=top_p,
            returns_vemb=1,
            ratio_Bl1=None,
            gumbel=0,
            norm_cfg=False,
            cfg_exp_k=0.0,
            cfg_insertion_layer=[cfg_insertion_layer],
            vae_type=vae_type,
            softmax_merge_topk=-1,
            ret_img=True,
            trunk_scale=1000,
            gt_leak=-1,
            gt_ls_Bl=None,
            inference_mode=True,
            sampling_per_bits=sampling_per_bits,
        )
        infer_time = time.time() - infer_start
    
    total_time = time.time() - start_time
    print(f"✓ Generation complete: {total_time:.2f}s (inference: {infer_time:.2f}s)")
    
    img = img_list[0]
    return img


def main():
    """主函数"""
    args = parse_args()
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 构建模型并加载 LoRA 权重
    vae, gpt_model = build_model_with_lora(args, device)
    
    # 获取文本通道数（从模型配置）
    text_channels = gpt_model.Ct5 if hasattr(gpt_model, 'Ct5') else 2048
    text_maxlen = 256  # 与训练脚本保持一致
    
    # 获取 scale schedule（分辨率配置）
    h_div_w_template = args.h_div_w_template
    pn = args.pn
    scale_schedule = dynamic_resolution_h_w[h_div_w_template][pn]['scales']
    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
    tgt_h, tgt_w = dynamic_resolution_h_w[h_div_w_template][pn]['pixel']
    
    print(f"\nResolution: {tgt_h}x{tgt_w}")
    print(f"Scale schedule: {len(scale_schedule)} scales")
    print(f"Text channels: {text_channels}, Text maxlen: {text_maxlen}")
    
    # 生成图像
    with autocast(dtype=torch.bfloat16):
        generated_image = gen_one_img(
            infinity_model=gpt_model,
            vae=vae,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            cfg_list=args.cfg,
            tau_list=args.tau,
            scale_schedule=scale_schedule,
            top_k=args.top_k,
            top_p=args.top_p,
            cfg_insertion_layer=args.cfg_insertion_layer,
            vae_type=args.vae_type,
            g_seed=args.seed,
            sampling_per_bits=args.sampling_per_bits,
            device=device,
            text_maxlen=text_maxlen,
            text_channels=text_channels,
        )
    
    # 保存图像
    if args.save_file:
        save_path = args.save_file
    else:
        import hashlib
        from datetime import datetime
        
        # 生成唯一的文件名
        prompt_hash = hashlib.md5(args.prompt.encode('utf-8')).hexdigest()[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 添加配置信息使文件名更有描述性
        lora_suffix = "_lora" if (args.use_lora and not args.no_lora) else "_no_lora"
        seed_suffix = f"_seed{args.seed}" if args.seed is not None else ""
        cfg_suffix = f"_cfg{args.cfg:.1f}"
        
        # 组合文件名: 时间戳_prompt哈希_配置信息
        filename = f"{timestamp}_{prompt_hash}{lora_suffix}{cfg_suffix}{seed_suffix}.jpg"
        save_path = os.path.join(args.output_dir, filename)
    
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    cv2.imwrite(save_path, generated_image.cpu().numpy())
    
    print(f"\n{'='*80}")
    print(f"✓ Image saved to: {os.path.abspath(save_path)}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
