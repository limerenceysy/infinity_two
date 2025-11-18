#!/usr/bin/env python
# coding: UTF-8
"""
    @date:  2025.01
    @func:  Training script for EraseInfinity
            基于 Infinity 自回归模型的 nude 内容擦除训练脚本
"""

import os
import sys

import warnings
# 抑制 pydantic 的警告
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")


import yaml
import argparse
import random
import time
import math
from pathlib import Path
from typing import List, Tuple, Optional
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
# T5已禁用，不再导入
# from transformers import AutoTokenizer, T5EncoderModel, T5TokenizerFast
from tqdm.auto import tqdm
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt

# 添加 Infinity 到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 添加本地 peft 库到路径（如果 peft 安装在 EraseInfinity/peft 目录下）
_eraseinfinity_dir = os.path.dirname(os.path.abspath(__file__))
_peft_local_path = os.path.join(_eraseinfinity_dir, 'peft', 'src')
if os.path.exists(_peft_local_path):
    sys.path.insert(0, _peft_local_path)
    print(f"[DEBUG] Added local peft to path: {_peft_local_path}")

# 导入 Infinity 相关模块
import infinity.utils.dist as dist
from infinity.models import Infinity
from infinity.utils.load import build_vae_gpt
from infinity.utils import arg_util, misc

# 导入 EraseInfinity 模块
from dataset import EraseInfinityDataset, collate_fn
from dataset_prompt_only import PromptOnlyDataset, collate_fn as collate_fn_prompt_only
from utils.calc_loss import calculate_first_esd_loss
from utils.esd_utils import get_default_scale_schedule

# Wandb - 已禁用，不需要日志记录
has_wandb = False
# try:
#     import wandb
#     has_wandb = True
# except ImportError:
#     has_wandb = False
#     print("Wandb not available, logging disabled")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Train EraseInfinity model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=0,  # 默认使用 1 号 GPU
        help="Local rank for distributed training"
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    # 如果路径是相对路径，确保相对于当前工作目录解析
    if not os.path.isabs(config_path):
        config_path = os.path.abspath(config_path)
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 确保配置文件中的路径都是绝对路径（相对于项目根目录）
    # 配置文件路径格式：.../Infinity-main/EraseInfinity/config/xxx.yaml
    # 项目根目录应该是：.../Infinity-main
    config_dir = os.path.dirname(config_path)  # EraseInfinity/config/
    eraseinfinity_dir = os.path.dirname(config_dir)  # EraseInfinity/
    project_root = os.path.dirname(eraseinfinity_dir)  # Infinity-main/
    
    print(f"[DEBUG] Config path: {config_path}")
    print(f"[DEBUG] Config dir: {config_dir}")
    print(f"[DEBUG] EraseInfinity dir: {eraseinfinity_dir}")
    print(f"[DEBUG] Project root: {project_root}")
    
    # 转换路径配置为绝对路径
    path_keys = ['vae_ckpt', 'gpt_ckpt', 'instance_data_dir', 'output_dir', 'logging_dir']
    for key in path_keys:
        if key in config and config[key] and not os.path.isabs(str(config[key])):
            # 如果是相对路径，转换为相对于项目根目录的绝对路径
            if config[key] != 'null' and config[key] is not None:
                config[key] = os.path.join(project_root, config[key])
                print(f"[DEBUG] {key}: {config[key]}")
    
    return config


def setup_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_models(config: dict, device: torch.device):
    """
    构建模型：VAE + GPT + Text Encoder
    """
    print("=" * 80)
    print("Building models...")
    
    # ==================== 加载 VAE ====================
    vae_ckpt_path = config['vae_ckpt']
    print(f"Loading VAE from {vae_ckpt_path}")
    
    if os.path.exists(vae_ckpt_path):
        vae_ckpt = torch.load(vae_ckpt_path, map_location='cpu')
    else:
        raise FileNotFoundError(f"VAE checkpoint not found: {vae_ckpt_path}")
    
    # ==================== 加载 GPT ====================
    gpt_ckpt_path = config['gpt_ckpt']
    print(f"Loading GPT from {gpt_ckpt_path}")
    
    # 创建临时 args 用于 build_vae_gpt
    class TempArgs:
        def __init__(self, config):
            # VAE 相关参数
            self.vae_type = 32  # 根据权重文件名 infinity_vae_d32reg.pth
            self.fake_vae_input = False  # 是否使用假的 VAE 输入（用于调试）
            self.apply_spatial_patchify = config.get('apply_spatial_patchify', 0)
            
            # 设备相关
            self.device = device
            self.model_init_device = 'cpu'
            
            # 模型架构参数
            # 从 checkpoint 文件名推断模型名称
            gpt_ckpt_name = os.path.basename(config['gpt_ckpt'])  # infinity_2b_reg.pth
            model_name = gpt_ckpt_name.replace('.pth', '').replace('_reg', '')  # infinity_2b
            self.model = config.get('model_name', model_name)
            self.pn = config.get('pn', '0.06M')
            self.always_training_scales = config.get('always_training_scales', 20)
            
            # 文本编码器参数
            self.tlen = config.get('max_sequence_length', 256)
            self.Ct5 = config.get('text_channels', 2048)  # 文本特征维度
            
            # 训练相关参数
            self.cond_drop_rate = config.get('cond_drop_rate', 0.1)
            self.cfg = self.cond_drop_rate  # classifier-free guidance
            self.use_bit_label = config.get('use_bit_label', 1)
            
            # 注意力相关参数
            self.flash = False  # customized_flash_attn
            self.fuse = False  # fused_mlp
            self.fused_norm = False
            self.enable_checkpointing = None
            self.pad_to_multiplier = 0
            self.use_flex_attn = config.get('use_flex_attn', False)
            self.batch_size = config.get('train_batch_size', 4)
            
            # 层归一化参数
            self.norm_eps = 1e-6
            self.rms = False  # rms_norm
            self.saln = True  # shared_aln
            self.haln = True  # head_aln
            
            # Dropout 和正则化
            self.drop = 0.0  # dropout rate
            self.rand_uncond = False
            
            # Cross Attention 参数
            self.ca_gamma = -1.0  # cross_attn_layer_scale
            
            # 其他模型参数
            self.nm0 = False
            self.tau = 1
            self.cos = False  # cos_attn
            self.swi = False  # swiglu
            
            # Scale schedule
            self.scale_schedule = None  # raw_scale_schedule
            
            # Head 参数
            self.dec = 1  # head_depth
            self.tp = 0.0  # top_p
            self.tk = 0.0  # top_k
            
            # RoPE 2D 参数（必须与预训练权重匹配，infinity_2b_reg 需要 rope2d_each_sa_layer=1）
            self.rope2d_each_sa_layer = config.get('rope2d_each_sa_layer', 1)  # 默认改为1，与预训练权重匹配
            self.rope2d_normalized_by_hw = config.get('rope2d_normalized_by_hw', 2)  # 默认改为2，与预训练权重匹配
            self.add_lvl_embeding_only_first_block = config.get('add_lvl_embeding_only_first_block', 1)
            
            # 训练相关
            self.train_h_div_w_list = None
            self.dp = -1  # drop_path_rate (如果 >= 0 会设置)
            self.hd = 0  # num_heads (如果 > 0 会设置)
            
            # EMA 相关
            self.use_fsdp_model_ema = False
            
            # 其他参数（可能不需要，但保留以防万一）
            self.diva = 1
            self.alng = 1e-5
            self.aln = True
            self.hd0 = 1
            self.online_t5 = False  # T5已禁用
            self.t5_path = config.get('t5_path', 'google/flan-t5-xl')  # 保留但不使用
    
    temp_args = TempArgs(config)
    
    # 使用 Infinity 的 build_vae_gpt 函数
    vae_local, gpt_wo_ddp, _ = build_vae_gpt(temp_args, vae_ckpt, skip_gpt=False, device='cpu')
    
    # 加载 GPT checkpoint
    if os.path.exists(gpt_ckpt_path):
        print(f"Loading GPT checkpoint...")
        gpt_state = torch.load(gpt_ckpt_path, map_location='cpu')
        
        # 处理不同的 checkpoint 格式
        if 'trainer' in gpt_state:
            # 训练中的 checkpoint
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
    else:
        print(f"Warning: GPT checkpoint not found at {gpt_ckpt_path}, using random initialization")
    
    # 移动到设备
    vae_local = vae_local.to(device)
    gpt_wo_ddp = gpt_wo_ddp.to(device)
    
    # 冻结 VAE
    vae_local.requires_grad_(False)
    vae_local.eval()
    
    # ==================== 添加 LoRA 到 GPT ====================
    if config.get('use_lora', True):
        print("=" * 80)
        print("Adding LoRA adapters to GPT...")
        print("=" * 80)
        
        try:
            from peft import LoraConfig, inject_adapter_in_model
        except ImportError:
            print("❌ Error: PEFT library not found!")
            print("Please install PEFT first:")
            print("  cd EraseInfinity/peft && pip install -e .")
            raise
        
        # ==================== LoRA Target Modules 配置 ====================
        # 仿照 EraseAnything/train_flux_lora.py 的写法
        # 想微调哪部分就取消哪部分的注释
        # 目前只微调交叉注意力层的 proj
        
        # 如果配置文件中指定了 target_modules，优先使用配置文件的
        if config.get('lora_target_modules'):
            target_modules = config.get('lora_target_modules')
            print(f"\nUsing target_modules from config: {target_modules}")
            # 如果配置文件中是 "proj"，需要转换为更精确的匹配
            if target_modules == ["proj"] or target_modules == "proj":
                print("  ⚠ Warning: 'proj' matches all proj layers (including sa.proj)")
                print("  → Converting to 'ca.proj' to only match Cross-Attention layers")
                target_modules = ["ca.proj"]
        else:
            # 默认配置：可注释/取消注释来选择要微调的部分
            target_modules = [
                # Cross-Attention 相关（推荐，用于概念擦除）
                "ca.proj",           # Cross-Attention 的投影层
                 #"ca.mat_q",        # Cross-Attention 的 Q 矩阵
                 #"ca.mat_kv",       # Cross-Attention 的 KV 矩阵
                
                # Self-Attention 相关（通常不需要）
                # "sa.mat_qkv",  
                
                # Feed-Forward Network 相关（通常不需要）
                # "ffn.fc1",         # Feed-Forward 第一层
                # "ffn.fc2",         # Feed-Forward 第二层
            ]
            print(f"\nUsing default target_modules: {target_modules}")
        
        print(f"\nLoRA target modules: {target_modules}")
        
        # 先检查模型结构，找到实际的模块名称（用于验证）
        ca_proj_names = []
        sa_proj_names = []
        all_proj_names = []
        
        for name, module in gpt_wo_ddp.named_modules():
            if 'proj' in name and isinstance(module, torch.nn.Linear):
                all_proj_names.append(name)
                if 'ca' in name and 'proj' in name:
                    ca_proj_names.append(name)
                elif 'sa' in name and 'proj' in name:
                    sa_proj_names.append(name)
        
        print(f"\nFound module statistics:")
        print(f"  CrossAttention.proj layers: {len(ca_proj_names)}")
        print(f"  SelfAttention.proj layers: {len(sa_proj_names)}")
        print(f"  Total proj layers: {len(all_proj_names)}")
        
        if len(ca_proj_names) > 0:
            print(f"\nExample CrossAttention.proj: {ca_proj_names[0]}")
        
        # 创建 LoRA config
        lora_config = LoraConfig(
            r=config.get('lora_rank', 8),
            lora_alpha=config.get('lora_alpha', 8),
            lora_dropout=config.get('lora_dropout', 0.0),
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",  # Infinity 是自回归模型
            inference_mode=False,  # 训练模式
        )
        
        # 应用 LoRA（使用 inject_adapter_in_model 避免 prepare_inputs_for_generation 问题）
        print("\nApplying LoRA adapters...")
        print(f"Target modules: {target_modules}")
        
        # 先检查target_modules是否能匹配到模型中的模块
        print("\nChecking if target modules exist in model...")
        matched_modules = []
        for name, module in gpt_wo_ddp.named_modules():
            for target in target_modules:
                if target in name and isinstance(module, torch.nn.Linear):
                    matched_modules.append(name)
                    break
        
        if len(matched_modules) == 0:
            print(f"❌ ERROR: No modules matched target_modules {target_modules}!")
            print("Available modules with 'proj' in name:")
            for name, module in gpt_wo_ddp.named_modules():
                if 'proj' in name.lower() and isinstance(module, torch.nn.Linear):
                    print(f"  - {name}")
            raise ValueError(f"No modules matched target_modules {target_modules}. Please check the module names.")
        else:
            print(f"✓ Found {len(matched_modules)} matching modules")
            if len(matched_modules) <= 10:
                for name in matched_modules[:10]:
                    print(f"  - {name}")
            else:
                for name in matched_modules[:5]:
                    print(f"  - {name}")
                print(f"  ... and {len(matched_modules) - 5} more")
        
        try:
            # 保存原始模型状态（用于验证不破坏原权重）
            original_state_dict = {k: v.clone() for k, v in gpt_wo_ddp.state_dict().items() if 'lora' not in k.lower()}
            
            # 尝试使用 inject_adapter_in_model（它内部调用 get_peft_model，但可能仍然失败）
            # 如果失败，我们添加一个假的 prepare_inputs_for_generation 方法
            try:
                gpt_wo_ddp = inject_adapter_in_model(lora_config, gpt_wo_ddp)
            except AttributeError as e:
                if 'prepare_inputs_for_generation' in str(e):
                    # 添加一个假的 prepare_inputs_for_generation 方法
                    def dummy_prepare_inputs_for_generation(self, *args, **kwargs):
                        return kwargs if kwargs else {}
                    gpt_wo_ddp.prepare_inputs_for_generation = dummy_prepare_inputs_for_generation.__get__(gpt_wo_ddp, type(gpt_wo_ddp))
                    # 再次尝试
                    from peft import get_peft_model
                    gpt_wo_ddp = get_peft_model(gpt_wo_ddp, lora_config)
                else:
                    raise
            except Exception as e:
                print(f"❌ ERROR during LoRA injection: {e}")
                import traceback
                traceback.print_exc()
                raise
            
            print("✓ LoRA adapters added successfully")
            
            # 打印可训练参数统计
            trainable_params = sum(p.numel() for p in gpt_wo_ddp.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in gpt_wo_ddp.parameters())
            print(f"Trainable parameters: {trainable_params / 1e6:.2f}M / {total_params / 1e6:.2f}M ({trainable_params / total_params * 100:.2f}%)")
            
            # 验证：检查哪些模块有 LoRA
            print("\n" + "=" * 80)
            print("Verifying LoRA target modules:")
            print("=" * 80)
            lora_modules = []
            for name, module in gpt_wo_ddp.named_modules():
                if hasattr(module, 'lora_A') or hasattr(module, 'lora_B'):
                    lora_modules.append(name)
                    # 检查是否符合预期
                    is_expected = False
                    for target in target_modules:
                        if target in name:
                            is_expected = True
                            break
                    if is_expected:
                        print(f"  ✓ {name} has LoRA")
                    else:
                        print(f"  ⚠ {name} has LoRA (unexpected)")
            
            if len(lora_modules) == 0:
                print("  ⚠ Warning: No LoRA modules found!")
            else:
                print(f"\n✓ Total LoRA modules: {len(lora_modules)}")
            
            # 验证原权重未被修改
            print("\n" + "=" * 80)
            print("Verifying original weights are preserved:")
            print("=" * 80)
            weight_preserved = True
            for key in original_state_dict:
                if key in gpt_wo_ddp.state_dict():
                    if not torch.equal(original_state_dict[key], gpt_wo_ddp.state_dict()[key]):
                        print(f"  ⚠ Warning: {key} was modified!")
                        weight_preserved = False
            
            if weight_preserved:
                print("  ✓ All original weights are preserved (LoRA only modifies adapter weights)")
            else:
                print("  ⚠ Warning: Some original weights were modified!")
            
        except Exception as e:
            print(f"\n❌ CRITICAL ERROR: Failed to add LoRA: {e}")
            import traceback
            traceback.print_exc()
            print("\n" + "=" * 80)
            print("ERROR: LoRA application failed!")
            print("This will cause all target modules to be trained directly (not using LoRA).")
            print("This will significantly increase trainable parameters (from ~0.05% to ~6%).")
            print("=" * 80)
            print("\nPlease check:")
            print("1. Are the target_modules correctly specified?")
            print("2. Does the model structure match expectations?")
            print("3. Is PEFT library correctly installed?")
            print("\nTo continue with selective fine-tuning (NOT RECOMMENDED),")
            print("uncomment the fallback code below.")
            print("=" * 80)
            
            # 默认情况下，如果LoRA失败，应该停止训练而不是fallback
            # 取消下面的注释以启用fallback（不推荐）
            raise RuntimeError(
                "LoRA application failed. Please fix the issue before training. "
                "Training without LoRA will use ~6% of parameters instead of ~0.05%."
            )
            
            # FALLBACK CODE (NOT RECOMMENDED - 取消注释以启用)
            # print("\n⚠️ WARNING: Falling back to selective fine-tuning (no LoRA)")
            # print("⚠️ This will train ALL target modules directly, increasing trainable parameters significantly!")
            # gpt_wo_ddp.requires_grad_(False)
            # for name, param in gpt_wo_ddp.named_parameters():
            #     for target in target_modules:
            #         if target in name:
            #             param.requires_grad = True
            #             print(f"  Training {name}")
            #             break
    else:
        # 不使用 LoRA，只训练 CrossAttention.proj 层
        print("Training CrossAttention.proj layers only (no LoRA)")
        gpt_wo_ddp.requires_grad_(False)
        for name, param in gpt_wo_ddp.named_parameters():
            if 'ca' in name and 'proj' in name:
                param.requires_grad = True
                print(f"  Training {name}")
    
    # 打印可训练参数数量
    trainable_params = sum(p.numel() for p in gpt_wo_ddp.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in gpt_wo_ddp.parameters())
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M / {total_params / 1e6:.2f}M ({trainable_params / total_params * 100:.2f}%)")
    
    
    # 不加载T5，返回None
    text_tokenizer = None
    text_encoder = None
    
    print("Models built successfully!")
    print("=" * 80)
    
    return vae_local, gpt_wo_ddp, text_tokenizer, text_encoder


def build_optimizer(config: dict, model: nn.Module):
    """构建优化器"""
    optimizer_name = config.get('optimizer', 'adamw').lower()
    
    # 辅助函数：确保参数是浮点数（YAML可能将科学计数法解析为字符串）
    def to_float(value, default):
        if value is None:
            return default
        if isinstance(value, str):
            return float(value)
        return float(value)
    
    learning_rate = to_float(config.get('learning_rate'), 1e-4)
    adam_beta1 = to_float(config.get('adam_beta1'), 0.9)
    adam_beta2 = to_float(config.get('adam_beta2'), 0.999)
    adam_epsilon = to_float(config.get('adam_epsilon'), 1e-8)
    adam_weight_decay = to_float(config.get('adam_weight_decay'), 1e-4)
    
    # 获取可训练参数
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    if optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=learning_rate,
            betas=(adam_beta1, adam_beta2),
            eps=adam_epsilon,
            weight_decay=adam_weight_decay,
        )
    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam(
            trainable_params,
            lr=learning_rate,
            betas=(adam_beta1, adam_beta2),
            eps=adam_epsilon,
        )
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(
            trainable_params,
            lr=learning_rate,
            momentum=0.9,
            weight_decay=adam_weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    return optimizer


def build_dataloader(config: dict):
    """构建数据加载器"""
    print("=" * 80)
    print("Building dataloader...")
    
    # 检查是否使用 prompt-only 模式
    use_prompt_only = config.get('use_prompt_only', False)
    
    if use_prompt_only:
        print("✓ Using Prompt-Only Dataset (no real nude images needed!)")
        dataset = PromptOnlyDataset(
            instance_prompt=config['instance_prompt'],
            key_word=config['key_word'],
            size=config['resolution'],
            num_samples=config.get('num_samples', 100),
            use_random_noise=config.get('use_random_noise', True),
        )
        collate_func = collate_fn_prompt_only
    else:
        print("✓ Using Standard Dataset (requires real images)")
        dataset = EraseInfinityDataset(
            instance_data_root=config['instance_data_dir'],
            instance_prompt=config['instance_prompt'],
            key_word=config['key_word'],
            size=config['resolution'],
            repeats=config.get('repeats', 1),
            center_crop=config.get('center_crop', False),
            random_flip=config.get('random_flip', False),
        )
        collate_func = collate_fn
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['train_batch_size'],
        shuffle=True,
        collate_fn=collate_func,
        num_workers=config.get('dataloader_num_workers', 4),
        pin_memory=True,
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Dataloader batches: {len(dataloader)}")
    print("=" * 80)
    
    return dataloader


def plot_loss_curves(
    loss_history: dict,
    output_dir: str,
    epoch: int,
    plot_index: int = None,
):
    """
    绘制loss曲线（按epoch）
    
    Args:
        loss_history: 字典，包含各个loss的历史记录
            {
                'esd_loss': [loss1, loss2, ...],  # 每个epoch的平均loss
                'total_loss': [total1, total2, ...],  # 每个epoch的平均loss
                'epochs': [epoch1, epoch2, ...],  # epoch编号
            }
        output_dir: 输出目录
        epoch: 当前epoch
        plot_index: 图片编号（用于命名），如果为None则自动查找下一个可用编号
    """
    # 确保有数据可绘制
    if len(loss_history['epochs']) == 0:
        return
    
    # 确定图片编号
    if plot_index is None:
        # 查找已存在的loss_curves文件，找到下一个可用编号
        plot_index = 1
        while os.path.exists(os.path.join(output_dir, f'loss_curves{plot_index}.png')):
            plot_index += 1
    
    # 创建图形
    fig, axes = plt.subplots(1, 1, figsize=(12, 6))
    
    # 绘制 ESD loss
    if len(loss_history['esd_loss']) > 0:
        axes.plot(
            loss_history['epochs'],
            loss_history['esd_loss'],
            label='ESD Loss',
            color='blue',
            linewidth=1.5,
            alpha=0.8,
            marker='o',
            markersize=4
        )
    
    # 绘制 Total loss（如果有多个loss，total_loss会不同）
    if len(loss_history['total_loss']) > 0:
        axes.plot(
            loss_history['epochs'],
            loss_history['total_loss'],
            label='Total Loss',
            color='red',
            linewidth=1.5,
            alpha=0.8,
            linestyle='--',
            marker='s',
            markersize=4
        )
    
    axes.set_xlabel('Epoch', fontsize=12)
    axes.set_ylabel('Loss', fontsize=12)
    axes.set_title(f'Training Loss Curves (Epoch {epoch})', fontsize=14, fontweight='bold')
    axes.legend(fontsize=10)
    axes.grid(True, alpha=0.3)
    
    # 设置y轴为对数刻度（如果loss值变化很大）
    if len(loss_history['esd_loss']) > 0:
        max_loss = max(loss_history['esd_loss'])
        positive_losses = [l for l in loss_history['esd_loss'] if l > 0]
        if positive_losses:  # 确保有大于0的loss
            min_loss = min(positive_losses)
            if max_loss / min_loss > 100:
                axes.set_yscale('log')
    
    plt.tight_layout()
    
    # 保存图片（使用编号命名）
    save_path = os.path.join(output_dir, f'loss_curves{plot_index}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # 同时保存loss值到文本文件（按epoch记录）
    log_path = os.path.join(output_dir, f'loss_log{plot_index}.txt')
    with open(log_path, 'w') as f:
        f.write('Epoch\tESD_Loss\tTotal_Loss\n')
        for i in range(len(loss_history['epochs'])):
            f.write(f"{loss_history['epochs'][i]}\t{loss_history['esd_loss'][i]:.6f}\t{loss_history['total_loss'][i]:.6f}\n")
    
    return plot_index


def create_text_features_from_prompts(
    prompts: list,
    model: nn.Module,
    device: torch.device,
    text_maxlen: int = 256,
    text_channels: int = 4096,
) -> Tuple:
    """
    不使用T5，直接从prompt创建文本特征
    基于模型内部的cfg_uncond创建文本特征
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
    # 基于prompt的长度和内容，对base_features进行简单的修改
    text_features_list = []
    text_lens_list = []
    
    for prompt in prompts:
        # 根据prompt长度决定使用的特征长度（简单策略）
        # 使用prompt字符串的hash来决定特征的变化
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
    # 而不是batch格式 [B, L, Ct5]
    text_features_compact = []
    for feat in text_features_list:
        text_features_compact.append(feat)
    
    # Concatenate所有特征
    text_features = torch.cat(text_features_compact, dim=0).to(device)  # [total_len, Ct5]
    
    # 创建cu_seqlens_k（累积序列长度）
    cu_seqlens_k = F.pad(torch.tensor(text_lens_list, dtype=torch.int32, device=device).cumsum_(0), (1, 0))
    Ltext = max(text_lens_list)
    
    return (text_features, text_lens_list, cu_seqlens_k, Ltext)


def train_one_epoch(
    epoch: int,
    model: nn.Module,
    vae: nn.Module,
    text_encoder: nn.Module,  # 已禁用，为None
    text_tokenizer,  # 已禁用，为None
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: dict,
    device: torch.device,
    loss_history: dict = None,
    global_step: int = 0,
):
    """训练一个 epoch"""
    
    model.train()
    vae.eval()
    # T5已禁用，text_encoder为None，不需要调用eval()
    # if text_encoder is not None:
    #     text_encoder.eval()
    
    # 准备参数
    weight_dtype = torch.bfloat16 if config.get('mixed_precision') == 'bf16' else torch.float32
    gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
    max_grad_norm = config.get('max_grad_norm', 1.0)
    
    # 获取 scale schedule
    scale_schedule = get_default_scale_schedule(config['resolution'])
    
    # 添加到 config（用于 loss 计算）
    class ConfigWithSchedule:
        def __init__(self, config_dict, scale_schedule):
            for k, v in config_dict.items():
                setattr(self, k, v)
            self.scale_schedule = scale_schedule
            self.always_training_scales = len(scale_schedule)
    
    config_obj = ConfigWithSchedule(config, scale_schedule)
    
    # Loss function
    criteria = nn.MSELoss()
    
    # Progress bar
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    total_loss = 0.0
    num_steps = 0
    
    # 初始化loss历史记录（如果未提供）
    if loss_history is None:
        loss_history = {
            'esd_loss': [],
            'total_loss': [],
            'epochs': [],
        }
    
    # 用于记录当前epoch的所有loss值（用于计算epoch平均loss）
    epoch_losses_esd = []
    epoch_losses_total = []
    
    for step, batch in enumerate(progress_bar):
        # ==================== 编码文本（不使用T5） ====================
        prompts = batch['prompts']
        
        with torch.no_grad():
            # 不使用T5，直接从prompt创建文本特征
            # 使用模型内部的cfg_uncond作为基础
            text_cond_tuple = create_text_features_from_prompts(
                prompts=prompts,
                model=model,
                device=device,
                text_maxlen=config.get('max_sequence_length', 256),
                text_channels=model.Ct5 if hasattr(model, 'Ct5') else 4096,
            )
        
        # ==================== 计算第一个 ESD Loss ====================
        # 使用从 EraseAnything 迁移的第一个 ESD loss
        loss_esd = calculate_first_esd_loss(
            args=config_obj,
            batch=batch,
            gpt_model=model,
            vae_model=vae,
            text_cond_tuple=text_cond_tuple,
            text_cond_tuple_uncond=None,  # 会自动生成
            criteria=criteria,
            negative_guidance=config.get('negative_guidance', 1.0),
            device=device,
            weight_dtype=weight_dtype,
        )
        
        # 当前只有一个loss，所以total_loss = esd_loss
        loss = loss_esd
        
        # 缩放 loss
        loss = loss / gradient_accumulation_steps
        
        # Backward
        loss.backward()
        
        # 梯度累积
        if (step + 1) % gradient_accumulation_steps == 0:
            # 梯度裁剪
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            # 优化器步骤
            optimizer.step()
            optimizer.zero_grad()
            
            # 更新统计
            loss_value = loss.item() * gradient_accumulation_steps
            total_loss += loss_value
            num_steps += 1
            
            # 记录当前epoch的loss（用于计算epoch平均loss）
            epoch_losses_esd.append(loss_esd.item())
            epoch_losses_total.append(loss_value)
            
            # 更新进度条
            avg_loss = total_loss / num_steps
            progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
    
    # 计算epoch平均loss并记录
    if len(epoch_losses_esd) > 0:
        avg_esd_loss = sum(epoch_losses_esd) / len(epoch_losses_esd)
        avg_total_loss = sum(epoch_losses_total) / len(epoch_losses_total)
        loss_history['esd_loss'].append(avg_esd_loss)
        loss_history['total_loss'].append(avg_total_loss)
        loss_history['epochs'].append(epoch)
    
    # Wandb logging - 已禁用
    # if has_wandb and config.get('report_to') == 'wandb':
    #     wandb.log({
    #         'train/loss': loss.item() * gradient_accumulation_steps,
    #         'train/avg_loss': avg_loss,
    #         'train/lr': optimizer.param_groups[0]['lr'],
    #         'train/epoch': epoch,
    #         'train/step': step,
    #     })
    
    avg_loss = total_loss / num_steps if num_steps > 0 else 0.0
    return avg_loss


def main():
    """主函数"""
    
    # ==================== 解析参数 ====================
    args = parse_args()
    config = load_config(args.config)
    
    # ==================== 设置设备 ====================
    device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # ==================== 设置随机种子 ====================
    setup_seed(config.get('seed', 42))
    
    # ==================== 创建输出目录 ====================
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # ==================== 初始化 Wandb ====================
    # Wandb 已禁用，不需要日志记录
    # if has_wandb and config.get('report_to') == 'wandb':
    #     wandb.init(
    #         project=config.get('project_name', 'EraseInfinity'),
    #         name=config.get('exp_name', 'erase_nude'),
    #         config=config,
    #     )
    
    # ==================== 构建模型 ====================
    vae, model, text_tokenizer, text_encoder = build_models(config, device)
    
    # ==================== 构建优化器 ====================
    optimizer = build_optimizer(config, model)
    
    # ==================== 构建数据加载器 ====================
    dataloader = build_dataloader(config)
    
    # ==================== 训练循环 ====================
    # 只使用 num_train_epochs 控制训练，移除 max_train_steps 限制
    num_train_epochs = config.get('num_train_epochs', 15)
    gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
    
    # 计算每个 epoch 的步数
    num_update_steps_per_epoch = math.ceil(len(dataloader) / gradient_accumulation_steps)
    total_train_steps = num_train_epochs * num_update_steps_per_epoch
    
    print("=" * 80)
    print("Starting training...")
    print(f"Dataset size: {len(dataloader.dataset)}")
    print(f"Batches per epoch: {len(dataloader)}")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"Update steps per epoch: {num_update_steps_per_epoch}")
    print(f"Total epochs: {num_train_epochs}")
    print(f"Total training steps: {total_train_steps}")
    print("=" * 80)
    
    global_step = 0
    first_epoch = 0
    
    # 初始化loss历史记录
    loss_history = {
        'esd_loss': [],
        'total_loss': [],
        'epochs': [],
    }
    
    # 用于确定loss曲线图片编号
    plot_index = None
    
    for epoch in range(first_epoch, num_train_epochs):
        avg_loss = train_one_epoch(
            epoch=epoch,
            model=model,
            vae=vae,
            text_encoder=text_encoder,
            text_tokenizer=text_tokenizer,
            dataloader=dataloader,
            optimizer=optimizer,
            config=config,
            device=device,
            loss_history=loss_history,
            global_step=global_step,
        )
        
        print(f"Epoch {epoch+1}/{num_train_epochs} completed. Average loss: {avg_loss:.4f}")
        
        # 每个epoch结束时绘制loss曲线
        plot_index = plot_loss_curves(
            loss_history=loss_history,
            output_dir=output_dir,
            epoch=epoch,
            plot_index=plot_index,  # 第一次调用时自动查找编号，后续使用相同编号
        )
        
        # 更新global_step（每个epoch的步数）
        global_step += num_update_steps_per_epoch
    
    # ==================== 保存最终模型 ====================
    # 自动检测现有checkpoint编号并递增
    import re
    import glob
    
    print("\n" + "=" * 80)
    print("Saving final checkpoint...")
    
    # 查找现有的checkpoint目录
    existing_checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    checkpoint_numbers = []
    for cp in existing_checkpoints:
        match = re.search(r'checkpoint-(\d+)', cp)
        if match:
            checkpoint_numbers.append(int(match.group(1)))
    
    # 确定下一个checkpoint编号
    if checkpoint_numbers:
        next_checkpoint_num = max(checkpoint_numbers) + 1
        print(f"Found existing checkpoints: {sorted(checkpoint_numbers)}")
    else:
        next_checkpoint_num = 1
        print("No existing checkpoints found.")
    
    checkpoint_save_path = os.path.join(output_dir, f"checkpoint-{next_checkpoint_num}")
    os.makedirs(checkpoint_save_path, exist_ok=True)
    print(f"Saving to: {checkpoint_save_path}")
    
    # 保存模型
    if config.get('use_lora', True):
        # 保存 LoRA 权重
        try:
            from peft import get_peft_model_state_dict
            import json
            
            # 提取 LoRA 权重
            if hasattr(model, 'peft_config'):
                model_lora_state_dict = get_peft_model_state_dict(model)
            else:
                # 手动提取 LoRA 权重
                print("Warning: Model doesn't have peft_config, manually extracting LoRA weights...")
                model_lora_state_dict = {}
                for name, param in model.named_parameters():
                    if 'lora_A' in name or 'lora_B' in name or 'lora_embedding' in name:
                        model_lora_state_dict[name] = param.data.clone()
                
                if len(model_lora_state_dict) == 0:
                    print("Warning: No LoRA weights found! Saving all trainable parameters...")
                    for name, param in model.named_parameters():
                        if param.requires_grad:
                            model_lora_state_dict[name] = param.data.clone()
            
            if len(model_lora_state_dict) == 0:
                print("Error: No weights to save!")
            else:
                # 保存权重文件
                try:
                    from safetensors.torch import save_file
                    save_file(model_lora_state_dict, os.path.join(checkpoint_save_path, "adapter_model.safetensors"))
                    print(f"  ✓ Saved {len(model_lora_state_dict)} LoRA weights as safetensors")
                except ImportError:
                    torch.save(model_lora_state_dict, os.path.join(checkpoint_save_path, "adapter_model.bin"))
                    print(f"  ✓ Saved {len(model_lora_state_dict)} LoRA weights as .bin")
                except Exception as e:
                    print(f"Warning: Failed to save as safetensors: {e}")
                    torch.save(model_lora_state_dict, os.path.join(checkpoint_save_path, "adapter_model.bin"))
                    print(f"  ✓ Saved {len(model_lora_state_dict)} LoRA weights as .bin")
                
                # 保存配置文件
                if hasattr(model, 'peft_config'):
                    peft_config_dict = {}
                    for adapter_name, config_obj in model.peft_config.items():
                        if hasattr(config_obj, 'to_dict'):
                            config_dict = config_obj.to_dict()
                            for key, value in config_dict.items():
                                if isinstance(value, set):
                                    config_dict[key] = list(value)
                            peft_config_dict[adapter_name] = config_dict
                        else:
                            peft_config_dict[adapter_name] = str(config_obj)
                    
                    # 提取标准格式配置
                    if "default" in peft_config_dict:
                        final_config = peft_config_dict["default"]
                    elif len(peft_config_dict) > 0:
                        final_config = list(peft_config_dict.values())[0]
                    else:
                        final_config = None
                else:
                    final_config = None
                
                # 如果无法从peft_config获取，创建标准配置
                if final_config is None:
                    target_modules_list = config.get('lora_target_modules', ["ca.proj"])
                    final_config = {
                        "peft_type": "LORA",
                        "task_type": "CAUSAL_LM",
                        "inference_mode": False,
                        "r": config.get('lora_rank', 8),
                        "lora_alpha": config.get('lora_alpha', 8),
                        "lora_dropout": config.get('lora_dropout', 0.0),
                        "target_modules": target_modules_list,
                        "bias": "none",
                        "fan_in_fan_out": False,
                        "init_lora_weights": True,
                    }
                
                # 保存adapter_config.json
                config_file_path = os.path.join(checkpoint_save_path, "adapter_config.json")
                with open(config_file_path, "w", encoding='utf-8') as f:
                    json.dump(final_config, f, indent=2, ensure_ascii=False)
                print(f"  ✓ Saved adapter_config.json")
                
        except Exception as e:
            print(f"Error saving LoRA weights: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: 保存所有可训练参数
            print("Falling back to saving all trainable parameters...")
            trainable_state_dict = {name: param.data.clone() for name, param in model.named_parameters() if param.requires_grad}
            torch.save(trainable_state_dict, os.path.join(checkpoint_save_path, "trainable_params.bin"))
            print(f"  ✓ Saved {len(trainable_state_dict)} trainable parameters")
    else:
        # 保存完整模型
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
        }, os.path.join(checkpoint_save_path, "model.pth"))
        print(f"  ✓ Saved full model")
    
    print(f"\n✓ Checkpoint saved to: {checkpoint_save_path}")
    print("=" * 80)
    print("Training completed!")
    print("=" * 80)
    
    # ==================== 关闭 Wandb ====================
    # Wandb 已禁用，不需要关闭
    # if has_wandb and config.get('report_to') == 'wandb':
    #     wandb.finish()


if __name__ == "__main__":
    main()

