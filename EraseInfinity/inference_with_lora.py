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
import torch.nn.functional as F
from transformers import AutoTokenizer, T5EncoderModel
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
    parser.add_argument("--t5_path", type=str, default="google/flan-t5-xl", help="Path to T5 model")
    
    # 推理参数
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
    
    return parser.parse_args()


def load_text_encoder(t5_path: str, device):
    """加载 T5 文本编码器"""
    print(f"Loading T5 text encoder from {t5_path}")
    text_tokenizer = AutoTokenizer.from_pretrained(t5_path)
    text_encoder = T5EncoderModel.from_pretrained(t5_path, torch_dtype=torch.bfloat16).to(device)
    text_encoder.eval()
    print("✓ T5 loaded")
    return text_tokenizer, text_encoder


def encode_prompt(text_tokenizer, text_encoder, prompt):
    """编码文本 prompt"""
    captions = [prompt]
    tokens = text_tokenizer(
        text=captions,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = tokens.input_ids.cuda(non_blocking=True)
    mask = tokens.attention_mask.cuda(non_blocking=True)
    
    with torch.no_grad():
        text_features = text_encoder(input_ids=input_ids, attention_mask=mask)['last_hidden_state'].float()
    
    lens: List[int] = mask.sum(dim=-1).tolist()
    cu_seqlens_k = F.pad(mask.sum(dim=-1).to(dtype=torch.int32).cumsum_(0), (1, 0))
    Ltext = max(lens)
    
    kv_compact = []
    for len_i, feat_i in zip(lens, text_features.unbind(0)):
        kv_compact.append(feat_i[:len_i])
    kv_compact = torch.cat(kv_compact, dim=0)
    
    text_cond_tuple = (kv_compact, lens, cu_seqlens_k, Ltext)
    return text_cond_tuple


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
            self.model = '2bc8'
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
            self.t5_path = args_in.t5_path
    
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
        
        # 尝试使用 PEFT 加载
        try:
            from peft import PeftModel, LoraConfig, inject_adapter_in_model
            
            config_path = os.path.join(args.lora_ckpt, "adapter_config.json") if os.path.isdir(args.lora_ckpt) else None
            
            if config_path and os.path.exists(config_path):
                print("Loading LoRA using PeftModel.from_pretrained...")
                gpt_wo_ddp = PeftModel.from_pretrained(gpt_wo_ddp, args.lora_ckpt)
            else:
                print("Loading LoRA weights manually with PEFT...")
                lora_config = LoraConfig(
                    r=8,
                    lora_alpha=8,
                    lora_dropout=0.0,
                    target_modules=["ca.proj"],
                    bias="none",
                    task_type="CAUSAL_LM",
                    inference_mode=True,
                )
                
                # 添加假方法避免错误
                def dummy_prepare_inputs_for_generation(self, *args, **kwargs):
                    return kwargs if kwargs else {}
                gpt_wo_ddp.prepare_inputs_for_generation = dummy_prepare_inputs_for_generation.__get__(gpt_wo_ddp, type(gpt_wo_ddp))
                
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
    text_tokenizer,
    text_encoder,
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
):
    """生成单张图像"""
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
    
    # 编码文本
    text_cond_tuple = encode_prompt(text_tokenizer, text_encoder, prompt)
    
    if negative_prompt:
        negative_label_tuple = encode_prompt(text_tokenizer, text_encoder, negative_prompt)
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
    
    # 加载 T5 文本编码器
    text_tokenizer, text_encoder = load_text_encoder(args.t5_path, device)
    
    # 获取 scale schedule（分辨率配置）
    h_div_w_template = args.h_div_w_template
    pn = args.pn
    scale_schedule = dynamic_resolution_h_w[h_div_w_template][pn]['scales']
    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
    tgt_h, tgt_w = dynamic_resolution_h_w[h_div_w_template][pn]['pixel']
    
    print(f"\nResolution: {tgt_h}x{tgt_w}")
    print(f"Scale schedule: {len(scale_schedule)} scales")
    
    # 生成图像
    with autocast(dtype=torch.bfloat16):
        generated_image = gen_one_img(
            infinity_model=gpt_model,
            vae=vae,
            text_tokenizer=text_tokenizer,
            text_encoder=text_encoder,
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
        )
    
    # 保存图像
    if args.save_file:
        save_path = args.save_file
    else:
        import hashlib
        prompt_hash = hashlib.md5(args.prompt.encode('utf-8')).hexdigest()[:8]
        lora_suffix = "_lora" if (args.use_lora and not args.no_lora) else "_no_lora"
        save_path = os.path.join(args.output_dir, f"{prompt_hash}{lora_suffix}.jpg")
    
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    cv2.imwrite(save_path, generated_image.cpu().numpy())
    
    print(f"\n{'='*80}")
    print(f"✓ Image saved to: {os.path.abspath(save_path)}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
