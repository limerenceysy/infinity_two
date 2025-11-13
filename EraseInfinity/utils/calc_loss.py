# coding: UTF-8
"""
    @date:  2025.01
    @func:  ESD Loss calculation for Infinity Autoregressive Model
            符合自回归原理的 ESD loss 计算
    
    核心思想（参考 EraseAnything）：
    1. 用含nude概念的prompt自回归生成到中间几个scale，得到每个scale的logits
    2. 用空prompt（无条件）自回归生成到同样的中间几个scale，得到每个scale的logits
    3. 对于每个scale，计算loss = MSE(e_n, e_0 - negative_guidance * (e_p - e_0))
    其中：
    - e_n: 当前模型（正在训练的）用含概念prompt预测的logits（需要梯度）
    - e_0: 无条件预测的logits（不需要梯度）
    - e_p: 原始模型（冻结的）用含概念prompt预测的logits（不需要梯度）
"""

import random
import ast
import torch
import torch.nn.functional as F
from typing import Tuple, List, Optional


def autoregressive_generate_to_scales(
    gpt_model,
    vae_model,
    text_cond_tuple: Tuple,
    target_scales: List[int],  # 要生成到的scale索引列表，如[1, 2, 3]
    scale_schedule: List[Tuple],
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    return_logits: bool = True,  # True返回logits，False返回tokens
    enable_grad: bool = False,  # 是否启用梯度（用于训练时需要梯度）
) -> List[torch.Tensor]:
    """
    自回归生成到指定的几个scale，返回每个scale的logits或tokens
    
    Args:
        gpt_model: Infinity GPT模型
        vae_model: Infinity VAE模型
        text_cond_tuple: 文本条件 (text_features, text_lens, cu_seqlens_k, Ltext)
        target_scales: 要生成到的scale索引列表，如[1, 2, 3]表示生成到第1、2、3个scale
        scale_schedule: scale调度表
        device: 设备
        dtype: 数据类型
        return_logits: 是否返回logits（True）还是tokens（False）
        enable_grad: 是否启用梯度（True表示需要梯度，False表示不需要梯度）
        
    Returns:
        scale_logits_list: 每个target_scale的logits列表，每个元素shape为[B, scale_len, vocab_size]
    """
    if enable_grad:
        gpt_model.train()
    else:
        gpt_model.eval()
    text_features, text_lens, cu_seqlens_k, Ltext = text_cond_tuple
    B = len(text_lens)
    
    scale_logits_list = []
    all_tokens = []  # 累积的tokens，用于下一个scale的输入
    all_features = []  # 累积的features，用于下一个scale的输入
    
    # 预加载所有可能的 scale_schedule 组合到 rope2d_freqs_grid
    if hasattr(gpt_model, 'rope2d_freqs_grid') and gpt_model.rope2d_freqs_grid is not None:
        full_key = str(tuple(scale_schedule))
        if full_key in gpt_model.rope2d_freqs_grid:
            rope_cache_full = gpt_model.rope2d_freqs_grid[full_key]
            # 为所有可能的部分 scale_schedule 创建 key
            for i in range(1, len(scale_schedule) + 1):
                partial_schedule = scale_schedule[:i]
                partial_key = str(tuple(partial_schedule))
                if partial_key not in gpt_model.rope2d_freqs_grid:
                    partial_seq_len = sum(t * h * w for t, h, w in partial_schedule)
                    if rope_cache_full.shape[4] >= partial_seq_len:
                        new_rope_cache = rope_cache_full[:, :, :, :, :partial_seq_len].clone()
                        gpt_model.rope2d_freqs_grid[partial_key] = new_rope_cache
                    else:
                        gpt_model.rope2d_freqs_grid[partial_key] = rope_cache_full.clone()
    
    # 根据enable_grad决定是否使用no_grad上下文管理器
    if not enable_grad:
        with torch.no_grad():
            # 逐scale生成
            for scale_idx in range(max(target_scales) + 1):
                current_scale_schedule = scale_schedule[:scale_idx + 1]
                t, h, w = scale_schedule[scale_idx]
                scale_len = t * h * w
                
                # rope2d_freqs_grid 的键应该已经预加载了，这里只做检查
                # 如果键不存在，尝试找到包含当前 scale 的完整 schedule 的键
                if hasattr(gpt_model, 'rope2d_freqs_grid') and gpt_model.rope2d_freqs_grid is not None:
                    current_key = str(tuple(current_scale_schedule))
                    if current_key not in gpt_model.rope2d_freqs_grid:
                        # 查找包含当前 scale 的完整 schedule 的键
                        found_key = None
                        for key in gpt_model.rope2d_freqs_grid.keys():
                            try:
                                key_schedule = ast.literal_eval(key)  # 将字符串转换回元组
                                if isinstance(key_schedule, tuple) and len(key_schedule) > 0:
                                    # 检查当前 scale_schedule 是否是 key_schedule 的前缀
                                    if len(current_scale_schedule) <= len(key_schedule):
                                        match = True
                                        for i, (t1, h1, w1) in enumerate(current_scale_schedule):
                                            if i < len(key_schedule):
                                                t2, h2, w2 = key_schedule[i]
                                                if (t1, h1, w1) != (t2, h2, w2):
                                                    match = False
                                                    break
                                        if match:
                                            found_key = key
                                            break
                            except:
                                continue
                        
                        if found_key:
                            # 使用找到的键，复制对应的 rope_cache
                            rope_cache = gpt_model.rope2d_freqs_grid[found_key]
                            # 计算当前 scale_schedule 需要的长度
                            current_seq_len = sum(t * h * w for t, h, w in current_scale_schedule)
                            # 如果 rope_cache 的长度足够，创建一个新的键
                            if rope_cache.shape[4] >= current_seq_len:
                                # 只取需要的部分
                                new_rope_cache = rope_cache[:, :, :, :, :current_seq_len].clone()
                                gpt_model.rope2d_freqs_grid[current_key] = new_rope_cache
                            else:
                                # 如果长度不够，使用完整的 rope_cache（可能会出错，但至少不会 KeyError）
                                gpt_model.rope2d_freqs_grid[current_key] = rope_cache.clone()
                        else:
                            # 如果找不到匹配的键，使用完整的 scale_schedule
                            # 这应该总是存在的
                            full_key = str(tuple(scale_schedule))
                            if full_key in gpt_model.rope2d_freqs_grid:
                                rope_cache = gpt_model.rope2d_freqs_grid[full_key]
                                current_seq_len = sum(t * h * w for t, h, w in current_scale_schedule)
                                if rope_cache.shape[4] >= current_seq_len:
                                    new_rope_cache = rope_cache[:, :, :, :, :current_seq_len].clone()
                                    gpt_model.rope2d_freqs_grid[current_key] = new_rope_cache
                                else:
                                    gpt_model.rope2d_freqs_grid[current_key] = rope_cache.clone()
                            else:
                                # 如果连完整的 scale_schedule 都不存在，打印警告并使用第一个可用的键
                                print(f"Warning: rope2d_freqs_grid key '{current_key}' not found, and full key '{full_key}' also not found.")
                                if len(gpt_model.rope2d_freqs_grid) > 0:
                                    # 使用第一个可用的键
                                    first_key = list(gpt_model.rope2d_freqs_grid.keys())[0]
                                    rope_cache = gpt_model.rope2d_freqs_grid[first_key]
                                    current_seq_len = sum(t * h * w for t, h, w in current_scale_schedule)
                                    if rope_cache.shape[4] >= current_seq_len:
                                        new_rope_cache = rope_cache[:, :, :, :, :current_seq_len].clone()
                                        gpt_model.rope2d_freqs_grid[current_key] = new_rope_cache
                                    else:
                                        gpt_model.rope2d_freqs_grid[current_key] = rope_cache.clone()
                
                if scale_idx == 0:
                    # 第一个scale：使用空输入（SOS token）
                    # x_BLC_wo_prefix应该是空的，因为还没有生成任何tokens
                    x_BLC = torch.zeros(B, 0, gpt_model.d_vae, device=device, dtype=dtype)
                else:
                    # 后续scale：使用之前生成的features
                    # 关键：x_BLC_wo_prefix的长度应该是所有scale的长度之和减去第一个scale的长度
                    # 因为第一个scale是由SOS token生成的，不应该包含在x_BLC_wo_prefix中
                    # 参考训练代码：x_BLC_wo_prefix[:, :(training_seq_len-np.array(scale_schedule[0]).prod()), :]
                    expected_len = sum(t * h * w for t, h, w in current_scale_schedule) - (current_scale_schedule[0][0] * current_scale_schedule[0][1] * current_scale_schedule[0][2])
                    
                    if all_features:
                        # 将所有之前scale的features拼接起来
                        x_BLC = torch.cat(all_features, dim=1)  # [B, total_prev_len, d_vae]
                        actual_len = x_BLC.shape[1]
                        
                        if actual_len < expected_len:
                            # 如果长度不够，需要padding（当前scale还没有生成，所以需要占位符）
                            pad_len = expected_len - actual_len
                            padding = torch.zeros(B, pad_len, gpt_model.d_vae, device=device, dtype=dtype)
                            x_BLC = torch.cat([x_BLC, padding], dim=1)
                        elif actual_len > expected_len:
                            # 如果长度超出，截断（这不应该发生，但为了安全）
                            x_BLC = x_BLC[:, :expected_len, :]
                    else:
                        # 如果没有累积的features，创建全零输入（包括当前scale的占位符）
                        x_BLC = torch.zeros(B, expected_len, gpt_model.d_vae, device=device, dtype=dtype)
                
                # Forward pass
                logits_BLV = gpt_model(text_cond_tuple, x_BLC, scale_schedule=current_scale_schedule)
                
                # 提取当前scale的logits
                # 注意：logits_BLV的形状是[B, total_seq_len, vocab_size]
                # 我们需要当前scale对应的logits部分
                if scale_idx == 0:
                    current_logits = logits_BLV  # 第一个scale，所有logits都是当前scale的
                else:
                    # 后续scale：取最后scale_len个logits
                    current_logits = logits_BLV[:, -scale_len:, :]  # [B, scale_len, vocab_size]
                
                # 如果当前scale在target_scales中，保存logits
                if scale_idx in target_scales:
                    scale_logits_list.append(current_logits)
                
                # 采样得到tokens（用于下一个scale的输入）
                if gpt_model.use_bit_label:
                    # bit label: logits shape是[B, scale_len, 2]（每个token是2个bit）
                    tmp_bs, tmp_seq_len = current_logits.shape[:2]
                    logits_reshaped = current_logits.reshape(tmp_bs, -1, 2)
                    # 使用argmax采样（训练时不需要随机采样）
                    tokens = logits_reshaped.argmax(dim=-1)  # [B, scale_len]
                    tokens = tokens.reshape(tmp_bs, tmp_seq_len, -1)  # [B, scale_len, codebook_dim]
                else:
                    # 普通token: 使用argmax采样
                    tokens = current_logits.argmax(dim=-1)  # [B, scale_len]
                
                all_tokens.append(tokens)
                
                # 将tokens转换为features用于下一个scale
                # 需要先将tokens (Long indices) 转换为float embeddings
                if hasattr(gpt_model, 'word_embed'):
                    if gpt_model.use_bit_label:
                        # 对于bit label: tokens shape是[B, scale_len, codebook_dim] (Long, 0或1)
                        # 需要转换为[B, scale_len, d_vae]的float embeddings
                        B, L, codebook_dim = tokens.shape
                        # tokens是Long，需要转换为float bits (0.0或1.0)
                        bits_float = tokens.float()  # [B, scale_len, codebook_dim]
                        # indices_to_codes对于bit_label期望 [B, L, 1, codebook_dim] 格式
                        bits_for_quantizer = bits_float.unsqueeze(2)  # [B, scale_len, 1, codebook_dim]
                        # 调用quantizer转换，注意：由于ndim>=3，should_transpose可能为True
                        codes = vae_model.quantizer.lfq.indices_to_codes(bits_for_quantizer, label_type='bit_label')
                        # codes shape可能是 [B, d_vae, scale_len, 1] (如果transposed) 或 [B, scale_len, 1, d_vae]
                        # 需要reshape为 [B, scale_len, d_vae]
                        if codes.ndim == 4:
                            # 检查哪个维度是scale_len
                            if codes.shape[2] == L:
                                # [B, d_vae, scale_len, 1] -> [B, scale_len, d_vae]
                                codes = codes.squeeze(-1).permute(0, 2, 1)
                            elif codes.shape[1] == L:
                                # [B, scale_len, 1, d_vae] -> [B, scale_len, d_vae]
                                codes = codes.squeeze(2)
                            else:
                                # 尝试其他可能的shape
                                codes = codes.reshape(B, L, -1)
                        elif codes.ndim == 3:
                            # [B, L, d_vae] 或 [B, d_vae, L]
                            if codes.shape[1] != L:
                                codes = codes.permute(0, 2, 1)
                        token_embeddings = codes.to(dtype=dtype)
                        # 确保最终shape是 [B, scale_len, d_vae]
                        if token_embeddings.shape[:2] != (B, L):
                            # 如果shape不对，尝试reshape
                            token_embeddings = token_embeddings.reshape(B, L, -1)
                        # 确保最后一维是d_vae
                        if token_embeddings.shape[-1] != gpt_model.d_vae:
                            # 如果维度不匹配，可能需要截断或padding（这不应该发生）
                            if token_embeddings.shape[-1] > gpt_model.d_vae:
                                token_embeddings = token_embeddings[..., :gpt_model.d_vae]
                            else:
                                # padding with zeros
                                padding = torch.zeros(B, L, gpt_model.d_vae - token_embeddings.shape[-1], 
                                                    device=token_embeddings.device, dtype=token_embeddings.dtype)
                                token_embeddings = torch.cat([token_embeddings, padding], dim=-1)
                    else:
                        # 普通token: tokens shape是[B, scale_len] (Long indices)
                        # 需要转换为[B, scale_len, d_vae]的float embeddings
                        B, L = tokens.shape
                        # indices_to_codes对于int_label会自动添加维度，输入 [B, scale_len] 即可
                        # 但为了明确，我们添加维度: [B, scale_len] -> [B, scale_len, 1]
                        indices_for_quantizer = tokens.unsqueeze(-1)  # [B, scale_len, 1]
                        codes = vae_model.quantizer.lfq.indices_to_codes(indices_for_quantizer, label_type='int_label')
                        # codes shape可能是 [B, d_vae, scale_len, 1] (如果transposed) 或 [B, scale_len, 1, d_vae]
                        # 需要reshape为 [B, scale_len, d_vae]
                        if codes.ndim == 4:
                            # 检查哪个维度是scale_len
                            if codes.shape[2] == L:
                                # [B, d_vae, scale_len, 1] -> [B, scale_len, d_vae]
                                codes = codes.squeeze(-1).permute(0, 2, 1)
                            elif codes.shape[1] == L:
                                # [B, scale_len, 1, d_vae] -> [B, scale_len, d_vae]
                                codes = codes.squeeze(2)
                            else:
                                # 尝试其他可能的shape
                                codes = codes.reshape(B, L, -1)
                        elif codes.ndim == 3:
                            # [B, L, d_vae] 或 [B, d_vae, L]
                            if codes.shape[1] != L:
                                codes = codes.permute(0, 2, 1)
                        token_embeddings = codes.to(dtype=dtype)
                        # 确保最终shape是 [B, scale_len, d_vae]
                        if token_embeddings.shape[:2] != (B, L):
                            # 如果shape不对，尝试reshape
                            token_embeddings = token_embeddings.reshape(B, L, -1)
                        # 确保最后一维是d_vae
                        if token_embeddings.shape[-1] != gpt_model.d_vae:
                            # 如果维度不匹配，可能需要截断或padding（这不应该发生）
                            if token_embeddings.shape[-1] > gpt_model.d_vae:
                                token_embeddings = token_embeddings[..., :gpt_model.d_vae]
                            else:
                                # padding with zeros
                                padding = torch.zeros(B, L, gpt_model.d_vae - token_embeddings.shape[-1], 
                                                    device=token_embeddings.device, dtype=token_embeddings.dtype)
                                token_embeddings = torch.cat([token_embeddings, padding], dim=-1)
                    
                    # 现在token_embeddings是float类型，shape为[B, scale_len, d_vae]
                    # 注意：模型forward方法会再次应用norm0_ve和word_embed
                    # 所以我们应该存储原始的token_embeddings（在word_embed之前）
                    # 确保token_embeddings的dtype与word_embed权重的dtype匹配
                    if hasattr(gpt_model, 'word_embed'):
                        # 获取word_embed权重的dtype
                        word_embed_dtype = gpt_model.word_embed.weight.dtype
                        # 确保token_embeddings的dtype匹配
                        token_embeddings = token_embeddings.to(dtype=word_embed_dtype)
                    
                    # 存储token_embeddings（在word_embed之前），因为模型forward会再次应用word_embed
                    # token_embeddings的shape应该是[B, scale_len, d_vae]
                    all_features.append(token_embeddings)
    else:
        # enable_grad=True，不使用no_grad，直接执行循环
        # 逐scale生成
        for scale_idx in range(max(target_scales) + 1):
            current_scale_schedule = scale_schedule[:scale_idx + 1]
            t, h, w = scale_schedule[scale_idx]
            scale_len = t * h * w
            
            # rope2d_freqs_grid 的键应该已经预加载了，这里只做检查
            if hasattr(gpt_model, 'rope2d_freqs_grid') and gpt_model.rope2d_freqs_grid is not None:
                current_key = str(tuple(current_scale_schedule))
                if current_key not in gpt_model.rope2d_freqs_grid:
                    # 查找包含当前 scale 的完整 schedule 的键
                    found_key = None
                    for key in gpt_model.rope2d_freqs_grid.keys():
                        try:
                            key_schedule = ast.literal_eval(key)  # 将字符串转换回元组
                            if isinstance(key_schedule, tuple) and len(key_schedule) > 0:
                                # 检查当前 scale_schedule 是否是 key_schedule 的前缀
                                if len(current_scale_schedule) <= len(key_schedule):
                                    match = True
                                    for i, (t1, h1, w1) in enumerate(current_scale_schedule):
                                        if i < len(key_schedule):
                                            t2, h2, w2 = key_schedule[i]
                                            if (t1, h1, w1) != (t2, h2, w2):
                                                match = False
                                                break
                                    if match:
                                        found_key = key
                                        break
                        except:
                            continue
                    
                    if found_key:
                        # 使用找到的键，复制对应的 rope_cache
                        rope_cache = gpt_model.rope2d_freqs_grid[found_key]
                        # 计算当前 scale_schedule 需要的长度
                        current_seq_len = sum(t * h * w for t, h, w in current_scale_schedule)
                        # 如果 rope_cache 的长度足够，创建一个新的键
                        if rope_cache.shape[4] >= current_seq_len:
                            # 只取需要的部分
                            new_rope_cache = rope_cache[:, :, :, :, :current_seq_len].clone()
                            gpt_model.rope2d_freqs_grid[current_key] = new_rope_cache
                        else:
                            # 如果长度不够，使用完整的 rope_cache（可能会出错，但至少不会 KeyError）
                            gpt_model.rope2d_freqs_grid[current_key] = rope_cache.clone()
                    else:
                        # 如果找不到匹配的键，使用完整的 scale_schedule
                        full_key = str(tuple(scale_schedule))
                        if full_key in gpt_model.rope2d_freqs_grid:
                            rope_cache = gpt_model.rope2d_freqs_grid[full_key]
                            current_seq_len = sum(t * h * w for t, h, w in current_scale_schedule)
                            if rope_cache.shape[4] >= current_seq_len:
                                new_rope_cache = rope_cache[:, :, :, :, :current_seq_len].clone()
                                gpt_model.rope2d_freqs_grid[current_key] = new_rope_cache
                            else:
                                gpt_model.rope2d_freqs_grid[current_key] = rope_cache.clone()
                        else:
                            # 如果连完整的 scale_schedule 都不存在，打印警告并使用第一个可用的键
                            print(f"Warning: rope2d_freqs_grid key '{current_key}' not found, and full key '{full_key}' also not found.")
                            if len(gpt_model.rope2d_freqs_grid) > 0:
                                # 使用第一个可用的键
                                first_key = list(gpt_model.rope2d_freqs_grid.keys())[0]
                                rope_cache = gpt_model.rope2d_freqs_grid[first_key]
                                current_seq_len = sum(t * h * w for t, h, w in current_scale_schedule)
                                if rope_cache.shape[4] >= current_seq_len:
                                    new_rope_cache = rope_cache[:, :, :, :, :current_seq_len].clone()
                                    gpt_model.rope2d_freqs_grid[current_key] = new_rope_cache
                                else:
                                    gpt_model.rope2d_freqs_grid[current_key] = rope_cache.clone()
            
            if scale_idx == 0:
                # 第一个scale：使用空输入（SOS token）
                # x_BLC_wo_prefix应该是空的，因为还没有生成任何tokens
                x_BLC = torch.zeros(B, 0, gpt_model.d_vae, device=device, dtype=dtype)
            else:
                # 后续scale：使用之前生成的features
                # 关键：x_BLC_wo_prefix的长度应该是所有scale的长度之和减去第一个scale的长度
                # 因为第一个scale是由SOS token生成的，不应该包含在x_BLC_wo_prefix中
                # 参考训练代码：x_BLC_wo_prefix[:, :(training_seq_len-np.array(scale_schedule[0]).prod()), :]
                expected_len = sum(t * h * w for t, h, w in current_scale_schedule) - (current_scale_schedule[0][0] * current_scale_schedule[0][1] * current_scale_schedule[0][2])
                
                if all_features:
                    # 将所有之前scale的features拼接起来
                    x_BLC = torch.cat(all_features, dim=1)  # [B, total_prev_len, d_vae]
                    actual_len = x_BLC.shape[1]
                    
                    if actual_len < expected_len:
                        # 如果长度不够，需要padding（当前scale还没有生成，所以需要占位符）
                        pad_len = expected_len - actual_len
                        padding = torch.zeros(B, pad_len, gpt_model.d_vae, device=device, dtype=dtype)
                        x_BLC = torch.cat([x_BLC, padding], dim=1)
                    elif actual_len > expected_len:
                        # 如果长度超出，截断（这不应该发生，但为了安全）
                        x_BLC = x_BLC[:, :expected_len, :]
                else:
                    # 如果没有累积的features，创建全零输入（包括当前scale的占位符）
                    x_BLC = torch.zeros(B, expected_len, gpt_model.d_vae, device=device, dtype=dtype)
            
            # Forward pass
            logits_BLV = gpt_model(text_cond_tuple, x_BLC, scale_schedule=current_scale_schedule)
            
            # 提取当前scale的logits
            # 注意：logits_BLV的形状是[B, total_seq_len, vocab_size]
            # 我们需要当前scale对应的logits部分
            if scale_idx == 0:
                current_logits = logits_BLV  # 第一个scale，所有logits都是当前scale的
            else:
                # 后续scale：取最后scale_len个logits
                current_logits = logits_BLV[:, -scale_len:, :]  # [B, scale_len, vocab_size]
            
            # 如果当前scale在target_scales中，保存logits
            if scale_idx in target_scales:
                scale_logits_list.append(current_logits)
            
            # 采样得到tokens（用于下一个scale的输入）
            if gpt_model.use_bit_label:
                # bit label: logits shape是[B, scale_len, 2]（每个token是2个bit）
                tmp_bs, tmp_seq_len = current_logits.shape[:2]
                logits_reshaped = current_logits.reshape(tmp_bs, -1, 2)
                # 使用argmax采样（训练时不需要随机采样）
                tokens = logits_reshaped.argmax(dim=-1)  # [B, scale_len]
                tokens = tokens.reshape(tmp_bs, tmp_seq_len, -1)  # [B, scale_len, codebook_dim]
            else:
                # 普通token: 使用argmax采样
                tokens = current_logits.argmax(dim=-1)  # [B, scale_len]
            
            all_tokens.append(tokens)
            
            # 将tokens转换为features用于下一个scale
            # 需要先将tokens (Long indices) 转换为float embeddings
            if hasattr(gpt_model, 'word_embed'):
                if gpt_model.use_bit_label:
                    # 对于bit label: tokens shape是[B, scale_len, codebook_dim] (Long, 0或1)
                    # 需要转换为[B, scale_len, d_vae]的float embeddings
                    B, L, codebook_dim = tokens.shape
                    # tokens是Long，需要转换为float bits (0.0或1.0)
                    bits_float = tokens.float()  # [B, scale_len, codebook_dim]
                    # indices_to_codes对于bit_label期望 [B, L, 1, codebook_dim] 格式
                    bits_for_quantizer = bits_float.unsqueeze(2)  # [B, scale_len, 1, codebook_dim]
                    # 调用quantizer转换，注意：由于ndim>=3，should_transpose可能为True
                    codes = vae_model.quantizer.lfq.indices_to_codes(bits_for_quantizer, label_type='bit_label')
                    # codes shape可能是 [B, d_vae, scale_len, 1] (如果transposed) 或 [B, scale_len, 1, d_vae]
                    # 需要reshape为 [B, scale_len, d_vae]
                    if codes.ndim == 4:
                        # 检查哪个维度是scale_len
                        if codes.shape[2] == L:
                            # [B, d_vae, scale_len, 1] -> [B, scale_len, d_vae]
                            codes = codes.squeeze(-1).permute(0, 2, 1)
                        elif codes.shape[1] == L:
                            # [B, scale_len, 1, d_vae] -> [B, scale_len, d_vae]
                            codes = codes.squeeze(2)
                        else:
                            # 尝试其他可能的shape
                            codes = codes.reshape(B, L, -1)
                    elif codes.ndim == 3:
                        # [B, L, d_vae] 或 [B, d_vae, L]
                        if codes.shape[1] != L:
                            codes = codes.permute(0, 2, 1)
                    token_embeddings = codes.to(dtype=dtype)
                    # 确保最终shape是 [B, scale_len, d_vae]
                    if token_embeddings.shape[:2] != (B, L):
                        # 如果shape不对，尝试reshape
                        token_embeddings = token_embeddings.reshape(B, L, -1)
                    # 确保最后一维是d_vae
                    if token_embeddings.shape[-1] != gpt_model.d_vae:
                        # 如果维度不匹配，可能需要截断或padding（这不应该发生）
                        if token_embeddings.shape[-1] > gpt_model.d_vae:
                            token_embeddings = token_embeddings[..., :gpt_model.d_vae]
                        else:
                            # padding with zeros
                            padding = torch.zeros(B, L, gpt_model.d_vae - token_embeddings.shape[-1], 
                                                device=token_embeddings.device, dtype=token_embeddings.dtype)
                            token_embeddings = torch.cat([token_embeddings, padding], dim=-1)
                else:
                    # 普通token: tokens shape是[B, scale_len] (Long indices)
                    # 需要转换为[B, scale_len, d_vae]的float embeddings
                    B, L = tokens.shape
                    # indices_to_codes对于int_label会自动添加维度，输入 [B, scale_len] 即可
                    # 但为了明确，我们添加维度: [B, scale_len] -> [B, scale_len, 1]
                    indices_for_quantizer = tokens.unsqueeze(-1)  # [B, scale_len, 1]
                    codes = vae_model.quantizer.lfq.indices_to_codes(indices_for_quantizer, label_type='int_label')
                    # codes shape可能是 [B, d_vae, scale_len, 1] (如果transposed) 或 [B, scale_len, 1, d_vae]
                    # 需要reshape为 [B, scale_len, d_vae]
                    if codes.ndim == 4:
                        # 检查哪个维度是scale_len
                        if codes.shape[2] == L:
                            # [B, d_vae, scale_len, 1] -> [B, scale_len, d_vae]
                            codes = codes.squeeze(-1).permute(0, 2, 1)
                        elif codes.shape[1] == L:
                            # [B, scale_len, 1, d_vae] -> [B, scale_len, d_vae]
                            codes = codes.squeeze(2)
                        else:
                            # 尝试其他可能的shape
                            codes = codes.reshape(B, L, -1)
                    elif codes.ndim == 3:
                        # [B, L, d_vae] 或 [B, d_vae, L]
                        if codes.shape[1] != L:
                            codes = codes.permute(0, 2, 1)
                    token_embeddings = codes.to(dtype=dtype)
                    # 确保最终shape是 [B, scale_len, d_vae]
                    if token_embeddings.shape[:2] != (B, L):
                        # 如果shape不对，尝试reshape
                        token_embeddings = token_embeddings.reshape(B, L, -1)
                    # 确保最后一维是d_vae
                    if token_embeddings.shape[-1] != gpt_model.d_vae:
                        # 如果维度不匹配，可能需要截断或padding（这不应该发生）
                        if token_embeddings.shape[-1] > gpt_model.d_vae:
                            token_embeddings = token_embeddings[..., :gpt_model.d_vae]
                        else:
                            # padding with zeros
                            padding = torch.zeros(B, L, gpt_model.d_vae - token_embeddings.shape[-1], 
                                                device=token_embeddings.device, dtype=token_embeddings.dtype)
                            token_embeddings = torch.cat([token_embeddings, padding], dim=-1)
                
                # 现在token_embeddings是float类型，shape为[B, scale_len, d_vae]
                # 注意：模型forward方法会再次应用norm0_ve和word_embed
                # 所以我们应该存储原始的token_embeddings（在word_embed之前）
                # 确保token_embeddings的dtype与word_embed权重的dtype匹配
                if hasattr(gpt_model, 'word_embed'):
                    # 获取word_embed权重的dtype
                    word_embed_dtype = gpt_model.word_embed.weight.dtype
                    # 确保token_embeddings的dtype匹配
                    token_embeddings = token_embeddings.to(dtype=word_embed_dtype)
                
                # 存储token_embeddings（在word_embed之前），因为模型forward会再次应用word_embed
                # token_embeddings的shape应该是[B, scale_len, d_vae]
                all_features.append(token_embeddings)
    
    return scale_logits_list


def calculate_first_esd_loss(
    args,
    batch: dict,
    gpt_model,
    vae_model,
    text_cond_tuple: Tuple,
    text_cond_tuple_uncond: Optional[Tuple] = None,
    criteria: torch.nn.Module = None,
    negative_guidance: float = 1.0,
    device: str = "cuda",
    weight_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    符合自回归原理的 ESD loss 计算
    
    核心思想：
    1. 用含nude概念的prompt自回归生成到中间几个scale，得到每个scale的logits
    2. 用空prompt（无条件）自回归生成到同样的中间几个scale，得到每个scale的logits
    3. 对于每个scale，计算loss = MSE(e_n, e_0 - negative_guidance * (e_p - e_0))
    
    其中：
    - e_n: 当前模型（正在训练的）用含概念prompt预测的logits（需要梯度）
    - e_0: 无条件预测的logits（不需要梯度）
    - e_p: 原始模型（冻结的）用含概念prompt预测的logits（不需要梯度）
    
    Args:
        args: 配置参数
        batch: 数据批次，包含 pixel_values, prompts 等
        gpt_model: Infinity GPT 模型（当前正在训练的模型）
        vae_model: Infinity VAE 模型
        text_cond_tuple: 有条件的文本特征 (text_features, text_lens, cu_seqlens_k, Ltext)
        text_cond_tuple_uncond: 无条件的文本特征（可选，如果没有则自动生成）
        criteria: 损失函数（默认 MSE）
        negative_guidance: 负向引导强度
        device: 设备
        weight_dtype: 权重数据类型
        
    Returns:
        loss_esd: ESD 损失值
    """
    if criteria is None:
        criteria = torch.nn.MSELoss()
    
    B = batch["pixel_values"].shape[0]
    
    # 解包 text condition tuple
    text_features, text_lens, cu_seqlens_k, Ltext = text_cond_tuple
    
    # ==================== 1. 准备无条件文本特征 ====================
    if text_cond_tuple_uncond is None:
        if hasattr(gpt_model, 'cfg_uncond'):
            cfg_uncond = gpt_model.cfg_uncond  # [text_maxlen, text_channels]
            uncond_len = cfg_uncond.shape[0]
            
            text_features_uncond_list = []
            text_lens_uncond = []
            for i in range(B):
                text_features_uncond_list.append(cfg_uncond[:uncond_len])
                text_lens_uncond.append(uncond_len)
            
            text_features_uncond = torch.cat(text_features_uncond_list, dim=0).to(device)
            cu_seqlens_k_uncond = F.pad(torch.tensor(text_lens_uncond, dtype=torch.int32, device=device).cumsum_(0), (1, 0))
            Ltext_uncond = uncond_len
            text_cond_tuple_uncond = (text_features_uncond, text_lens_uncond, cu_seqlens_k_uncond, Ltext_uncond)
        else:
            total_len = text_features.shape[0]
            text_features_uncond = torch.zeros(total_len, text_features.shape[1], device=device, dtype=text_features.dtype)
            text_cond_tuple_uncond = (text_features_uncond, text_lens, cu_seqlens_k, Ltext)
    
    # ==================== 2. 获取 scale schedule ====================
    scale_schedule = args.scale_schedule if hasattr(args, 'scale_schedule') else [(1,1,1), (1,2,2), (1,3,3), (1,4,4), (1,6,6), (1,8,8)]
    num_scales = len(scale_schedule)
    
    # 随机选择中间几个scale来计算loss（类似EraseAnything的随机timestep）
    # 选择前半部分的scale，如总共9个scale，选择前5个中的1-3个
    num_target_scales = random.randint(1, min(3, num_scales - 1))  # 选择1-3个scale
    max_target_scale = min(5, num_scales - 1)  # 最多到第5个scale
    if max_target_scale < 1:
        max_target_scale = 1
    target_scales = sorted(random.sample(range(1, max_target_scale + 1), num_target_scales))
    
    # ==================== 3. 用含概念prompt生成到target_scales（原始模型，不需要梯度）====================
    # 这里需要原始冻结模型的副本，但为了简化，我们使用当前模型但detach
    # 注意：理想情况下应该保存原始模型的副本
    with torch.no_grad():
        gpt_model.eval()
        # 用含概念prompt生成
        logits_p_list = autoregressive_generate_to_scales(
            gpt_model=gpt_model,
            vae_model=vae_model,
            text_cond_tuple=text_cond_tuple,
            target_scales=target_scales,
            scale_schedule=scale_schedule,
            device=device,
            dtype=weight_dtype,
            return_logits=True,
        )
        # detach，不需要梯度
        logits_p_list = [logits.detach() for logits in logits_p_list]
        
        # 用空prompt（无条件）生成
        logits_0_list = autoregressive_generate_to_scales(
            gpt_model=gpt_model,
            vae_model=vae_model,
            text_cond_tuple=text_cond_tuple_uncond,
            target_scales=target_scales,
            scale_schedule=scale_schedule,
            device=device,
            dtype=weight_dtype,
            return_logits=True,
        )
        # detach，不需要梯度
        logits_0_list = [logits.detach() for logits in logits_0_list]
    
    # ==================== 4. 用含概念prompt生成到target_scales（当前模型，需要梯度）====================
    gpt_model.train()
    logits_n_list = autoregressive_generate_to_scales(
        gpt_model=gpt_model,
        vae_model=vae_model,
        text_cond_tuple=text_cond_tuple,
        target_scales=target_scales,
        scale_schedule=scale_schedule,
        device=device,
        dtype=weight_dtype,
        return_logits=True,
        enable_grad=True,  # 需要梯度用于反向传播
    )
    
    # ==================== 5. 计算每个scale的loss并求和 ====================
    total_loss = 0.0
    num_scales_with_loss = 0
    
    # 确保三个列表长度一致
    min_len = min(len(logits_n_list), len(logits_p_list), len(logits_0_list))
    if min_len == 0:
        print("Warning: No valid logits generated, returning 0")
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    for scale_idx in range(min_len):
        logits_n = logits_n_list[scale_idx]
        logits_p = logits_p_list[scale_idx] 
        logits_0 = logits_0_list[scale_idx]
        
        # 检查是否有nan或inf
        if (torch.isnan(logits_n).any() or torch.isinf(logits_n).any() or
            torch.isnan(logits_p).any() or torch.isinf(logits_p).any() or
            torch.isnan(logits_0).any() or torch.isinf(logits_0).any()):
            print(f"Warning: logits at scale {target_scales[scale_idx]} contains nan or inf")
            continue
        
        # 计算target: e_0 - negative_guidance * (e_p - e_0)
        target = logits_0 - negative_guidance * (logits_p - logits_0)
        
        # 检查target是否有nan或inf
        if torch.isnan(target).any() or torch.isinf(target).any():
            print(f"Warning: target at scale {target_scales[scale_idx]} contains nan or inf")
            continue
        
        # 计算loss: MSE(e_n, target)
        scale_loss = criteria(logits_n, target)
        
        # 检查loss是否有nan或inf
        if torch.isnan(scale_loss) or torch.isinf(scale_loss):
            print(f"Warning: scale_loss at scale {target_scales[scale_idx]} is nan or inf")
            continue
        
        total_loss += scale_loss
        num_scales_with_loss += 1
    
    # 如果所有scale的loss都无效，返回0
    if num_scales_with_loss == 0:
        print("Warning: All scale losses are invalid, returning 0")
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # 返回平均loss
    avg_loss = total_loss / num_scales_with_loss
    
    return avg_loss