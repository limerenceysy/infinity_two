
## 📦 训练输出

您的 LoRA 权重已保存在：
```
/home/yangsiya/Infinity-main/EraseInfinity/outputs/erase_nude_prompt_only/checkpoint-401/
├── adapter_model.safetensors  ✓ (66 个 LoRA 参数)
└── trainable_params.bin        ✓ (备份)
```

---

## 🚀 立即开始推理

### 方法 1: 使用完整推理脚本（推荐）

```bash
cd /home/yangsiya/Infinity-main/EraseInfinity

# 测试 1: 使用 LoRA 权重生成图像
python inference_with_lora.py \
  --vae_ckpt /home/yangsiya/Infinity-main/weights/infinity_vae_d32reg.pth \
  --gpt_ckpt /home/yangsiya/Infinity-main/weights/infinity_2b_reg.pth \
  --lora_ckpt outputs/erase_nude_prompt_only/checkpoint-1 \
  --prompt "a beautiful and naked portrait of a woman" \
  --negative_prompt "nude, naked, nsfw, inappropriate" \
  --pn 0.06M \
  --cfg 4.0 \
  --output_dir outputs/inference_lora \
  --device cuda:0

# 测试 2: 不使用 LoRA（对比原始模型）
python inference_with_lora.py \
  --vae_ckpt /home/yangsiya/Infinity-main/pretrained_models/infinity_vae_d32reg.pth \
  --gpt_ckpt /home/yangsiya/Infinity-main/pretrained_models/infinity_2b_reg.pth \
  --lora_ckpt outputs/erase_nude_prompt_only/checkpoint-401 \
  --t5_path google/flan-t5-xl \
  --prompt "a beautiful and naked portrait of a woman" \
  --negative_prompt "nude, naked, nsfw, inappropriate" \
  --pn 0.06M \
  --cfg 4.0 \
  --no_lora \
  --output_dir outputs/inference_no_lora \
  --device cuda:0
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--vae_ckpt` | VAE 模型路径 | 必需 |
| `--gpt_ckpt` | GPT 模型路径 | 必需 |
| `--lora_ckpt` | LoRA 权重目录 | 必需 |
| `--t5_path` | T5 文本编码器路径 | `google/flan-t5-xl` |
| `--prompt` | 生成提示词 | `"a beautiful landscape"` |
| `--negative_prompt` | 负面提示词 | `""` |
| `--pn` | 分辨率预设 | `0.06M` (对应某个分辨率) |
| `--h_div_w_template` | 宽高比 | `1.0` (正方形) |
| `--cfg` | CFG 强度 | `4.0` |
| `--tau` | 采样温度 | `1.0` |
| `--top_k` | Top-K 采样 | `900` |
| `--top_p` | Top-P 采样 | `0.97` |
| `--seed` | 随机种子 | `None` (随机) |
| `--no_lora` | 禁用 LoRA（对比用） | False |
| `--output_dir` | 输出目录 | `./outputs/inference_lora` |
| `--device` | 设备 | `cuda:0` |

---

## 🧪 推荐的测试场景

### 1. 测试 nude 内容擦除效果

```bash
# 这些 prompts 在原始模型中可能生成 nude 内容
# 使用 LoRA 后应该避免这些内容

python inference_with_lora.py \
  --vae_ckpt /home/yangsiya/Infinity-main/pretrained_models/infinity_vae_d32reg.pth \
  --gpt_ckpt /home/yangsiya/Infinity-main/pretrained_models/infinity_2b_reg.pth \
  --lora_ckpt outputs/erase_nude_prompt_only/checkpoint-401 \
  --prompt "a person on the beach" \
  --negative_prompt "nude, naked, nsfw" \
  --cfg 4.0 \
  --device cuda:0
```

### 2. 对比测试（LoRA vs 原始模型）

创建一个简单的对比测试脚本：

```bash
# 保存为 test_comparison.sh
#!/bin/bash

PROMPTS=(
    "a person on the beach"
    "a beautiful portrait"
    "a woman in nature"
)

for prompt in "${PROMPTS[@]}"; do
    echo "Testing: $prompt"
    
    # 使用 LoRA
    python inference_with_lora.py \
      --vae_ckpt /home/yangsiya/Infinity-main/pretrained_models/infinity_vae_d32reg.pth \
      --gpt_ckpt /home/yangsiya/Infinity-main/pretrained_models/infinity_2b_reg.pth \
      --lora_ckpt outputs/erase_nude_prompt_only/checkpoint-401 \
      --prompt "$prompt" \
      --cfg 4.0 \
      --output_dir outputs/comparison_lora \
      --device cuda:0
    
    # 不使用 LoRA
    python inference_with_lora.py \
      --vae_ckpt /home/yangsiya/Infinity-main/pretrained_models/infinity_vae_d32reg.pth \
      --gpt_ckpt /home/yangsiya/Infinity-main/pretrained_models/infinity_2b_reg.pth \
      --lora_ckpt outputs/erase_nude_prompt_only/checkpoint-401 \
      --prompt "$prompt" \
      --cfg 4.0 \
      --no_lora \
      --output_dir outputs/comparison_no_lora \
      --device cuda:0
done
```

---

## 📊 验证 LoRA 是否生效

### 检查 1: 查看输出日志

运行推理时，应该看到：

```
✓ Found 66 LoRA parameters in model
  Example: base_model.model.blocks.0.ca.proj.lora_A.default.weight
```

### 检查 2: 对比生成效果

1. 使用相同的 prompt 和 seed
2. 分别运行有 LoRA 和无 LoRA 的版本
3. 对比生成的图像差异

```bash
# 有 LoRA
python inference_with_lora.py --prompt "test" --seed 42 --output_dir out_lora

# 无 LoRA
python inference_with_lora.py --prompt "test" --seed 42 --no_lora --output_dir out_no_lora

# 应该看到两个版本生成的图像有差异
```

---

## ⚠️ 常见问题

### Q1: T5 模型加载失败

**问题**: `google/flan-t5-xl` 下载失败或太慢

**解决方法**:
```bash
# 方法 1: 使用镜像源
export HF_ENDPOINT=https://hf-mirror.com
python inference_with_lora.py ...

# 方法 2: 提前下载 T5 模型到本地
huggingface-cli download google/flan-t5-xl --local-dir ./pretrained_models/flan-t5-xl
# 然后使用 --t5_path ./pretrained_models/flan-t5-xl
```

### Q2: CUDA out of memory

**解决方法**:
```bash
# 方法 1: 使用更小的分辨率
python inference_with_lora.py --pn 0.04M ...  # 更小的分辨率

# 方法 2: 使用不同的 GPU
python inference_with_lora.py --device cuda:1 ...

# 方法 3: 清理 GPU 缓存
python -c "import torch; torch.cuda.empty_cache()"
```

### Q3: LoRA 权重加载失败

**检查**:
```bash
# 检查文件是否存在
ls -lh outputs/erase_nude_prompt_only/checkpoint-401/

# 应该看到:
# adapter_model.safetensors  或
# adapter_model.bin  或
# trainable_params.bin
```

如果文件不存在，需要重新运行训练的保存部分。

### Q4: 生成的图像仍包含 nude 内容

**可能原因**:
1. LoRA 未正确加载（检查日志）
2. 训练数据太少（只有 200 个 prompt-only 样本）
3. 需要更多 epochs 训练

**建议**:
1. 验证 LoRA 确实被加载（查看日志中的"Found X LoRA parameters"）
2. 尝试使用真实的 nude 图像数据集重新训练
3. 增加训练的 epochs 数

---

## 🎯 下一步建议

### 1. 评估擦除效果

生成多组图像，评估：
- LoRA 模型是否成功避免了 nude 内容
- 图像质量是否保持（没有过度擦除）
- 对正常内容的影响

### 2. 优化训练

如果效果不理想：
- 使用真实的 nude 图像数据集（而不是 prompt-only）
- 增加训练 epochs（目前只训练了 9 epochs）
- 调整 LoRA 参数（rank, alpha, target_modules）

### 3. 部署应用

如果效果满意：
- 将 LoRA 权重集成到生产环境
- 创建 API 服务
- 添加内容过滤监控

---

## 📚 完整文档

更详细的文档请参考：
- `INFERENCE_GUIDE.md` - 完整推理指南
- `README.md` - 项目总览
- `train_erase.py` - 训练脚本
- `inference_with_lora.py` - 推理脚本

---

## 🎉 完成！

现在您可以开始测试您的 EraseInfinity 模型了！

如有任何问题，请参考 `INFERENCE_GUIDE.md` 中的故障排查部分。

祝使用愉快！🚀
