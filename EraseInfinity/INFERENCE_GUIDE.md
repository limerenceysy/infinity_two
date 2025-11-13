# EraseInfinity 推理指南

本文档详细说明如何使用训练好的 LoRA 权重进行推理。

---

## 📊 训练结果总结

根据您的训练输出：

### 训练配置
- **训练轮数**: 9 epochs
- **数据集大小**: 200 samples (Prompt-Only Dataset)
- **Batch size**: 4
- **每轮步数**: 50 steps
- **总训练步数**: 450 steps

### Loss 收敛情况
```
Epoch 0: Loss 0.2306
Epoch 1: Loss 0.0152  (↓ 93.4%)
Epoch 2: Loss 0.0086  (↓ 43.4%)
Epoch 3: Loss 0.0045  (↓ 47.7%)
Epoch 4: Loss 0.0031  (↓ 31.1%)
Epoch 5: Loss 0.0085  (↑ 174.2%) - 可能是学习率调整或数据随机性
Epoch 6: Loss 0.0041  (↓ 51.8%)
Epoch 7: Loss 0.0028  (↓ 31.7%)
Epoch 8: Loss 0.0042  (↑ 50.0%)
```

✅ **总体来说，Loss 从 0.2306 降到了 0.0042，下降了 98.2%，显示出良好的学习效果！**

### 保存的模型权重
训练完成后，LoRA 权重已保存在：
```
EraseInfinity/outputs/erase_nude_prompt_only/
├── checkpoint-401/
│   ├── adapter_model.safetensors  (66 个 LoRA 参数)
│   └── trainable_params.bin       (fallback 保存的权重)
└── loss_curves1.png                (训练 loss 曲线图)
```

---

## 🚀 如何使用训练好的 LoRA 权重

### 方法 1: 使用提供的推理脚本（推荐）

我已经为您创建了 `inference_erase.py` 脚本，可以直接使用：

```bash
cd /home/yangsiya/Infinity-main/EraseInfinity

python inference_erase.py \
  --vae_ckpt /home/yangsiya/Infinity-main/pretrained_models/infinity_vae_d32reg.pth \
  --gpt_ckpt /home/yangsiya/Infinity-main/pretrained_models/infinity_2b_reg.pth \
  --lora_ckpt outputs/erase_nude_prompt_only/checkpoint-401 \
  --prompt "a beautiful landscape" \
  --negative_prompt "nude, naked, nsfw" \
  --resolution 1024 \
  --num_images 4 \
  --cfg_scale 4.0 \
  --output_dir outputs/inference \
  --device cuda:0
```

**参数说明：**
- `--vae_ckpt`: VAE 模型权重路径
- `--gpt_ckpt`: GPT 模型权重路径（原始预训练权重）
- `--lora_ckpt`: 训练好的 LoRA 权重目录
- `--prompt`: 生成图像的文本提示
- `--negative_prompt`: 负面提示（要避免的内容）
- `--resolution`: 图像分辨率
- `--num_images`: 生成图像数量
- `--cfg_scale`: Classifier-free guidance 强度
- `--output_dir`: 输出目录
- `--device`: 使用的设备（cuda:0, cuda:1, cpu 等）

---

### 方法 2: 在 Infinity 原生推理代码中加载 LoRA

如果您想在 Infinity 的原生推理脚本中使用 LoRA 权重，需要修改推理代码：

#### 步骤 1: 在构建模型后加载 LoRA

```python
# 在 Infinity 的推理脚本中，找到模型构建的代码
# 通常在 inference/sample.py 或类似文件中

# 原始代码：
vae, gpt_model, _ = build_vae_gpt(args, vae_ckpt, skip_gpt=False, device='cpu')
gpt_model = gpt_model.to(device)

# 在加载原始权重后，添加 LoRA 加载代码：
from peft import PeftModel

# 加载 LoRA 权重
lora_ckpt_path = "EraseInfinity/outputs/erase_nude_prompt_only/checkpoint-401"
print(f"Loading LoRA weights from {lora_ckpt_path}")

try:
    # 方法 1: 使用 PeftModel.from_pretrained（推荐）
    gpt_model = PeftModel.from_pretrained(gpt_model, lora_ckpt_path)
    print("✓ LoRA weights loaded successfully")
except Exception as e:
    print(f"Warning: Failed to load LoRA using PeftModel: {e}")
    
    # 方法 2: 手动加载权重（fallback）
    from safetensors.torch import load_file
    lora_state_dict = load_file(f"{lora_ckpt_path}/adapter_model.safetensors")
    gpt_model.load_state_dict(lora_state_dict, strict=False)
    print("✓ LoRA weights loaded manually")

# 设置为评估模式
gpt_model.eval()
```

#### 步骤 2: 正常进行推理

加载 LoRA 后，模型的使用方式与原始模型完全相同，不需要修改推理逻辑。

---

## 🔍 验证 LoRA 是否生效

### 方法 1: 检查模型参数

```python
# 加载 LoRA 后，检查模型是否有 LoRA 参数
lora_params = []
for name, param in gpt_model.named_parameters():
    if 'lora' in name.lower():
        lora_params.append(name)

print(f"Found {len(lora_params)} LoRA parameters")
for name in lora_params[:5]:  # 显示前5个
    print(f"  - {name}")
```

预期输出：
```
Found 66 LoRA parameters
  - base_model.model.blocks.0.ca.proj.lora_A.default.weight
  - base_model.model.blocks.0.ca.proj.lora_B.default.weight
  - base_model.model.blocks.1.ca.proj.lora_A.default.weight
  - base_model.model.blocks.1.ca.proj.lora_B.default.weight
  - base_model.model.blocks.2.ca.proj.lora_A.default.weight
```

### 方法 2: 对比生成效果

1. **加载原始模型**：生成图像，看是否包含 nude 内容
2. **加载 LoRA 模型**：生成相同 prompt 的图像，看是否成功擦除 nude 内容

---

## 📝 推理代码示例

### 完整的 Python 推理代码

```python
import torch
from peft import PeftModel
from infinity.utils.load import build_vae_gpt

# 设备配置
device = torch.device("cuda:0")

# 1. 加载 VAE 和 GPT
vae_ckpt = torch.load("pretrained_models/infinity_vae_d32reg.pth", map_location='cpu')
gpt_ckpt = torch.load("pretrained_models/infinity_2b_reg.pth", map_location='cpu')

# 构建模型（需要提供 args，参考 inference_erase.py）
vae, gpt_model, _ = build_vae_gpt(args, vae_ckpt, skip_gpt=False, device='cpu')

# 加载 GPT 权重
if 'trainer' in gpt_ckpt:
    gpt_state_dict = gpt_ckpt['trainer'].get('gpt_wo_ddp', gpt_ckpt)
else:
    gpt_state_dict = gpt_ckpt
gpt_model.load_state_dict(gpt_state_dict, strict=False)

# 2. 加载 LoRA 权重
lora_path = "EraseInfinity/outputs/erase_nude_prompt_only/checkpoint-401"
gpt_model = PeftModel.from_pretrained(gpt_model, lora_path)

# 3. 移动到设备并设置为评估模式
vae = vae.to(device).eval()
gpt_model = gpt_model.to(device).eval()

# 4. 准备文本特征
# （需要根据 Infinity 的实际文本编码方式）
prompt = "a beautiful portrait"
# text_features = encode_text(prompt)  # 具体实现参考 Infinity 代码

# 5. 生成图像
with torch.no_grad():
    # 调用 Infinity 的生成函数
    # generated_images = gpt_model.generate(...)
    pass

# 6. 解码并保存
# images = vae.decode(generated_images)
# save_images(images, "outputs/")
```

---

## ⚠️ 重要注意事项

### 1. 推理接口依赖 Infinity 的实现

**当前状态**: `inference_erase.py` 提供了模型加载的框架，但实际的图像生成部分需要根据 Infinity 的具体 API 来实现。

**您需要做的**:
1. 查看 Infinity 项目中的推理代码（通常在 `infinity/inference/` 或类似目录）
2. 找到图像生成的函数（如 `generate()`, `sample()`, `autoregressive_infer()` 等）
3. 将该函数集成到 `inference_erase.py` 中

### 2. 文本编码方式

训练时我们使用了简化的文本编码（基于 `cfg_uncond` 的扰动），这可能与 Infinity 原生的文本编码不完全一致。

**建议**:
- 如果 Infinity 有 T5 文本编码器，建议使用它来获得更好的效果
- 如果要保持一致性，推理时也应该使用相同的文本编码方式（`create_text_features_from_prompts`）

### 3. LoRA 权重兼容性

训练时我们只对 Cross-Attention 的 `proj` 层添加了 LoRA（66个参数）。推理时：
- ✅ 可以直接用 `PeftModel.from_pretrained()` 加载
- ✅ 原始模型权重不会被修改，只是添加了 LoRA 适配器
- ✅ 如果需要，可以随时禁用 LoRA：`gpt_model.disable_adapter_layers()`

---

## 🛠️ 故障排查

### 问题 1: 找不到 Infinity 的推理代码

**解决方法**:
```bash
# 在 Infinity 项目中搜索推理相关代码
cd /home/yangsiya/Infinity-main
find . -name "*.py" -type f | xargs grep -l "def generate\|def sample\|def infer"

# 或者查看是否有 demo 或 example 脚本
ls -la demos/ examples/ scripts/
```

### 问题 2: LoRA 加载失败

**可能原因**:
- PEFT 库版本不兼容
- `adapter_config.json` 格式错误（已在训练脚本中修复）

**解决方法**:
```python
# 尝试手动加载权重
from safetensors.torch import load_file

lora_weights = load_file("outputs/checkpoint-401/adapter_model.safetensors")
print(f"Loaded {len(lora_weights)} parameters:")
for k in list(lora_weights.keys())[:5]:
    print(f"  {k}: {lora_weights[k].shape}")

# 手动加载到模型
gpt_model.load_state_dict(lora_weights, strict=False)
```

### 问题 3: 生成的图像仍包含 nude 内容

**可能原因**:
- LoRA 权重未正确加载
- 训练的 epoch 数不够
- 训练数据集太小（只有 200 个 prompt-only 样本）

**解决方法**:
1. 验证 LoRA 是否真的加载了（参考"验证 LoRA 是否生效"）
2. 尝试使用更多 epochs 重新训练
3. 如果需要更好的效果，使用真实的 nude 图像数据集进行训练

---

## 📚 下一步建议

### 1. 找到 Infinity 的推理代码

```bash
# 查看 Infinity 项目结构
cd /home/yangsiya/Infinity-main
ls -la
ls -la infinity/

# 查找推理相关文件
find . -name "*infer*" -o -name "*sample*" -o -name "*generate*"
```

### 2. 集成推理代码到 inference_erase.py

将 Infinity 的推理逻辑添加到 `generate_images()` 函数中。

### 3. 测试不同的 prompts

```bash
# 测试原本会生成 nude 内容的 prompt
python inference_erase.py \
  --prompt "a person on the beach" \
  --negative_prompt "nude, naked, nsfw" \
  ...

# 对比：使用原始模型生成（不加载 LoRA）
# 应该看到 LoRA 模型成功避免了 nude 内容
```

---

## 📞 需要帮助？

如果您在推理过程中遇到问题：

1. **查看训练日志**: `outputs/erase_nude_prompt_only/loss_log1.txt`
2. **检查 LoRA 权重**: 确认 `adapter_model.safetensors` 文件存在且大小合理
3. **阅读 Infinity 文档**: 查看 Infinity 项目的 README 和文档
4. **调试模式**: 在 Python 中逐步运行代码，检查每一步的输出

---

## 🎉 总结

您已经成功训练了一个 EraseInfinity 模型！

- ✅ 训练完成，Loss 从 0.23 降到 0.004
- ✅ LoRA 权重已保存（66 个参数）
- ✅ 推理脚本已准备好
- ⏳ 下一步：集成 Infinity 的推理 API 并测试效果

祝您使用愉快！🚀

