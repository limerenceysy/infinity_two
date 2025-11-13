# EraseInfinity LoRA 微调使用指南

本指南说明如何使用 LoRA 微调 Infinity 模型，并使用微调后的权重进行推理。

## 📋 目录

1. [准备工作](#准备工作)
2. [启动 LoRA 微调](#启动-lora-微调)
3. [使用微调后的权重进行推理](#使用微调后的权重进行推理)
4. [注意事项](#注意事项)

---

## 准备工作

### 1. 下载模型权重

确保你已经下载了以下权重文件（**原权重不会被修改**）：

```bash
# Infinity 基础模型
weights/infinity_2b_reg.pth

# VAE 模型
weights/infinity_vae_d32reg.pth

# T5 文本编码器（会自动下载到 ~/.cache/huggingface）
# 或者手动下载到 weights/flan-t5-xl/
```

### 2. 准备训练数据

编辑 `EraseInfinity/config/erase_nude.yaml`，设置：

```yaml
instance_data_dir: "/path/to/your/training/images"
instance_prompt: "your prompt here"
key_word: "your keyword"
```

### 3. 配置训练参数

编辑 `EraseInfinity/config/erase_nude.yaml`，确保路径正确：

```yaml
vae_ckpt: "weights/infinity_vae_d32reg.pth"  # 修改为实际路径
gpt_ckpt: "weights/infinity_2b_reg.pth"      # 修改为实际路径
t5_path: "google/flan-t5-xl"                  # 或 "weights/flan-t5-xl"

# LoRA 配置（只针对 CrossAttention.proj 层）
use_lora: true
lora_rank: 8
lora_alpha: 8
lora_dropout: 0.0

# 训练配置
resolution: 256
pn: "0.06M"  # 对应 256x256 分辨率
train_batch_size: 1
num_train_epochs: 1
learning_rate: 1e-3

# ESD Loss 配置
negative_guidance: 1.0

# 输出目录
output_dir: "EraseInfinity/outputs/erase_nude"
```

---

## 启动 LoRA 微调

### 方法 1: 使用训练脚本（推荐）

```bash
cd /home/yangsiya/Infinity-main/EraseInfinity

# 使用默认配置
bash train.sh

# 或指定配置文件
python train_erase.py --config config/erase_nude.yaml
```

### 方法 2: 直接运行 Python 脚本

```bash
cd /home/yangsiya/Infinity-main/EraseInfinity

python train_erase.py \
    --config config/erase_nude.yaml \
    --local_rank 0
```

### 训练过程

训练过程中会：
1. **加载基础模型**（原权重不会被修改）
2. **添加 LoRA 适配器**到 CrossAttention.proj 层
3. **使用第一个 ESD loss**进行训练
4. **定期保存 LoRA 权重**到输出目录

### 训练输出

训练完成后，你会在输出目录找到：

```
EraseInfinity/outputs/erase_nude/
├── lora_final/                    # 最终 LoRA 权重（PEFT 格式）
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── ...
├── checkpoint_epoch_1.pth         # 训练 checkpoint（如果不用 LoRA）
└── training.log                   # 训练日志
```

**重要**：原权重文件 `weights/infinity_2b_reg.pth` **不会被修改**！

---

## 使用微调后的权重进行推理

### 方法 1: 使用专用推理脚本（推荐）

```bash
cd /home/yangsiya/Infinity-main/EraseInfinity

python inference_with_lora.py \
    --model_path ../weights/infinity_2b_reg.pth \
    --lora_path outputs/erase_nude/lora_final \
    --vae_path ../weights/infinity_vae_d32reg.pth \
    --vae_type 32 \
    --text_encoder_ckpt google/flan-t5-xl \
    --model_type infinity_2b \
    --pn 0.06M \
    --prompt "a beautiful landscape" \
    --cfg 3 \
    --tau 0.5 \
    --seed 42 \
    --save_file output_with_lora.jpg
```

### 方法 2: 修改现有推理脚本

你也可以修改 `tools/run_infinity.py` 或创建新的推理脚本来加载 LoRA 权重。

**示例代码**（在加载模型后）：

```python
from peft import PeftModel

# 加载基础模型
infinity = load_infinity(...)

# 加载 LoRA 权重
infinity = PeftModel.from_pretrained(
    infinity,
    "EraseInfinity/outputs/erase_nude/lora_final",
    device_map="cuda",
)

# 继续正常推理
generated_image = gen_one_img(...)
```

---

## 完整示例

### 1. 训练 LoRA

```bash
# 编辑配置文件
vim EraseInfinity/config/erase_nude.yaml

# 启动训练
cd EraseInfinity
bash train.sh
```

训练输出示例：
```
Adding LoRA adapters to GPT (CrossAttention.proj only)...
LoRA target modules: ['proj']
Found 1 unique module names
LoRA adapters added successfully
trainable params: 2.05M || all params: 2048.00M || trainable%: 0.10

Verifying LoRA target modules:
  ✓ unregistered_blocks.0.ca.proj has LoRA
  ✓ unregistered_blocks.1.ca.proj has LoRA
  ...
```

### 2. 使用 LoRA 推理

```bash
# 使用 LoRA 权重生成图像
python EraseInfinity/inference_with_lora.py \
    --model_path weights/infinity_2b_reg.pth \
    --lora_path EraseInfinity/outputs/erase_nude/lora_final \
    --vae_path weights/infinity_vae_d32reg.pth \
    --vae_type 32 \
    --text_encoder_ckpt google/flan-t5-xl \
    --model_type infinity_2b \
    --pn 0.06M \
    --prompt "nude person" \
    --cfg 3 \
    --save_file test_lora.jpg
```

### 3. 对比原模型和 LoRA 模型

```bash
# 使用原模型生成
python tools/run_infinity.py \
    --model_path weights/infinity_2b_reg.pth \
    --vae_path weights/infinity_vae_d32reg.pth \
    --prompt "nude person" \
    --save_file output_original.jpg

# 使用 LoRA 模型生成
python EraseInfinity/inference_with_lora.py \
    --model_path weights/infinity_2b_reg.pth \
    --lora_path EraseInfinity/outputs/erase_nude/lora_final \
    --vae_path weights/infinity_vae_d32reg.pth \
    --prompt "nude person" \
    --save_file output_lora.jpg

# 对比两张图片观察效果
```

---

## 注意事项

### 1. 原权重保护

- ✅ **原权重文件不会被修改**：所有训练都只修改 LoRA 权重
- ✅ **LoRA 权重独立保存**：保存在 `output_dir/lora_final/`
- ✅ **可以随时切换**：可以选择使用原模型或 LoRA 模型

### 2. 分辨率配置

确保训练和推理时使用相同的 `--pn` 参数：

```yaml
pn: "0.06M"  # 256x256 分辨率
pn: "0.25M"  # 512x512 分辨率  
pn: "1M"     # 1024x1024 分辨率
```

### 3. LoRA 目标层

当前实现**只对 CrossAttention.proj 层进行 LoRA 微调**，这是为了：
- 减少可训练参数（约 0.1%）
- 保持模型其他部分不变
- 专注于跨模态注意力机制

如果需要修改目标层，编辑 `EraseInfinity/train_erase.py` 中的 `target_modules` 配置。

### 4. 模型兼容性

- LoRA 权重必须与基础模型版本匹配
- 确保使用相同的 `model_type`（如 `infinity_2b`）
- 确保使用相同的 `pn` 配置

### 5. 内存优化

如果遇到内存不足：
- 减小 `train_batch_size`
- 启用 `gradient_checkpointing`
- 使用 `mixed_precision: "bf16"`

---

## 故障排除

### 问题 1: LoRA 加载失败

```
错误: Failed to load LoRA from directory
```

**解决方案**：
- 检查 LoRA 路径是否正确
- 确保 `adapter_model.bin` 或 `adapter_config.json` 存在
- 尝试使用完整路径而非相对路径

### 问题 2: 模型不匹配

```
RuntimeError: Error(s) in loading state_dict
```

**解决方案**：
- 确保使用相同的基础模型权重
- 确保使用相同的 `model_type`
- 检查 LoRA 是否与基础模型兼容

### 问题 3: 推理结果异常

**检查**：
- 训练时和推理时使用的 `pn` 是否一致
- `cfg` 和 `tau` 参数是否合理
- LoRA 权重是否正确加载（查看日志中的 "✓ LoRA weights loaded"）

---

## 快速开始

```bash
# 1. 配置训练参数
vim EraseInfinity/config/erase_nude.yaml

# 2. 启动训练
cd EraseInfinity
bash train.sh

# 3. 等待训练完成（查看 outputs/erase_nude/training.log）

# 4. 使用 LoRA 推理
python inference_with_lora.py \
    --model_path ../weights/infinity_2b_reg.pth \
    --lora_path outputs/erase_nude/lora_final \
    --vae_path ../weights/infinity_vae_d32reg.pth \
    --vae_type 32 \
    --text_encoder_ckpt google/flan-t5-xl \
    --model_type infinity_2b \
    --pn 0.06M \
    --prompt "your prompt here" \
    --save_file output.jpg
```

---

## 技术细节

### LoRA 配置

当前 LoRA 配置：
- **Rank**: 8
- **Alpha**: 8
- **Dropout**: 0.0
- **Target**: CrossAttention.proj 层
- **可训练参数**: ~2M（约 0.1%）

### ESD Loss

使用第一个 ESD loss：
```
loss_esd = MSE(e_n, e_0 - negative_guidance * (e_p - e_0))
```

其中：
- `e_n`: 当前模型的预测（需要梯度）
- `e_0`: 无条件预测（空文本）
- `e_p`: 原始模型的有条件预测（冻结）
- `negative_guidance`: 负向引导强度（默认 1.0）

---

## 参考

- [Infinity 官方 README](../README.md) - 了解基础模型
- [EraseAnything](../EraseAnything/) - 原始擦除方法实现
- [PEFT 文档](https://huggingface.co/docs/peft/) - LoRA 技术细节

