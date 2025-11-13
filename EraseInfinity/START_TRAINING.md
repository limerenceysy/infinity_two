# EraseInfinity LoRA 微调启动指南 - 擦除 Nudity 概念

本指南将帮助你启动 LoRA 微调，使 Infinity 模型无法生成 nudity 相关内容。

---

## 🎯 目标

通过 LoRA 微调，使 Infinity 模型：
- ✅ **无法生成 nudity 相关内容**，即使提示词明确包含 "nude"、"naked" 等词
- ✅ **无法通过同义词生成**，如 "naked"、"exposed"、"bare" 等
- ✅ **保持其他生成能力**，只擦除 nudity 概念

---

## 📋 步骤 1: 准备训练数据

### 1.1 创建数据目录

```bash
mkdir -p EraseInfinity/data/nude_images
```

### 1.2 准备训练图像

将包含 nudity 内容的图像放入 `EraseInfinity/data/nude_images/` 目录。

**要求**：
- 图像格式：`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`
- 图像数量：建议至少 10-50 张（越多越好）
- 图像内容：包含需要擦除的 nudity 概念

**示例**：
```bash
EraseInfinity/data/nude_images/
├── image1.jpg
├── image2.png
├── image3.jpeg
└── ...
```

---

## 📋 步骤 2: 配置训练参数

### 2.1 编辑配置文件

编辑 `EraseInfinity/config/erase_nude.yaml`：

```yaml
# ==================== 模型配置 ====================
vae_ckpt: "weights/infinity_vae_d32reg.pth"  # ⚠️ 修改为实际路径
gpt_ckpt: "weights/infinity_2b_reg.pth"      # ⚠️ 修改为实际路径
t5_path: "google/flan-t5-xl"                  # 或 "weights/flan-t5-xl"

# ==================== 数据配置 ====================
instance_data_dir: "EraseInfinity/data/nude_images"  # ⚠️ 修改为实际路径
instance_prompt: "nude person, naked body, explicit content, nudity"
key_word: "nude"  # 要擦除的关键词

# ==================== 训练配置 ====================
resolution: 256
pn: "0.06M"      # 256x256 分辨率
train_batch_size: 1
num_train_epochs: 1
max_train_steps: 200
learning_rate: 1e-3

# ==================== ESD Loss 配置 ====================
negative_guidance: 1.0  # ESD 负向引导强度（越大擦除效果越强）

# ==================== 输出配置 ====================
output_dir: "EraseInfinity/outputs/erase_nude"  # ⚠️ 修改为实际路径
```

### 2.2 重要参数说明

- **`instance_prompt`**: 描述训练图像内容的 prompt，应包含要擦除的概念
- **`key_word`**: 要擦除的核心关键词（如 "nude"）
- **`negative_guidance`**: 
  - `1.0`: 标准擦除强度
  - `1.5-2.0`: 更强的擦除效果（可能影响其他生成能力）
  - `0.5-0.8`: 较弱的擦除效果
- **`learning_rate`**: 
  - `1e-3`: 标准学习率
  - `5e-4`: 更保守（推荐用于首次训练）
  - `2e-3`: 更激进（可能不稳定）

---

## 📋 步骤 3: 启动训练

### 方法 1: 使用快速启动脚本（推荐）

```bash
cd /home/yangsiya/Infinity-main/EraseInfinity
bash quick_start.sh train
```

### 方法 2: 使用训练脚本

```bash
cd /home/yangsiya/Infinity-main/EraseInfinity
bash train.sh
```

### 方法 3: 直接运行 Python

```bash
cd /home/yangsiya/Infinity-main/EraseInfinity

python train_erase.py \
    --config config/erase_nude.yaml \
    --local_rank 0
```

---

## 📋 步骤 4: 监控训练过程

### 4.1 查看训练日志

```bash
# 实时查看日志
tail -f EraseInfinity/outputs/erase_nude/training.log

# 或查看完整日志
cat EraseInfinity/outputs/erase_nude/training.log
```

### 4.2 检查训练输出

训练过程中会显示：
- LoRA 适配器添加成功
- 可训练参数数量（约 0.1%）
- 每个 epoch 的平均 loss
- LoRA 权重保存路径

**预期输出示例**：
```
Adding LoRA adapters to GPT (CrossAttention.proj only)...
Found 32 CrossAttention.proj layers
  Example: unregistered_blocks.0.ca.proj
Using 'proj' to match all proj layers (only ca.proj found)
LoRA adapters added successfully
trainable params: 2.05M || all params: 2048.00M || trainable%: 0.10

Epoch 0: Average loss: 0.1234
Checkpoint saved to EraseInfinity/outputs/erase_nude/lora_final/
```

---

## 📋 步骤 5: 验证擦除效果

### 5.1 使用 LoRA 权重生成图像

```bash
cd /home/yangsiya/Infinity-main/EraseInfinity

# 测试直接包含 "nude" 的提示词
python inference_with_lora.py \
    --model_path ../weights/infinity_2b_reg.pth \
    --lora_path outputs/erase_nude/lora_final \
    --vae_path ../weights/infinity_vae_d32reg.pth \
    --vae_type 32 \
    --text_encoder_ckpt google/flan-t5-xl \
    --model_type infinity_2b \
    --pn 0.06M \
    --prompt "nude person, naked body" \
    --cfg 3 \
    --save_file test_nude_direct.jpg

# 测试同义词（如 "naked"）
python inference_with_lora.py \
    --model_path ../weights/infinity_2b_reg.pth \
    --lora_path outputs/erase_nude/lora_final \
    --vae_path ../weights/infinity_vae_d32reg.pth \
    --vae_type 32 \
    --text_encoder_ckpt google/flan-t5-xl \
    --model_type infinity_2b \
    --pn 0.06M \
    --prompt "naked person, exposed body" \
    --cfg 3 \
    --save_file test_nude_synonym.jpg

# 测试正常提示词（应该不受影响）
python inference_with_lora.py \
    --model_path ../weights/infinity_2b_reg.pth \
    --lora_path outputs/erase_nude/lora_final \
    --vae_path ../weights/infinity_vae_d32reg.pth \
    --vae_type 32 \
    --text_encoder_ckpt google/flan-t5-xl \
    --model_type infinity_2b \
    --pn 0.06M \
    --prompt "a beautiful landscape with mountains" \
    --cfg 3 \
    --save_file test_normal.jpg
```

### 5.2 对比原模型和 LoRA 模型

```bash
# 使用原模型生成（应该能生成 nudity 内容）
python ../tools/run_infinity.py \
    --model_path ../weights/infinity_2b_reg.pth \
    --vae_path ../weights/infinity_vae_d32reg.pth \
    --vae_type 32 \
    --text_encoder_ckpt google/flan-t5-xl \
    --model_type infinity_2b \
    --pn 0.06M \
    --prompt "nude person" \
    --save_file output_original.jpg

# 使用 LoRA 模型生成（应该无法生成 nudity 内容）
python inference_with_lora.py \
    --model_path ../weights/infinity_2b_reg.pth \
    --lora_path outputs/erase_nude/lora_final \
    --vae_path ../weights/infinity_vae_d32reg.pth \
    --vae_type 32 \
    --text_encoder_ckpt google/flan-t5-xl \
    --model_type infinity_2b \
    --pn 0.06M \
    --prompt "nude person" \
    --save_file output_lora.jpg
```

**预期效果**：
- ✅ 原模型：能生成 nudity 相关内容
- ✅ LoRA 模型：无法生成 nudity 相关内容（可能生成 clothed 版本或其他内容）
- ✅ 正常提示词：两个模型都能正常生成

---

## 🔧 调优建议

### 如果擦除效果不够强

1. **增加 `negative_guidance`**：
   ```yaml
   negative_guidance: 1.5  # 从 1.0 增加到 1.5
   ```

2. **增加训练步数**：
   ```yaml
   num_train_epochs: 2
   max_train_steps: 500
   ```

3. **增加学习率**（谨慎）：
   ```yaml
   learning_rate: 2e-3  # 从 1e-3 增加到 2e-3
   ```

### 如果影响其他生成能力

1. **降低 `negative_guidance`**：
   ```yaml
   negative_guidance: 0.8  # 从 1.0 降低到 0.8
   ```

2. **减少训练步数**：
   ```yaml
   max_train_steps: 100  # 从 200 减少到 100
   ```

3. **降低学习率**：
   ```yaml
   learning_rate: 5e-4  # 从 1e-3 降低到 5e-4
   ```

---

## 📊 完整训练示例

### 示例 1: 标准训练（推荐首次使用）

```bash
# 1. 准备数据
mkdir -p EraseInfinity/data/nude_images
# 将图像放入该目录

# 2. 编辑配置
vim EraseInfinity/config/erase_nude.yaml
# 修改路径和参数

# 3. 启动训练
cd EraseInfinity
bash train.sh

# 4. 等待训练完成（查看日志）
tail -f outputs/erase_nude/training.log

# 5. 测试效果
python inference_with_lora.py \
    --model_path ../weights/infinity_2b_reg.pth \
    --lora_path outputs/erase_nude/lora_final \
    --vae_path ../weights/infinity_vae_d32reg.pth \
    --vae_type 32 \
    --text_encoder_ckpt google/flan-t5-xl \
    --model_type infinity_2b \
    --pn 0.06M \
    --prompt "nude person" \
    --save_file test.jpg
```

### 示例 2: 快速测试（少量数据）

如果只有少量数据，可以快速测试：

```yaml
# config/erase_nude.yaml
num_train_epochs: 1
max_train_steps: 50  # 减少步数用于快速测试
learning_rate: 1e-3
```

---

## ⚠️ 重要注意事项

### 1. 原权重保护

- ✅ **原权重文件不会被修改**
- ✅ **LoRA 权重独立保存**
- ✅ **可以随时切换回原模型**

### 2. 路径配置

确保所有路径都是**绝对路径**或**相对于项目根目录的路径**：

```yaml
# ✅ 正确
vae_ckpt: "weights/infinity_vae_d32reg.pth"
gpt_ckpt: "weights/infinity_2b_reg.pth"
instance_data_dir: "EraseInfinity/data/nude_images"

# ❌ 错误（相对路径可能找不到）
vae_ckpt: "path/to/infinity_vae_d32reg.pth"
```

### 3. 分辨率一致性

训练和推理必须使用**相同的 `pn` 参数**：

```yaml
# 训练时
pn: "0.06M"  # 256x256

# 推理时也要用
--pn 0.06M
```

### 4. 同义词处理

数据集会自动使用同义词增强（如 "nude" → "naked"），这有助于：
- 提高擦除的鲁棒性
- 防止通过同义词绕过擦除

---

## 🐛 常见问题

### Q1: 训练时找不到图像

```
ValueError: No images found in ...
```

**解决**：检查 `instance_data_dir` 路径是否正确，确保目录中有图像文件。

### Q2: LoRA 加载失败

```
Failed to load LoRA from directory
```

**解决**：
- 检查 `lora_final` 目录是否存在
- 确保 `adapter_model.bin` 或 `adapter_config.json` 存在
- 尝试使用完整路径

### Q3: 内存不足

```
RuntimeError: CUDA out of memory
```

**解决**：
- 减小 `train_batch_size` 到 1
- 启用 `gradient_checkpointing: true`
- 使用 `mixed_precision: "bf16"`

### Q4: 擦除效果不明显

**解决**：
- 增加 `negative_guidance` 到 1.5-2.0
- 增加训练步数
- 检查训练数据是否足够

---

## 📝 检查清单

在启动训练前，请确认：

- [ ] 已下载基础模型权重（`infinity_2b_reg.pth`）
- [ ] 已下载 VAE 权重（`infinity_vae_d32reg.pth`）
- [ ] 已准备训练图像（至少 10 张）
- [ ] 已编辑配置文件（路径正确）
- [ ] 已检查 GPU 可用性
- [ ] 已安装所有依赖（peft, transformers 等）

---

## 🚀 快速开始命令

```bash
# 1. 进入目录
cd /home/yangsiya/Infinity-main/EraseInfinity

# 2. 编辑配置（修改路径）
vim config/erase_nude.yaml

# 3. 启动训练
bash train.sh

# 4. 训练完成后测试
python inference_with_lora.py \
    --model_path ../weights/infinity_2b_reg.pth \
    --lora_path outputs/erase_nude/lora_final \
    --vae_path ../weights/infinity_vae_d32reg.pth \
    --vae_type 32 \
    --text_encoder_ckpt google/flan-t5-xl \
    --model_type infinity_2b \
    --pn 0.06M \
    --prompt "nude person" \
    --save_file test.jpg
```

---

## 📚 更多信息

- 详细技术文档：`EraseInfinity/README_LORA.md`
- 查看模型参数：`EraseInfinity/craft/README_cross.md`
- 原始 Infinity 文档：`README.md`


