# CrossAttention 参数查看和修改工具使用指南

## 工具列表

本目录包含两个工具：
1. **`inspect_crossattention_params.py`** - 查看 CrossAttention 的 mat_kv 和 proj 参数
2. **`manual_modify_mat_q.py`** - 手动修改 CrossAttention 的 mat_q 权重

---

## 工具 1: 查看 mat_kv 和 proj 参数

### 功能说明

`inspect_crossattention_params.py` 用于查看 Infinity 模型中 CrossAttention 层的以下参数：
- **mat_kv**: 键值对的线性变换矩阵（将 KV 输入投影到 Q 空间）
- **v_bias**: mat_kv 相关的偏置项
- **proj**: 输出投影矩阵

### 使用方法

#### 1. 查看所有参数（mat_kv 和 proj）

```bash
cd /home/yangsiya/Infinity-main
python EraseInfinity/craft/inspect_crossattention_params.py \
    --model_path weights/infinity_2b_reg.pth \
    --all
```

#### 2. 只查看 mat_kv 参数（包括 v_bias）

```bash
python EraseInfinity/craft/inspect_crossattention_params.py \
    --model_path weights/infinity_2b_reg.pth \
    --mat_kv
```

#### 3. 只查看 proj 参数

```bash
python EraseInfinity/craft/inspect_crossattention_params.py \
    --model_path weights/infinity_2b_reg.pth \
    --proj
```

#### 4. 同时查看 mat_kv 和 proj（显式指定）

```bash
python EraseInfinity/craft/inspect_crossattention_params.py \
    --model_path weights/infinity_2b_reg.pth \
    --mat_kv \
    --proj
```

### 输出说明

脚本会显示每个 CrossAttention 层的：
- **参数名称和类型**（linear_weight 或 parameter）
- **形状**（shape）
- **统计信息**（均值、标准差、最小值、最大值）

示例输出：
```
层 0: unregistered_blocks.0.ca
  mat_kv (linear_weight):
    形状: (4096, 4096)
    统计: mean=0.000123, std=0.015623, min=-0.045678, max=0.052341
  v_bias (parameter):
    形状: (2048,)
    统计: mean=0.000000, std=0.000000, min=0.000000, max=0.000000
  proj (linear_weight):
    形状: (2048, 2048)
    统计: mean=0.000234, std=0.012456, min=-0.038901, max=0.041234
```

---

## 工具 2: 手动修改 mat_q 权重

### 功能说明

这个工具允许你手动修改 Infinity 模型中 CrossAttention 层的 `mat_q` 权重，用于：
- 找到对模型影响最大的层
- 测试不同权重修改对生成图像的影响
- 保存修改后的权重而不破坏原权重文件

### 使用方法

#### 1. 列出所有 CrossAttention 层

```bash
cd /home/yangsiya/Infinity-main
python EraseInfinity/craft/manual_modify_mat_q.py \
    --model_path weights/infinity_2b_reg.pth \
    --list_layers
```

这会显示所有层的索引、名称、形状和统计信息。

#### 2. 修改指定层的权重

#### 方法 1: 置零（最明显的效果）
```bash
python EraseInfinity/craft/manual_modify_mat_q.py \
    --model_path weights/infinity_2b_reg.pth \
    --layer_idx 0 \
    --method zero \
    --save_path weights/infinity_2b_reg_layer0_zero.pth
```

#### 方法 2: 缩放（放大或缩小权重）
```bash
# 放大 10 倍（会产生强烈变化）
python EraseInfinity/craft/manual_modify_mat_q.py \
    --model_path weights/infinity_2b_reg.pth \
    --layer_idx 5 \
    --method scale \
    --scale 10.0 \
    --save_path weights/infinity_2b_reg_layer5_scale10.pth

# 缩小到 0.1 倍
python EraseInfinity/craft/manual_modify_mat_q.py \
    --model_path weights/infinity_2b_reg.pth \
    --layer_idx 5 \
    --method scale \
    --scale 0.1 \
    --save_path weights/infinity_2b_reg_layer5_scale0.1.pth
```

#### 方法 3: 添加噪声（随机扰动）
```bash
# 添加较大噪声（标准差=1.0，会产生明显变化）
python EraseInfinity/craft/manual_modify_mat_q.py \
    --model_path weights/infinity_2b_reg.pth \
    --layer_idx 10 \
    --method noise \
    --noise_std 1.0 \
    --save_path weights/infinity_2b_reg_layer10_noise1.0.pth
```

#### 方法 4: 设置为特定值
```bash
# 全部设置为 1.0
python EraseInfinity/craft/manual_modify_mat_q.py \
    --model_path weights/infinity_2b_reg.pth \
    --layer_idx 15 \
    --method set \
    --target_value 1.0 \
    --save_path weights/infinity_2b_reg_layer15_set1.0.pth
```

#### 方法 5: 临时修改（不保存，用于快速测试）
```bash
python EraseInfinity/craft/manual_modify_mat_q.py \
    --model_path weights/infinity_2b_reg.pth \
    --layer_idx 0 \
    --method zero \
    --temporary
```

注意：临时修改需要在同一个 Python 进程中继续使用模型，如果退出程序，修改会丢失。

### 3. 在推理脚本中使用修改后的权重

修改 `tools/run_infinity.py` 或你的推理脚本，将 `--model_path` 指向修改后的权重：

```bash
python tools/run_infinity.py \
    --model_path weights/infinity_2b_reg_layer0_zero.pth \
    --model_type infinity_2b \
    --pn 1M \
    --prompt "a beautiful landscape" \
    --save_file output_modified.jpg
```

或者在 Python 代码中：

```python
args.model_path = "weights/infinity_2b_reg_layer0_zero.pth"
infinity = load_transformer(vae, args)
# 继续使用修改后的模型进行推理
```

## 推荐测试策略

### 策略 1: 系统性测试（找到影响最大的层）

1. **先列出所有层**：
   ```bash
   python EraseInfinity/craft/manual_modify_mat_q.py \
       --model_path weights/infinity_2b_reg.pth \
       --list_layers
   ```

2. **测试前几层（通常影响较大）**：
   ```bash
   for i in 0 1 2 3 4 5; do
       python EraseInfinity/craft/manual_modify_mat_q.py \
           --model_path weights/infinity_2b_reg.pth \
           --layer_idx $i \
           --method zero \
           --save_path weights/infinity_2b_reg_layer${i}_zero.pth
   done
   ```

3. **对每层运行推理，观察图像变化**

4. **测试中间层和后层**：
   ```bash
   # 中间层（如第10-15层）
   for i in 10 11 12 13 14 15; do
       python EraseInfinity/craft/manual_modify_mat_q.py \
           --model_path weights/infinity_2b_reg.pth \
           --layer_idx $i \
           --method zero \
           --save_path weights/infinity_2b_reg_layer${i}_zero.pth
   done
   ```

### 策略 2: 渐进式测试（逐步增大修改幅度）

对于同一层，逐步增大修改幅度：

```bash
# 轻微缩放（0.5倍）
python EraseInfinity/craft/manual_modify_mat_q.py \
    --model_path weights/infinity_2b_reg.pth \
    --layer_idx 5 \
    --method scale --scale 0.5 \
    --save_path weights/infinity_2b_reg_layer5_scale0.5.pth

# 中等缩放（2倍）
python EraseInfinity/craft/manual_modify_mat_q.py \
    --model_path weights/infinity_2b_reg.pth \
    --layer_idx 5 \
    --method scale --scale 2.0 \
    --save_path weights/infinity_2b_reg_layer5_scale2.0.pth

# 大幅缩放（10倍）
python EraseInfinity/craft/manual_modify_mat_q.py \
    --model_path weights/infinity_2b_reg.pth \
    --layer_idx 5 \
    --method scale --scale 10.0 \
    --save_path weights/infinity_2b_reg_layer5_scale10.0.pth
```

然后分别运行推理，观察变化程度。

### 策略 3: 快速筛选（使用临时修改）

如果想快速测试多个层，可以使用临时修改：

```python
# 在 Python 脚本中
import sys
sys.path.insert(0, '/home/yangsiya/Infinity-main')

from tools.run_infinity import load_transformer, load_visual_tokenizer
import argparse

# 加载模型
args = argparse.Namespace(...)  # 你的参数
vae = load_visual_tokenizer(args)
model = load_transformer(vae, args)

# 临时修改第0层
layers = [m for name, m in model.named_modules() 
          if m.__class__.__name__ == "CrossAttention"]
layers[0].mat_q.weight.zero_()  # 置零

# 立即运行推理测试
# ...
```

## 注意事项

1. **原权重文件不会被修改**：所有修改都会保存为新文件
2. **修改幅度建议**：
   - 置零 (`zero`): 最明显的效果，适合快速筛选
   - 大倍数缩放 (`scale 10.0`): 会产生强烈变化
   - 大噪声 (`noise_std 1.0`): 会产生随机但明显的变化
3. **层选择建议**：
   - 前层（0-5）: 通常影响低级特征
   - 中间层（10-20）: 影响中级特征
   - 后层（25+）: 影响高级语义
4. **保存文件命名**：建议包含层索引和修改方法，如 `layer5_scale10.pth`

## 批量测试脚本示例

创建 `test_all_layers.sh`:

```bash
#!/bin/bash
MODEL_PATH="weights/infinity_2b_reg.pth"
METHOD="zero"  # 或 scale, noise 等

for i in {0..31}; do
    echo "修改层 $i..."
    python EraseInfinity/craft/manual_modify_mat_q.py \
        --model_path $MODEL_PATH \
        --layer_idx $i \
        --method $METHOD \
        --save_path weights/infinity_2b_reg_layer${i}_${METHOD}.pth
done
```

然后对每个生成的权重文件运行推理，比较结果。



