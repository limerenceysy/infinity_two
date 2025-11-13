#!/bin/bash
# ===================================================================
# EraseInfinity Training Script (Prompt-Only Mode)
# 只使用 prompt 的nude内容擦除训练 - 不需要真实nude图像！
# ===================================================================

set -e  # 遇到错误立即退出

# ==================== 配置路径 ====================
# 获取脚本所在目录（EraseInfinity）
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# 项目根目录（EraseInfinity 的父目录）
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# 配置文件路径（相对于 EraseInfinity 目录）
CONFIG_FILE="$SCRIPT_DIR/config/erase_nude_prompt_only.yaml"

# 切换到项目根目录（权重文件在这里）
cd "$PROJECT_ROOT"

echo "=========================================="
echo "EraseInfinity Training Script (Prompt-Only)"
echo "Project root: $PROJECT_ROOT"
echo "Script directory: $SCRIPT_DIR"
echo "Config file: $CONFIG_FILE"
echo "Working directory: $(pwd)"
echo "=========================================="


# ==================== 检查配置文件 ====================
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "✓ Config file found"


# ==================== 环境检查 ====================
echo "=========================================="
echo "Environment Check"
echo "=========================================="

# 检查 Python
if ! command -v python &> /dev/null; then
    echo "Error: Python not found"
    exit 1
fi

PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $PYTHON_VERSION"

# 检查 CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "✓ CUDA available"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
else
    echo "Warning: nvidia-smi not found, GPU may not be available"
fi

# ==================== 准备配置文件相对路径 ====================
# 计算配置文件相对于项目根目录的路径（用于传递给 train_erase.py）
CONFIG_RELATIVE="${CONFIG_FILE#$PROJECT_ROOT/}"

# ==================== 创建输出目录 ====================
OUTPUT_DIR_REL=$(grep "output_dir:" "$CONFIG_FILE" | awk '{print $2}' | tr -d '"')
if [ -n "$OUTPUT_DIR_REL" ]; then
    # 转换为绝对路径（相对于项目根目录）
    OUTPUT_DIR="$PROJECT_ROOT/$OUTPUT_DIR_REL"
    mkdir -p "$OUTPUT_DIR"
    echo "✓ Output directory: $OUTPUT_DIR"
else
    OUTPUT_DIR="$PROJECT_ROOT/EraseInfinity/outputs/erase_nude_prompt_only"
    mkdir -p "$OUTPUT_DIR"
    echo "✓ Output directory (default): $OUTPUT_DIR"
fi

# ==================== 设置环境变量 ====================
# Hugging Face 镜像（如果在国内）
# export HF_ENDPOINT="https://hf-mirror.com"

# 防止 tokenizers 并行警告
export TOKENIZERS_PARALLELISM=false

# CUDA 设备（从 config 读取或使用默认值）
DEVICES=$(grep "devices:" "$CONFIG_FILE" | awk '{print $2}' | tr -d '"')
if [ -z "$DEVICES" ]; then
    DEVICES="0"
fi
export CUDA_VISIBLE_DEVICES="$DEVICES"


# ==================== 开始训练 ====================
echo ""
echo "Starting training..."
echo "Press Ctrl+C to stop"
echo ""

# 使用 python 运行训练脚本（使用相对于项目根目录的路径）
# train_erase.py 会从项目根目录运行，所以配置文件路径需要相对于项目根目录
set +e  # 暂时允许错误，以便检查退出码
python "$SCRIPT_DIR/train_erase.py" \
    --config "$CONFIG_RELATIVE" \
    2>&1 | tee "${OUTPUT_DIR}/training.log"
TRAIN_EXIT_CODE=${PIPESTATUS[0]}  # 获取python命令的退出码
set -e  # 恢复错误即退出

# ==================== 检查训练结果 ====================
echo ""
echo "=========================================="
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training failed with exit code: $TRAIN_EXIT_CODE"
    echo "Please check the log file for details: ${OUTPUT_DIR}/training.log"
    exit $TRAIN_EXIT_CODE
fi
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"
echo "Log file: ${OUTPUT_DIR}/training.log"
echo ""
echo "To use the trained LoRA weights for inference:"
echo "  python inference_with_lora.py \\"
echo "    --model_path ../weights/infinity_2b_reg.pth \\"
echo "    --lora_path ${OUTPUT_DIR}/lora_final \\"
echo "    --vae_path ../weights/infinity_vae_d32reg.pth \\"
echo "    --prompt \"nude person\" \\"
echo "    --save_file test_nude.jpg"
echo ""
echo "=========================================="
echo "✓ Training successful - No real nude images were used!"
echo "=========================================="

