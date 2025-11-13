#!/bin/bash
# ===================================================================
# EraseInfinity LoRA 微调启动脚本
# 使用本地 peft 库进行 LoRA 微调，擦除 nude 相关内容
# ===================================================================

set -e

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 切换到项目根目录
cd "$PROJECT_ROOT"

echo "=========================================="
echo "EraseInfinity LoRA Fine-tuning"
echo "=========================================="
echo "Project root: $PROJECT_ROOT"
echo "Working directory: $(pwd)"
echo ""

# 检查配置文件
CONFIG_FILE="$SCRIPT_DIR/config/erase_nude_prompt_only.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "✓ Config file: $CONFIG_FILE"

# 检查 peft 库
if [ -d "$SCRIPT_DIR/peft" ]; then
    echo "✓ Local peft library found: $SCRIPT_DIR/peft"
else
    echo "⚠ Warning: Local peft library not found at $SCRIPT_DIR/peft"
    echo "  Will try to use system-installed peft"
fi

# 设置环境变量
export CUDA_VISIBLE_DEVICES=$(grep "devices:" "$CONFIG_FILE" | awk '{print $2}' | tr -d '"' || echo "0")
export TOKENIZERS_PARALLELISM=false

echo ""
echo "Starting LoRA fine-tuning..."
echo "=========================================="
echo ""

# 运行训练脚本
python "$SCRIPT_DIR/train_erase.py" \
    --config "$CONFIG_FILE" \
    2>&1 | tee "$SCRIPT_DIR/outputs/training_$(date +%Y%m%d_%H%M%S).log"

TRAIN_EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "=========================================="
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "✓ Training completed successfully!"
    echo ""
    OUTPUT_DIR=$(grep "output_dir:" "$CONFIG_FILE" | awk '{print $2}' | tr -d '"')
    echo "LoRA weights saved to: $PROJECT_ROOT/$OUTPUT_DIR/lora_final"
    echo ""
    echo "To use the trained LoRA for inference:"
    echo "  python $SCRIPT_DIR/inference_with_lora.py \\"
    echo "    --model_path weights/infinity_2b_reg.pth \\"
    echo "    --lora_path $OUTPUT_DIR/lora_final \\"
    echo "    --vae_path weights/infinity_vae_d32reg.pth \\"
    echo "    --prompt \"nude person\" \\"
    echo "    --save_file test_output.jpg"
else
    echo "❌ Training failed with exit code: $TRAIN_EXIT_CODE"
    exit $TRAIN_EXIT_CODE
fi
echo "=========================================="

