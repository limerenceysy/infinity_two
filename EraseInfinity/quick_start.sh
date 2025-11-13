#!/bin/bash
# ===================================================================
# EraseInfinity Quick Start Script
# 快速启动 LoRA 微调和推理
# ===================================================================

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "EraseInfinity Quick Start"
echo "=========================================="

# 检查参数
if [ "$1" == "train" ]; then
    echo "Starting LoRA fine-tuning..."
    echo ""
    bash train.sh
    
elif [ "$1" == "infer" ]; then
    if [ -z "$2" ] || [ -z "$3" ]; then
        echo "Usage: $0 infer <lora_path> <prompt> [output_file]"
        echo ""
        echo "Example:"
        echo "  $0 infer outputs/erase_nude/lora_final 'a beautiful landscape' output.jpg"
        exit 1+
    fi
    
    LORA_PATH="$2"
    PROMPT="$3"
    OUTPUT_FILE="${4:-output_lora.jpg}"
    
    echo "Running inference with LoRA weights..."
    echo "  LoRA path: $LORA_PATH"
    echo "  Prompt: $PROMPT"
    echo "  Output: $OUTPUT_FILE"
    echo ""
    
    python inference_with_lora.py \
        --model_path ../weights/infinity_2b_reg.pth \
        --lora_path "$LORA_PATH" \
        --vae_path ../weights/infinity_vae_d32reg.pth \
        --vae_type 32 \
        --text_encoder_ckpt google/flan-t5-xl \
        --model_type infinity_2b \
        --pn 0.06M \
        --prompt "$PROMPT" \
        --cfg 3 \
        --tau 0.5 \
        --seed 42 \
        --save_file "$OUTPUT_FILE"
    
elif [ "$1" == "help" ] || [ -z "$1" ]; then
    echo "Usage: $0 {train|infer|help}"
    echo ""
    echo "Commands:"
    echo "  train                    - Start LoRA fine-tuning"
    echo "  infer <lora> <prompt>    - Run inference with LoRA weights"
    echo "  help                     - Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Start training"
    echo "  $0 train"
    echo ""
    echo "  # Run inference"
    echo "  $0 infer outputs/erase_nude/lora_final 'a beautiful landscape'"
    echo ""
    echo "For more details, see README_LORA.md"
    
else
    echo "Unknown command: $1"
    echo "Use '$0 help' for usage information"
    exit 1
fi

