#!/bin/bash
# 快速测试示例：修改第0层的 mat_q 权重并生成图像对比

MODEL_PATH="weights/infinity_2b_reg.pth"
PROMPT="a beautiful landscape with mountains"

echo "=========================================="
echo "步骤1: 列出所有 CrossAttention 层"
echo "=========================================="
python EraseInfinity/craft/manual_modify_mat_q.py \
    --model_path $MODEL_PATH \
    --list_layers

echo ""
echo "=========================================="
echo "步骤2: 修改第0层的 mat_q（置零，效果最明显）"
echo "=========================================="
python EraseInfinity/craft/manual_modify_mat_q.py \
    --model_path $MODEL_PATH \
    --layer_idx 0 \
    --method zero \
    --save_path weights/infinity_2b_reg_layer0_zero.pth

echo ""
echo "=========================================="
echo "步骤3: 使用原权重生成图像"
echo "=========================================="
python EraseInfinity/craft/test_modified_weight.py \
    --model_path $MODEL_PATH \
    --model_type infinity_2b \
    --pn 1M \
    --prompt "$PROMPT" \
    --save_file output_original.jpg \
    --text_encoder_ckpt model_cache/google/flan-t5-xl \
    --vae_path model_cache/FoundationVision/Infinity/infinity_vae_d32reg.pth

echo ""
echo "=========================================="
echo "步骤4: 使用修改后的权重生成图像"
echo "=========================================="
python EraseInfinity/craft/test_modified_weight.py \
    --model_path weights/infinity_2b_reg_layer0_zero.pth \
    --model_type infinity_2b \
    --pn 1M \
    --prompt "$PROMPT" \
    --save_file output_modified_layer0_zero.jpg \
    --text_encoder_ckpt model_cache/google/flan-t5-xl \
    --vae_path model_cache/FoundationVision/Infinity/infinity_vae_d32reg.pth

echo ""
echo "=========================================="
echo "完成！对比 output_original.jpg 和 output_modified_layer0_zero.jpg"
echo "=========================================="


