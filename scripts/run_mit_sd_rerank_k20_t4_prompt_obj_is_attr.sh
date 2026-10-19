#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
# 激活包含 CAMS 依赖的环境（按本机环境修改）
# conda activate <env>

GPU_ID=${GPU_ID:-1}
DATASET_PATH=${DATASET_PATH:-data/mit-states}
LOAD_MODEL=${LOAD_MODEL:-output/cams_mit_seed0/final_model.pt}
SD_MODEL=${SD_MODEL:-./pretrained/stable-diffusion-2-1}
OUT_DIR=${OUT_DIR:-output/mit_sd_rerank_k20_t4_prompt_obj_is_attr}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-8}
SD_BATCH_SIZE=${SD_BATCH_SIZE:-4}
PROMPT_TEMPLATE=${PROMPT_TEMPLATE:-"a photo of the {obj} is {attr}"}

mkdir -p "$OUT_DIR"

CUDA_VISIBLE_DEVICES="$GPU_ID" python -u sd_rerank_eval.py \
  --config configs/mit_seed0.yml \
  --dataset_path "$DATASET_PATH" \
  --load_model "$LOAD_MODEL" \
  --sd_model_path "$SD_MODEL" \
  --output_dir "$OUT_DIR" \
  --phase both \
  --topk_pairs 20 \
  --num_timesteps 4 \
  --sd_batch_size "$SD_BATCH_SIZE" \
  --eval_batch_size "$EVAL_BATCH_SIZE" \
  --prompt_template "$PROMPT_TEMPLATE" \
  --alpha_grid "0,0.02,0.04,0.06,0.08,0.1,0.12,0.14,0.16,0.18,0.2,0.22,0.24,0.26,0.28,0.3" \
  2>&1 | tee "$OUT_DIR/run.log"
