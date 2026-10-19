#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
# 激活包含 CAMS 依赖的环境（按本机环境修改）
# conda activate <env>

GPU_ID=${GPU_ID:-1}
DATASET_PATH=${DATASET_PATH:-data/mit-states}
LOAD_MODEL=${LOAD_MODEL:-output/cams_mit_seed0/final_model.pt}
SD_MODEL=${SD_MODEL:-./pretrained/stable-diffusion-2-1}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-8}
SD_BATCH_SIZE=${SD_BATCH_SIZE:-4}
PROMPT_SET=${PROMPT_SET:-obj_is_attr,plain,image}

IFS=',' read -r -a PROMPT_NAMES <<< "$PROMPT_SET"

for PROMPT_NAME in "${PROMPT_NAMES[@]}"; do
  case "$PROMPT_NAME" in
    obj_is_attr)
      PROMPT_TEMPLATE="a photo of the {obj} is {attr}"
      OUT_DIR="output/mit_sd_rerank_k20_t1_prompt_obj_is_attr"
      ;;
    plain)
      PROMPT_TEMPLATE="a {attr} {obj}"
      OUT_DIR="output/mit_sd_rerank_k20_t1_prompt_plain"
      ;;
    image)
      PROMPT_TEMPLATE="an image of a {attr} {obj}"
      OUT_DIR="output/mit_sd_rerank_k20_t1_prompt_image"
      ;;
    *)
      echo "Unknown PROMPT_NAME: $PROMPT_NAME" >&2
      exit 2
      ;;
  esac

  mkdir -p "$OUT_DIR"
  echo "Running prompt variant: $PROMPT_NAME"
  echo "Prompt template: $PROMPT_TEMPLATE"
  echo "Output dir: $OUT_DIR"

  CUDA_VISIBLE_DEVICES="$GPU_ID" python -u sd_rerank_eval.py \
    --config configs/mit_seed0.yml \
    --dataset_path "$DATASET_PATH" \
    --load_model "$LOAD_MODEL" \
    --sd_model_path "$SD_MODEL" \
    --output_dir "$OUT_DIR" \
    --phase both \
    --topk_pairs 20 \
    --num_timesteps 1 \
    --sd_batch_size "$SD_BATCH_SIZE" \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --prompt_template "$PROMPT_TEMPLATE" \
    --alpha_grid "0,0.02,0.04,0.06,0.08,0.1,0.12,0.14,0.16,0.18,0.2,0.22,0.24,0.26,0.28,0.3" \
    2>&1 | tee "$OUT_DIR/run.log"
done
