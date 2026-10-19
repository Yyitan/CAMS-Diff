#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
# 激活包含 CAMS 依赖的环境（按本机环境修改）
# conda activate <env>

DATASET_PATH=${1:-data/mit-states}
GPU_ID=${GPU_ID:-1}

run_one() {
  local config=$1
  local out_dir=$2

  mkdir -p "$out_dir"
  echo "============================================================"
  echo "$(date '+%F %T') | start $config on GPU $GPU_ID"
  echo "log: $out_dir/train.log"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES="$GPU_ID" python -u train.py \
    --config "$config" \
    --dataset_path "$DATASET_PATH" \
    2>&1 | tee "$out_dir/train.log"

  echo "$(date '+%F %T') | finished $config"
}

run_one configs/mit_diff_v3_lam001.yml output/mit_diff_v3_lam001
run_one configs/mit_diff_v3_lam0005.yml output/mit_diff_v3_lam0005
run_one configs/mit_diff_v3_detach.yml output/mit_diff_v3_detach
