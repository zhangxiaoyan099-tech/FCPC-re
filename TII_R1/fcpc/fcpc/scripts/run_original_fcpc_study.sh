#!/usr/bin/env bash
set -euo pipefail

source /home/zxy/miniconda3/etc/profile.d/conda.sh
conda activate fcpc-core

study_repo=/home/zxy/FCPC-re/TII_R1/fcpc/fcpc
cd "$study_repo"

mkdir -p outputs/jsdn_gradient_update_quick
mkdir -p outputs/original_fcpc_ablation

CUDA_VISIBLE_DEVICES=0 python -u -m scripts.run_jsdn_gradient_update_audit \
  --config configs/original_fcpc/cifar10_jsdn_gradient_update_quick.yaml \
  2>&1 | tee outputs/jsdn_gradient_update_quick/console.log

CUDA_VISIBLE_DEVICES=0 python -u -m scripts.run_original_fcpc_ablation \
  --methods all \
  --seeds 42 \
  --rounds 50 \
  --beta 0.01 \
  --continue-on-error \
  2>&1 | tee outputs/original_fcpc_ablation/runner_seed42_r50.log

python -m scripts.summarize_full_comparison \
  --log-dir outputs/original_fcpc_ablation/logs \
  --console-dir outputs/original_fcpc_ablation/console \
  --seeds 42 \
  --rounds 50 \
  --clients-per-round 6 \
  --output outputs/original_fcpc_ablation/summary_seed42_r50.csv \
  2>&1 | tee outputs/original_fcpc_ablation/summary_seed42_r50.txt

python -m scripts.summarize_original_fcpc_mechanisms \
  --rounds 50 \
  --seed 42 \
  --output outputs/original_fcpc_ablation/mechanism_summary_seed42_r50.csv \
  2>&1 | tee outputs/original_fcpc_ablation/mechanism_summary_seed42_r50.txt
