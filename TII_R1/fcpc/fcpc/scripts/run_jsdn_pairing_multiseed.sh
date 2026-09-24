#!/usr/bin/env bash
set -euo pipefail

source /home/zxy/miniconda3/etc/profile.d/conda.sh
conda activate fcpc-core

study_repo=/home/zxy/FCPC-re/TII_R1/fcpc/fcpc
cd "$study_repo"
mkdir -p outputs/original_fcpc_ablation

# Registered comparison: the three treatments differ only in the partner
# pairing rule.  Seeds are run in blocks so an interim paired comparison is
# available as soon as each block completes.
CUDA_VISIBLE_DEVICES=0 python -u -m scripts.run_original_fcpc_ablation \
  --methods similar_partner,random_partner,jsdn_partner \
  --seeds 45,46,47 \
  --rounds 200 \
  --beta 0.01 \
  --continue-on-error \
  2>&1 | tee outputs/original_fcpc_ablation/jsdn_pairing_3seed_r200_driver.log

python -m scripts.summarize_full_comparison \
  --log-dir outputs/original_fcpc_ablation/logs \
  --console-dir outputs/original_fcpc_ablation/console \
  --seeds 45,46,47 \
  --rounds 200 \
  --clients-per-round 6 \
  --output outputs/original_fcpc_ablation/jsdn_pairing_3seed_r200_summary.csv \
  2>&1 | tee outputs/original_fcpc_ablation/jsdn_pairing_3seed_r200_summary.txt
