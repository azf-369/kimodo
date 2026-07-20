#!/usr/bin/env bash
set -euo pipefail
cd /home/zhengjk/PyProject/kimodo
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TEXT_ENCODER_DEVICE=cpu
RP=~/.cache/huggingface/hub/models--nvidia--Kimodo-G1-RP-v1/snapshots/3020ad8c419c244e0429d360163730c63c4ed011

exec python -m kimodo.distill.scripts.train_pd \
  --local --formal --stage 100to50 --text-mode encoder \
  --teacher-checkpoint "$RP" --teacher-dtype fp32 \
  --data-root datasets/bones-seed \
  --split-path datasets/kimodo-benchmark/splits/train_split_paths.txt \
  --stats-path checkpoints/Kimodo-G1-SEED-v1.hf/stats/motion \
  --output-dir outputs/pd_g1_rp_teacher_seed/stage_100to50_formal_cons_v4_rerun \
  --device cuda \
  --wandb --wandb-project kimodo-pd-g1 \
  --wandb-run-name pd-100to50-formal-cons-v4-rerun
