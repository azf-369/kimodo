#!/usr/bin/env bash
# Stage-2 PD: 50→25, teacher = Stage-1 best student @ step_30000
set -euo pipefail
cd /home/zhengjk/PyProject/kimodo
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TEXT_ENCODER_DEVICE=cpu

STAGE1="${STAGE1:-outputs/pd_g1_rp_teacher_seed/stage_100to50_best/step_30000}"
OUT="${OUT:-outputs/pd_g1_rp_teacher_seed/stage_50to25}"

if [[ ! -f "$STAGE1/model.safetensors" ]]; then
  echo "ERROR: Stage-1 checkpoint missing: $STAGE1/model.safetensors" >&2
  exit 1
fi

exec python -m kimodo.distill.scripts.train_pd \
  --local --recipe pd_g1_stage2_50to25_local --stage 50to25 \
  --teacher-checkpoint "$STAGE1" --teacher-dtype fp32 \
  --data-root datasets/bones-seed \
  --split-path datasets/kimodo-benchmark/splits/train_split_paths.txt \
  --stats-path checkpoints/Kimodo-G1-SEED-v1.hf/stats/motion \
  --output-dir "$OUT" \
  --device cuda \
  --wandb --wandb-project kimodo-pd-g1 \
  --wandb-run-name pd-50to25-from-best
