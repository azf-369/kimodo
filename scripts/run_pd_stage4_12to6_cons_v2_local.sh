#!/usr/bin/env bash
# Stage-4 PD cons_v2: 12→6 full retrain from Stage-3 (cold start, no resume)
set -euo pipefail
cd /home/zhengjk/PyProject/kimodo
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TEXT_ENCODER_DEVICE=cpu

STAGE3="${STAGE3:-outputs/pd_g1_rp_teacher_seed/stage_24to12_best/step_8000}"
OUT="${OUT:-outputs/pd_g1_rp_teacher_seed/stage_12to6_cons_v2}"

if [[ ! -f "$STAGE3/model.safetensors" ]]; then
  echo "ERROR: Stage-3 checkpoint missing: $STAGE3/model.safetensors" >&2
  exit 1
fi

exec python -m kimodo.distill.scripts.train_pd \
  --local --recipe pd_g1_stage4_12to6_cons_v2_local --stage 12to6 \
  --teacher-checkpoint "$STAGE3" --teacher-dtype fp32 \
  --data-root datasets/bones-seed \
  --split-path datasets/kimodo-benchmark/splits/train_split_paths.txt \
  --stats-path checkpoints/Kimodo-G1-SEED-v1.hf/stats/motion \
  --output-dir "$OUT" \
  --device cuda \
  --wandb --wandb-project kimodo-pd-g1 \
  --wandb-run-name pd-12to6-cons-v2
