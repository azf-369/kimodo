#!/usr/bin/env bash
# Stage-3 PD: 24→12, teacher = Stage-2 best student @ step_10000
set -euo pipefail
cd /home/zhengjk/PyProject/kimodo
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TEXT_ENCODER_DEVICE=cpu

STAGE2="${STAGE2:-outputs/pd_g1_rp_teacher_seed/stage_50to25_best/step_10000}"
OUT="${OUT:-outputs/pd_g1_rp_teacher_seed/stage_24to12_gentle}"

if [[ ! -f "$STAGE2/model.safetensors" ]]; then
  echo "ERROR: Stage-2 checkpoint missing: $STAGE2/model.safetensors" >&2
  exit 1
fi

exec python -m kimodo.distill.scripts.train_pd \
  --local --recipe pd_g1_stage3_24to12_gentle_local --stage 24to12 \
  --teacher-checkpoint "$STAGE2" --teacher-dtype fp32 \
  --data-root datasets/bones-seed \
  --split-path datasets/kimodo-benchmark/splits/train_split_paths.txt \
  --stats-path checkpoints/Kimodo-G1-SEED-v1.hf/stats/motion \
  --output-dir "$OUT" \
  --device cuda \
  --wandb --wandb-project kimodo-pd-g1 \
  --wandb-run-name pd-24to12-gentle
