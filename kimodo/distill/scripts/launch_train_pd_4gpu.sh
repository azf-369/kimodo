#!/usr/bin/env bash
# Launch 4-GPU Progressive Distillation (Stage 100→50) on H100 80GB nodes.
#
# Usage:
#   export RP=/path/to/Kimodo-G1-RP-v1   # checkpoint dir with model.safetensors
#   export CUDA_VISIBLE_DEVICES=0,1,2,3
#   bash kimodo/distill/scripts/launch_train_pd_4gpu.sh --wandb --wandb-run-name pd-100to50-4xh100
#
# Extra CLI args are appended after the defaults below.
set -euo pipefail

export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

NPROC="${NPROC:-4}"
EXTRA_ARGS=("$@")

if [[ -z "${RP:-}" ]]; then
  echo "ERROR: set RP to the Kimodo-G1-RP-v1 checkpoint directory, e.g.:"
  echo "  export RP=~/.cache/huggingface/hub/models--nvidia--Kimodo-G1-RP-v1/snapshots/<hash>"
  exit 1
fi

exec torchrun \
  --standalone \
  --nproc_per_node="${NPROC}" \
  -m kimodo.distill.scripts.train_pd \
  --server \
  --server-4gpu \
  --stage 100to50 \
  --text-mode encoder \
  --teacher-checkpoint "${RP}" \
  --teacher-dtype fp32 \
  --data-root datasets/bones-seed \
  --split-path datasets/kimodo-benchmark/splits/train_split_paths.txt \
  --stats-path checkpoints/Kimodo-G1-SEED-v1.hf/stats/motion \
  --output-dir outputs/pd_g1_rp_teacher_seed/stage_100to50_formal_4gpu \
  --device cuda \
  "${EXTRA_ARGS[@]}"
