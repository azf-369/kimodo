#!/usr/bin/env bash
# Launch 4-GPU Flow Matching training on H100 80GB nodes.
#
# GPU selection:
#   - Default: uses visible devices 0,1,2,3 (torchrun LOCAL_RANK 0..3).
#   - To pin GPUs: export CUDA_VISIBLE_DEVICES=0,1,2,3
#   - To use another set: export CUDA_VISIBLE_DEVICES=4,5,6,7
#
# Official-aligned run: max_steps=1_000_000 (~3-5 days on 4xH100).
set -euo pipefail

# NCCL / CUDA stability (avoid silent collective hangs after checkpoint save).
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

NPROC="${NPROC:-4}"
EXTRA_ARGS=("$@")

torchrun \
  --standalone \
  --nproc_per_node="${NPROC}" \
  -m kimodo.train.scripts.train_fm \
  --server \
  --server-4gpu \
  --text-mode encoder \
  --data-root datasets/bones-seed \
  --split-path datasets/kimodo-benchmark/splits/train_split_paths.txt \
  --stats-path checkpoints/Kimodo-G1-SEED-v1/stats/motion \
  --output-dir outputs/fm_g1_seed_4gpu_1m \
  --device cuda \
  "${EXTRA_ARGS[@]}"
