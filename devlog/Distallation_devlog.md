# Distillation Devlog

## 7.14

### 开发机测试：
```bash
# 文本cache到cpu
export TEXT_ENCODER_DEVICE=cpu
python -m kimodo.distill.scripts.train_pd \
  --local --precompute-text-only --text-mode encoder --text-encoder-device cpu \
  --data-root datasets/bones-seed \
  --split-path datasets/kimodo-benchmark/splits/train_split_paths.txt \
  --stats-path checkpoints/Kimodo-G1-SEED-v1.hf/stats/motion

# 正式训练
RP=~/.cache/huggingface/hub/models--nvidia--Kimodo-G1-RP-v1/snapshots/3020ad8c419c244e0429d360163730c63c4ed011
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m kimodo.distill.scripts.train_pd \
  --local --formal --stage 100to50 --text-mode encoder \
  --teacher-checkpoint "$RP" --teacher-dtype bf16 \
  --data-root datasets/bones-seed \
  --split-path datasets/kimodo-benchmark/splits/train_split_paths.txt \
  --stats-path checkpoints/Kimodo-G1-SEED-v1.hf/stats/motion \
  --output-dir outputs/pd_g1_rp_teacher_seed/stage_100to50_formal \
  --device cuda \
  --wandb --wandb-project kimodo-pd-g1 \
  --wandb-run-name pd-100to50-formal-f64-ga8

# 要点：`max_frames=64`，`grad_accum=8`，`lr=5e-5`（warmup→cosine），`constraint_prob=0.8`，`max_steps=30000`。
```

### H100 服务器训练
```bash
# 配置文件：
# - 单卡：`kimodo/distill/config/pd_g1_formal_server.yaml`（`--server`）
# - 4 卡：再叠 `pd_g1_formal_server_4gpu.yaml`（`--server --server-4gpu` + `torchrun`）

#### 0) 环境（与 FM 相同）

```bash
source .venv/bin/activate
export http_proxy=http://10.127.48.11:3128/
export https_proxy=http://10.127.48.11:3128/
export HTTP_PROXY=$http_proxy
export HTTPS_PROXY=$https_proxy
export WANDB_API_KEY="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"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RP=~/.cache/huggingface/hub/models--nvidia--Kimodo-G1-RP-v1/snapshots/<hash>
# 确认 RP、SEED 数据、stats 路径存在（stats 也可用 checkpoints/Kimodo-G1-SEED-v1/stats/motion）

# 不需要先 `--text-encoder-device cpu` 单独跑 cache：首次 `--server` 训练时 **rank0 在 CUDA 上自动补全** cache，再加载 teacher/student。

#### 1) 单卡 H100
python -m kimodo.distill.scripts.train_pd \
  --server --stage 100to50 --text-mode encoder \
  --teacher-checkpoint "$RP" --teacher-dtype fp32 \
  --data-root datasets/bones-seed \
  --split-path datasets/kimodo-benchmark/splits/train_split_paths.txt \
  --stats-path checkpoints/Kimodo-G1-SEED-v1.hf/stats/motion \
  --output-dir outputs/pd_g1_rp_teacher_seed/stage_100to50_formal_h100 \
  --device cuda \
  --wandb --wandb-project kimodo-pd-g1 \
  --wandb-run-name pd-100to50-formal-h100-f300-b8

# 可选：只预热 text cache（仍用 GPU，不加载 PD 模型）：

# ```bash
# python -m kimodo.distill.scripts.train_pd \
#   --server --precompute-text-only --text-mode encoder \
#   --text-encoder-device cuda \
#   --data-root datasets/bones-seed \
#   --split-path datasets/kimodo-benchmark/splits/train_split_paths.txt \
#   --stats-path checkpoints/Kimodo-G1-SEED-v1.hf/stats/motion
# ```

#### 2) 4 卡 H100（DDP）

export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_ASYNC_ERROR_HANDLING=1

torchrun --standalone --nproc_per_node=4 \
  -m kimodo.distill.scripts.train_pd \
  --server --server-4gpu --stage 100to50 --text-mode encoder \
  --teacher-checkpoint "$RP" --teacher-dtype fp32 \
  --data-root datasets/bones-seed \
  --split-path datasets/kimodo-benchmark/splits/train_split_paths.txt \
  --stats-path checkpoints/Kimodo-G1-SEED-v1.hf/stats/motion \
  --output-dir outputs/pd_g1_rp_teacher_seed/stage_100to50_formal_4gpu \
  --device cuda \
  --wandb --wandb-project kimodo-pd-g1 \
  --wandb-run-name pd-100to50-formal-4xh100-f300-b8x4


# 或：

# ```bash
# export RP=...   # 同上
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# bash kimodo/distill/scripts/launch_train_pd_4gpu.sh \
#   --wandb --wandb-project kimodo-pd-g1 \
#   --wandb-run-name pd-100to50-formal-4xh100
# ```

<!-- 要点：必须用 `torchrun`；只开 `--server-4gpu` 而不用多进程会落成单卡并打印 WARNING。checkpoint 仅 rank0 写入。 -->
```
