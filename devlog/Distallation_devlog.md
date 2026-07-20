# Distillation Devlog

## 7.14

### 配置与阶段

| 阶段 | 配置文件 | teacher | student init |
|------|----------|---------|--------------|
| Stage-1 100→50 | `pd_g1_stage1_100to50_local.yaml`（归档） | 官方 RP | 从 RP 拷贝 |
| Stage-2 50→25 | `pd_g1_stage2_50to25_local.yaml` | **Stage-1 `step_30000`** | 从该 ckpt 拷贝 |

已训好的 Stage-1：`outputs/pd_g1_rp_teacher_seed/stage_100to50_best/step_30000`  
（与 `formal_cons` / wandb `pd-100to50-formal-cons-v4` 同源。）

### 开发机：Stage-2（50→25）训练

```bash
export TEXT_ENCODER_DEVICE=cpu
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
STAGE1=outputs/pd_g1_rp_teacher_seed/stage_100to50_best/step_30000

# 确认 ckpt 存在：ls "$STAGE1/model.safetensors" "$STAGE1/config.yaml"
# 关掉占用 GPU 的 demo / 旧训练。

python -m kimodo.distill.scripts.train_pd \
  --local --recipe pd_g1_stage2_50to25_local --stage 50to25 \
  --teacher-checkpoint "$STAGE1" --teacher-dtype fp32 \
  --data-root datasets/bones-seed \
  --split-path datasets/kimodo-benchmark/splits/train_split_paths.txt \
  --stats-path checkpoints/Kimodo-G1-SEED-v1.hf/stats/motion \
  --output-dir outputs/pd_g1_rp_teacher_seed/stage_50to25 \
  --device cuda \
  --wandb --wandb-project kimodo-pd-g1 \
  --wandb-run-name pd-50to25-from-best

# 或: bash scripts/run_pd_stage2_50to25_local.sh
#
# 启动日志应显示: PD stage 50->25 | global_batch=8 | w_cons=2
# Teacher / student 均来自 STAGE1（勿再用官方 RP 当 teacher）。
```

### 开发机：复现 Stage-1（可选）

```bash
RP=~/.cache/huggingface/hub/models--nvidia--Kimodo-G1-RP-v1/snapshots/3020ad8c419c244e0429d360163730c63c4ed011
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m kimodo.distill.scripts.train_pd \
  --local --recipe pd_g1_stage1_100to50_local --stage 100to50 \
  --teacher-checkpoint "$RP" --teacher-dtype fp32 \
  --data-root datasets/bones-seed \
  --split-path datasets/kimodo-benchmark/splits/train_split_paths.txt \
  --stats-path checkpoints/Kimodo-G1-SEED-v1.hf/stats/motion \
  --output-dir outputs/pd_g1_rp_teacher_seed/stage_100to50_rerun \
  --device cuda \
  --wandb --wandb-project kimodo-pd-g1 \
  --wandb-run-name pd-100to50-stage1-reproduce
```

### H100 服务器训练

配置文件：
- 单卡：`pd_g1_formal_server.yaml`（`--server`）
- 4 卡：再叠 `pd_g1_formal_server_4gpu.yaml`（`--server --server-4gpu` + `torchrun`）

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
  --output-dir outputs/pd_g1_rp_teacher_seed/stage_100to50_formal_h100_ee_root \
  --device cuda \
  --wandb --wandb-project kimodo-pd-g1 \
  --wandb-run-name pd-100to50-formal-h100-ee-root-v5

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
  --output-dir outputs/pd_g1_rp_teacher_seed/stage_100to50_formal_4gpu_ee_root \
  --device cuda \
  --wandb --wandb-project kimodo-pd-g1 \
  --wandb-run-name pd-100to50-formal-4xh100-ee-root-v5

<!-- 要点：必须用 `torchrun`；只开 `--server-4gpu` 而不用多进程会落成单卡并打印 WARNING。checkpoint 仅 rank0 写入。 -->
```

## 7.15
### PD 配方 v4（约束优先）

f300 评测：travel OK，但 **约束/skate 变差**。根因与改动：[logs/7.15/pd_constraint_first_v4.md](logs/7.15/pd_constraint_first_v4.md)。

v4 `formal_cons` @30k：约束全面优于 RP，skate 仍差一截。

### PD 配方 v5（已回退）

v5（EE/root/skate 通道损失 + mix）难以同时兼顾约束与滑步；**`--local --formal` 已恢复为 v4 loss**，仅保留 **batch=4 / accum=2**。

| 项 | v5 ee_root | **现 formal（v4 + bs4）** |
|----|------------|---------------------------|
| 约束类型 | EE/root mix | **FullBody only** |
| constraint_anchor | 1.5 | **2.0** |
| root_xz / contact / skate | 1 / 0.5 / 1 | **0 / 0 / 0** |
| batch / accum | 4 / 2 | **4 / 2（保留）** |

```bash
RP=~/.cache/huggingface/hub/models--nvidia--Kimodo-G1-RP-v1/snapshots/3020ad8c419c244e0429d360163730c63c4ed011
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m kimodo.distill.scripts.train_pd \
  --local --formal --stage 100to50 --text-mode encoder \
  --teacher-checkpoint "$RP" --teacher-dtype fp32 \
  --data-root datasets/bones-seed \
  --split-path datasets/kimodo-benchmark/splits/train_split_paths.txt \
  --stats-path checkpoints/Kimodo-G1-SEED-v1.hf/stats/motion \
  --output-dir outputs/pd_g1_rp_teacher_seed/stage_100to50_formal_cons_bs4 \
  --device cuda \
  --wandb --wandb-project kimodo-pd-g1 \
  --wandb-run-name pd-100to50-formal-cons-bs4
```

验收：

```bash
python -m kimodo.distill.scripts.eval_pd_vs_rp \
  --student-checkpoint outputs/pd_g1_rp_teacher_seed/stage_100to50_formal_cons_root_ee/step_8000 \
  --baseline-checkpoint "$RP" --steps 50 --device cuda
```

### 模型测试 / 无头评估

```bash
export TEXT_ENCODER_DEVICE=cpu
kimodo_demo \
  --model g1-rp \
  --checkpoint outputs/pd_g1_rp_teacher_seed/stage_24to12_best/step_8000 \
  --examples-dir kimodo/assets/demo/examples/kimodo-g1-rp

python -m kimodo.distill.scripts.eval_pd_vs_rp \
  --student-checkpoint outputs/pd_g1_rp_teacher_seed/stage_24to12_best/step_8000 \
  --baseline-model g1-rp --baseline-checkpoint "$RP" --steps 12 --device cuda

# 25 step (baseline):
# | `constraint_root2d_err` | 0.1036 | 0.0733 | -0.0303 |
# | `constraint_root2d_acc` | 0.4296 | 0.7901 | +0.3605 |
# | `constraint_fullbody_keyframe` | 0.1203 | 0.0280 | -0.0922 |
# | `constraint_end_effector` | 0.0587 | 0.0341 | -0.0247 |

# stage2：
# | `constraint_root2d_err` | 0.1036 | 0.0680 | -0.0356 |
# | `constraint_root2d_acc` | 0.4296 | 0.8066 | +0.3771 |
# | `constraint_fullbody_keyframe` | 0.1203 | 0.0233 | -0.0970 |
# | `constraint_end_effector` | 0.0587 | 0.0377 | -0.0210 |

# stage3：

# 12 steps (baseline)：
# | `constraint_root2d_err` | 0.2956 | 0.1384 | -0.1573 |
# | `constraint_root2d_acc` | 0.2804 | 0.3135 | +0.0331 |
# | `constraint_fullbody_keyframe` | 0.3062 | 0.0733 | -0.2328 |
# | `constraint_end_effector` | 0.3482 | 0.1483 | -0.1999 |

# stage2：
# | `constraint_root2d_err` | 0.2956 | 0.1091 | -0.1865 |
# | `constraint_root2d_acc` | 0.2804 | 0.5939 | +0.3135 |
# | `constraint_fullbody_keyframe` | 0.3062 | 0.0419 | -0.2643 |
# | `constraint_end_effector` | 0.3482 | 0.0924 | -0.2558 |

# stage3：
# | `constraint_root2d_err` | 0.2956 | 0.0996 | -0.1960 |
# | `constraint_root2d_acc` | 0.2804 | 0.5546 | +0.2742 |
# | `constraint_fullbody_keyframe` | 0.3062 | 0.0247 | -0.2815 |
# | `constraint_end_effector` | 0.3482 | 0.0813 | -0.2669 |


# 6 steps:
# stage2:
# | `constraint_root2d_err` | 0.9664 | 0.3557 | -0.6107 |
# | `constraint_root2d_acc` | 0.1637 | 0.3163 | +0.1526 |
# | `constraint_fullbody_keyframe` | 1.0109 | 0.2030 | -0.8079 |
# | `constraint_end_effector` | 1.2966 | 0.7568 | -0.5398 |

# stage3:
# | `constraint_root2d_err` | 0.9664 | 0.2899 | -0.6765 |
# | `constraint_root2d_acc` | 0.1637 | 0.3191 | +0.1554 |
# | `constraint_fullbody_keyframe` | 1.0109 | 0.1269 | -0.8839 |
# | `constraint_end_effector` | 1.2966 | 0.6150 | -0.6816 |

# stage4:
# | `constraint_root2d_err` | 0.9664 | 0.1156 | -0.8508 |
# | `constraint_root2d_acc` | 0.1637 | 0.5849 | +0.4213 |
# | `constraint_fullbody_keyframe` | 1.0109 | 0.0394 | -0.9715 |
# | `constraint_end_effector` | 1.2966 | 0.2104 | -1.0861 |
```

## 7.17
```bash
# 在stage1配置基础上修改配置训练stage2：
STAGE1=outputs/pd_g1_rp_teacher_seed/stage_100to50_best/step_30000
python -m kimodo.distill.scripts.train_pd \
  --local --recipe pd_g1_stage2_50to25_gentle_local --stage 50to25 \
  --teacher-checkpoint "$STAGE1" --teacher-dtype fp32 \
  --data-root datasets/bones-seed \
  --split-path datasets/kimodo-benchmark/splits/train_split_paths.txt \
  --stats-path checkpoints/Kimodo-G1-SEED-v1.hf/stats/motion \
  --output-dir outputs/pd_g1_rp_teacher_seed/stage_50to25_gentle \
  --device cuda --wandb --wandb-project kimodo-pd-g1 \
  --wandb-run-name pd-50to25-gentle
# 扫 run 内所有 step_*，汇总并选出最优 ckpt：
python -m kimodo.distill.scripts.eval_pd_run \
  --run-dir outputs/pd_g1_rp_teacher_seed/stage_12to6_cons_rerun \
  --baseline-checkpoint "$RP" --steps 6 --device cuda
# → outputs/pd_eval/stage_50to25_gentle_sweep_s25/{leaderboard.md,ranking.json}

# Stage-3：24→12（gentle），teacher = Stage-2 best step_10000
STAGE2=outputs/pd_g1_rp_teacher_seed/stage_50to25_best/step_10000
python -m kimodo.distill.scripts.train_pd \
  --local --recipe pd_g1_stage3_24to12_gentle_local --stage 24to12 \
  --teacher-checkpoint "$STAGE2" --teacher-dtype fp32 \
  --data-root datasets/bones-seed \
  --split-path datasets/kimodo-benchmark/splits/train_split_paths.txt \
  --stats-path checkpoints/Kimodo-G1-SEED-v1.hf/stats/motion \
  --output-dir outputs/pd_g1_rp_teacher_seed/stage_24to12_gentle \
  --device cuda --wandb --wandb-project kimodo-pd-g1 \
  --wandb-run-name pd-24to12-gentle
# 或: bash scripts/run_pd_stage3_24to12_gentle_local.sh
# 详记: devlog/logs/7.17/stage1_stage2_summary_and_stage3_plan.md

# Stage-4：12→6（gentle），teacher = Stage-3 best step_8000
STAGE3=outputs/pd_g1_rp_teacher_seed/stage_24to12_best/step_8000
python -m kimodo.distill.scripts.train_pd \
  --local --recipe pd_g1_stage4_12to6_gentle_local --stage 12to6 \
  --teacher-checkpoint "$STAGE3" --teacher-dtype fp32 \
  --data-root datasets/bones-seed \
  --split-path datasets/kimodo-benchmark/splits/train_split_paths.txt \
  --stats-path checkpoints/Kimodo-G1-SEED-v1.hf/stats/motion \
  --output-dir outputs/pd_g1_rp_teacher_seed/stage_12to6_gentle \
  --device cuda --wandb --wandb-project kimodo-pd-g1 \
  --wandb-run-name pd-12to6-gentle
# 或: bash scripts/run_pd_stage4_12to6_gentle_local.sh

# Stage-4 cons 复训（与历史 peak step_4000 同配方；勿覆盖旧 stage_12to6_cons）
# 启动应见: PD stage 12->6 | w_cons=2.5 w_root=1.5 w_ee=2 | lr=1.5e-6 | max_steps=12000
bash scripts/run_pd_stage4_12to6_cons_local.sh
# 默认 OUT=outputs/pd_g1_rp_teacher_seed/stage_12to6_cons_rerun
# 对比基线: stage_12to6_cons/step_4000 @ --steps 6
```
