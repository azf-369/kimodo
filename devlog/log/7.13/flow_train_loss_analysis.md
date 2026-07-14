# Flow Training Loss 计算分析

## 文件位置

- `kimodo/train/flow_train.py` — 训练流程主逻辑
- `kimodo/model/flow_matching.py` — `FlowMatchingLoss` 类定义

---

## Loss 计算完整链路

### 1. 入口：`flow_matching_batch_step`（flow_train.py:99）

预处理 batch 数据，最终调用 `flow_matching_train_step`。

```
batch ──> x1 (ground truth motion data), pad_mask, text_feat ──> flow_matching_train_step
                                              ↑
                    (可选) CFG dropout ───────┤
                    (可选) motion constraints ─┤
```

### 2. 核心步骤：`flow_matching_train_step`（flow_train.py:49）

```python
t, xt, ut = flow_loss.sample_path(x1)           # (1) OT-CFM 路径采样
xt = apply_motion_constraints(xt, ...)           # (2) 对 xt 施加运动约束（hard inpainting）
v_pred = denoiser(xt, pad_mask, text_feat, t)    # (3) Denoiser 预测速度场
loss = _masked_mse_loss(v_pred, ut, pad_mask)    # (4) 计算 masked MSE loss
```

#### (1) `FlowMatchingLoss.sample_path`（flow_matching.py:58）

```python
def sample_path(self, x1: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    x0 = torch.randn_like(x1)                                       # 采样高斯噪声
    t, xt, ut = self.matcher.sample_location_and_conditional_flow(x0, x1)
    return t, xt, ut
```

- **`x0`**: 从标准正态分布采样的噪声 `N(0, I)`
- **`t`**: 从 `U[0,1]` 采样的时间步，shape `[B]`
- **`xt`**: 噪声与数据之间的线性插值: **`xt = (1 - t) * x0 + t * x1`**，shape `[B, T, D]`
- **`ut`**: 目标速度场 (conditional flow): **`ut = x1 - x0`**，shape `[B, T, D]`

默认使用 `ExactOptimalTransportConditionalFlowMatcher` (OT-CFM)，OT 匹配通过最优运输对齐 x0 和 x1 的批次配对，减少路径交叉。

#### (2) `apply_motion_constraints`（flow_matching.py:86）

```python
def apply_motion_constraints(x, motion_mask, observed_motion):
    if motion_mask is None or observed_motion is None:
        return x
    return x * (1 - motion_mask) + observed_motion * motion_mask
```

对 xt 中受约束的关节/帧做 hard inpainting，用观测值直接替换。

#### (3) Denoiser 预测速度

```python
v_pred = denoiser(xt, pad_mask, text_feat, text_pad_mask, t,
                  first_heading_angle=..., motion_mask=..., observed_motion=...)
```

Denoiser 接收加噪后的 motion `xt` 和时间 `t`，结合文本特征，预测速度场 `v_pred`，shape `[B, T, D]`。

#### (4) `_masked_mse_loss`（flow_train.py:41）

```python
def _masked_mse_loss(pred: Tensor, target: Tensor, pad_mask: Tensor) -> Tensor:
    frame_mask = pad_mask.unsqueeze(-1).expand_as(pred)
    if not frame_mask.any():
        return F.mse_loss(pred, target)
    return F.mse_loss(pred[frame_mask], target[frame_mask])
```

- `pad_mask` shape `[B, T]`，值为 `True` 的位置表示有效帧，`False` 为 padding
- 将 `pad_mask` 扩展到 `[B, T, D]`，与 pred/target 对齐
- **只对有效帧（非 padding 帧）计算 MSE**
- 如果全部为 padding（边界情况），回退到完整 MSE

---

## Loss 数学公式

最终 loss 即为 **OT-CFM 的 Masked MSE Loss**：

$$ \mathcal{L} = \mathbb{E}_{t \sim U[0,1], x_0 \sim \mathcal{N}(0,I), x_1 \sim \text{data}} \left[ \frac{1}{|M|} \sum_{(i,j) \in M} \| v_\theta(x_t^{(i,j)}, t, \text{text}) - (x_1^{(i,j)} - x_0^{(i,j)}) \|^2 \right] $$

其中：

- $x_t = (1-t) \cdot x_0 + t \cdot x_1$（线性插值路径）
- $v_\theta$ 是 denoiser 网络输出的预测速度
- $M$ 是 pad_mask 标记的有效帧集合（排除 padding）
- 可选：`apply_motion_constraints` 对 xt 施加 hard inpainting，`CFG dropout` 对 text/constraint 进行随机丢弃

---

## 训练流程总图

```
flow_matching_batch_step
  │
  ├── _randomize_batch_heading      ← 随机旋转 heading 做数据增强
  ├── text_provider.encode          ← 获取文本特征
  ├── sample_training_constraints   ← 采样运动约束 (keyframe mask + observed)
  ├── apply_separated_cfg_dropout   ← CFG dropout
  │
  └── flow_matching_train_step
        │
        ├── flow_loss.sample_path(x1)        ──→  t, xt, ut
        ├── apply_motion_constraints(xt)     ──→  对 xt 做硬约束
        ├── denoiser(xt, t, text_feat, ...)  ──→  v_pred
        └── _masked_mse_loss(v_pred, ut, pad_mask)  ──→  loss
```

## 关键要点

| 要素 | 说明 |
|------|------|
| Loss 类型 | Masked MSE Loss |
| 目标值 ut | `x1 - x0`，即数据与噪声的差值 |
| 预测值 v_pred | Denoiser 网络对速度场的估计 |
| 加噪路径 xt | `(1-t) * x0 + t * x1`，线性插值 |
| 时间采样 t | `U[0, 1]`，通过 torchcfm 自动采样 |
| 对齐方式 | OT-CFM (Exact Optimal Transport)，减少路径交叉 |
| Mask 处理 | 只对 pad_mask=True 的有效帧计算 MSE |
| 运动约束 | Hard inpainting：用观测值直接覆盖 xt 的对应维度 |
| CFG Dropout | 随机丢弃 text/constraint 条件，支持 classifier-free guidance |
