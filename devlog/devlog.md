# 7.3
## 环境安装：
```bash
uv python install 3.10
uv python pin 3.10
uv venv --python 3.10

source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
# 测试：
# python -c "
# import torch
# print('PyTorch:', torch.__version__)
# print('Built with CUDA:', torch.version.cuda)
# print('CUDA available:', torch.cuda.is_available())
# if torch.cuda.is_available():
#     print('GPU:', torch.cuda.get_device_name(0))
# "
# 预期输出：
# PyTorch: 2.x.x+cu126
# Built with CUDA: 12.6
# CUDA available: True
# GPU: NVIDIA ...

uv pip install -e ".[all]"

# 访问 https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct 注册 Hugging Face 账号，获取访问令牌（token），提交访问申请并登录（使用非中文账号）：
hf auth login --force
# 输入 token （read权限）

# 测试：直接运行 'kimodo_demo' 显存不足，使用cpu进行测试
export TEXT_ENCODER_DEVICE=cpu
kimodo_demo
```


## 数据集准备：
```bash
# 访问 https://huggingface.co/datasets/bones-studio/seed 填写申请信息得到许可

# 创建数据集目录并下载数据集：
mkdir -p datasets/bones-seed
cd datasets/bones-seed

# 完整仓库含 soma_uniform.tar.gz、soma_proportional.tar.gz、g1.tar.gz。我们只需 G1 包和 metadata：
hf download bones-studio/seed \
  --repo-type dataset \
  --local-dir . \
  --include "g1.tar.gz" \
  --include "metadata/*" \
  --include "LICENSE.md" \
  --include "README.md"

# 使用tmux在后台下载：
# 新建名为 download 的会话
tmux new -s download
# 在 tmux 里正常执行下载（建议只下 g1，metadata 已有可跳过）
cd ~/PyProject/kimodo/datasets/bones-seed
source ~/PyProject/kimodo/.venv/bin/activate
hf download bones-studio/seed \
  --repo-type dataset \
  --local-dir . \
  --include "g1.tar.gz" \
  --include "metadata/*" \
  --include "LICENSE.md" \
  --include "README.md"
# 重连 tmux 会话：
tmux attach -t download

# 验证：
## 确认 tar 包存在且体积正确（约 23.5G）
ls -lh ~/PyProject/kimodo/datasets/bones-seed/g1.tar.gz
# 解压并统计 CSV 数量
cd ~/PyProject/kimodo/datasets/bones-seed
tar -xzf g1.tar.gz
find g1/csv -name "*.csv" | wc -l   # 应接近 142220
```

## H100服务器连接配置：
```bash
# 查看 H100 服务器资源占用情况：
# │  gpu-monitor   查看 H100 占用 / 释放卡
# │  run-mjlab      扫描空闲 GPU 并启动 mjlab 训练拉 GPU 利用率
watch -n 1 nvidia-smi
nvitop
nvtop
```

# 7.7
<!-- prompt1: @devlog/devlog.md 我现在已经按照这个文档中的记录配置了相应的uv环境并下载了相应的训练数据，现在我想要在本地对当前的pipeline进行最小化修改以实现替换当前数据处理流程中使用到的DDPM和DDIM部分为Flow Matching。请先详细分析当前训练流程中的相关代码文件，分析并制定相应的修改计划的md文档到/home/zhengjk/PyProject/kimodo/devlog/logs/7.7文件夹中。
经过我现在的调研，Flow Matching也许可以用现在已有的python包torchcfm来实现，这样就能减少我们自己编写的代码量来降低错误率。

prompt2: 按照你编写的文档中的分析，我们当前如果要实现flow matching的pipeline，就主要还需要编写flow matching的实现、训练脚本的实现、以及对应模型调用时推理部分的替换这三个部分对吧？
我现在想要在自己复训模型的过程中，简化模型能力（不需要文本推理的功能，仅需要关键帧约束推理的功能），这样进行训练是否能简化模型的训练难度？另外应该在后续调用模型时也能增加一些推理速度吧？

请直接按照你进行的代码修改计划对当前代码文件进行最小化修改（尽量复用已有代码内容以及已经实现好的如torchcfm这样的开源代码实现），请在编写完成后进行小规模自动化训练测试确保训练过程能够正常进行（在本地uv环境下跑，如需补充配置一些新的环境请进行记录），之后编写相应的开发日志到/home/zhengjk/PyProject/kimodo/devlog/logs/7.7文件夹中 -->

## 安装训练环境：
```bash
uv pip install torchcfm
# 或：uv pip install -e ".[train]"
```

<!-- prompt3:那我使用当前实现的代码进行训练就能训练出一个和官方模型相同（除了不支持文本输入、架构中DDPM/DDIM替换成了Flow-matching）的模型了嘛？完整的训练代码有实现嘛？ 

请继续实现完整的训练流程（参数配置尽量与官方给出的配置相同，便于复现），文本输入的支持如果现在还没有去除就不要去除了，最终实现为和官方模型除了flow-matching部分的替换其他完全相同的训练pipeline。另外请进行冒烟测试确保实现的代码能正常跑通。
在编写代码时尽量调用相关工具包和复用已有代码，自研代码尽量少且需确保逻辑和实现的正确性。 -->

```bash
# 无文本环境训练命令
# 冒烟测试（CPU，小模型，8 步）
cd /home/zhengjk/PyProject/kimodo
source .venv/bin/activate
python -m kimodo.train.scripts.train_fm \
  --smoke \
  --no-text \
  --device cpu \
  --max-files 4
# 全量训练（官方尺度 denoiser）
python -m kimodo.train.scripts.train_fm \
  --no-text \
  --data-root datasets/bones-seed \
  --output-dir outputs/fm_g1_seed_no_text \
  --device cuda

# 有文本：
python -m kimodo.train.scripts.train_fm \
  --data-root datasets/bones-seed \
  --output-dir outputs/fm_g1_seed \
  --device cuda \
  --text-mode encoder

# 下载官方 train split
hf download nvidia/Kimodo-Motion-Gen-Benchmark \
  splits/train_split_paths.txt \
  --repo-type dataset \
  --local-dir datasets/kimodo-benchmark

# 训练时指定
python -m kimodo.train.scripts.train_fm \
  --data-root datasets/bones-seed \
  --split-path datasets/kimodo-benchmark/splits/train_split_paths.txt \
  --output-dir outputs/fm_g1_seed \
  --device cuda \
  --text-mode encoder   # 有文本训练时

# 无文本：
python -m kimodo.train.scripts.train_fm \
  --no-text \
  --data-root datasets/bones-seed \
  --split-path datasets/kimodo-benchmark/splits/train_split_paths.txt \
  --output-dir outputs/fm_g1_seed_no_text \
  --device cuda
```

## 在服务器上进行训练：
```bash
cd /data_sjy/jf
git clone 

# 按照之前的日志配置环境：

```