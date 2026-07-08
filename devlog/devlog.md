# 7.3

## 配置提交信息

```bash
git remote set-url origin https://azf-369@github.com/azf-369/kimodo.git

git push origin main

# 配置全局用户名
git config user.name "azf-369"

# 配置全局邮箱
git config user.email "azf-157@sjtu.edu.cn"
```

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



# 7.7 - 7.8



## 安装训练环境：

```bash
uv pip install torchcfm
# 或：uv pip install -e ".[train]"
```

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

# 本地训练
tmux new -s train
cd /home/zhengjk/PyProject/kimodo
source .venv/bin/activate
conda deactivate

tmux attach -t train

python -m kimodo.train.scripts.train_fm \
  --no-text \
  --data-root datasets/bones-seed \
  --split-path datasets/kimodo-benchmark/splits/train_split_paths.txt \
  --output-dir outputs/fm_g1_seed_no_text \
  --device cuda

# 增加wandb训练图片绘制：
uv pip install wandb
# 或
uv pip install -e ".[train]"

wandb login

python -m kimodo.train.scripts.train_fm \
  --no-text \
  --data-root datasets/bones-seed \
  --split-path datasets/kimodo-benchmark/splits/train_split_paths.txt \
  --output-dir outputs/fm_g1_seed_no_text \
  --device cuda \
  --wandb \
  --wandb-project kimodo-fm-g1 \
  --wandb-run-name g1-notext-local
```



## 在服务器上进行训练：

```bash
cd /data_sjy/jf
git clone https://github.com/azf-369/kimodo.git

# 按照之前的日志配置环境：
uv python install 3.10
uv python pin 3.10
uv venv --python 3.10

source .venv/bin/activate
# uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
UV_HTTP_TIMEOUT=300 uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 2. 用独立缓存 + 更长超时
# export UV_CACHE_DIR=/data_sjy/jf/kimodo/.uv-cache
# export UV_HTTP_TIMEOUT=300
# export SKIP_MOTION_CORRECTION_IN_SETUP=1   # 跳过 CMake 编译，加快安装

uv pip install -e ".[all]"

# 访问 https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct 注册 Hugging Face 账号，获取访问令牌（token），提交访问申请并登录（使用非中文账号）：
hf auth login --force
# 输入 token （read权限）

# 创建数据集目录并下载数据集：
mkdir -p datasets/bones-seed
cd datasets/bones-seed

# 使用tmux在后台下载：
# 新建名为 download 的会话
tmux new -s download1
# 在 tmux 里正常执行下载（建议只下 g1，metadata 已有可跳过）
cd datasets/bones-seed
source /data_sjy/jf/kimodo/.venv/bin/activate
hf download bones-studio/seed \
  --repo-type dataset \
  --local-dir . \
  --include "g1.tar.gz" \
  --include "metadata/*" \
  --include "LICENSE.md" \
  --include "README.md"
# 重连 tmux 会话：
tmux attach -t download1

## 确认 tar 包存在且体积正确（约 23.5G）
ls -lh /data_sjy/jf/kimodo/datasets/bones-seed/g1.tar.gz
# 解压并统计 CSV 数量
cd /data_sjy/jf/kimodo/datasets/bones-seed
tar -xzf g1.tar.gz
find g1/csv -name "*.csv" | wc -l   # 应接近 142220

uv pip install torchcfm
uv pip install wandb

wandb login

# 在 kimodo 项目目录下，临时用另一个账号
cd /data_sjy/jf/kimodo
export WANDB_API_KEY="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"

# 验证当前用的是哪个账号
python -c "import wandb; wandb.login(); print(wandb.api.viewer())"

# 下载官方 stats 文件：
cd /data_sjy/jf/kimodo
hf download nvidia/Kimodo-G1-SEED-v1 \
  stats/motion/global_root/mean.npy \
  stats/motion/global_root/std.npy \
  stats/motion/local_root/mean.npy \
  stats/motion/local_root/std.npy \
  stats/motion/body/mean.npy \
  stats/motion/body/std.npy \
  --local-dir checkpoints/Kimodo-G1-SEED-v1

# 下载官方 train split
hf download nvidia/Kimodo-Motion-Gen-Benchmark \
  splits/train_split_paths.txt \
  --repo-type dataset \
  --local-dir datasets/kimodo-benchmark


# 冒烟测试（CPU，小模型，8 步）
cd /data_sjy/jf/kimodo
source .venv/bin/activate
python -m kimodo.train.scripts.train_fm \
  --smoke \
  --no-text \
  --device cpu \
  --max-files 4

tmux new -s train
export http_proxy=http://10.127.48.11:3128/
export https_proxy=http://10.127.48.11:3128/
export HTTP_PROXY=$http_proxy
export HTTPS_PROXY=$https_proxy
export WANDB_API_KEY="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"
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=3
python -m kimodo.train.scripts.train_fm \
  --no-text \
  --data-root datasets/bones-seed \
  --split-path datasets/kimodo-benchmark/splits/train_split_paths.txt \
  --stats-path checkpoints/Kimodo-G1-SEED-v1/stats/motion \
  --output-dir outputs/fm_g1_seed_no_text \
  --device cuda \
  --wandb \
  --wandb-project kimodo-fm-g1 \
  --wandb-run-name g1-notext-server
```

## 代理：
```bash
# # 1、开发机使用apt更新失败
# # 解决方式：输入以下命令指定代理
# export http_proxy=http://10.127.48.4:3128/
# export https_proxy=http://10.127.48.4:3128/

# # 2、gpu使用pip和apt的网络代理连接
# # apt代理使用设置：
# tee /etc/apt/sources.list << 'EOF'
# deb http://10.127.48.3:30081/repository/apt/ jammy main restricted universe multiverse
# deb http://10.127.48.3:30081/repository/apt/ jammy-updates main restricted universe multiverse
# deb http://10.127.48.3:30081/repository/apt/ jammy-backports main restricted universe multiverse
# deb http://10.127.48.3:30081/repository/apt/ jammy-security main restricted universe multiverse
# EOF

# # pip代理使用设置：
# pip config set global.index-url http://10.127.48.3:30081/repository/pip/simple/
# pip config set global.trusted-host 10.127.48.3

# 编译：
cd /data_sjy/jf/kimodo
source .venv/bin/activate
export http_proxy=http://10.127.48.11:3128/
export https_proxy=http://10.127.48.11:3128/
export HTTP_PROXY=$http_proxy
export HTTPS_PROXY=$https_proxy
git config --global http.proxy http://10.127.48.11:3128/
git config --global https.proxy http://10.127.48.11:3128/
export UV_DEFAULT_INDEX=http://10.127.48.3:30081/repository/pip/simple/
export UV_HTTP_TIMEOUT=300
uv pip install -v -e ".[all]"

# 不编译：
export http_proxy=http://10.127.48.11:3128/
export https_proxy=http://10.127.48.11:3128/
export HTTP_PROXY=$http_proxy
export HTTPS_PROXY=$https_proxy
export UV_DEFAULT_INDEX=http://10.127.48.3:30081/repository/pip/simple/
export UV_HTTP_TIMEOUT=300
export SKIP_MOTION_CORRECTION_IN_SETUP=1
uv pip install -e ".[all]"

# 下载数据集：
mkdir -p datasets/bones-seed
cd datasets/bones-seed
export http_proxy=http://10.127.48.11:3128/
export https_proxy=http://10.127.48.11:3128/
export HTTP_PROXY=$http_proxy
export HTTPS_PROXY=$https_proxy
cd /data_sjy/jf/kimodo/datasets/bones-seed
source /data_sjy/jf/kimodo/.venv/bin/activate
hf download bones-studio/seed \
  --repo-type dataset \
  --local-dir . \
  --include "g1.tar.gz" \
  --include "metadata/*" \
  --include "LICENSE.md" \
  --include "README.md"
```