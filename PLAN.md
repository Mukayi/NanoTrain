# NanoTrain Training Engine Plan

## Project Goal

NanoTrain 的目标不是简单复现某篇论文，也不是只写一个能跑的训练脚本，而是做一个真正可扩展的轻量级 Training Engine。

这个项目的核心价值在于：先以 `nanoGPT/` 作为可运行、可对照的训练 baseline，再逐步把模型、训练循环、分布式通信、优化器和 Runtime 抽象到 `nanotrain/` 包中。后续继续实现 Tensor Parallel、ZeRO、Checkpoint、Mixed Precision、Benchmark 等训练框架能力，最终形成一个结构清晰、可测试、可扩展的 AI Infra 项目。

推荐按照以下阶段推进：

```text
MVP -> Core Parallel Features -> Runtime Infrastructure -> Benchmark -> Extension
```

每个阶段都要有明确的可展示成果，保证项目不是想到什么写什么，而是持续演进。

## Phase 0: Project Preparation

预计时间：1 到 2 天

### Project Setup

- [x] 初始化本地 Git Repo：`NanoTrain`
- [ ] 创建 GitHub 远端仓库并 push 初始代码
- [x] 配置 `.gitignore`
- [x] 保留 `nanoGPT/` 作为 baseline/reference implementation
- [x] 设计整体目录结构
- [x] 配置 `pre-commit`
- [x] 配置 `black`
- [x] 配置 `ruff`
- [x] 配置 `pytest`
- [x] 编写基础 `README.md`
- [x] 配置 Conda / Docker 环境
- [x] 准备最小本地检查命令

### Basic Knowledge

- 熟悉 `torch.distributed`
- 熟悉 `ProcessGroup`
- 熟悉 NCCL Backend
- 熟悉 `torchrun` 启动方式
- 熟悉 DistributedDataParallel 的基本概念
- 熟悉 PyTorch autograd 与自定义通信算子的关系

### Milestone

项目结构初始化完成，`nanoGPT/` baseline 保持可参考，开发环境可复现，能够运行最小单元测试。

## Phase 1: MVP Single-GPU GPT Training

预计时间：约 1 周

目标：先跑通 `nanoGPT/` 单卡 baseline，确认 loss 能正常下降；然后逐步把模型定义、配置和训练循环迁移到 `nanotrain/` 包中。

### Model

- Token Embedding
- Position Embedding 或 Rotary Position Embedding
- Multi-Head Attention
- MLP
- LayerNorm / RMSNorm
- GPT Block
- GPT Model
- 权重初始化

### Training

- CrossEntropy Loss
- AdamW
- Learning Rate Scheduler
- Gradient Clipping
- Gradient Accumulation 可选
- Training / Validation loop

### Dataset

- Tiny Shakespeare
- WikiText2
- 简单 tokenizer
- dataloader
- train / val split

### Runtime

- `train.py`
- `config.yaml`
- logging
- seed 固定
- device 管理
- checkpoint 保存最小版本可选

### Validation

- loss 能稳定下降
- 单元测试覆盖模型 shape
- 单步 forward / backward 正常
- 小数据集可以 overfit

### Milestone

`python train.py` 可以在单卡上训练 GPT，loss 正常下降。

## Phase 2: Tensor Parallel

预计时间：约 2 周

目标：实现项目最核心的训练框架能力：Tensor Parallel。

这一阶段不要一开始追求复杂模型，先保证线性层、embedding、attention、MLP 的并行逻辑正确，再集成到 GPT 中。

### Process Group

- 初始化 Tensor Parallel Group
- rank / world size 管理
- global rank 到 TP rank 的映射
- 单机多卡 `torchrun` 启动
- group 内通信封装

### Communication Primitives

- AllReduce
- Broadcast
- AllGather
- ReduceScatter 可选
- 同步梯度
- 通信接口封装到 `distributed` 或 `parallel` 模块

### Tensor Parallel Linear

#### Column Parallel Linear

- forward 切分 output dimension
- backward 梯度正确同步
- 支持 gather output 可选
- 与普通 `nn.Linear` 做数值对齐

#### Row Parallel Linear

- forward 切分 input dimension
- forward 后 AllReduce
- backward 正确传播
- 与普通 `nn.Linear` 做数值对齐

### Parallel Embedding

- Vocab Parallel Embedding
- embedding weight 按 vocab dimension 切分
- masked input 处理
- AllReduce 合并 embedding output

### Parallel Attention

- QKV projection 并行
- attention heads 按 TP rank 切分
- output projection 并行
- causal mask 正确
- 与单卡 attention logits 对齐

### Parallel MLP

- FC1 使用 Column Parallel Linear
- FC2 使用 Row Parallel Linear
- activation 保持一致
- 与单卡 MLP 输出对齐

### Integration

- Tensor Parallel GPT Block
- Tensor Parallel GPT Model
- 支持 `tp_size=1`
- 支持 `tp_size=2`
- 支持配置切换普通模型与 TP 模型

### Validation

- 1 GPU 与 2 GPU loss 对齐
- 同样 seed 下 logits 对齐
- 单步 forward / backward 梯度对齐
- 小模型短序列训练 loss 曲线接近

### Milestone

使用 2 GPU Tensor Parallel 训练时，loss 与单卡训练基本一致。

## Phase 3: ZeRO-1 Optimizer State Sharding

预计时间：约 1 周

目标：实现 ZeRO-1 的核心思想：切分 optimizer state，降低显存占用。

### Optimizer

- Optimizer state sharding
- AdamW state 分片
- 每个 rank 只维护部分 optimizer state
- 参数更新前后同步

### Parameter Management

- 参数分片映射
- 参数到 optimizer state 的映射
- rank-local optimizer step
- 更新后广播或同步参数

### Communication

- optimizer step 后同步参数
- 必要时 broadcast updated parameters
- 验证不同 rank 参数一致

### Memory Validation

- 统计 model parameter memory
- 统计 gradient memory
- 统计 optimizer state memory
- 对比 without ZeRO 与 with ZeRO

### Tests

- ZeRO-1 与普通 AdamW 单步更新对齐
- 多步训练 loss 接近
- 参数在不同 rank 上保持一致
- 显存占用下降有可观测数据

### Milestone

ZeRO-1 可以正常训练，并展示 optimizer state 显存占用下降。

## Phase 4: Runtime Infrastructure

预计时间：约 1 周

目标：把项目从一组功能模块整理成一个真正可复用的训练引擎。

### Engine

- `Trainer`
- `TrainingEngine`
- `ModelBuilder`
- `OptimizerBuilder`
- `DataBuilder`
- 训练状态管理

### Scheduler

- epoch-based training
- step-based training
- warmup
- cosine decay
- resume 后 scheduler state 恢复

### Checkpoint

- save checkpoint
- resume checkpoint
- model state
- optimizer state
- scheduler state
- random state
- distributed checkpoint layout 可选

### Activation Checkpointing

- Transformer Block checkpoint
- 配置是否开启
- 对比开启前后显存
- 验证 loss 不受影响

### Mixed Precision

- FP16 AMP
- BF16 AMP
- GradScaler
- overflow 处理
- 与 FP32 loss 做基本对比

### Config

- yaml config
- model config
- train config
- distributed config
- optimizer config
- dataset config

### CLI

- `train.py`
- `benchmark.py`
- 支持 `torchrun`
- 支持命令行覆盖 config

### Logging

- step loss
- learning rate
- grad norm
- tokens/s
- GPU memory
- checkpoint path

### Milestone

形成完整 Runtime：配置、训练、checkpoint、resume、AMP、activation checkpointing 都可以通过统一入口使用。

## Phase 5: Benchmark And Profiling

预计时间：约 3 天

目标：用数据证明训练引擎的能力，而不是只展示代码。

### Throughput

- tokens/s
- samples/s
- step time
- forward time
- backward time
- optimizer step time

### Memory

- peak GPU memory
- parameter memory
- gradient memory
- optimizer state memory
- activation memory 粗略估计

### Communication

- NCCL communication time
- AllReduce time
- Broadcast time
- Tensor Parallel communication overhead

### Scaling

测试配置：

- 1 GPU
- 2 GPU
- 4 GPU，如果条件允许

对比内容：

- 单卡 baseline
- Tensor Parallel
- Tensor Parallel + ZeRO-1
- Activation Checkpoint on / off
- AMP on / off

### Outputs

- Throughput curve
- Memory curve
- Scaling curve
- Communication overhead table

### Milestone

完成可复现 benchmark，并能在 README 中展示吞吐、显存和扩展性结果。

## Phase 6: Documentation

预计时间：约 2 天

目标：让项目具备完整工程展示能力。

### README

介绍：

- Motivation
- Architecture
- Features
- Quick Start
- Benchmark Results
- Roadmap

### Architecture

整体结构：

```text
Trainer
  |
Runtime
  |
Parallel
  |
Communication
  |
Model
```

### API Documentation

重点说明：

- `ColumnParallelLinear`
- `RowParallelLinear`
- `VocabParallelEmbedding`
- `Trainer`
- `TrainingEngine`
- `CheckpointManager`

### Examples

单卡训练：

```bash
python train.py --config configs/gpt_single_gpu.yaml
```

Tensor Parallel 训练：

```bash
torchrun --nproc_per_node=2 train.py --config configs/gpt_tp.yaml
```

ZeRO-1 训练：

```bash
torchrun --nproc_per_node=2 train.py --config configs/gpt_tp_zero1.yaml
```

### Milestone

README、架构图、API 说明、训练示例和 benchmark 结果完整，可以作为一个独立 AI Infra 项目展示。

## Phase 7: Future Extensions

这些内容不需要在前期完成，但要保证架构可以自然扩展。

### Parallel Training

- Sequence Parallel
- Pipeline Parallel
- ZeRO-2
- ZeRO-3
- FSDP
- Context Parallel

### Kernel Optimization

- FlashAttention 接入
- FlashAttention Kernel 自己实现
- Fused LayerNorm / RMSNorm
- Fused AdamW
- Triton kernel

### Precision

- BF16
- FP8
- Transformer Engine
- mixed precision policy

### MoE

- Expert Parallel
- Token dispatch
- Load balancing
- MoE all-to-all communication

### Performance

- Communication overlap
- Gradient bucket
- CUDA Graph
- Async checkpoint
- distributed dataloader optimization

## Recommended Directory Structure

```text
NanoTrain/
├── nanotrain/
│   ├── model/
│   ├── parallel/
│   ├── optimizer/
│   ├── runtime/
│   ├── distributed/
│   ├── checkpoint/
│   ├── data/
│   ├── config/
│   └── utils/
├── benchmark/
├── examples/
├── tests/
├── docs/
├── configs/
├── scripts/
├── train.py
├── benchmark.py
├── pyproject.toml
└── README.md
```

## Milestone Roadmap

| Version | Deliverable |
| --- | --- |
| v0.1 | GPT 单卡训练，loss 正常收敛 |
| v0.2 | Tensor Parallel，多 GPU loss 与单卡基本一致 |
| v0.3 | ZeRO-1 + Activation Checkpointing，显存占用下降并完成对比 |
| v0.4 | 完整 Runtime，支持 checkpoint、resume、yaml config、AMP、profiling |
| v0.5 | 完成 benchmark，展示 throughput、memory、scaling 曲线 |
| v0.6 | 接入 FlashAttention、Sequence Parallel、Pipeline Parallel 或 FSDP 中的一个扩展方向 |

## Development Principles

- 先保证训练正确性，再做性能优化。
- 每个阶段都要有可运行命令、可验证测试和可展示结果。
- Tensor Parallel 是项目核心，不要只停留在 API 封装。
- 分布式功能必须与单卡 baseline 做数值或 loss 对齐。
- Runtime 要和具体模型解耦，避免把所有逻辑写进 `train.py`。
- Benchmark 要固定模型规模、batch size、sequence length、硬件环境。
- 文档和 benchmark 结果要随着 milestone 更新，形成清晰的项目演进记录。

