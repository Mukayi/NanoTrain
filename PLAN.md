# NanoTrain Training Engine Plan

## Project Goal

NanoTrain 的目标不是简单复现某篇论文，也不是只写一个能跑的训练脚本，而是做一个真正可扩展的轻量级 Training Engine。

这个项目的核心价值在于：先以 `nanoGPT/` 作为可运行、可对照的训练 baseline，再逐步把模型、训练循环、分布式通信、优化器和 Runtime 抽象到 `nanotrain/` 包中。`nanoGPT` 已经具备 DDP、AMP、checkpoint、learning rate schedule 等基础训练能力，NanoTrain 不需要重复造这些轮子，而是要先复用和封装它们，再把精力放在 Tensor Parallel、ZeRO、Runtime 抽象、Benchmark 等训练框架能力上。最终形成一个结构清晰、可测试、可扩展的 AI Infra 项目。

推荐按照以下阶段推进：

```text
MVP -> Core Parallel Features -> Runtime Infrastructure -> Benchmark -> Extension
```

每个阶段都要有明确的可展示成果，保证项目不是想到什么写什么，而是持续演进。

## Phase 0: Project Preparation

预计时间：1 到 2 天

### Project Setup

- [x] 初始化本地 Git Repo：`NanoTrain`
- [x] 创建 GitHub 远端仓库并 push 初始代码
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

## Phase 1: MVP Baseline And NanoTrain Migration

预计时间：约 1 周

目标：先确认 `nanoGPT/` baseline 能训练、能复现 loss 下降；然后把必要能力迁移成 NanoTrain 的最小可用训练入口。这个阶段不重新实现 DDP、AMP、checkpoint、AdamW、LR scheduler 等 `nanoGPT` 已有能力，只做验证、整理、封装和最小迁移。

### nanoGPT Already Provides

- [x] GPT model：embedding、attention、MLP、LayerNorm、GPT block
- [x] CrossEntropy loss
- [x] AdamW optimizer
- [x] learning rate scheduler
- [x] gradient clipping
- [x] gradient accumulation
- [x] DDP training path
- [x] FP16 / BF16 mixed precision path
- [x] checkpoint save / resume
- [x] Shakespeare character dataset
- [x] OpenWebText data preparation path
- [x] baseline `train.py`
- [x] baseline config override mechanism



### Phase 1 To Implement

- [x] 跑通 `nanoGPT` Shakespeare character baseline，并记录命令、硬件、loss 曲线
  - Command: `python train.py config/train_shakespeare_char.py`
  - Result: iter 5000 completed, train loss 0.6282, val loss 1.6915
  - Observation: training loss kept decreasing while validation loss increased late in training, so the baseline overfits as expected on the small Shakespeare character dataset
- [x] 跑通最小 smoke training 配置，减少 `max_iters`，用于快速验证
  - Command: `python train.py ../configs/nanogpt_shakespeare_char_smoke.py`
  - Result: CPU smoke run completed at iter 20, train loss 3.2556, val loss 3.1504
- [x] 新增 NanoTrain 根目录训练入口，例如 `train.py` 或 `scripts/train_nanogpt_baseline.py`
  - Command: `python train.py --config configs/nanotrain_shakespeare_char_smoke.yaml`
  - Result: iter 20 completed, train loss 3.2556, val loss 3.1504
- [x] 新增 `nanotrain/model/gpt.py`，从 `nanoGPT/model.py` 迁移 GPT 相关代码
- [x] 新增 `nanotrain/config` 的配置加载逻辑，支持 yaml config
- [x] 把 `configs/gpt_single_gpu.yaml` 映射到 nanoGPT 等价训练参数
- [x] 新增 `nanotrain/runtime/trainer.py` 的最小 Trainer 骨架
- [x] Trainer 先调用单卡训练流程，不急着支持 TP/ZeRO
- [x] 保留 `nanoGPT/` 原始代码不改，作为对照 baseline
- [x] 增加单元测试：GPTConfig 构造、model forward shape、loss forward
- [x] 增加 smoke test：极小 batch/sequence 下完成一次 forward/backward



### Model Migration

- [x] 迁移 `GPTConfig`
- [x] 迁移 `LayerNorm`
- [x] 迁移 `CausalSelfAttention`
- [x] 迁移 `MLP`
- [x] 迁移 `Block`
- [x] 迁移 `GPT`
- [x] 保持与 `nanoGPT/model.py` 的参数命名尽量一致，方便加载和对齐
- [x] 添加基础 shape 和数值对齐测试



### Training Migration

- [x] 抽出 batch loading 逻辑
- [x] 抽出 loss 计算逻辑
- [x] 抽出 optimizer builder
- [x] 抽出 scheduler / learning rate 计算逻辑
- [x] 抽出 eval loop
- [x] 抽出 checkpoint manager 的接口，但底层逻辑可先复用 nanoGPT 思路
- [x] 明确哪些能力先复用，哪些后续重构



### Dataset Migration

- [x] 优先支持 `nanoGPT/data/shakespeare_char`
- [x] 复用 `prepare.py` 生成的 `train.bin` / `val.bin`
- [x] 新增 NanoTrain dataset loader
- [x] 后续再考虑 Tiny Shakespeare token-level 或 WikiText2



### Runtime Migration

- [x] 统一入口读取 yaml config
- [x] seed 固定
- [x] device / dtype 配置
- [x] logging 最小可用
- [x] 输出 loss、lr、tokens/s
- [x] 支持保存 config snapshot
- [x] 当前阶段只要求单进程单卡路径清晰



### Validation

- [x] `nanoGPT` baseline loss 能稳定下降
- [x] NanoTrain 迁移后的 GPT forward shape 正确
- [x] 单步 forward / backward 正常
- [x] 小数据集可以 overfit
- [x] 同一配置下，NanoTrain loss 和 `nanoGPT` baseline 在合理范围内接近



### Milestone

`nanoGPT` baseline 可以稳定训练；NanoTrain 拥有最小单卡训练入口、yaml config、GPT 模型迁移和基础测试，loss 能正常下降。

### Completion Summary

Phase 1 已经把 `nanoGPT` 的单卡训练主干迁移到 NanoTrain：新增了 GPT 模型、yaml 配置、Shakespeare char 数据读取、optimizer builder、checkpoint manager、最小 Trainer 和根目录 `train.py`。当前 `nanoGPT` baseline 与 NanoTrain smoke 训练都已跑通，NanoTrain smoke 在 iter 20 达到 train loss 3.2556、val loss 3.1504，并通过 `ruff`、`black` 和 `pytest` 检查。

## Phase 2: DistributedDataParallel Migration

预计时间：约 2 到 3 天

目标：先把 `nanoGPT/train.py` 中已经验证过的 DDP 训练路径迁移到 NanoTrain，作为后续 Tensor Parallel 和 ZeRO 的分布式 runtime 基础。

这一阶段不修改 GPT 模型结构，也不实现 Tensor Parallel。重点是把 `torchrun` 启动、rank / local rank / world size 管理、进程组初始化、DDP wrapper、master rank 日志与 checkpoint、gradient accumulation 同步策略迁移到 `nanotrain/runtime/trainer.py` 和 `nanotrain/distributed/`。

### nanoGPT DDP Patterns To Migrate

- `RANK` / `LOCAL_RANK` / `WORLD_SIZE` 环境变量检测
- `torch.distributed.init_process_group`
- `torch.cuda.set_device(f"cuda:{LOCAL_RANK}")`
- `master_process = rank == 0`
- `seed_offset = rank`
- `gradient_accumulation_steps //= world_size`
- `DistributedDataParallel(model, device_ids=[local_rank])`
- gradient accumulation 中只在最后一个 micro step 同步梯度
- master rank 才执行日志输出和 checkpoint 保存
- 训练结束后 `destroy_process_group`

### NanoTrain DDP Implementation

- 新增 distributed runtime context，封装 DDP 状态：
  - 是否由 `torchrun` 启动
  - global rank
  - local rank
  - world size
  - master rank 判断
  - DDP device 解析
  - process group 初始化和销毁
- Trainer 初始化时先建立 distributed context
- 单进程路径保持现有行为不变
- DDP 路径下使用 local rank 绑定 CUDA device
- DDP 路径下每个 rank 使用 `seed + rank`
- DDP 路径下要求 `gradient_accumulation_steps` 可以被 `world_size` 整除
- DDP 路径下保存全局 tokens per iteration 统计
- 模型保持现有 `GPT`，在 compile 之后用 DDP 包装
- `raw_model` 能正确 unwrap DDP 和 `torch.compile`
- eval、日志、checkpoint 只在 master rank 执行

### DDP Config And Launch

- 新增 `configs/gpt_ddp_smoke.yaml`
- 使用小模型和短训练步数验证 DDP 路径
- 示例命令：

```bash
torchrun --standalone --nproc_per_node=2 train.py --config configs/gpt_ddp_smoke.yaml
```

### Validation

- 原有单进程 smoke training 继续通过
- `pytest` 继续通过
- DDP smoke config 可以正常加载
- 2 GPU DDP smoke training 可以完成数个 iteration
- 只有 rank 0 输出 eval / log / checkpoint 信息

### Milestone

NanoTrain 支持从单进程训练平滑切换到 `torchrun` DDP 训练，具备后续 Tensor Parallel 和 ZeRO 所需的基础分布式 runtime。

## Phase 3: Tensor Parallel

预计时间：约 2 周

目标：在 DDP runtime 基础上实现项目最核心的训练框架能力：Tensor Parallel。

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

## Phase 4: ZeRO-1 Optimizer State Sharding

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

## Phase 5: Runtime Infrastructure

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

- 封装 `nanoGPT` 已有 save checkpoint 逻辑
- 封装 `nanoGPT` 已有 resume checkpoint 逻辑
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

- 复用 `nanoGPT` 已有 FP16 / BF16 AMP 路径
- 复用 `nanoGPT` 已有 GradScaler 逻辑
- 抽象 dtype / AMP 配置接口
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

## Phase 6: Benchmark And Profiling

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

## Phase 7: Documentation

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

## Phase 8: Future Extensions

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


| Version | Deliverable                                                           |
| ------- | --------------------------------------------------------------------- |
| v0.1    | 基于 `nanoGPT` baseline 完成 NanoTrain 单卡训练入口，loss 正常收敛                         |
| v0.2    | Tensor Parallel，多 GPU loss 与单卡基本一致                                    |
| v0.3    | ZeRO-1 + Activation Checkpointing，显存占用下降并完成对比                         |
| v0.4    | 完整 Runtime，封装 checkpoint、resume、yaml config、AMP、profiling             |
| v0.5    | 完成 benchmark，展示 throughput、memory、scaling 曲线                          |
| v0.6    | 接入 FlashAttention、Sequence Parallel、Pipeline Parallel 或 FSDP 中的一个扩展方向 |




## Development Principles

- 先保证训练正确性，再做性能优化。
- 每个阶段都要有可运行命令、可验证测试和可展示结果。
- Tensor Parallel 是项目核心，不要只停留在 API 封装。
- 分布式功能必须与单卡 baseline 做数值或 loss 对齐。
- Runtime 要和具体模型解耦，避免把所有逻辑写进 `train.py`。
- Benchmark 要固定模型规模、batch size、sequence length、硬件环境。
- 文档和 benchmark 结果要随着 milestone 更新，形成清晰的项目演进记录。

