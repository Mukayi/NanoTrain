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
- [x] 新增通用 `BinTokenDataset`，支持 nanoGPT-style `train.bin` / `val.bin` token 数据格式
- [x] 支持 `data.dataset: openwebtext` / `bin_token`，为 GPT-2 124M + OpenWebText benchmark 做准备
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

### Implementation Status

- [x] 新增 Megatron-LM 风格通信映射：copy、scatter、gather、reduce tensor-parallel region
- [x] 新增 `ColumnParallelLinear`
- [x] 新增 `RowParallelLinear`
- [x] 新增 `VocabParallelEmbedding`
- [x] 新增 vocab-parallel LM head，并在当前 MVP 中 gather logits 后复用普通 cross entropy
- [x] 新增 `TensorParallelGPT`，保留原 `GPT` baseline 不变
- [x] Trainer 支持 `distributed.tp_size > 1` 时切换到纯 TP 模型
- [x] 支持 2 GPU pure Tensor Parallel smoke training
- [x] 保留原单进程和 DDP 路径可运行
- [ ] TP checkpoint save/resume：等待 Phase 5 distributed checkpoint layout
- [ ] Hybrid TP + DDP：后续扩展

## Phase 4: ZeRO-1 And ZeRO-2 MVP

预计时间：约 1 周

目标：先实现 ZeRO-1 的核心思想：切分 optimizer state，降低显存占用；随后实现 ZeRO-2 MVP，在 DDP 场景下把梯度 reduce 到参数 owner rank，降低每个 rank 需要保留的梯度范围。

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
- ZeRO-2 backward 后将每个参数的梯度 reduce 到 owner rank
- ZeRO-2 非 owner rank 清空对应梯度



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
- ZeRO-2 smoke training loss 能正常下降
- `zero_stage: 0`、DDP、TP、ZeRO-1 路径不被 ZeRO-2 改动破坏



### Milestone

ZeRO-1 可以正常训练并展示 optimizer state 显存占用下降；ZeRO-2 MVP 可以在 DDP 场景下完成梯度分片训练，并保持 loss 正常下降。

### Implementation Status

- [x] 新增 ZeRO-1 AdamW wrapper
- [x] 每个 DDP rank 只为本 rank 负责的参数建立 optimizer state
- [x] 参数按 deterministic round-robin owner rank 分片
- [x] local optimizer step 后从 owner rank broadcast 参数，保持所有 DDP replica 同步
- [x] Trainer 支持 `distributed.zero_stage: 1`
- [x] `zero_stage: 0` 保持原单进程、DDP、TP 路径不变
- [x] 新增 ZeRO-1 smoke config
- [x] 新增 ZeRO-2 MVP：每个参数梯度 reduce 到 owner rank
- [x] ZeRO-2 非 owner rank 清空对应梯度，只在 owner rank 保留梯度并更新参数
- [x] ZeRO-2 跳过 DDP wrapper 的默认 gradient all-reduce，改用手动 gradient reduce
- [x] 新增 ZeRO-2 smoke config
- [x] 验证 DDP smoke、ZeRO-1 smoke、ZeRO-2 smoke 均可运行
- [x] `pytest`、`ruff`、`black --check` 通过
- [ ] ZeRO-1 checkpoint save/resume：等待 Phase 5 distributed checkpoint layout
- [ ] ZeRO-2 checkpoint save/resume：等待 Phase 5 distributed checkpoint layout
- [ ] ZeRO-2 bucket / reduce-scatter / communication overlap：后续优化
- [ ] Hybrid TP + ZeRO：后续扩展

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

### Implementation Status

- [x] 新增 `TrainingEngine` 作为可复用 runtime 入口，`run_training` 通过 engine 执行
- [x] 新增 `DataBuilder`、`ModelBuilder`、`OptimizerBuilder`，Trainer 通过 builder 构造核心组件
- [x] `DataBuilder` 支持 `shakespeare_char`、`bin_token` 和 `openwebtext` 三类 nanoGPT-style 数据入口
- [x] 保留 step-based training，warmup + cosine decay scheduler 由统一配置驱动
- [x] resume 后通过 checkpoint 中的 `iter_num` 恢复 scheduler 进度
- [x] 复用 `nanoGPT` checkpoint 格式保存 model、optimizer、model args、iter、best val loss 和 config
- [x] 新增 checkpoint resume，按 `nanoGPT` 逻辑用 checkpoint 中的关键 model args 恢复模型结构
- [x] checkpoint 保存并恢复 FP16 GradScaler state
- [x] checkpoint 保存并恢复 Python、NumPy、PyTorch 和 CUDA random state
- [x] Trainer 支持通过 `train.init_from: resume` 和可选 `train.resume_path` 从 checkpoint 继续训练
- [x] 保持现有 AMP autocast / GradScaler 路径，并纳入 checkpoint 保存恢复
- [x] GPT 与 TensorParallelGPT 支持 Transformer Block activation checkpointing
- [x] `runtime.activation_checkpointing` 支持 yaml 配置和命令行 override 开关
- [x] `train.py` 支持 `--override section.field=value`
- [x] 新增最小 `benchmark.py`，复用统一 config / override / runtime 入口
- [x] 日志输出 step loss、learning rate、grad norm、tokens/s、GPU memory 和 checkpoint path
- [x] 新增 GPT-2 124M / OpenWebText benchmark config family：`configs/bench_gpt2_owt.yaml`、`configs/bench_gpt2_owt_ddp.yaml`、`configs/bench_gpt2_owt_tp.yaml`、`configs/bench_gpt2_owt_zero.yaml`
- [x] TP / ZeRO distributed checkpoint layout 明确保留为后续扩展项；当前 Phase 5 checkpoint 覆盖单卡和普通 DDP 路径

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
- 4 GPU
- 8 GPU，如果条件允许

对比内容：

- 单卡 baseline
- Tensor Parallel
- DDP
- ZeRO-1
- ZeRO-2 MVP
- Tensor Parallel + ZeRO-1 / ZeRO-2，如果后续支持 hybrid parallel
- Activation Checkpoint on / off
- AMP on / off



### Outputs

- Throughput curve
- Memory curve
- Scaling curve
- Communication overhead table



### Milestone

完成可复现 benchmark，并能在 README 中展示吞吐、显存和扩展性结果。

### Implementation Status

- [x] `benchmark.py` 支持独立 run name、results dir、config override 和 structured summary 输出
- [x] Trainer 写出逐步 `metrics.jsonl`，包含 loss、lr、grad norm、step time、forward time、backward time、optimizer step time、communication time、tokens/s、samples/s
- [x] Trainer 记录 peak GPU memory、step-level peak GPU memory、parameter memory、gradient memory、optimizer state memory、activation memory 粗略估计
- [x] 每个 train step 前 reset CUDA peak memory stats，保证 activation checkpointing / ZeRO 的单步峰值显存可比
- [x] `benchmark.py` summary 增加 `step_peak_gpu_memory_mb`
- [x] 新增 `benchmark/aggregate_results.py`，自动聚合多个 `summary.json`，派生 speedup、scaling efficiency、peak memory reduction 等指标
- [x] 新增 `scripts/run_phase6_benchmarks.sh`，包含 CPU、单卡 GPU、activation checkpointing、2-GPU DDP、2-GPU TP、2-GPU ZeRO-1、2-GPU ZeRO-2 的完整评测命令
- [x] 新增 `scripts/run_scaling_benchmarks.sh`，覆盖 1/2/4/8 GPU DDP scaling
- [x] 新增 `scripts/run_memory_benchmarks.sh`，覆盖 activation checkpoint on/off、DDP、ZeRO-1、ZeRO-2 memory stress
- [x] 新增 `scripts/run_nsys_profiles.sh`，生成 DDP、TP、ZeRO-2 的 Nsight Systems profile
- [x] 实际 Phase 6 早期评测结果已写入 `benchmark/results/phase6/`
- [x] 新一轮 DDP scaling 结果已写入 `benchmark/results/scaling_medium/`
- [x] 新一轮 memory stress 结果已写入 `benchmark/results/memory_stress/`
- [x] 新一轮 Tensor Parallel medium 结果已写入 `benchmark/results/tp_medium/`
- [x] Nsight Systems profile 产物已写入 `benchmark/profiles/`
- [x] 汇总结果已写入各 results 目录下的 `aggregate.json` 和 `aggregate.md`
- [x] 每组评测保留 `summary.json`、`summary.md`、`out/metrics.jsonl` 和 `out/train.log`
- [x] 当前硬件评测环境：NVIDIA RTX A6000 节点；已完成 1/2/4/8 GPU DDP scaling，memory stress 使用 1/2 GPU，TP medium 使用 1/2/4 GPU
- [x] README 与 `benchmark/README.md` 已记录 benchmark 结果、复现实验命令和可直接使用的简历 bullet
- [ ] Tensor Parallel + ZeRO hybrid：当前 runtime 暂不支持 hybrid parallel，留到后续扩展

### Benchmark Configs Added

- `configs/bench_gpt_medium.yaml`：DDP scaling medium GPT 配置
- `configs/bench_gpt_memory_stress.yaml`：activation checkpointing memory stress 配置
- `configs/bench_gpt_tp_medium.yaml`：Tensor Parallel medium GPT 配置
- `configs/bench_gpt_zero_memory.yaml`：DDP / ZeRO memory stress 配置
- `configs/bench_gpt2_owt.yaml`：GPT-2 124M / OpenWebText 单卡配置，对齐 nanoGPT `train_gpt2.py` 的模型规模
- `configs/bench_gpt2_owt_ddp.yaml`：GPT-2 124M / OpenWebText DDP 配置
- `configs/bench_gpt2_owt_tp.yaml`：GPT-2 124M / OpenWebText pure Tensor Parallel 配置
- `configs/bench_gpt2_owt_zero.yaml`：GPT-2 124M / OpenWebText ZeRO 配置

### Benchmark Commands Added

DDP scaling:

```bash
RESULTS_DIR=benchmark/results/scaling_medium \
BENCH_ITERS=1000 \
EVAL_INTERVAL=200 \
EVAL_ITERS=20 \
LOG_INTERVAL=10 \
bash scripts/run_scaling_benchmarks.sh
```

Memory stress:

```bash
RESULTS_DIR=benchmark/results/memory_stress \
BENCH_ITERS=500 \
EVAL_INTERVAL=100 \
EVAL_ITERS=10 \
LOG_INTERVAL=10 \
bash scripts/run_memory_benchmarks.sh
```

Nsight Systems profiles:

```bash
PROFILE_DIR=benchmark/profiles \
PROFILE_ITERS=20 \
EVAL_INTERVAL=1000000 \
EVAL_ITERS=1 \
LOG_INTERVAL=10 \
bash scripts/run_nsys_profiles.sh
```

### Latest Benchmark Results

DDP scaling on `configs/bench_gpt_medium.yaml`:

- 1 GPU BF16 baseline：376,090 tokens/s，mean step time 32.83 ms，step peak GPU memory 517 MB，final val loss 1.7369
- 2 GPU DDP：609,324 tokens/s，1.62x speedup，81.0% scaling efficiency，step peak GPU memory 557 MB，final val loss 1.5860
- 4 GPU DDP：1,124,311 tokens/s，2.99x speedup，74.7% scaling efficiency，step peak GPU memory 556 MB，final val loss 1.5235
- 8 GPU DDP：2,143,187 tokens/s，5.70x speedup，71.2% scaling efficiency，step peak GPU memory 560 MB，final val loss 1.5593

Activation checkpointing on `configs/bench_gpt_memory_stress.yaml`:

- AC off：61,271 tokens/s，mean step time 189.31 ms，step peak GPU memory 1866 MB，final val loss 2.4821
- AC on：49,928 tokens/s，mean step time 267.64 ms，step peak GPU memory 1410 MB，final val loss 2.4813
- 结论：activation checkpointing 将单卡 step peak memory 从 1866 MB 降到 1410 MB，约 24.4% reduction；代价是 throughput 从 61.3K tokens/s 降到 49.9K tokens/s

ZeRO memory stress on `configs/bench_gpt_zero_memory.yaml`:

- 2 GPU DDP baseline：82,687 tokens/s，step peak GPU memory 2183 MB，optimizer state memory 652 MB，gradient memory 326 MB，final val loss 2.4090
- ZeRO-1：65,164 tokens/s，step peak GPU memory 1797 MB，optimizer state memory 270 MB，gradient memory 326 MB，final val loss 2.3802
- ZeRO-2：61,229 tokens/s，step peak GPU memory 1483 MB，optimizer state memory 270 MB，gradient memory 135 MB，final val loss 2.4534
- 结论：ZeRO-1 将 optimizer state memory 从 652 MB 降到 270 MB；ZeRO-2 进一步将 gradient memory 从 326 MB 降到 135 MB，并将 2-GPU peak memory 相比 DDP 降低 32.1%

Tensor Parallel medium on `configs/bench_gpt_tp_medium.yaml`:

- 1 GPU TP baseline：90,705 tokens/s，step peak GPU memory 1256 MB，parameter memory 217 MB，gradient memory 217 MB，optimizer state memory 434 MB，final val loss 2.1618
- 2 GPU TP：79,721 tokens/s，step peak GPU memory 732 MB，parameter memory 109 MB，gradient memory 109 MB，optimizer state memory 218 MB，final val loss 2.2045
- 4 GPU TP：77,019 tokens/s，step peak GPU memory 464 MB，parameter memory 55 MB，gradient memory 55 MB，optimizer state memory 110 MB，final val loss 2.4200
- 结论：TP 明显降低 per-rank model/gradient/optimizer/memory footprint，4-GPU TP 相比 1-GPU baseline step peak memory 下降 63.1%；当前小模型下 throughput 低于单卡，通信和并行开销可被清晰观测

### GPT-2 / OpenWebText Status

- 已确认 `nanoGPT` 原生包含更正式的大模型实验路径：GPT-2 124M 结构和 OpenWebText 数据准备流程
- NanoTrain 已新增 GPT-2 124M / OpenWebText YAML 配置和通用 `BinTokenDataset`
- 当前 workspace 中 `nanoGPT/data/openwebtext/train.bin` 与 `val.bin` 尚未准备，因此 GPT-2/OpenWebText 配置已经 ready，但没有纳入本轮实测数字
- 后续只需先运行 `nanoGPT/data/openwebtext/prepare.py` 生成 `.bin` 文件，即可用 `configs/bench_gpt2_owt*.yaml` 复现实验

### Profiling Notes

- Lightweight profiler 使用 `time.perf_counter()` + `torch.cuda.synchronize()` 包住 forward / backward / optimizer / communication 区间，避免 CUDA async 导致计时提前返回
- `nsys` profile 脚本可生成 `.nsys-rep` / `.qdstrm`，用于查看 CUDA kernel、NCCL collective、CPU launch overhead 与 timeline
- 当前机器的 Nsight Systems / driver 组合在 report import analysis 阶段出现 non-fatal errors，例如 unknown driver API function index；报告文件仍已生成，但应在 GUI 中检查 Diagnostics Summary
- 4-GPU TP run 在 summary 写出后，shutdown 阶段出现 NCCL watchdog warning；本轮 TP 结果可用于观察趋势，但后续正式展示建议重跑确认稳定性

### Resume Bullet Draft

- 基于 `nanoGPT` 重构 NanoTrain 轻量级分布式 GPT 训练引擎，支持 YAML 配置、DDP、Megatron-style Tensor Parallel、ZeRO-1/2 optimizer sharding、activation checkpointing、checkpoint resume 与 benchmark profiling；在 RTX A6000 实验中，8-GPU DDP 达到 2.14M tokens/s 和 5.70x speedup，ZeRO-2 将 2-GPU peak memory 降低 32.1%，4-GPU TP 将 per-rank peak memory 降低 63.1%。

### Verification

- `conda run -n nanotrain pytest`：25 passed
- `conda run -n nanotrain ruff check .`：passed
- `conda run -n nanotrain black --check .`：passed
- IDE diagnostics：no linter errors

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
torchrun --standalone --nproc_per_node=2 train.py --config configs/gpt_tp_smoke.yaml
```

ZeRO-1 训练：

```bash
torchrun --standalone --nproc_per_node=2 train.py --config configs/gpt_zero1_smoke.yaml
```

ZeRO-2 MVP 训练：

```bash
torchrun --standalone --nproc_per_node=2 train.py --config configs/gpt_zero2_smoke.yaml
```



### Milestone

README、架构图、API 说明、训练示例和 benchmark 结果完整，可以作为一个独立 AI Infra 项目展示。

## Phase 8: Future Extensions

这些内容不需要在前期完成，但要保证架构可以自然扩展。

### Parallel Training

- Sequence Parallel
- Pipeline Parallel
- ZeRO-2 bucket / reduce-scatter / communication overlap
- ZeRO-3 parameter sharding
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
| v0.2    | DDP training 迁移完成，支持 `torchrun` smoke training                         |
| v0.3    | Megatron-style Tensor Parallel，多 GPU loss 能正常下降                         |
| v0.4    | ZeRO-1 optimizer-state sharding 与 ZeRO-2 gradient sharding MVP             |
| v0.5    | 完整 Runtime，封装 checkpoint、resume、yaml config、AMP、profiling             |
| v0.6    | 完成 benchmark，展示 throughput、memory、scaling 曲线                          |
| v0.7    | 接入 FlashAttention、Sequence Parallel、Pipeline Parallel 或 FSDP 中的一个扩展方向 |




## Development Principles

- 先保证训练正确性，再做性能优化。
- 每个阶段都要有可运行命令、可验证测试和可展示结果。
- Tensor Parallel 是项目核心，不要只停留在 API 封装。
- 分布式功能必须与单卡 baseline 做数值或 loss 对齐。
- Runtime 要和具体模型解耦，避免把所有逻辑写进 `train.py`。
- Benchmark 要固定模型规模、batch size、sequence length、硬件环境。
- 文档和 benchmark 结果要随着 milestone 更新，形成清晰的项目演进记录。

