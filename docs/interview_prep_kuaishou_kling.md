# 快手「可灵」大模型训练与推理引擎工程师 — 面试准备指南

> 岗位定位：训练框架 / 微调平台 / 推理平台研发与优化。核心考点集中在
> **Transformer 架构、分布式训练并行、显存与精度优化、集合通信/CUDA、推理加速、
> 训练稳定性与大规模异构集群**。本指南以「面试问题 + 知识点 + 参考答案要点」组织，
> 并在末尾结合你自研的 `NanoTrain` 项目给出深挖问答与自我介绍话术。

---

## 目录

1. [Transformer / GPT 架构基础](#1-transformer--gpt-架构基础)
2. [分布式训练并行策略](#2-分布式训练并行策略)
3. [显存优化与混合精度 / FP8](#3-显存优化与混合精度--fp8)
4. [集合通信与 CUDA / HPC](#4-集合通信与-cuda--hpc)
5. [推理优化](#5-推理优化)
6. [微调技术（LoRA / PEFT / RLHF / 蒸馏）](#6-微调技术)
7. [训练稳定性与大规模异构集群](#7-训练稳定性与大规模异构集群)
8. [主流框架对比](#8-主流框架对比)
9. [视频生成 / 可灵相关加分知识](#9-视频生成--可灵相关加分知识)
10. [结合 NanoTrain 的自我介绍与项目深挖](#10-结合-nanotrain-的自我介绍与项目深挖)
11. [高频手撕题与系统设计](#11-高频手撕题与系统设计)

---

## 1. Transformer / GPT 架构基础

### 核心问题清单
- Self-Attention 的计算流程与复杂度？
- 为什么要除以 √d_k？
- Multi-Head Attention 的意义？MHA / MQA / GQA 的区别？
- 位置编码有哪些？绝对 vs 相对 vs RoPE vs ALiBi？
- LayerNorm vs RMSNorm？Pre-LN vs Post-LN？
- Encoder-Decoder / Encoder-only / Decoder-only 的区别与代表模型？
- 为什么现代大模型都是 Decoder-only？
- FFN/MLP 里的激活函数演进（ReLU → GELU → SwiGLU）？

### 知识点要点

**Self-Attention**
- 公式：`Attention(Q,K,V) = softmax(QKᵀ/√d_k) V`
- 复杂度：序列长度 `n`、维度 `d`，时间/空间复杂度 `O(n²·d)`，其中 `n²` 是长序列瓶颈的根源（催生 FlashAttention、sparse attention）。
- 除以 `√d_k`：点积随维度增大方差变大，softmax 进入饱和区导致梯度消失，缩放使方差稳定在 1 附近。

**MHA / MQA / GQA**
- MHA：每个 head 有独立的 Q/K/V。表达力强，但 KV cache 大。
- MQA (Multi-Query)：所有 head 共享一份 K/V，KV cache 缩小 `h` 倍，推理快，略有精度损失。
- GQA (Grouped-Query)：折中，`g` 组共享 K/V（Llama2-70B、Llama3 用），是当前主流。
- **面试高频**：为什么推理要用 GQA/MQA？→ KV cache 是显存和带宽瓶颈，减少 KV head 直接降低显存占用和访存量。

**位置编码**
- 绝对位置编码（原始 Transformer 的正弦 / 可学习）：外推能力差。
- RoPE（旋转位置编码）：把位置信息以旋转矩阵作用在 Q/K 上，相对位置感知，外推性好，主流（Llama、Qwen）。可配合 NTK-aware / YaRN 做长度外推。
- ALiBi：在 attention score 上加线性偏置，天然支持长度外推。

**归一化**
- LayerNorm：减均值除标准差 + 缩放偏移。
- RMSNorm：只除以均方根，不减均值，少一半统计量，更快，效果相当（Llama 系）。
- Pre-LN（LN 在残差分支内、attention/FFN 之前）：训练更稳定，梯度更好，现代大模型标配；Post-LN 深层难训练。

**激活函数**
- SwiGLU：`SwiGLU(x) = (Swish(xW) ⊙ xV)`，门控结构，效果优于 GELU，Llama/PaLM 采用。注意它把 FFN 拆成 3 个矩阵（gate/up/down），中间维通常取 `8/3·d` 以保持参数量。

### 参考回答模板（"讲讲 attention 复杂度瓶颈"）
> 标准 attention 对序列长度是 `O(n²)` 的时间和显存，长序列时 `n²` 的注意力矩阵成为主要开销。FlashAttention 通过 tiling + online softmax 把注意力矩阵分块在 SRAM 内计算，不落地整个 `n×n` 矩阵，把显存降到 `O(n)`，并显著减少 HBM 访存实现加速；进一步长序列可用 sparse attention（只算局部/选定 block）把复杂度降到亚二次。

---

## 2. 分布式训练并行策略

> **这是本岗位最核心的考点**，务必能画图讲清每种并行切什么、通信什么、放在哪个维度。

### 核心问题清单
- DP / DDP 原理？DDP 相比 DP 的优势？
- DDP 的梯度同步机制（bucket、overlap、Ring-AllReduce）？
- ZeRO-1/2/3 分别切分什么？通信量各是多少？
- Tensor Parallel（Megatron）怎么切 attention 和 MLP？通信在哪？
- Pipeline Parallel 的 bubble 是什么？GPipe vs 1F1B vs interleaved？
- 序列并行 / 上下文并行（Sequence / Context Parallel）？
- 3D / 4D 并行怎么组合？如何决定各维度大小？
- Expert Parallel（MoE）？

### 知识点要点

**数据并行**
- DP（单进程多线程，`nn.DataParallel`）：已淘汰，GIL 瓶颈、主卡负载不均。
- DDP：每个 GPU 一个进程，各自前反向，反向时对梯度做 **Ring-AllReduce**。通信量 `2·(N-1)/N·|params|` ≈ 2×参数量，与 GPU 数无关。
- 关键优化：**梯度分桶（bucketing）** + **通信与反向计算 overlap**（梯度一算完就异步 all-reduce）。

**ZeRO（DeepSpeed）**——把 DP 冗余的状态切分
- 训练显存三大块：参数 P、梯度 G、优化器状态 OS（Adam 的 m/v，fp32 主权重）。混合精度下每参数约 `16 bytes`（2 P + 2 G + 12 OS）。
- ZeRO-1：切 **优化器状态**。通信量与 DDP 相当，显存省最多。
- ZeRO-2：切 **优化器状态 + 梯度**。梯度用 reduce-scatter 到 owner。
- ZeRO-3：切 **优化器状态 + 梯度 + 参数**。前反向时按需 all-gather 参数，通信量约增加 50%，显存最省，可训超大模型。
- ZeRO-Offload / Infinity：把状态卸载到 CPU / NVMe。
- **面试高频**：ZeRO-3 vs TP 区别？→ ZeRO-3 仍是数据并行范式（按需聚合完整层），通信是参数 all-gather；TP 是把单层算子切开，每步都要 all-reduce 激活，通信更频繁但延迟低，通常限制在单机 NVLink 内。

**Tensor Parallel（Megatron-LM）**
- MLP：第一个线性层按 **列切**（column parallel，无需通信），第二个按 **行切**（row parallel），前向末尾 all-reduce，反向开头 all-reduce。一层 MLP 前反向各一次 all-reduce。
- Attention：按 head 切分（每个 rank 算一部分 head），QKV 列切、输出投影行切，同样一次 all-reduce。
- 通信量大且在关键路径，故 **TP 一般只在单机内（NVLink/NVSwitch）**。
- Embedding / LM head：vocab 维度切分（vocab parallel），需要 all-reduce logits 或用 vocab-parallel cross entropy。

**Sequence Parallel（Megatron）**
- 在 TP 基础上，把 LayerNorm/Dropout 这些非 TP 区域沿 **序列维** 切分，减少激活显存。all-reduce 被拆成 reduce-scatter + all-gather（通信量不变但省显存）。

**Context Parallel / Ring Attention**
- 针对超长序列，把序列切到多卡，attention 通过 ring 方式传递 K/V 分块计算，解决单卡放不下长序列 KV 的问题（视频生成 / 长文本关键）。

**Pipeline Parallel**
- 按层切分到不同 stage。朴素做法有 **bubble（空泡）**：`bubble ratio = (p-1)/(m+p-1)`，`p` stage 数，`m` micro-batch 数 → 增大 micro-batch 数降低 bubble。
- GPipe：先所有 micro-batch forward 再 backward，激活显存高。
- 1F1B（PipeDream-Flush）：一前一后交错，稳态显存更低，Megatron 默认。
- Interleaved 1F1B（virtual pipeline）：每卡放多个不连续的 stage 段，进一步降 bubble，代价是通信次数增加。

**3D 并行组合**
- 典型：TP（机内 NVLink）× PP（跨机流水）× DP（最外层，ZeRO 加持）。
- 经验：TP ≤ 单机 GPU 数（如 8）；PP 由模型层数 / 网络拓扑决定；DP 铺满剩余卡。
- 决策依据：让高频大通信（TP）走最快互联，低频通信（DP/PP）走较慢互联。

**MoE 与 Expert Parallel**
- 每个 token 经 router 选 top-k 专家；专家分布到不同卡，token 通过 **all-to-all** 分发/回收（EP 通信瓶颈是 all-to-all）。
- 挑战：负载均衡（load balancing loss）、all-to-all 通信、专家容量（capacity factor）。

### 通信量速记表

| 策略 | 切分对象 | 主要通信原语 | 通信频率 |
|---|---|---|---|
| DDP | 无（复制） | AllReduce 梯度 | 每 step 一次（可分桶 overlap） |
| ZeRO-1 | 优化器状态 | AllGather 参数（step 后） | 每 step |
| ZeRO-2 | +梯度 | ReduceScatter 梯度 | 每 step |
| ZeRO-3 | +参数 | AllGather 参数（前反向） | 每层 |
| TP | 层内算子 | AllReduce 激活 | 每层前反向 |
| SP | LN/序列维 | ReduceScatter+AllGather | 每层 |
| PP | 层 | P2P send/recv 激活 | 每 micro-batch |
| EP(MoE) | 专家 | All-to-All | 每 MoE 层 |

---

## 3. 显存优化与混合精度 / FP8

### 核心问题清单
- 训练显存都花在哪？如何估算？
- 激活重计算（gradient/activation checkpointing）原理与代价？
- FP16 vs BF16 区别？为什么大模型偏爱 BF16？
- 混合精度训练流程？loss scaling 为什么需要？
- FP8 训练怎么做？E4M3 / E5M2？per-tensor / per-block scaling？
- Offload（CPU/NVMe）机制？
- Paged / fused optimizer、8-bit optimizer？

### 知识点要点

**显存构成**
- 参数、梯度、优化器状态、**激活值**、临时 buffer/碎片。
- 激活值随 batch、序列长度、层数线性增长，长序列训练时往往是最大头 → 重计算 + 序列并行主攻这块。
- 估算：AdamW 混合精度 ≈ 每参数 16 bytes（fp16 参数2 + fp16 梯度2 + fp32 master 4 + m 4 + v 4）。7B 模型仅状态就 ~112GB → 必须切分/卸载。

**激活重计算（Activation Checkpointing）**
- 前向只保存 checkpoint 处激活，反向时重新前向计算中间激活。用 ~33% 额外计算换取大幅激活显存下降（`O(n)`→`O(√n)` 经典策略）。
- 可选择性重计算（selective recompute）：只重算便宜的算子（如 attention），保留贵的，Megatron 支持。

**混合精度**
- FP16：范围小（易上溢/下溢），需 **loss scaling**（放大 loss 使小梯度不下溢，更新前再缩回）。
- BF16：指数位与 FP32 相同（8 位），动态范围大，几乎不需 loss scaling，训练更稳，大模型主流（需 Ampere+）。
- 混合精度流程：fp16/bf16 做前反向计算，fp32 保存 master weights 与优化器状态做更新。

**FP8 训练/推理（本岗位加分点，可灵有 FP8 方案）**
- 两种格式：**E4M3**（精度高、范围小，用于前向/权重/激活）、**E5M2**（范围大、精度低，用于反向梯度）。
- 需要 **scaling**：per-tensor（配 delayed scaling，用历史 amax 估计缩放）或更细粒度的 per-block/per-channel（如 DeepSeek-V3 的细粒度 FP8）。
- 硬件：Hopper (H100) 的 Transformer Engine 原生支持 FP8 GEMM。
- 收益：GEMM 吞吐翻倍、显存/带宽下降；难点：数值稳定性（outlier）、amax 统计、哪些层保留高精度（通常 LN、softmax、最后一层保留 BF16）。

**优化器 / Offload**
- 8-bit Adam（bitsandbytes）：量化优化器状态省显存。
- ZeRO-Offload：优化器状态与 fp32 更新放 CPU，GPU 只算前反向。
- Fused optimizer（apex / FusedAdam）：把逐元素更新融合成一个 kernel，减少 kernel launch 与访存。

---

## 4. 集合通信与 CUDA / HPC

### 核心问题清单
- 常见集合通信原语及语义？
- Ring-AllReduce 为什么带宽最优？AllReduce = ReduceScatter + AllGather？
- NCCL 是什么？怎么调优（NCCL_ALGO、拓扑感知）？
- NVLink / NVSwitch / PCIe / InfiniBand / RDMA 的层级与带宽差异？
- GPU 架构：SM、warp、shared memory、寄存器、global memory？
- CUDA 优化：合并访存、bank conflict、occupancy、kernel fusion？
- 什么是 kernel launch overhead？CUDA Graph 解决什么？
- 计算与通信如何 overlap？

### 知识点要点

**集合通信原语**
- Broadcast、Reduce、AllReduce、ReduceScatter、AllGather、All-to-All、Gather/Scatter。
- **AllReduce = ReduceScatter + AllGather**，Ring 实现下每卡收发 `2(N-1)/N·数据量`，带宽最优、与卡数近似无关。
- All-to-All：MoE / 序列重排的关键，通信量大。

**互联层级（带宽由高到低）**
- 卡内：HBM。
- 机内：NVLink / NVSwitch（H100 NVLink ~900GB/s 级别）≫ PCIe。
- 机间：InfiniBand + RDMA（GPUDirect RDMA 让网卡直接读写 GPU 显存，绕过 CPU）。
- **拓扑感知调度**：把大通信（TP、all-to-all）放在 NVLink 域内，跨机走 IB。

**NCCL 调优**
- 选择算法（Ring / Tree，`NCCL_ALGO`）、`NCCL_PROTO`、绑核绑网卡、`NCCL_IB_HCA`、`NCCL_SOCKET_IFNAME`；调 `NCCL_DEBUG=INFO` 看拓扑。
- 通信/计算 overlap：DDP 的梯度 bucket 边算边通信；TP 的 sequence parallel 用异步。

**GPU / CUDA**
- 层级：Grid → Block → Warp(32 线程 SIMT) → Thread。
- 存储层级：寄存器 > shared memory（片上，block 内共享）> L2 > global（HBM）。
- 优化要点：
  - **合并访存（coalesced access）**：同 warp 连续地址。
  - **避免 bank conflict**：shared memory 32 banks。
  - **提高 occupancy**：平衡寄存器/shared memory 使用与活跃 warp 数。
  - **kernel fusion**：减少 kernel launch 和中间结果写回 HBM（如 fused LayerNorm、FlashAttention）。
  - **CUDA Graph**：把大量小 kernel 的 launch 序列录制成图，消除 CPU launch overhead（小 batch 推理/训练收益大）。
  - **Tensor Core**：矩阵乘专用单元，要求维度对齐（8/16 的倍数）与低精度输入。

**FlashAttention（必背）**
- 动机：标准 attention 反复读写 `n×n` 矩阵到 HBM，访存受限。
- 做法：分块（tiling）把 Q/K/V 分块加载到 SRAM，**online softmax** 增量维护 max 与 sum，不落地完整注意力矩阵；反向重算而非存储。
- 收益：显存 `O(n²)`→`O(n)`，速度提升数倍，是 IO-aware 算法的典范。v2 提升并行度与 work partitioning，v3 面向 Hopper + FP8。

---

## 5. 推理优化

### 核心问题清单
- LLM 推理两阶段（prefill / decode）差异？各自瓶颈？
- KV cache 原理？显存怎么算？如何优化（MQA/GQA、量化、PagedAttention）？
- 什么是 continuous batching？相比静态 batching 好在哪？
- PagedAttention（vLLM）解决什么问题？
- 投机解码（speculative decoding）原理？draft-verify？Medusa/EAGLE？
- 量化推理：GPTQ / AWQ / SmoothQuant / KV cache 量化？W8A8 vs W4A16？
- 吞吐 vs 延迟指标：TTFT、TPOT、throughput？
- Chunked prefill、prefix caching？
- Disaggregated serving（PD 分离）？

### 知识点要点

**Prefill vs Decode**
- Prefill：并行处理整个 prompt，**计算密集**（compute-bound），一次算出 KV cache。
- Decode：逐 token 自回归，每步只算 1 个 token，**访存密集**（memory-bound），受 KV cache 与权重读取带宽限制。
- 优化方向不同 → 催生 **PD 分离（disaggregated serving）**：prefill 与 decode 用不同实例/配置。

**KV Cache**
- 缓存历史 token 的 K/V 避免重复计算。大小 ≈ `2 · layers · seq_len · n_kv_heads · head_dim · batch · dtype_bytes`。
- 优化：GQA/MQA 减 KV head、KV cache 量化（int8/fp8）、PagedAttention 减碎片、offload/淘汰。

**Continuous Batching（in-flight batching）**
- 传统静态 batch 要等最慢的样本结束；continuous batching 在 token 级别动态加入/移出请求，GPU 利用率大幅提升（vLLM、TGI）。

**PagedAttention（vLLM 核心）**
- 借鉴 OS 虚拟内存分页：KV cache 按 block 分页管理，逻辑连续物理不连续，消除内存碎片，支持 block 级共享（prefix caching、beam search 共享前缀）。

**投机解码（Speculative Decoding）**
- 用小 draft 模型（或同模型的浅层/额外头）一次生成多个候选 token，大模型一次前向 **并行验证**，接受正确前缀。因为 decode 是访存瓶颈，一次验证多 token 几乎免费提速。
- 变体：Medusa（多头预测）、EAGLE（特征级自回归 draft）、Lookahead。

**量化推理**
- **权重量化**：GPTQ（基于二阶信息逐层量化）、AWQ（激活感知，保护重要权重通道）→ W4A16 常见，省显存、decode 提速。
- **权重+激活量化**：SmoothQuant（把激活的量化难度迁移到权重）、W8A8 → prefill 也能用 int8 GEMM 提速。
- **KV cache 量化**：int8/fp8，省显存、增大 batch。
- 权衡：低比特掉点、outlier 处理、是否需要校准集。

**服务化指标**
- TTFT（首 token 延迟，受 prefill 影响）、TPOT/ITL（每 token 延迟，受 decode 影响）、Throughput（token/s）、QPS。
- 工程手段：chunked prefill（把长 prefill 切块与 decode 混跑，平衡 TTFT 与吞吐）、prefix caching（系统 prompt 复用 KV）。

---

## 6. 微调技术

### 核心问题清单
- 全量微调 vs PEFT 的取舍？
- LoRA 原理？为什么有效？秩 r、alpha、作用在哪些层？
- QLoRA 的四个关键点（NF4、双量化、分页优化器、LoRA）？
- Adapter / Prefix Tuning / P-Tuning / (IA)³ 区别？
- RLHF 全流程（SFT → RM → PPO）？DPO 相比 PPO 的优势？
- 知识蒸馏：logit 蒸馏、特征蒸馏、on-policy 蒸馏？
- 灾难性遗忘怎么缓解？

### 知识点要点

**LoRA**
- 冻结原权重 `W`，学习低秩增量 `ΔW = B·A`（`A∈r×d`, `B∈d×r`, r≪d）。推理时可合并回 `W`，零额外延迟。
- 为什么有效：微调更新的本征秩很低。
- 超参：`r`（4~64）、`alpha`（缩放 `α/r`）、通常作用于 attention 的 q/k/v/o 投影（也可加 MLP）。
- 优点：可训参数降到 <1%，多任务可切换不同 LoRA。

**QLoRA**
- ①4-bit NF4 量化基座权重；②双量化（对量化常数再量化）；③分页优化器（避免 OOM 峰值）；④在量化基座上训 LoRA（前向反量化到 BF16 计算）。单卡即可微调超大模型。

**其他 PEFT**
- Adapter：层间插入小瓶颈 MLP（有额外推理延迟）。
- Prefix/Prompt/P-Tuning：学习可训练的虚拟 token/前缀。
- (IA)³：学习缩放向量作用在激活上。

**RLHF / 对齐**
- 三阶段：SFT → 训练奖励模型 RM（用人类偏好对）→ PPO 用 RM 优化策略（含 KL 惩罚防偏离 SFT）。
- **DPO**：跳过显式 RM 与 RL，直接用偏好数据做分类式损失优化策略，稳定、易实现，当前流行。相关：GRPO（去 critic，组内相对优势，DeepSeek 用于推理模型）。

**知识蒸馏（可灵关注的小型化）**
- 软标签蒸馏：学生拟合教师的 softmax（带温度 T）分布，KL 损失。
- 特征/中间层蒸馏、注意力蒸馏。
- 扩散模型蒸馏（视频/图像生成加速关键）：Progressive Distillation、Consistency Models、LCM、DMD——把多步采样蒸馏成几步甚至一步，大幅降推理成本。

---

## 7. 训练稳定性与大规模异构集群

> 岗位职责明确提到「集群建模、热插拔、快速恢复、冗余调度」，务必准备。

### 核心问题清单
- 万卡训练常见故障有哪些？如何检测与恢复？
- Checkpoint 策略：如何降低 checkpoint 开销？异步 checkpoint？
- 什么是 elastic training / 热插拔？
- Loss spike / 发散怎么排查与处理？
- 慢节点（straggler）如何发现与处理？
- 数据加载与 packing / 变长序列如何高效处理？
- 如何做集群性能建模与并行配置搜索（solver / 自动并行）？

### 知识点要点

**故障与恢复**
- 大规模训练故障率高（GPU 掉卡、ECC error、NCCL timeout、网络抖动、节点宕机）。MTBF 随规模线性下降。
- **快速恢复**：
  - 频繁但低开销的 checkpoint：**异步 checkpoint**（先拷到 CPU/内存再后台落盘）、分布式 checkpoint（每 rank 存自己的分片）。
  - In-memory / 冗余 checkpoint（如 Gemini、CheckFreq）：把状态复制到邻居节点内存，坏一个节点从邻居恢复，避免读远端存储。
  - **热插拔 / elastic**：坏节点剔除后用备份节点顶替，重建 process group（torchrun elastic、自研调度）。
- **冗余调度**：预留 hot spare 节点，故障时快速顶上，减少重排等待。

**Loss 稳定性**
- Loss spike 常见于大 LR、bf16/fp8 数值问题、坏数据。手段：梯度裁剪、warmup、跳过异常 batch、回滚到上个 checkpoint、降低 LR、检查数据。
- 监控：grad norm、loss、各 rank 一致性、激活/权重的 max（amax）。

**Straggler / 异构**
- 慢节点拖慢同步。检测：每 step 各 rank 耗时、通信时间。处理：剔除、重调度、负载再均衡；异构集群（不同型号 GPU）需按算力分配不同 micro-batch / 层数（负载均衡建模）。

**数据调度（岗位提到 packing/split、自动并行）**
- 变长序列 **packing**：把多条短序列拼进一个 sample 减少 padding 浪费（需 attention mask 隔离，或用 FlashAttention 的 varlen 接口）。
- 长序列 **split** / context parallel。
- **自动并行 / solver**：给定模型、集群拓扑、显存约束，搜索最优 (TP, PP, DP, micro-batch, recompute) 配置，最小化端到端时间。代表：Alpa、Galvatron、Megatron 的自动配置；本质是 cost model + 搜索/ILP。

**Checkpoint 一致性**
- 分布式 checkpoint（PyTorch DCP）支持改变并行度后 resume（reshard）。

---

## 8. 主流框架对比

### 核心问题清单
- Megatron-LM / Megatron-Core 提供什么？
- DeepSpeed 的定位？ZeRO 与 Megatron 怎么结合（3D 并行）？
- PyTorch FSDP 与 ZeRO-3 的关系？
- vLLM / TensorRT-LLM / SGLang 推理框架区别？
- Ray 在训练/推理中的角色？

### 知识点要点

| 框架 | 定位 | 关键能力 |
|---|---|---|
| **Megatron-LM/Core** | 训练并行库 | TP、SP、PP(1F1B/interleaved)、context parallel、高效 kernel；Core 是模块化可复用版本 |
| **DeepSpeed** | 训练优化 | ZeRO-1/2/3、Offload、Infinity、MoE、ZeRO++；常与 Megatron 结合（Megatron-DeepSpeed） |
| **PyTorch FSDP** | 原生数据并行 | 等价 ZeRO-3 的参数/梯度/状态分片，`FULL_SHARD`；FSDP2 改进 |
| **vLLM** | 推理服务 | PagedAttention、continuous batching、prefix caching、投机解码 |
| **TensorRT-LLM** | 推理（NV） | 编译优化、FP8/INT8 kernel、in-flight batching，极致延迟/吞吐 |
| **SGLang** | 推理服务 | RadixAttention（前缀树共享 KV）、结构化输出 |
| **Ray** | 分布式编排 | 任务调度、RLHF（RLlib/ray）、多组件编排（如 verl、OpenRLHF 底座） |

- **FSDP vs ZeRO-3**：思想一致（分片 + 按需 all-gather），FSDP 是 PyTorch 原生、与生态结合好；DeepSpeed 功能更全（offload/MoE/推理）。
- **Megatron vs DeepSpeed**：Megatron 强在 TP/PP 的模型并行与 kernel；DeepSpeed 强在 ZeRO 数据并行显存优化。大模型训练常二者结合。

---

## 9. 视频生成 / 可灵相关加分知识

> 不是硬性要求，但了解会让面试官眼前一亮（可灵是视频生成 DiT 路线）。

- **Diffusion 基础**：前向加噪、反向去噪；DDPM vs DDIM（确定性、少步采样）；classifier-free guidance。
- **DiT（Diffusion Transformer）**：用 Transformer 替代 UNet 做扩散骨干，Sora/可灵路线；视频是 3D（时空）token，序列极长 → **长序列 attention 是核心瓶颈**（呼应岗位的 sparse attention、context parallel、FP8）。
- **视频特有**：时空 patchify、3D VAE 压缩、时序一致性；序列长导致显存/算力爆炸 → sparse attention（如时空稀疏、滑窗）、序列并行、FP8。
- **采样加速**：一致性蒸馏 / LCM / DMD 把几十步降到几步（推理成本关键）。
- **训练挑战**：超长序列 packing/split、异构集群、显存 —— 正好对应岗位职责。

---

## 10. 结合 NanoTrain 的自我介绍与项目深挖

> 你自研了 `NanoTrain`——从 nanoGPT baseline 起步，逐步实现了 DDP、Megatron 式 Tensor Parallel、
> ZeRO-1/ZeRO-2，并有 benchmark 工具。**这是最强的差异化亮点**，一定要主动讲。

### 自我介绍话术（1 分钟）
> "我对训练引擎的兴趣促使我从零实现了一个轻量训练框架 NanoTrain。它以 nanoGPT 为基线，
> 我逐步把它重构成可复用的训练引擎组件：先迁移单卡训练与 torchrun DDP，然后实现了
> Megatron 风格的纯张量并行（column/row parallel linear、vocab-parallel embedding 与 LM head），
> 又实现了 ZeRO-1 的优化器状态切分和 ZeRO-2 的 owner-rank 梯度归约，并配了 benchmark
> 工具测吞吐、显存和扩展性。通过这个项目我把分布式训练里最核心的并行与通信机制都亲手
> 落地了一遍，对 TP 的通信位置、ZeRO 的显存收益、DDP 的梯度同步有很具体的理解。"

### 可能被深挖的问题（务必能答）
1. **你的 TP 是怎么切的？通信在哪？**
   - 答：MLP 列切+行切，attention 按 head 切，前向末尾 all-reduce；embedding/LM head 用 vocab parallel。能结合 `nanotrain/model/gpt.py` 和 `parallel/` 讲实现。
2. **ZeRO-1 和 ZeRO-2 你具体切了什么？两者差别？**
   - ZeRO-1 切优化器状态，step 后 broadcast 更新后的参数；ZeRO-2 在反向后把每个参数梯度 reduce 到 owner rank、清非 owner 梯度、本地 step 再 broadcast。（正是你 README 里描述的实现）
3. **为什么 ZeRO-2 你要绕过 DDP 默认的 all-reduce？**
   - 因为 owner-rank 只需要该参数的归约梯度，用 reduce 到 owner 比全 all-reduce 省通信/显存。
4. **TP + ZeRO / TP + DDP 为什么还没做？下一步怎么设计？**
   - 诚实说是 roadmap（README 写了 hybrid 是后续里程碑）；讲清楚 3D 并行的 process group 设计（TP group 在机内、DP group 跨机）。
5. **你的 vocab-parallel cross entropy 怎么算的？**
   - logits 按 vocab 切，需要跨 rank 求 max 和 sum-exp（两次 all-reduce）得到正确 softmax/loss。
6. **benchmark 里 phase6 的结果说明什么？** —— 打开 `benchmark/results/phase6/aggregate.json` 能讲出吞吐/显存/scaling 结论。
7. **踩过的坑？** —— 如 process group 初始化、TP 下随机种子/初始化一致性、梯度累积与并行的交互、checkpoint reshard。

### 建议补充/可讲的 roadmap（展示成长性）
- Pipeline Parallel（1F1B）、ZeRO-3/FSDP、序列并行、激活重计算、BF16/FP8、分布式/异步 checkpoint、自动并行 solver —— 恰好对应岗位职责，可作为"我接下来想加的能力"来聊。

---

## 11. 高频手撕题与系统设计

### 手撕代码（准备能白板写）
- Multi-Head Self-Attention（含 causal mask、scaling）。
- LayerNorm / RMSNorm 前向。
- 从零实现 Ring-AllReduce 逻辑（讲清 reduce-scatter + all-gather）。
- 简易 KV cache 的 decode 循环。
- Softmax 数值稳定实现（减 max）；online softmax（FlashAttention 思想）。
- LoRA 层封装（`W + B@A * scaling`）。
- 位置编码 RoPE 的旋转实现。

### 系统设计题
- 设计一个支持 3D 并行的训练框架的模块划分（process group 管理、并行层、优化器分片、checkpoint、runtime）。→ **直接用 NanoTrain 的分层来答**。
- 设计一个高吞吐 LLM 推理服务（continuous batching + PagedAttention + PD 分离 + 投机解码 + 量化）。
- 给定模型规模和集群（如 512×H100），如何选并行配置？→ 讲 cost model：TP 限机内、算 bubble、显存约束、通信/计算比。
- 万卡训练如何做容错？→ 异步/冗余 checkpoint + elastic 热插拔 + straggler 检测。

### 反问面试官（体现兴趣）
- 团队目前训练的最大规模与主要瓶颈（通信/显存/稳定性）？
- FP8 训练在生产中的收敛与稳定性经验？
- sparse attention 在视频长序列上的算法-工程协同细节？
- 自动并行 solver 目前落地到什么程度？

---

## 附：速记「一句话」清单（考前扫一遍）

- Attention 复杂度 `O(n²d)`；除 `√d_k` 稳定 softmax 梯度。
- GQA 减 KV cache；RoPE 相对位置可外推；RMSNorm 更快；Pre-LN 更稳；SwiGLU 更强。
- DDP=AllReduce 梯度（Ring 带宽最优）；ZeRO 1/2/3 切 状态/+梯度/+参数。
- TP 切层内、每层 all-reduce、限机内；PP 有 bubble、1F1B 降显存；SP 省激活；CP 解长序列。
- 显存四大块：参数/梯度/优化器/激活；激活重计算换计算；BF16 免 loss scaling；FP8 E4M3 前向/E5M2 反向 + scaling。
- AllReduce=ReduceScatter+AllGather；NVLink≫PCIe；RDMA 绕 CPU；CUDA Graph 消 launch 开销；FlashAttention IO-aware。
- 推理：prefill compute-bound / decode memory-bound；KV cache 是瓶颈；PagedAttention 去碎片；continuous batching 提利用率；投机解码免费提速；量化 W4A16/W8A8。
- 微调：LoRA 低秩、可合并、零延迟；QLoRA 4bit+NF4+双量化+分页；DPO 免 RM。
- 稳定性：异步/冗余 checkpoint、elastic 热插拔、straggler 检测、梯度裁剪防 spike。
- 视频生成：DiT + 超长时空序列 → sparse attention / context parallel / FP8 / 采样蒸馏。
