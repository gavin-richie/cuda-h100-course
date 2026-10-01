---
title: "第 9 课 - 多 GPU 第 1 部分"
lesson_number: "9"
lesson_slug: "multi-gpu-part-1"
instructor: "Prateek Shukla"
course: "CUDA Programming for NVIDIA H100s"
language: "zh-CN"
original_markdown: "/markdown/lessons/lesson-09-multi-gpu-part-1.md"
source_page: "pages/lesson-9.html"
source_slide_pdf: "H100-Course/slides/9. Multi GPU.pdf"
published_lesson_page: "/pages/lesson-9.html"
published_markdown_path: "/markdown/lessons/lesson-09-multi-gpu-part-1.md"
topics:
  - "NVLink"
  - "NVSwitch"
  - "ConnectX"
  - "UVA"
  - "P2P"
code_refs: []
generated_from:
  - "pages/lesson-9.html"
  - "H100-Course/slides/9. Multi GPU.pdf"
---

# 第 9 课 - 多 GPU 第 1 部分

本文件将已发布的课程页面与完整幻灯片文本合并，方便智能体直接抓取和检索。

## 来源

- 课程页面：`pages/lesson-9.html`
- 幻灯片：`H100-Course/slides/9. Multi GPU.pdf`
- 已发布课程 URL：`https://cudacourseh100.github.io/pages/lesson-9.html`
- 已发布 Markdown URL：`https://cudacourseh100.github.io/markdown/lessons/lesson-09-multi-gpu-part-1.md`

## 课程摘要

第 9 课离开单 GPU kernel，追问真实的训练系统必须让数据流经什么。答案是一张 fabric：节点（node）内部的 NVLink 与 NVSwitch，节点外部的 ConnectX 与 rail 对齐的 InfiniBand，再加上作为这些硬件之上最底层软件的 CUDA 对等访问（peer access）与 UVA。

## 本课的重要性

**性能不再只是 kernel 问题，而变成 fabric 问题。**

Hopper kernel 解释一块 GPU 如何保持忙碌。多 GPU（multi-GPU）系统解释的是：一旦内存、带宽（bandwidth）、拓扑（topology）与软件发现全部活在单个 SM 之外，许多 GPU 如何保持协调。

## 主题

- NVLink
- NVSwitch
- ConnectX
- UVA
- P2P

## 课程信息

- **课程位置：**第 09 课，共 10 课
- **主要转变：**瓶颈（bottleneck）从一条 kernel 流水线（pipeline）转移到加速器之间的通信 fabric。
- **配套幻灯片：**`9. Multi GPU.pdf`
- **系统重点：**`NVLink`、`NVSwitch`、`ConnectX-7`、rail，以及 `cudaDeviceEnablePeerAccess`

## 关键要点

- 多 GPU 训练的存在，是因为即使一块极快的 H100，对前沿规模的工作负载也远远不够。
- 节点内部，NVLink 与 NVSwitch 打造出快速路径。节点外部，ConnectX 与网络 fabric 主导着扩展的故事。
- CUDA P2P 与 UVA 提供了基本的直接访问，但一旦拓扑不再简单，仍然需要更高层的通信机制。

## 网页课程正文

### 为什么要横向扩展

幻灯片从一个直白的计算开始。如果训练大约每参数每 token 花费 6 次 FLOP，那么一个 1T 参数模型在 10T token 上就会落到 `6 x 10^25` FLOPs 附近。即使假装你能持续维持 1000 TFLOP/s，在单台设备上也要以世纪来计量。多 GPU 系统不是奢侈品。它是让训练预算进入人类时间尺度的唯一方式。

#### 更多吞吐量

额外的 GPU 把数学摊到更多张量核心（tensor core）和更大的 HBM 容量上，从而缩短墙钟时间。

#### 更大的内存面

更大的模型和优化器状态不再装得进单台设备，因此系统需要许多 HBM 池，外加一条足够快、能让它们在训练中保持足够一致的 fabric。

> **本课的要点：**一旦你离开单块 GPU，机器就不再只是 SM、张量核心和共享内存（shared memory）。机器现在是一个通信层级。

### 一个 DGX H100 节点本身已经是一个小型分布式系统

笔记把 DGX H100 节点描述为：八块 SXM5 H100 GPU 由四颗 NVSwitch 芯片连接，外加八张 ConnectX-7 网卡，以及四个用于外部计算 fabric 的 OSFP 笼位。这一点很重要，因为这个节点不是一块 GPU 带一堆配件。它是一个由加速器和网络端点组成的、相互协调的网格。

| 组件 | 角色 | 为什么重要 |
| --- | --- | --- |
| 8 x H100 SXM5 GPUs | 节点内的算力与 HBM 容量。 | 它们是需要交换激活、梯度和模型状态的端点。 |
| 4 x NVSwitch | 内部全速交换 fabric。 | 让每块 GPU 都能到达其他任何一块 GPU，而不会塌缩到缓慢的 PCIe 式路径。 |
| 8 x ConnectX-7 NICs | 面向计算 fabric 的外部网络接口。 | 一旦流量离开节点，就由这些设备掌管关键的带宽过渡。 |
| 4 x OSFP cages | 物理的对外高速链路。 | 把节点接入 rail 对齐的 InfiniBand fabric。 |

本课还把计算 fabric 与存储 fabric 区分开。OSFP 路径用于节点间的 GPU 通信。数据集摄取与 checkpoint 流量使用另一组接口和另一种网络设计。

### 节点内部，NVLink 与 NVSwitch 定义了快速路径

H100 的第四代 NVLink 给每块 GPU 约 900 GB/s 的双向带宽。笔记将其表述为 GPU 能够跳过 CPU 路径、直接从 HBM 出发通信，而不是在普通的以主机为中心的链路上来回弹跳的原因。

#### NVLink

直接的 GPU 到 GPU 链路层。笔记特别点名每块 H100 有 18 条独立的 NVLink 链路，并把相对 PCIe 的带宽跃升视为现代多 GPU 系统行为不同的主要原因之一。

#### NVSwitch

覆盖整个节点的交换层，使互连（interconnect）完全无阻塞。每块 GPU 都能以全速到达其他任何 GPU，而不会被困在菊花链的故事里。

重要的概念转变是：节点开始像一张 fabric，而不是八个孤立的设备。笔记甚至强调交换机侧操作很重要，因为交换层不只是一根愚蠢的电缆交叉开关。

> **带宽悬崖：**在一个 DGX 节点内部，GPU 通过 NVLink 交谈。一旦流量离开这台机器，它就会撞上网卡和外部网络。整个扩展故事的好坏，取决于你如何管理这次带宽下降与距离增加。

### 节点外部，ConnectX、rail 与网络拓扑决定扩展能否守住

ConnectX-7 是节点的出口。笔记描述了每台 DGX H100 有八张计算网卡，以及一个 rail 对齐的系统：所有节点上的 GPU 0 共享一条 rail，GPU 1 共享另一条，依此类推。这创造了八个并行的流量平面，而不是一个巨大的混合队列。

| 概念 | 含义 | 为什么重要 |
| --- | --- | --- |
| rail 对齐（Rail alignment） | 集群中相同的 GPU 索引各自映射到独立的网络平面。 | 减少干扰，让 leaf 层的流量模式更干净。 |
| Leaf / spine / core | 分层的 InfiniBand 胖树（fat-tree），从一个可扩展单元（scalable unit）扩展到更大的 pod。 | 决定路径多样性、超额订阅（oversubscription）行为与集群级可达性。 |
| 自适应路由（Adaptive routing） | 交换机硬件根据拥塞情况在多条可行路径之间做选择。 | 防止在有替代路径可用时，流量仍堆积到一条热门上行链路上。 |
| SHIELD 与网内逻辑（in-network logic） | 帮助隔离故障或卸载部分通信行为的硬件特性。 | 如果 fabric 无法高效处理故障与归约，扩展会很快失败。 |

笔记把 SuperPOD 式部署描述为用 Quantum-2 InfiniBand 交换机构建的 rail 优化三层 fabric。具体数字不如本课要点重要：一旦扩展到单节点之外，拓扑就不再是背景细节。它是性能工程的一部分。

### CUDA P2P 与 UVA 提供了基本的直接路径，但没有解决整个通信问题

CUDA 在这个层级给出两个关键机制。统一虚拟寻址（UVA，Unified Virtual Addressing）意味着一个指针空间就能识别数据是位于主机内存还是某块 GPU 的 HBM。对等访问则允许一块 GPU 通过现有 fabric 直接访问另一块 GPU 的内存。

#### `cudaDeviceEnablePeerAccess`

启用直接的对等访问。笔记强调它是单向的，因此双向访问需要在两块设备上都调用它。

#### `cudaMemcpyPeer`

在设备之间直接发起内存拷贝、不经主机内存中转，前提是对等访问与相应拓扑可用。

但本课也指出，裸 P2P 并不是完整的答案。在 HGX H100 板上，拓扑是由交换机连接的网格，不是一条直线。从一个 SM 到另一块 GPU 的 HBM 的手动 load/store 路径，既不会自动打满 fabric，也解决不了多方协调。这正是下一课转向 NCCL、PMIx 和 Slurm 的原因。

```text
// Conceptual CUDA-side flow
cudaDeviceEnablePeerAccess(peer, 0);
cudaMemcpyPeer(dst_ptr, dst_device, src_ptr, src_device, bytes);
```

### 实践指导

1. **测量带宽层级，而不只是测量 GPU。**节点内 NVLink 行为与节点间 NIC 行为属于截然不同的扩展形态。
2. **把节点当作一张 fabric 来读。**NVSwitch、ConnectX、OSFP、rail 和拓扑都是你的 kernel 实际运行于其中的机器的一部分。
3. **审慎地使用对等访问。**直接的 GPU 内存访问很强大，但它并不是集合通信（collective）库或拓扑感知调度的自动替代品。
4. **预期拓扑会塑造性能。**rail 对齐、自适应路由和交换机层级决定了集群增长时通信能否保持均衡。
5. **把第 9 课当作第 10 课的硬件序言。**一旦理解了 fabric，编排栈就会容易理解得多。

#### 术语表

| 术语 | 定义 |
| --- | --- |
| NVLink | 节点内用于高带宽通信的直接 GPU 到 GPU 互连。 |
| NVSwitch | 让节点内的 GPU 彼此以全速通信的交换 fabric。 |
| ConnectX-7 | 节点用于外部 GPU 到 GPU 通信的高速网络接口。 |
| Rail | 一个隔离的网络平面，把所有节点上相同的 GPU 索引对齐到同一条通信路径上。 |
| UVA | 统一虚拟寻址（Unified Virtual Addressing），让系统能从指针值本身确定内存位置。 |

### 继续课程

第 9 课解释了硬件与 CUDA 侧的路径。最后一课进入编排：作业如何跨节点启动、rank 如何相互发现，以及集合通信实际如何被初始化和调度。

## 完整幻灯片文本

从 `H100-Course/slides/9. Multi GPU.pdf` 使用 `pdftotext -layout` 提取。共 26 张幻灯片。

### 幻灯片 1：多 GPU

Prateek Shukla

### 幻灯片 2：让我们用 H100 在 10T token 上训练一个 1T 参数模型

H100
实证观察表明，训练数据中每个 token、每个参数大约需要 6 次运算（FLOPs）（前向传播 + 反向传播 + 更新）。

总 FLOPs=6x(10^12)x(10^13)=6x10^25 FLOPs

假设我们每秒能拿到 1000tflops

那大约是 1900 年！！

这就是为什么我们需要多块 GPU

### 幻灯片 3：多 GPU

与其花费数千年等待运算完成，我们可以直接把多块 GPU 连接在一起，获得更高吞吐量并运行更大的模型。这由两项技术支撑

NVLink Gen 4

绕开 CPU 的直接 GPU 到 GPU 互连。

900 GB/s 双向带宽。

比 PCIe Gen 5 快 7 倍。

NVSwitch

让节点中的每块 GPU 都能以全速与其他任何 GPU 通信

### 幻灯片 4：H100 集群与旧式集群对比

在传统计算中，GPU 是通过较慢的 PCIe 通道通信的离散单元。H100 范式改变了这一点：它创造了一个"网格（mesh）"，其中每块 GPU 都能以极高的速度访问其他任何 GPU 的内存。

一个 8 GPU 集群可以表现得好像拥有单一的内存池和计算核心池

这种架构让系统可以从单台服务器中的 8 块 GPU，扩展到 256 块 GPU 的集群（使用 NVLink Switch System），并以原生芯片速度通信。

### 幻灯片 5：H100 DGX 系统或一个节点

它配备 8 块采用 SXM5 形态的 NVIDIA H100 GPU。

该板载有 4 颗第三代 NVSwitch

GPU 与交换机通过第四代 NVLink 连接，为每块 GPU 提供高达 900 GB/s 的带宽

8 张网络接口卡（NIC）：NVIDIA ConnectX-7，一种专门设计用于把网络任务从主 CPU 卸载的处理器。

4x OSFP（Octal Small Form-factor Pluggable）笼位

### 幻灯片 6：Nvidia H100 Superpod

一台 NVIDIA DGX H100 SuperPod 大约有 127-128 块 GPU，组织为 4 个 SU（Scalable Unit，可扩展单元），每个 SU 拥有 32 个 DGX H100 节点。

数字 32 使得使用标准交换机端口数（通常每台交换机 64 个端口）的"Level 1"网络达到完美均衡。

H100 SuperPod 的设计基于这样一个事实：单个 DGX H100 节点拥有 8 个独立的 NVIDIA ConnectX-7 网络接口（HCA）用于计算，每块 GPU 一个。

### 幻灯片 7：Nvlink

每块 H100 GPU 具有 900 GB/s 的双向带宽。这大约比用于连接 GPU 与 CPU 的标准 PCIe Gen5 接口快 7 倍。Nvlink 让 GPU 利用这一点在自己的 HBM 之间通信，并跳过 CPU 路径。

H100 使用 18 条独立的 NVLink"链路"来达到这一速度。与以往世代不同，第四代 NVLink 更注重密度优化，每条链路只使用两对高速差分对（从四对减少），从而能在同样的空间里塞进更多链路。

### 幻灯片 8：Nvswitch 3

在标准的 8 GPU HGX 板上，使用四颗 NVSwitch 芯片连接全部八块 H100 GPU。每台交换机连接 4 - 5 个 Nvlink 端口

这些交换机构成完全无阻塞的互连。这意味着每块 GPU 都能同时以 900 GB/s 的全速与任何其他 GPU 通信，而无需排队等待数据通道腾空

NVIDIA 把 NVSwitch 芯片放进称为 NVLink Switch System 的外部托盘。这些系统把多达 256 块 H100 GPU（32 个服务器机架）连接成单个"SuperPOD"

NVSwitch 本身带有 ALU，可以在交换机内部执行加法/归约，这是带来巨大性能提升的最重要特性之一。

### 幻灯片 9：Connectx

在服务器机箱内部，GPU 通过 NVLink（900 GB/s）交谈。一旦数据需要离开机箱去往另一台服务器，它就会撞上 ConnectX-7。每台 DGX 还有 8 个计算网络接口（ConnectX-7）。

这就是瓶颈。你的训练速度取决于你管理这次 18 倍带宽骤降的效率。ConnectX-7 的存在就是为了把离开机箱的代价降到最小。

ConnectX7 让网络可以在 CPU 不知情的情况下读写 GPU 内存。

NIC 与网络交换机协作，在传输途中对梯度求和。这把网络流量从 O(N)（随集群规模线性增长）变为 O(1)（恒定流量）。这是大规模训练能够线性扩展的唯一原因

### 幻灯片 10：OSFP 笼位

这 4 个物理笼位通过"双端口（Twin-port）"技术提供 8 条独立的 400Gb/s 网络链路（合计 3.2 Tb/s 带宽）。它们在内部连接到 8x NVIDIA ConnectX-7 网卡。

这使得 GPU Direct RDMA 成为可能，让 GPU 能与其他 DGX 节点中的 GPU 交谈，而不用给系统 CPU 增加负担。

这 4 个 OSFP 笼位专用于计算 fabric（Compute Fabric）。存储流量绝不经过那些线缆。

### 幻灯片 11：存储 fabric

它帮助把海量数据集加载进 GPU 并保存 checkpoint。

它使用位于机箱背面标准 PCIe 插槽中的独立 PCIe 卡，而不是 OSFP 笼位。

这通常是胖树或标准的 leaf-spine 网络。与"rail"网络不同，任何 DGX 节点都需要能够与任何存储阵列交谈。

### 幻灯片 12：rail 对齐系统

在标准网络中，一台服务器可能只用一根线缆承载其所有组件的流量。在 DGX SuperPOD 中，网络在物理上被拆分为 8 个并行的、相互隔离的网络——这些就是"rail"。

一个节点内部有 8 块 GPU（编号 0 到 7）。Rail 1 把集群中每个节点的 GPU 0 连接到同一组交换机。Rail 2 把每个节点的 GPU 1 连接到另一组交换机。依此类推

在 leaf switch 层面，Rail 1 上的流量绝不会干扰 Rail 2 上的流量。这在整个集群中创造了 8 个独立的连接平面。

rail 系统利用 NVIDIA 的 SHARP 技术，把数据操作卸载到网络交换机本身。

### 幻灯片 13：rail 优化的胖树

SuperPOD 使用三层层级（Leaf、Spine、Core）把各个 SU 连接在一起

第 1 层：Leaf 层（与节点的连接）

第 2 层：Spine 层（SU 之间的连接）

第 3 层：Core / Super-Spine 层（最大规模）

这些层不过是一堆连接在一起的 Quantum-2 InfiniBand 交换机

### 幻灯片 14：Quantum-2 QM9700 InfiniBand 交换机

它提供超高带宽、低延迟的 GPU-GPU 互连

它拥有 64 个 400Gb/s 连接端口

当你看向交换机正面时，会看到 32 个笼位。它们采用 OSFP（Octal Small Form-factor Pluggable）标准。每个 OSFP 笼位实际承载两条独立的 400Gb/s 链路。32 笼位 x 2 链路 = 64 逻辑端口。

由于 QM9700 有 64 个端口，它正好是一个 32 节点 SU 的完美尺寸。

32 个端口（下行）：连接到 SU 中的 32 个节点（例如 Rail 1 连接到全部 32 个节点上的 GPU 0）。

32 个端口（上行）：向上连接到 spine switch（第 2 层），与其他 SU 交谈。

### 幻灯片 15：leaf 层

这里就是线缆从 DGX H100 服务器背面物理引出的地方。

rail 隔离（Rail Segregation）：有 8 组独立的交换机"平面"：

Rail 1 交换机：只连接每个节点的第 1 张网卡（GPU 0）。

Rail 8 交换机：只连接每个节点的第 8 张网卡（GPU 7）。

leaf switch 处理本地可扩展单元（Scalable Unit）内的流量。如果节点 1 上的 GPU 0 需要与节点 2（同一 SU 内）上的 GPU 0 交谈，流量路径是节点 -> Leaf Switch -> 节点。它从不需要沿链条走向更高的层级。

QM9700 使用 SHARPv3，其能力是上一代的 32 倍，使它能够处理复杂的 AI 数学。

### 幻灯片 16：leaf 层中的自适应路由

在 leaf 层，自适应路由对上行流量至关重要。

当数据包从一块 GPU 到达 leaf switch、并且需要前往另一个 SU 时，它必须上行到某台 spine switch。在无阻塞甚至超额订阅的胖树中，有多台可用的 spine switch 供它选用。

Quantum-2 交换机硬件不使用静态哈希（静态哈希总是把特定流发送到同一台 spine switch，可能造成碰撞），而是监视所有上行端口的队列深度与拥塞水平

交换机以逐包或逐消息的方式，动态地把数据包发送到最不拥挤的 spine switch 链路。这保证了即使通往 spine 的某条路径被堵死，流量也能顺畅地从其他路径流过。

### 幻灯片 17：spine 层

Spine 层把各台 Leaf switch 连接在一起。这使得 SU-1 中的节点可以与 SU-2 中的节点交谈。

Spine 组：spine switch 也被组织成与 rail 对齐的组。

Spine 组 1：只连接处理 Rail 1 的 Leaf switch。

隔离：这保证 Rail 1 的流量绝不会意外"泄漏"到 Rail 2 的线缆上，否则会造成拥塞（阻塞）。

如果 SU-1 中的 GPU 0 需要与 SU-2 中的 GPU 0 交谈，流量路径为：节点（SU1）-> Leaf（Rail 1）-> Spine（组 1）-> Leaf（Rail 1）-> 节点（SU2）

### 幻灯片 18：spine 层中的自适应路由

在 spine 层，自适应路由管理穿越网络核心的流量。

通常，在标准的两层胖树中，从某台特定 spine switch 到某个目标 leaf switch 只有唯一的"下行"路径。不过，如果存在并行链路，自适应路由在处理故障与多路径方面仍然至关重要

如果通往某台特定 leaf switch 的缓冲区已满，spine switch 之间会传递这种背压。Quantum-2 的自适应路由特性（如 SHIELD）帮助隔离这种拥塞，使其不会扩散到 spine 中其他未受影响的流量流。

### 幻灯片 19：Core / Super-Spine 层（最大规模）

对于超大规模集群（例如 127 个节点或更多），会加入第三层。

功能：这一层把多个"pod"（SU 的集群）连接在一起。

光模块：在这一层，系统通常改用 800G 光收发器以减少所需线缆数量，同时在逻辑上再拆分回两条 400G 链路。

### 幻灯片 20：SHIELD

在传统 InfiniBand 网络中，如果线缆断裂或链路抖动，软件控制器需要大约 5 - 30 秒来计算新的路由表并下发给所有交换机。这会让整个训练运行崩溃

SHIELD 把这套恢复逻辑直接搬进交换机硬件 ASIC。

当一台交换机看到某条链路断开时，它不会只是丢包。它立即在自己的本地硬件表中检查是否存在替代的有效路径。如果找到，它就更新自己的表以绕开出问题的节点、转向健康的邻居；如果失败，它就向该邻居发送硬件信号，使今后不会有任何流量再发给它。

### 幻灯片 21：GPU 间通信的 P2P 机制

CUDA 的 P2P（点对点，Peer-to-Peer）机制是一种允许两块 GPU 在不借助 CPU 的情况下通信的特性

我们可以显式地做 P2P 内存拷贝，或者获得 P2P 直接访问，整个传输都走 Nvlink。

统一虚拟寻址（Unified Virtual Addressing）支撑起这整个系统

### 幻灯片 22：统一虚拟寻址（UVA）

UVA 让 CPU 和 GPU 共享单一的虚拟地址空间。

在 UVA 之前：CPU 有自己的内存指针，GPU 也有自己的。你必须手动管理哪个指针指向哪块物理内存。

有了 UVA：系统仅凭指针值本身，就能确定数据在物理上的确切位置（在系统 RAM 中，还是在 H100 的 HBM3 内存上）。

你需要用 cudaDeviceEnablePeerAccess 启用对等访问，否则这个特性无法工作

### 幻灯片 23：cudaDeviceEnablePeerAccess

如果没有这个调用，GPU 可能不会使用 Nvlink，而是走 PCIe

### 幻灯片 24：几个要点

cudaDeviceEnablePeerAccess(peerDevice, 0) 是单向的。如果你需要来回拷贝，或者两侧都要有 kernel 访问内存，你必须在两块设备上都调用它。

如果不启用它，你可能会退回到较慢的 PCIe 路径

在 H100 服务器上，如果可用，这个函数会自动使用 NVLink（900 GB/s）；否则，它使用 PCIe Gen5

### 幻灯片 25：cudaMemCpyPeer

让数据可以在两块独立 GPU 的内存之间直接传输，而不涉及主机（CPU）内存

要使用 Nvlink 进行拷贝，必须先启用 cudaDeviceEnablePeerAccess

### 幻灯片 26：裸 P2P 的真相

在 H100 HGX 板上，你面对的并不是 8 块连成一条线的 GPU，而是一张通过 NVSwitch 连接的复杂网格。

P2P：你必须手动管理自己正在穿越哪条链路。如果 GPU 0 与 GPU 7 交谈，它是直连吗？还是经由 GPU 3 跳转（hop）？

H100 拥有 900 GB/s 的双向 NVLink 带宽。如果你从某个 SM 跨越 NVLink fabric 发出标准的 LD/ST（Load/Store）指令，你很可能只在使用单条独立的 NVLink 通道或其中一部分。你会很难打满那条管道。这就像用一根吸管去喝消防水带里的水。

正是因为这些原因，我们需要像 NCCL 这样的库来管理多 GPU 连接
