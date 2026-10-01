---
title: "第 8.2 课 - Kernel 启动"
lesson_number: "8.2"
lesson_slug: "kernel-launch"
instructor: "Prateek Shukla"
course: "CUDA Programming for NVIDIA H100s"
language: "zh-CN"
original_markdown: "/markdown/lessons/lesson-08-2-kernel-launch.md"
source_page: "pages/lesson-8.2.html"
source_slide_pdf: "H100-Course/slides/8.2 Kernel Launch.pdf"
published_lesson_page: "/pages/lesson-8.2.html"
published_markdown_path: "/markdown/lessons/lesson-08-2-kernel-launch.md"
topics:
  - "启动边界（Launch Bounds）"
  - "grid 常量（Grid Constants）"
  - "依赖 grid（Dependent Grids）"
  - "GDC"
  - "L2 预取（L2 Prefetch）"
code_refs:
  - "matmul_12.cuh"
  - "pingpong_experimental.cuh"
generated_from:
  - "pages/lesson-8.2.html"
  - "H100-Course/slides/8.2 Kernel Launch.pdf"
---

# 第 8.2 课 - Kernel 启动

本文件将已发布的课程页面与完整幻灯片文本合并，方便智能体直接抓取和检索。

## 来源

- 课程页面：`pages/lesson-8.2.html`
- 幻灯片：`H100-Course/slides/8.2 Kernel Launch.pdf`
- 已发布课程 URL：`https://cudacourseh100.github.io/pages/lesson-8.2.html`
- 已发布 Markdown URL：`https://cudacourseh100.github.io/markdown/lessons/lesson-08-2-kernel-launch.md`

## 课程摘要

第 8.2 课是关于交接（handoff）的补充课。重点不仅在于如何启动一个 kernel，还在于 Hopper 如何让你塑造启动驻留（launch residency）、标记 grid 生命周期参数，并放宽流串行化（stream serialization），让下一个 grid 能在前一个 grid 全局退出之前启动。

## 本课的重要性

**grid 的启动时间与数据安全成为两个独立的控制点。**

Hopper 让启动策略成为优化的一部分。本课聚焦于这个转折点：多 kernel 流水线（pipeline）从这里起不再是"先 kernel A 后 kernel B"，而是变成一个牵动调度器、内存与主机启动的协调重叠问题。

## 主题

- 启动边界（Launch Bounds）
- grid 常量（Grid Constants）
- 依赖 grid（Dependent Grids）
- GDC
- L2 预取（L2 Prefetch）

## 课程信息

- **课程位置：**第 08.2 课，位于 Stream-K 之后、多 GPU 之前
- **主要转变：**机器可以让依赖 grid 提前准入，同时仍对那些一旦执行就会破坏正确性的读取精确地施加栅栏（fence）。
- **配套幻灯片：**`8.2 Kernel Launch.pdf`
- **具体锚点：**`matmul_12.cuh`、`pingpong_experimental.cuh`、`cudaLaunchKernelEx`

## 关键要点

- 多 kernel 边界往往就是正确的设计，而不是融合得不够彻底的失败。
- 依赖启动之所以可行，是因为它把调度器资格与内存读取安全分离开。
- 只有当启动窗口大到足以隐藏启动成本、又不至于大到制造有害竞争时，重叠才有帮助。

## 网页课程正文

### 为什么 kernel 启动值得单独一课

笔记明确指出，多 kernel 设计不只是"真正的"优化做完之后的清理工作。有些后处理阶段确实需要边界。epilogue 会撞上实际上限，融合工作会推高寄存器（register）压力，而 LayerNorm 或 Softmax 这类跨 tile 操作需要一个点：一个阶段完成之后，下一个阶段才能基于其输出进行推演。

#### 是什么迫使出现边界

跨 tile 归约、沉重的 epilogue，以及再也无法与累加器 mainloop 舒适共处的状态，让干净的交接成为正确的架构。

#### Hopper 改变了什么

交接不再必须意味着"生产者（producer）完全退出，然后消费者（consumer）冷启动"。只要启动包（launch packet）与指令流相互配合，Hopper 就能重叠启动路径。

#### 本课的核心规则

尽早启动，延后读取。让依赖 grid 在生产者全局完成之前预留上下文、执行安全的准备工作，但对依赖敏感的读取保持栅栏，直到真正安全。

> **优化目标：**不要问"如何避免多个 kernel？"要问"如何把边界做得足够便宜，让正确的分解仍然能以接近满速运行？"

### 启动控制横跨编译期形态与运行时权限

有些启动决策被嵌入 kernel 签名和代码生成（codegen）假设里。另一些则存在于主机侧启动包中。本补充课把这两个层级分开，因为 Hopper 的性能取决于你能否分清：哪个旋钮改变占用率（occupancy），哪个旋钮改变参数生命周期，哪个旋钮改变调度器策略。

| 控制项 | 表达了什么 | 为什么在这里重要 |
| --- | --- | --- |
| `__cluster_dims__(x, y, z)` | 编译期线程块集群（thread block cluster）形状。 | 在 SM90 上很重要，因为集群驻留与多播（multicast）行为是真实 kernel 设计的一部分。 |
| `__launch_bounds__(...)` | 线程数与占用率假设的上界。 | 启动形态是资源预算的一部分，后续的重叠决策必须在这个预算之内进行。 |
| `__maxnreg__(N)` | 每线程寄存器上限。 | 当融合工作与依赖启动重叠同时挤压驻留资源时，这一点很重要。 |
| `__grid_constant__` | 只读的 grid 生命周期 kernel 参数。 | 这正是 Hopper 用来承载描述符（descriptor）与启动期稳定控制数据的元数据路径。 |
| `cudaLaunchKernelEx` + 属性 | 主机侧对启动行为的运行时权限。 | 程序化流串行化许可正是在这里被真正授予的。 |

本地的 fast.cu 示例在这里很有用，尽管它们并不是依赖启动的演示。像 `matmul_12.cuh` 和 `pingpong_experimental.cuh` 这样的文件直接展示了启动形态这一面：集群维度、启动边界和 grid 常量描述符在任何运行时启动包出现之前，就已经被编码进 kernel。

```text
__global__ __launch_bounds__(NUM_THREADS)
void __cluster_dims__(CLUSTER_M * CLUSTER_N, 1, 1)
matmulKernel12(
    int M,
    int N,
    int K,
    const __grid_constant__ CUtensorMap tensorMapC,
    const __grid_constant__ CUtensorMap tensorMapA,
    const __grid_constant__ CUtensorMap tensorMapB,
    int* dspace);
```

### 默认流串行化留下一个尾延迟空洞

在默认规则下，在同一流上启动的 kernel 表现得像一个有序队列。kernel B 在 kernel A 完全退出之前不具备资格。这意味着交接被绑在整 grid 退出上，而不是绑在更早的那个时刻——生产者的工作已经完成到足够程度、下一阶段可以开始准备的时刻。

#### 为什么尾部有害

在生产者 grid 接近尾声时，只剩不断萎缩的 SM 子集还有工作。即使依赖 grid 已经确定并准备就绪，机器的其余部分也可能闲置。

#### 浪费了什么

代价不只是空转的周期。依赖 grid 稍后还要付出冷启动惩罚，因为没有任何上下文被准入，也没有任何安全的预取（prefetch）工作被允许提前开始。

```text
Default behavior:
producer grid fully retires
-> dependent grid becomes eligible
-> dependent grid starts booting
```

这正是本补充课想要软化的那堵串行化之墙。目标不是破坏正确性，而是把"有资格启动"与"可以安全读取依赖敏感数据"解开。

### 依赖启动能够成立，是因为 Hopper 把调度器许可与读取许可拆开了

grid 依赖控制（grid dependency control，GDC）引入了两个控制点。一个是生产者侧的，与调度器对话；另一个是消费者侧的，在依赖真正解除之前对执行施加栅栏。正是这个拆分构成了让重叠成为可能、又不把交接变成竞态的机制。

| 机制 | 做什么 | 不做什么 |
| --- | --- | --- |
| `griddepcontrol.launch_dependents` | 发出信号：依赖 grid 可以在生产者 grid 全局退出之前开始启动。 | 它本身并不能让依赖敏感的加载变安全。 |
| `griddepcontrol.wait` | 把依赖 warp 停在确切的栅栏位置，未解除的读取在那里会变得不安全。 | 它本身不授予提前资格；它只在依赖 grid 内部对执行施加栅栏。 |
| `cudaLaunchAttributeProgrammaticStreamSerialization` | 主机侧许可位，告诉启动机制遵循依赖启动协议。 | 它不能替代设备侧指令；启动包与指令流两者缺一不可。 |

笔记强有力地强调了主机权限这一点：设备指令本身不能覆盖流策略。如果主机启动描述符没有选择加入（opt-in），即使 kernel 包含相关指令，严格的同流串行化仍然生效。

```text
// producer kernel on the stream
producer_kernel<<<gridA, blockA, 0, stream>>>(...);

// dependent kernel with programmatic stream serialization permission
cudaLaunchAttribute attr[1];
attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
attr[0].val.programmaticStreamSerializationAllowed = 1;

cudaLaunchConfig_t cfg{};
cfg.gridDim = gridB;
cfg.blockDim = blockB;
cfg.attrs = attr;
cfg.numAttrs = 1;

cudaLaunchKernelEx(&cfg, consumer_kernel, ...);
```

> **概念上的拆分：**生产者代码说"依赖方现在可以启动了。"依赖代码说"我会就停在这里，直到那些不安全的读取真正安全。"两侧缺一不可。

### 重叠窗口是优化要么兑现要么崩塌的地方

一旦依赖 grid 可以提前到场，有趣的问题就变成了时机。启动得太晚，你几乎隐藏不到任何启动成本。启动得太早，生产者和消费者就会争夺同样的执行槽、寄存器、共享内存（shared memory）和缓存容量。

#### 依赖方的墙

依赖 grid 中的 warp 可以运行准备代码，然后停靠在 `griddepcontrol.wait`。这会提前预留执行上下文——有用，但不是免费的。

#### 生产者的绿灯

`launch_dependents` 应该放在既能带来真实重叠的最晚安全点。太保守会浪费这个特性。太激进会制造竞争，并可能降低总吞吐量。

#### 预取分工

笔记着重展示了一个非对称的依赖 kernel：受依赖约束的 warp 停在 wait 栅栏处，而另一些独立的 warp 把静态权重预取进 L2。这让内存系统保持忙碌，同时仍尊重激活读取的正确性。

| 重叠太少 | 重叠太多 | 健康目标 |
| --- | --- | --- |
| 依赖 grid 仍要付出大部分冷启动成本。 | 生产者与消费者争夺发射槽、驻留资源、L2 和内存队列。 | 在仍能隐藏可观启动延迟的最晚安全时刻启动。 |
| 从该特性获得的净收益很小。 | 预取的缓存行可能在计算需要它们之前就被搅出 L2。 | 只预取稳定的、可复用的张量，其安全性不依赖生产者的完成。 |
| 反正在生产者完成之前，依赖 grid 基本处于闲置。 | 即使并发在纸面上看起来更壮观，总运行时间也可能上升。 | 观察 L2 命中率、队列压力和驻留余量，而不是仅凭表象评判重叠。 |

```text
Dependent-launch protocol:
producer issues launch_dependents
-> dependent grid boots
-> dependency-bound warps stop at wait
-> dependency resolves
-> stalled warps continue and consume the produced data
```

### 实践指导

1. **不要把多 kernel 边界当作失败。**如果计算确实需要一条边界，正确的问题是如何隐藏它的代价，而不是否认它存在。
2. **让启动策略保持正确的分层。**编译期启动形态、grid 生命周期参数与主机侧启动许可解决的是不同的问题。
3. **记住许可由主机掌控。**程序化流串行化是选择加入（opt-in）的行为，而不是对同流工作的默认解释。
4. **把启动重叠与读取安全分开。**提前的调度器准入与正确的内存可见性相关，但它们不是同一个事件。
5. **为净吞吐量调优，而不是为好看的重叠截图调优。**损害缓存驻留或队列压力的并发，很容易输给一个稍晚但更干净的启动点。

#### 术语表

| 术语 | 定义 |
| --- | --- |
| 程序化流串行化（Programmatic stream serialization） | 由主机授权的、对默认同流排序的一种放宽，允许依赖 grid 参与启动/等待协议。 |
| 启动依赖方（Launch dependents） | 生产者侧信号，告诉调度器某个依赖 grid 可以开始启动。 |
| 依赖等待（Dependent wait） | 消费者侧栅栏，在依赖敏感读取变得合法之前使 warp 停滞。 |
| 重叠窗口（Overlap window） | 从依赖方被提前准入，到未解除的依赖真正可以安全消费之间的时间间隔。 |
| grid 常量（Grid constant） | 一种只读 kernel 参数，保证在单个已启动 grid 的生命周期内保持稳定。 |

### 继续课程

本补充课为单节点 kernel 交接的故事收尾。下一课将放下 kernel 边界，进入 fabric 本身：NVLink、NVSwitch、rail，以及决定多少块 H100 合成一台训练机器的系统拓扑。

## 完整幻灯片文本

从 `H100-Course/slides/8.2 Kernel Launch.pdf` 使用 `pdftotext -layout` 提取。共 10 张幻灯片。

### 幻灯片 1：Kernel 启动

Prateek Shukla

### 幻灯片 2：一些重要的约束与函数

__cluster_dims__(x,y,z) 用于编译期线程块集群形状。

__launch_bounds__(maxThreads[,minBlocksPerSM[,maxBlocksPerCluster]])。第 3 个参数是集群相关的那个，在 SM90 上更重要。

__maxnreg__(N) 用于限制每线程寄存器数量。

__grid_constant__ 用于只读的 grid 生命周期 kernel 参数。这在 CUtensorMap / TMA 描述符场景中尤其常见。

__forceinline__ 强制 nvcc 在单个翻译单元内内联该函数。

__restrict__ 告诉 nvcc：在该指针于当前作用域中的生命周期内，所指向的内存不会被任何其他用于访问同一数据的指针起别名。

### 幻灯片 3：为什么需要多 kernel 设置

epilogue 有一个硬上限。强行突破它会摧毁你的性能。融合操作会与 mainloop 的累加器 tile 竞争，可能造成寄存器压力。需要跨 tile 依赖的操作（LayerNorm、Softmax）在架构上与 GEMM epilogue 的独立 tile 处理不兼容。数学上严格要求一个 kernel 边界才能正确归约。你必须接受 kernel 边界和中间 DRAM 写入。工程挑战不是强行做成单体 kernel，而是让 kernel 之间的交接几乎零成本。这要靠部署依赖启动协议、L2 预取策略和跨 grid mbarrier 来实现

### 幻灯片 4：流串行化问题

通常，一个 grid 必须完全退出之后，同一流上的下一个 grid 才能开始发射工作。这种严格的完成排序造成了利用率缺口：当第一个 grid 进入其低占用率尾部时尤为明显。硬件调度器把同流的 grid 当作单个有序队列。在 Kernel A 报告全局完成之前，它不会向 Kernel B 分发线程块（thread block）槽位，即使一些 SM 已经利用不足。随着 Kernel A 接近完成，只剩不断萎缩的 SM 子集仍有活动工作。其余 SM 闲置，因为在默认流规则下没有任何下一 grid 的线程块具备资格。交接是一个整 grid 退出事件，而不是逐 tile 的传递。这意味着所有依赖安全的重叠机会都被忽略，除非显式启用依赖启动控制。关键代价是尾延迟放大：在 Kernel A 的最后阶段，硅片还在，却没有发射有用的指令。

### 幻灯片 5：硬件信号（GDC 指令）

GDC 指令在 GPU 执行时间线内控制 grid 交接的时机。其物理目的是把"调度器可以启动依赖方"与"依赖方可以读取依赖敏感内存"区分开。griddepcontrol.launch_dependents 是由运行中的 warp 发出的、面向调度器的信号。它告诉分发逻辑：依赖 grid 现在已有资格启动，甚至早在活动 grid 全局退出之前。griddepcontrol.wait 是依赖 grid 中的一道硬执行栅栏。到达该指令的 warp 会被扣住，直到启动依赖条件得到满足，从而防止对尚未定型数据的过早读取。两者共同构成一个两段式协议：提前启动许可加内存安全闸门。调度器可以重叠启动工作，同时栅栏为依赖敏感的加载保持正确性。这些是指令流中的物理控制点，因此摆放位置直接改变重叠时机与硬件竞争。

### 幻灯片 6：主机启动权限

只有 CPU 启动包才能为依赖重叠授权放宽的流行为。其物理目的是：除非主机软件显式选择加入，否则保持默认流语义完好。启动时，CPU 把属性写入由 GPU 命令处理器消费的命令描述符。其中一个属性授予在前驱完全退出之前调度依赖 grid 的许可。如果该许可位存在，调度器状态机会把 launch_dependents 和 wait 当作依赖控制指令来遵循。如果它不存在，调度器执行正常串行化，这些指令没有任何启用效果。因此，交接权限由主机发起、由硬件执行。除非启动元数据授权，设备侧信号无法覆盖流策略。这在混合负载下保护了正确性：某些流需要严格排序，另一些流需要受控重叠。

### 幻灯片 7：依赖方的墙（wait）

这一阶段定义依赖 grid 中生产者 warp 的早期启动停滞点。其物理目的是让依赖 kernel 提前预留执行上下文，同时阻断不安全的内存流量。

依赖 grid 可以被准入，并在可用的 SM 上开始执行准备指令。生产者 warp 一直运行到抵达 griddepcontrol.wait，在那里被硬件停靠。

停靠期间，这些 warp 不能发出依赖敏感的全局读取，例如由活动 kernel 产生的激活张量。这防止了因读取尚未定型数据而导致的缓存填充与内存序违规。

一旦依赖信号得到满足，栅栏释放，同一批 warp 恢复发出加载。结果就是立即向前推进，而不必在生产者完成之后再付出完整的冷启动延迟。

代价是资源被预留：停靠的 warp 仍占用调度槽位，并可能减少其他工作的瞬时余量。

### 幻灯片 8：生产者的绿灯（launch）

这一阶段是活动 kernel 发出那个精确信号、打开依赖调度的地方。其物理目的是在最后一个正确性安全的时刻触发重叠，同时仍能暴露启动延迟隐藏（latency hiding）的机会。活动 grid 中的生产者 warp 在其剩余写入已推进到足以支持依赖启动时，执行 griddepcontrol.launch_dependents。该信号针对的是调度器资格，而不是立即的内存读取许可。在常驻（persistent）kernel 中，正确的摆放位置与最后一个被调度的相关工作 tile 的生命周期绑定。信号应当与真正的流水线完成对齐，而不仅仅是代码区域的词法末尾。信号发出后，依赖线程块可以在活动 grid 排空剩余工作的同时被分发。依赖 grid 的 wait 栅栏仍会守卫依赖敏感的读取，直到所需条件得到满足。信号发得太晚会浪费重叠；发得太早会增加并发压力，并可能降低净吞吐量。

### 幻灯片 9：L2 预取技巧（拆分 DMA）

这一阶段通过在依赖 kernel 的 warp 之间划分角色来实现重叠。其物理目的是：当受依赖约束的生产者仍被栅栏拦住时，让内存 fabric 忙于与依赖无关的传输。

一个生产者 warp 到达依赖屏障（barrier）并在激活读取之前暂停。另一个预取 warp 继续运行，为与生产者完成无关的静态权重发出 DMA 式请求。

这些请求的目标是在计算需求之前填充 L2，从而在依赖栅栏打开时减少日后的缺失惩罚。因此，内存系统可以在原本停滞的依赖等待期间预热有用的缓存行。

这种交接是非对称但安全的：激活流量在栅栏后面等待，权重预取则不必。当栅栏释放时，生产者会看到更优的有效延迟，因为部分工作集已经驻留。

只有当预取的数据具有高复用度、并能在 L2 中存活到被消费时，这一策略才有效。

### 幻灯片 10：调优重叠窗口

重叠比例由依赖启动信号发出与真正依赖就绪之间的时间间隔决定。更大的间隔意味着更大的潜在隐藏空间，但前提是资源保持不拥堵。

如果信号发得太早，两个 kernel 会争夺 SM 发射槽、寄存器文件容量、共享内存分配和内存端口。这可能把每个 kernel 都压慢到总时间不降反升。如果预取过于激进，L2 缓存行会在被使用之前就被搅动逐出，DRAM 流量因回填而飙升。带宽随后被花在把被逐出的数据搬回来，抹掉了预取重叠的收益。

实际调优要借助硬件计数器：在最晚的安全点启动，只预取稳定/被复用的张量，并把 L2 命中率与内存队列压力作为主要的守护指标。
