---
title: "第 8.1 课 - Stream-K"
lesson_number: "8.1"
lesson_slug: "stream-k"
instructor: "Prateek Shukla"
course: "CUDA Programming for NVIDIA H100s"
language: "zh-CN"
original_markdown: "/markdown/lessons/lesson-08-1-stream-k.md"
source_page: "pages/lesson-8.1.html"
source_slide_pdf: "H100-Course/slides/8.1 Stream-K.pdf"
published_lesson_page: "/pages/lesson-8.1.html"
published_markdown_path: "/markdown/lessons/lesson-08-1-stream-k.md"
topics:
  - "Stream-K"
  - "Fixup"
  - "常驻调度"
  - "分组"
  - "L2 局部性"
code_refs:
  - "sm90_tile_scheduler_stream_k.hpp"
  - "sm90_tile_scheduler.hpp"
  - "sm90_tile_scheduler_group.hpp"
generated_from:
  - "pages/lesson-8.1.html"
  - "H100-Course/slides/8.1 Stream-K.pdf"
---

# 第 8.1 课 - Stream-K

本文件将已发布的课程页面与完整幻灯片文本合并，方便智能体直接抓取和检索。

## 来源

- 课程页面：`pages/lesson-8.1.html`
- 幻灯片：`H100-Course/slides/8.1 Stream-K.pdf`
- 已发布课程 URL：`https://cudacourseh100.github.io/pages/lesson-8.1.html`
- 已发布 Markdown URL：`https://cudacourseh100.github.io/markdown/lessons/lesson-08-1-stream-k.md`

## 课程摘要

第 8.1 课是让常驻（persistent）kernel 显得完整的调度器（scheduler）附录。Stream-K 不是把一个完整输出 tile 分配给一个 CTA、然后接受一条孱弱的尾部波（wave），而是把剩余的 K 工作切成一条均衡的工作带（tape），显式跟踪 split 的所有权，并且只在尾部真正需要的地方支付归约（reduction）成本。

## 本课的重要性

**调度的单元从输出 tile 转移到一份受控的 K 工作预算。**

Kernel Design 已经介绍了常驻调度。这个附录单独拎出了棘手的部分：如何消除半空的尾部波，而不把整个问题变成昂贵的 split-K fixup。

## 主题

- Stream-K
- Fixup
- 常驻调度
- 分组
- L2 局部性

## 课程信息

- **课程位置：**第 08.1 课，介于 Kernel Design 与 Multi GPU 之间
- **核心转变：**一些 CTA 不再拥有完整的 tile，转而拥有 tile 内部的数学区间。
- **配套幻灯片：**`8.1 Stream-K.pdf`
- **代码锚点：**`sm90_tile_scheduler_stream_k.hpp`、`sm90_tile_scheduler.hpp`、`sm90_tile_scheduler_group.hpp`

## 关键要点

- Stream-K 是一个尾部均衡调度器，而不是问题中每个 tile 的默认答案。
- 每个 split 跟踪自己在 K 中从哪里开始、拥有多少 K，以及是否负责最终 epilogue。
- 分组（Groups）负责找回 L2 局部性（locality）——否则 1D 的 K 工作带会把协作的 CTA 分散到整个输出空间。

## 网页课程正文

### Stream-K 为什么存在

问题不在于正确性。朴素的数据并行常驻调度是正确的。问题在于利用率（utilization）。如果输出 tile 的数量不是可用 SM 数量的整洁倍数，最后一条波就会让机器的很大一部分闲置，而只有少数几个 CTA 在收尾。

#### tile 并行视角

一个 CTA 拥有一个输出 tile。这很干净、也便宜——直到最后一条波变得稀疏，大量 SM 都在等最后几个 tile 退场。

#### Stream-K 视角

剩余的工作被当作一条沿 K 方向的数学迭代工作带。尾部 tile 可以被切开，让每个 SM 的完成时间更接近彼此。

讲义谨慎地指出，这通常是一种混合策略。早期的波往往保持纯数据并行，因为那样开销最低。当它瞄准尾部波——分解失配正在那里造成可见伤害——时，Stream-K 最有价值。

> **讲义中一个有用的经验法则：**如果尾部大体仍然占满，更好的选择可能是保留常规调度器、避免不必要的 split-K 归约开销。当尾部明显欠满时，Stream-K 才最有吸引力。

### 工作单元与 split 让调度器显式化

对一个输出 tile 而言，GEMM 是许多 K tile 之和。如果一个 CTA 把它们全部算完，就不需要归约。如果多个 CTA 各算这些 K tile 的一部分，那么每个 CTA 拥有该输出 tile 的一个 split，结果就需要 fixup。

| 字段 | 含义 | 为什么重要 |
| --- | --- | --- |
| `M_idx`, `N_idx`, `L_idx` | 输出 tile 坐标。 | 告诉 CTA 它为哪个 C tile 做贡献。 |
| `K_idx` | 该 split 在输出 tile 的 K 维度内部从哪里开始。 | 把第一个 split 与中间或 final split 区分开。 |
| `k_tile_count` | 该 split 计算多少个 K tile。 | 告诉你该 CTA 对这个输出 tile 实际拥有多少数学工作。 |
| `k_tile_remaining` | 工作单元还剩多少没有处理。 | 很重要，因为一个 CTA 沿工作带前进时可能跨越多个 split。 |

一个 split 就是一个 CTA 对一个输出 tile 的贡献。一个 CTA 可以在同一段被分配的数学工作范围内结束一个 tile、开始下一个 tile，这正是调度器同时跟踪 tile 坐标与 tile 内精确 K 起点的原因。

```text
// Conceptual split state
tile: (M_idx, N_idx, L_idx)
K_idx: where this CTA starts in the tile's K dimension
k_tile_count: how much K this CTA computes

is_final_split = (K_idx + k_tile_count) == k_tiles_per_output_tile
```

### 第一个、中间与 final split 的角色直接从这个状态中推出

一旦 CTA 知道了自己的 `K_idx` 和 `k_tile_count`，它的角色就不再神秘。幻灯片描述了三种情况，而本仓库里的 Stream-K 调度器头文件用 `is_final_split(...)` 和 `compute_epilogue(...)` 这样的辅助函数与它们对应。

#### 第一个 split

`K_idx == 0`。还没有人为这个 tile 写过工作区（workspace），所以第一个 split 负责初始化部分结果。

#### 中间 split

拥有严格位于 tile 内部的一段 K 区间。它把自己的部分和归约进已有工作区，并不拥有最终 epilogue。

#### final split

覆盖 K 维度的末端。它等待先前的 split，加载它们的部分和，加上自己的部分，然后拥有最终 epilogue。

这才是真正的概念转变。CTA 仍然可以是常驻的，仍然可以遵循一个确定性的（deterministic）工作循环，但它不再保证从头到尾拥有一个完整的输出 tile。

> **反向迭代帮了 final split：**讲义强调，worker 常以相反的 K 顺序遍历共享 tile，让收尾的 split 更晚到达 fixup 点，从而减少它等待先前 split 完成的时间。

### 工作区与锁协议是均衡尾部的代价

split 所有权之所以能成立，是因为部分累加器可以被暂存进全局工作区，而进度可以用一把锁来跟踪。讲义把这把锁描述为每个输出 tile 一个整数，它单调地编码已经完成并发布了多少 K 工作。

| 机制 | 目的 | 对执行的影响 |
| --- | --- | --- |
| 归约工作区 | 存储跨 CTA 拆分的 tile 的部分累加器。 | 创造出一个场所，让后续 split 可以加载并归约先前的工作。 |
| 锁/进度计数器 | 记录该 tile 已经完成了多少 K 工作。 | 让后续 split 知道自己可以按确定性方式还是机会主义方式进行归约。 |
| 独立归约单元 | 允许在某些调度器模式下把归约与 epilogue 工作显式建模。 | 当调度器判定有利时，把数学所有权与最终 fixup 所有权解耦。 |

本仓库的 Stream-K 头文件同时暴露确定性与非确定性两种归约模式。确定性路径等待它期望的精确累计 K 进度。非确定性路径更宽松，主要只关心工作区已初始化，随后中间 split 们便竞相往里归约。

```text
// Conceptual fixup flow
first split   -> store partials, publish progress
middle split  -> wait until workspace is valid, reduce partials, publish progress
final split   -> wait for prior progress, load-add all required partials, run epilogue
```

### 分组在 1D 工作带打散 CTA 之后找回局部性

朴素的 Stream-K 分解均衡了工作，却破坏了那种有利于 L2 的漂亮空间波模式。分组就是局部性修复机制。它们把 Stream-K 单元划进子组（sub-group），让协作的 worker 彼此更靠近输出空间的同一区域，复用更多有用的缓存状态。

#### 基础常驻调度器

保留标准的 tile 并行常驻循环、swizzle 与光栅（raster）顺序。当不需要 split-K fixup 时，它是那个廉价的基线。

#### 分组常驻调度器

把常驻性扩展到多个分组 GEMM 问题上，把它们当作一个长的线性 tile 空间，同时保留每组的 swizzle 与局部性元数据。

#### Stream-K 调度器

增加 split 跟踪、归约所有权与局部性分组，让尾部能被均衡，同时不必完全放弃复用这个故事。

| 文件 | 在这个故事中的角色 |
| --- | --- |
| `sm90_tile_scheduler.hpp` | 基础静态常驻调度器与 swizzle 机制。 |
| `sm90_tile_scheduler_group.hpp` | 面向单次 launch 中多个 GEMM 问题的分组常驻扩展。 |
| `sm90_tile_scheduler_stream_k.hpp` | Stream-K 路径所使用的 split-K、fixup、归约与分组局部性调度器。 |

讲义还提到 HyTiS，把它作为对抗波量化（wave quantization）的另一种方式。即使你不采用它，它的观点也有用：Stream-K 不是唯一答案。当 K 是天然的切分维度时，它是一种用额外归约机制换取更好尾部利用率的答案。

### 实践指导

1. **为失配而用 Stream-K，而不是为用而用。**目标是修好尾部波，而不是在常规常驻调度器已经高效时还去拆分每个 tile。
2. **显式跟踪 split 所有权。**`K_idx`、`k_tile_count` 与 final-split 状态，是决定一个 CTA 究竟是存储、归约还是运行 epilogue 的机制。
3. **记住 fixup 是真实的成本。**工作区流量、锁和跨 CTA 归约，正是混合策略常常胜过处处应用 Stream-K 的原因。
4. **均衡之后保护局部性。**分组很重要，因为一条纯 1D 工作带可以在解决利用率的同时悄悄摧毁 L2 复用。
5. **把调度器文件当作一个家族来读。**基础常驻调度器、分组调度器与 Stream-K 调度器，是对同一个利用率问题的三个互相关联的答案。

#### 术语表

| 术语 | 定义 |
| --- | --- |
| Split | 当 tile 沿 K 被划分时，一个 CTA 对一个输出 tile 的贡献。 |
| Fixup | 把来自多个 CTA 的部分累加器归约成一个最终输出 tile 的过程。 |
| Final split | K 区间到达 tile 末尾、因而拥有最终 epilogue 的那个 split。 |
| Separate reduction | 一种调度器模式，其中归约与 epilogue 工作可以被建模为不同的工作单元。 |
| Group | Stream-K 单元中保持局部性的一个子集，在工作带属于自己的那一段上协作。 |

### 继续课程

这个调度器附录为尾部均衡的故事画上句号，但还剩一个 kernel 补充。下一页看的是 launch 边界本身：launch bounds、依赖 grid、程序化 stream 串行化（programmatic stream serialization），以及 Hopper 如何让一个 grid 的启动与另一个 grid 的尾部相互重叠。

## 完整幻灯片文本

使用 `pdftotext -layout` 从 `H100-Course/slides/8.1 Stream-K.pdf` 提取。幻灯片总数：10。

### 幻灯片 1：Stream-K

Prateek Shukla

### 幻灯片 2：第一性原理

对一个输出 tile C_tile，GEMM 计算：

C_tile = sum over K-tiles of (A_tile_k * B_tile_k)

如果一个 CTA 为该输出 tile 计算全部 K-tile，就不需要跨 CTA 归约。

如果多个 CTA 各自只计算 K-tile 的一个子集，每个 CTA 产出一个部分和，而这些部分和必须被合并。

在这个调度器中，那个合并步骤被称为 fixup。

### 幻灯片 3：工作单元

每个工作单元包括：

- tile 坐标 (m_idx, n_idx, l_idx)

- k_idx：输出 tile 内的起始 k-tile

- k_tile_count：该工作单元为这个输出 tile 计算的 k-tile 数量

是否需要归约由以下决定：

- k 方向完整 tile：k_tile_count == k_tiles_per_output_tile -> 不需要归约

- k 方向部分 tile：k_tile_count != k_tiles_per_output_tile -> 需要归约

这就是 requires_fixup(...) 背后的关键谓词。

### 幻灯片 4：Split

一个 CTA 被分配的 k-tile 迭代区间可能落在一个输出 tile 的中间。例如，3 个输出 tile 各有 90 个 k-tile，而有 4 个 CTA 单元：

一个"split"是一个 CTA 对单个输出 tile 的贡献。上面的 Unit 1 有两个 split——一个对应 tile 0 的尾部，一个对应 tile 1 的头部。代码逐个处理它们（这就是 advance_to_next_work 中的 k_tile_remaining 循环）。

### 幻灯片 5：跟踪三元组

对一个单独的 split（一个 CTA 在一个输出 tile 上的工作）：

K_idx：该 split 在输出 tile 的 K 维度内部从哪里开始。

K_idx = tile_iter_start - output_tile_iter_start。

对 Unit 0 在 tile 0 上的工作，K_idx = 0。对 Unit 1 在 tile 0 上的工作，K_idx = 67。

k_tile_count：该 split 处理多少个 k-tile。对 tile 0 上的 Unit 0，是 67。对 tile 0 上的 Unit 1，是 23（= 90 - 67）。

is_final_split()：(K_idx + k_tile_count) == k_tiles_per_output_tile。当该 split 覆盖 K 维度的末尾时为真。Unit 1 在 tile 0 上的 split 是一个 final split（67 + 23 = 90）。

### 幻灯片 6：三种角色直接得出

给定一个输出 tile 可能有 2-4 个 CTA 各自计算 K 的一部分：

K_idx == 0 -> 你是第一个 split。你计算了 k-tile [0, N)。在你之前没有人向工作区写过任何东西。直接存储即可。

is_final_split() == true 且不是 separate reduction -> 你是 final split 和 epilogue 所有者。你计算了 k-tile [X, 90)。等待你之前的所有人，加载它们的累加结果，加上你的，运行 epilogue。

其余一切 -> 你是中间 split。你计算了 k-tile [A, B)，其中 0 < A < B < 90。你需要把自己的部分和归约进工作区里已有的内容。

### 幻灯片 7：锁：它在物理上是什么

全局内存中有一个连续的 int 数组，每个输出 tile 一个（多 warpgroup kernel 还要乘以 num_barriers）。
指向这个数组的指针紧挨着归约数据缓冲区，位于同一次分配之中。kernel 启动时，每把锁都从 0 开始。
锁是一个单独的整数，编码了针对给定输出 tile 已完成并写入工作区的 K 维度工作量。在普通（非 separate reduction）模式下，它统计已处理的累计 k-tile 数。它只会增加。
锁以 K-tile 空间编码进度。对确定性模式，每个 split 等待与其起始位置匹配的精确累计 K-tile 数，从而强制一个严格的从左到右归约顺序。对非确定性模式，中间 split 只需要知道工作区已被初始化（lock >= 1），然后它们就竞相以原子方式往里归约。

### 幻灯片 8：分组

分组是一种 L2 缓存局部性优化。它们把 stream-K 单元划分成 G 个独立子组，每个组只在自己的 stream-K tile 子集上协作。这是 stream-K 特有的优化，因为 stream-K 破坏了那种漂亮的、基于波的光栅化模式——单个 CTA 可能横跨来自输出 grid 不同区域的 tile，从而摧毁那种局部性。
没有分组（G=1）时，所有 stream-K 单元共享一个大池子，横跨所有 stream-K 输出 tile 的 K tile。Unit 0 可能在做 tile 0 和 tile 1，而 unit 7 在做 tile 5 和 tile 6——完全不同的空间位置。它们在 L2 缓存中的数据毫无重叠。
有了分组之后，每个组里的 unit 0 会计算相同 K 范围的 tile——按照数据并行表述的光栅化顺序，这些 tile 本会被分配到同一条波里

### 幻灯片 9：分组层级

分组（最多 8 个，为了 L2 局部性）

每个组包含多个 cluster-tile

每个集群（cluster）包含多个 CTA（线程块）

每个 CTA 处理 K-tile

分组沿光栅化维度确定。例如，如果你沿 M 光栅化，且 problem_blocks_m / cluster_m = 4，你会得到 4 个组。各组在输出空间中交错（interleave）。最终的 output_tile_id 计算如下：
output_tile_id = (output_tile_id_in_group * num_groups) + group_idx

### 幻灯片 10：HyTiS

HyTiS 通过让部分波（partial wave）使用更细粒度的 tile 来解决波量化，让更多 SM 保持忙碌。它是一种纯空间分解（MxN）、使用异构 tile 大小，对比 Stream-K 沿 K 维度分解、使用同构 tile 大小。
HyTiS 不用 stream-k，而是在一次 kernel launch 中使用两种不同的 tile 大小：
- 大 tile（例如 128x256）给完整的波 -> 最大吞吐量
- 小 tile（例如 64x64）给部分波 -> 最小延迟
没有归约、没有工作区、没有屏障、没有 fixup
代价是：当问题在 M 和 N 上很小但在 K 上很大时，HyTiS 帮不上忙
