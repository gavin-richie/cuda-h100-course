---
title: "第 4 课 - cuTensorMap"
lesson_number: "4"
lesson_slug: "cutensormap"
instructor: "Prateek Shukla"
course: "CUDA Programming for NVIDIA H100s"
language: "zh-CN"
original_markdown: "/markdown/lessons/lesson-04-cutensormap.md"
source_page: "pages/lesson-4.html"
source_slide_pdf: "H100-Course/slides/4. cuTensorMap.pdf"
published_lesson_page: "/pages/lesson-4.html"
published_markdown_path: "/markdown/lessons/lesson-04-cutensormap.md"
topics:
  - "CUtensorMap"
  - "TMA"
  - "Swizzle"
  - "交错"
  - "L2 提升"
code_refs:
  - "fast.cu/examples/matmul/matmul_12.cuh"
generated_from:
  - "pages/lesson-4.html"
  - "H100-Course/slides/4. cuTensorMap.pdf"
---

# 第 4 课 - cuTensorMap

本文件将已发布的课程页面与完整幻灯片文本合并，方便智能体直接抓取和检索。

## 来源

- 课程页面：`pages/lesson-4.html`
- 幻灯片：`H100-Course/slides/4. cuTensorMap.pdf`
- 已发布课程 URL：`https://cudacourseh100.github.io/pages/lesson-4.html`
- 已发布 Markdown URL：`https://cudacourseh100.github.io/markdown/lessons/lesson-04-cutensormap.md`

## 课程摘要

第 4 课是描述符（descriptor）一课。Hopper TMA 之所以能真正异步，是因为传输状态被提前编码：基址指针、维度、步长（stride）、元素遍历方式、swizzle、交错（interleave）、L2 抓取策略与越界行为，全部成为一个硬件可以据以执行的可复用对象。

## 本课的重要性

**地址算术不再活在指令流里，而是进入元数据。**

本课解释为什么 Hopper TMA 不只是"一种更快的拷贝"。它是一条描述符驱动的搬运路径，其性能与正确性取决于在异步操作开始之前就把张量结构、布局（layout）策略与抓取行为编码好。

## 主题

- CUtensorMap
- TMA
- Swizzle
- 交错
- L2 提升

## 课程信息

- **课程位置：**第 04 课，共 10 课
- **核心转变：**搬运成为拥有一份显式契约的一等对象。
- **配套幻灯片：**`4. cuTensorMap.pdf`
- **代码锚点：**`fast.cu/examples/matmul/matmul_12.cuh`

## 关键要点

- 描述符让 Hopper 得以只发起一次异步传输，然后把剩下的交给硬件完成。
- 维度、元素步长、字节步长与 box 维度不是可以互换的单位。
- swizzle、交错、L2 提升（L2 promotion）与 OOB 填充是性能与正确性的旋钮，不是装饰。

## 网页课程正文

### TMA 改变了什么

TMA 是 Hopper 在全局内存（global memory）与共享内存之间进行大规模异步张量拷贝的硬件路径。Hopper 不再把搬运与每线程地址生成绑在一起，而是让一个发起线程描述这次传输，然后继续执行，由硬件在后台处理该操作。

#### 描述符驱动搬运之前

线程大量参与地址生成，拷贝碎片化成更小的片段，SM 仍然被拴在搬运的机械细节上。

#### 有了 Hopper TMA 之后

一条指令加一个描述符就能描述整个 tile 传输，而搬运路径基本由硬件自行处理。

这就是本课不断回到描述符的原因。Hopper 的异步机器之所以能运转，是因为布局知识被提前编码，而不是每次都由发起线程重新计算。

> **核心思想：**`CUtensorMap` 把传输状态从指令流中移出，放进一个可复用的 128 字节元数据对象中，TMA 硬件可以据以执行。

### 描述符包含什么

一个 `CUtensorMap` 在 host 上创建，用 CUDA Driver API 编码，然后由发出 TMA 指令的运行时路径使用。概念上它回答三个问题：张量住在哪里、它如何布局、硬件应该如何搬运它。

#### 寻址元数据

基址指针、rank、全局维度与字节步长定义了内存中的源张量。

#### 遍历元数据

box 维度与元素步长定义 tile 大小与每个维度的移动行为。

#### 布局与策略元数据

数据类型、swizzle、交错、L2 提升与 OOB 填充塑造传输的行为方式。

#### 编码工作流

1. 在 host 内存中创建一个 `CUtensorMap` 对象。
2. 用 CUDA Driver API 对它编码，通常使用 `cuTensorMapEncodeTiled(...)`。
3. 把编码好的描述符传给将发起 TMA 搬运的路径。

```text
CUtensorMap tma_desc;
// host-side descriptor object, 128 bytes, 128B aligned

// encode with CUDA Driver API
// cuTensorMapEncodeTiled(&tma_desc, ...);
```

### 逐字段的关键规则

破坏一个 tensor map 最简单的方式就是混淆单位与不变量。课程材料非常明确：有些字段以元素为单位，有些以字节为单位，还有些带有取决于数据类型或交错模式的对齐约束。

| 字段 | 含义 | 单位 / 约束 |
| --- | --- | --- |
| 数据类型 | 定义源张量如何被解释，以及适用哪些对齐 / 大小规则。 | 类型枚举，如 FP16、BF16、FP32、TF32、整数。 |
| rank | 张量维度的数量，不是矩阵的秩。 | 笔记将支持的搬运范围界定为 1D 到 5D 张量拷贝。 |
| 全局地址 | HBM 中的基址设备指针。 | 基线要求 16B 对齐，启用交错时更严格。 |
| `globalDim` | 每个张量维度的大小。 | 以元素为单位。 |
| `globalStrides` | 在源内存中跨更高维度移动的距离。 | 以字节为单位。 |
| `elementStrides` | 异步拷贝期间遍历的逻辑步长。 | 以元素为单位，数组大小等于 rank。 |
| `boxDim` | 一次 TMA 传输的 tile 大小。 | 以元素为单位，必须符合下游布局假设。 |

> **注意区分：**全局维度与 box 维度以元素为单位，而全局步长以字节为单位。丢掉这个区别是编码出错误描述符的最快方式之一。

#### 数据类型与 rank

笔记列出了常见类型，包括无符号整数、有符号整数、FP16、BF16、FP32、FP64 与 TensorFloat-32 变体。这里的 rank 指张量维度数，不是线性代数中的矩阵秩。

#### 对齐与交错约束

交错模式会收紧地址与步长规则。笔记明确指出：16B 交错要求 16 字节对齐，32B 交错要求 32 字节对齐，步长粒度要与之匹配，并且使用交错时维度数至少为 3。

### swizzle 与交错是布局变得硬件友好的方式

swizzle 是共享内存一侧的布局变换。交错是全局内存中源布局的解码。两者相关，因为 Hopper 常常必须先解码打包的源布局，再把结果放进下游消费者（consumer）能够高效读取的共享内存（shared memory）排布中。

#### swizzle 为什么存在

共享内存有 32 个 bank。常规的 strided 布局可能反复命中相同的 bank，把本应并行的访问串行化。Hopper 在传输过程中使用硬件加速的地址置换，使共享内存中的物理放置更好地匹配消费 warp 的访问模式。

| 模式 | 改变什么 | 典型用途 |
| --- | --- | --- |
| 32B | 最小的 swizzle 跨度，缓存行利用率更低，tile 更窄。 | 非常小的内层维度。 |
| 64B | 比 32B 更好的 bank 分散度，但仍不是最强的布局保护。 | 中等宽度的 tile。 |
| 128B | 最大的常见 swizzle 跨度，与主流的共享内存 / 张量消费模式对齐。 | 常见的 FP16 / BF16 张量核心流水线（pipeline）。 |

笔记还区分了 atom 大小与跨度（span）。atom 是 swizzler 移动的不可分割块。跨度是 swizzle 模式的重复窗口。这就是笔记强调 bounding-box 约束的原因：如果内层维度以字节计超过 swizzle 跨度，模式就会重复，可能重新制造 bank 冲突。

#### 交错耦合

有些源布局在物理上是交错的，例如 NC/8HWC8 或 NC/16HWC16 风格的打包（packing）。交错模式告诉 TMA 如何解码这些块。课程笔记明确指出：32B 交错必须与 32B swizzle 配对，因为这两条硬件路径同步联动运行。

### L2 提升与越界填充补全搬运契约

描述符并不止步于布局。它还控制系统向 L2 抓取的激进程度，以及当 tile 超出有效张量边界时目的地应收到什么值。

#### L2 提升

| 模式 | 含义 | 最适合 |
| --- | --- | --- |
| NONE | 使用默认的较小抓取行为。 | 稀疏（sparse）或不规则访问，过度抓取会浪费带宽的场景。 |
| L2_64B | 提升到 64B 抓取宽度。 | 连续宽度较窄、密度中等的场景。 |
| L2_128B | 一次抓取完整缓存行。 | 常见的稠密 GEMM 或卷积类流水线。 |
| L2_256B | 逻辑上把两条完整缓存行一起抓取。 | 复用即时、局部性足以证明值得占用缓存空间的高密度工作负载。 |

本课的经验法则很简单：稠密 GEMM 与卷积通常受益于 128B 或 256B 提升，而稀疏查找通常不做提升更好，以避免抓取永远不会被触碰的邻居数据。

#### 越界填充

边缘 tile 仍然需要搬运。OOB 填充让传输无需手动逐元素防护即可继续：为越界位置合成目的地的值，同时保持 HBM 中的源张量不变。

- 零填充对 padding 与干净的边界计算很有用。
- 与 NaN 相关的填充模式对调试或特殊行为很有用。
- 这简化了那些天然会延伸到张量边界之外的 tiled kernel。

### 实践指导

1. **先把描述符的数学搞对。**rank、维度与字节步长必须描述真实的张量。
2. **遵守对齐规则。**交错模式与数据类型会改变硬件的预期。
3. **围绕消费者选择 tile 几何。**TMA 应该喂养你下游真正使用的计算模式。
4. **有意识地使用 swizzle。**只有当它匹配读取者的访问模式时才有帮助。
5. **让 L2 提升匹配连续宽度。**稠密可预测的流量与稀疏查找需要非常不同的抓取行为。

#### 术语表

| 术语 | 定义 |
| --- | --- |
| TMA | 张量内存加速器（Tensor Memory Accelerator），Hopper 的异步张量拷贝引擎。 |
| `CUtensorMap` | 由 host 编码的描述符，教会 TMA 如何解释并搬运一个张量 tile。 |
| Swizzle | 共享内存地址重映射，用于减少 bank 冲突并与消费者访问模式对齐。 |
| 交错（Interleave） | 针对打包全局内存格式的源布局解码。 |
| L2 提升 | 一种抓取宽度提示，在带宽效率与可能的缓存污染之间做权衡。 |
| OOB 填充 | 当 TMA tile 超出有效张量边界时使用的目的地填充策略。 |

### 继续学习本课程

第 4 课给了 TMA 描述符与布局词汇。下一课 `cp.async.bulk` 进入直接使用这套机制进行结构化批量搬运、分组、多播（multicast）与屏障（barrier）关联完成的指令家族。

## 完整幻灯片文本

提取自 `H100-Course/slides/4. cuTensorMap.pdf`，使用 `pdftotext -layout`。幻灯片总数：39。

### 幻灯片 1：cuTensorMap

Prateek Shukla

### 幻灯片 2：TMA 是什么

TMA 是让数据搬运能够以最少的线程参与、完全异步进行的新单元。

单个线程发起 TMA 指令后立即继续执行，整个操作由硬件在后台处理。

你得到的是描述符，而不是地址计算。

TMA 可以同时向多个 SM 的共享内存传输数据。

能处理 1d-5d 张量。

### 幻灯片 3：H100 如何获得完美的异步性

我们需要所有单元随时保持忙碌。

H100 的正确做法是：用 TMA 让多个缓冲同时加载，并让张量核心（Tensor Core）同时执行运算。

更重要的一点是：你不需要大量复杂工程。描述符包办了从数据到布局再到优化的一切，把大量繁重工作留给硬件。

### 幻灯片 4：为什么我们需要描述符

描述符是拼图中让 H100 具备异步本性的一块。
在更早的时候，整个数据获取都用索引完成：线程需要计算想要获取的数据的地址，然后才取回数据。
这在 Ampere 上有所改善——拷贝指令是非阻塞的——但线程仍然要为每 16 字节计算地址。SM 仍被拴在拷贝引擎上。
到了 H100，描述符封装了整个传输状态。由于完成这项工作所需的全部信息都包含在内存中那个 128 字节的对象里，硬件可以在后台完成任务，无需 SM 参与。

### 幻灯片 5：如何使用 TMA 进行拷贝

- 使用 CUDA API 创建一个 cuTensorMap
- 用所需的信息对它编码
- 发起该操作

### 幻灯片 6：cuTensorMap

它是一个描述符，存储关于以下内容的信息：

内存基址指针（设备地址）
张量形状（每个维度的元素数量）
步长（以字节计）
数据类型
对齐与 swizzle
内存空间（device、host 或 unified）
顺序与 rank（最多 32 个维度）
可选：tiling 或"交错"布局

### 幻灯片 7：创建 tensormap 对象

CUtensorMap tma_desc;

它在 host 内存中作为局部变量创建，大小为 128 字节，并且必须 128B 对齐。它不是普通指针，而是 NVIDIA 创造的一种特定数据结构。

cuTensorMapEncodeTiled() 函数用于把值编码进 tensormap。

### 幻灯片 8：CUtensorMap

这就是我们使用以下声明创建的 tensormap 对象：

CUtensorMap tma_desc，它将被填入编码后的描述符。

### 幻灯片 9：确定你要从 HBM 拷贝的实际数据的数据类型

TMA 引擎用它来自动获得内存对齐与传输大小。

### 幻灯片 10：数据类型

CU_TENSOR_MAP_DATA_TYPE_UINT8 - 无符号 8 位整数
CU_TENSOR_MAP_DATA_TYPE_UINT16 - 无符号 16 位整数
CU_TENSOR_MAP_DATA_TYPE_UINT32 - 无符号 32 位整数
CU_TENSOR_MAP_DATA_TYPE_INT32 - 有符号 32 位整数
CU_TENSOR_MAP_DATA_TYPE_UINT64 - 无符号 64 位整数
CU_TENSOR_MAP_DATA_TYPE_INT64 - 有符号 64 位整数
CU_TENSOR_MAP_DATA_TYPE_FLOAT16 - 16 位浮点（半精度）
CU_TENSOR_MAP_DATA_TYPE_FLOAT32 - 32 位浮点（单精度）
CU_TENSOR_MAP_DATA_TYPE_FLOAT64 - 64 位浮点（双精度）
CU_TENSOR_MAP_DATA_TYPE_BFLOAT16 - 16 位 brain 浮点
CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ - 带 flush-to-zero 模式的 32 位浮点
CU_TENSOR_MAP_DATA_TYPE_TFLOAT32 - TensorFloat-32 格式
CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ - 带 flush-to-zero 模式的 TensorFloat-32

### 幻灯片 11：Flush-to-zero（FTZ）模式是一种浮点算术优化，用于处理

非正规数（denormalized，即 subnormal）：用零替换它们，而不是正常地计算它们。

### 幻灯片 12：张量 rank 与全局内存地址

- 张量 rank 是张量的维度数量，不是线性代数中的矩阵秩
- 全局内存地址是张量在 HBM 中的内存地址

全局地址必须 16 字节对齐，以在硬件上获得高效的内存访问模式。

### 幻灯片 13：全局维度（globalDim）

一个数组，以元素数量（而不是字节）指定张量每个维度的大小。

对于一个 r 维数组，我们按以下方式组织数据：

globalDim[0] = 最内层维度

globalDim[r] = 最外层维度

注意该数组的每个元素取值范围是 0 到 2 ^ 32（约 40 亿）。

### 幻灯片 14：这些是每个维度中元素之间的字节步长

即硬件沿特定维度从一个坐标移动到下一个坐标时必须跳过多少字节。

最内层维度的步长是隐式的，由元素大小决定；该数组的大小为 (rank - 1)。

globalStrides[0] 是 globalDim[1]（次内层维度）的字节步长。

globalStrides[rank-2] 是 globalDim[rank-1]（最外层维度）的字节步长。

### 幻灯片 15：它是你想沿每个维度跳过的元素数量

即在执行异步拷贝的时候。

它也是一个大小为 rank 的数组，并且有特定要求：

它必须包含非零值。

步长值应小于或等于 8。

数组大小 = rank。

此外它以元素数量计算，而不是字节。

### 幻灯片 16：这是 tile 大小。它是一个大小为 rank 的数组，每个张量维度对应一个条目

与张量的 rank 相匹配。

它指定遍历 box 的大小（以元素计）——即每次 TMA 操作从全局内存传输到共享内存的那块数据。

以元素（而不是字节）为单位，与被传输的数据类型相对应。

内层维度必须 <= swizzle 大小。

### 幻灯片 17：共享内存

共享内存是附着在 SM 上的片上暂存存储（scratchpad）。你的线程块获得其中一块区域，该块中的每个线程都可以读写该区域内的任意地址。在程序层面，它看起来像一个平坦的地址空间。

硬件在底层做的是把这块地址空间拆分成 32 个 bank。一个 bank 就是共享内存阵列中一条可独立服务的通道。之所以是 32 个，原因很简单：一个 warp 有 32 个线程，理想情况是每个线程占一个 bank。

要点：bank 不是 32 个由你手动索引的独立数组。你仍然只是计算一个地址，bank 由地址选出。

### 幻灯片 18：swizzle 与 bank 冲突

全部 32 个 bank 可以在一个周期内同时访问。一个 warp 发出一条逻辑上的加载/存储指令，但 SMEM 子系统可能分若干轮执行它：

第 1 轮：来自每个 bank 的请求的一个子集

第 2 轮：剩余的冲突请求，依此类推

然而，如果多个线程访问同一个 bank，访问就会串行化，造成 bank 冲突。在共享内存中访问数据的常规方式是 strided 访问模式，这会导致串行化。

1 个请求 -> 无冲突，2 个请求 -> 2 路冲突，4 个请求 -> 4 路冲突，8 个请求 -> 8 路冲突，32 个请求 -> 最坏情况冲突

### 幻灯片 19：swizzle 如何工作

它是一个地址映射函数。它取逻辑地址（你的程序以为自己在写的 GmemAddress），对它的位进行"洗牌"，生成物理地址（数据真正落地的 SmemAddress）。

我们需要把顺序访问模式分散到所有 bank 上，以避免冲突。

为此我们有 3 种 swizzle 模式：32B、64B、128B。

128B、64B 与 32B 布局定义了内存 swizzle 的跨度（即块大小），从而也决定了内存事务的大小。根据你应用程序的内存访问模式选择正确的布局，是关键的性能调优步骤。

### 幻灯片 20：bank 方程

其中 a' 是硬件实际使用的共享内存字节地址。如果启用了 swizzle，a' 就是 swizzle 后的地址，而不是未 swizzle 的逻辑地址。这个 32 bank、4 字节步长的模型是推理冲突的标准方式，也与你之前使用的地址位视图一致。

这意味着 bank ID 来自地址位 a'[6:2]：

然后它每 32 * 4 = 128 字节重复一次。

### 幻灯片 21：swizzle 公式

a'= a^((a & Y_mask) >> 3)

其中 a 是字节地址，Y_mask 是 1<< 7，且低 4 位 a[3:0] 永远不被触碰。
32B: a[4]' = a[4] xor a[7]
64B: a[5:4]' = a[5:4] xor a[8:7]
128B: a[6:4]' = a[6:4] xor a[9:7]
被 XOR 的那些位是共享内存中 swizzle 模式行索引的低位，本质上并不是你数学矩阵的行。只有当你的布局把矩阵行映射到那些共享内存模式行上时，它们才成为你的矩阵行位。

### 幻灯片 22：swizzle 跨度与 atom

对于 SM90，完整的 swizzle 模式每以下字节数重复一次：

32B swizzle：256 字节

64B swizzle：512 字节

128B swizzle：1024 字节

每行 128 字节，每行被拆成 8 个 16 字节的 cell。swizzle 在行与行之间置换这 8 个 cell。所以 repeat_bytes = 128B * distinct_rows。

swizzle 行不是你的矩阵行。它是一条 128 字节的共享内存行。swizzle 只置换那些 16B 块。在 swizzle atom 内部，元素不是被随机打乱的——硬件/布局保持小的连续组完好无损，只是把这些组作为单元进行置换。

### 幻灯片 23：128B swizzle

Span = 128 字节，Atom = 16 字节
硬件把目的地的共享内存区域看作：
一行行 128 字节的行，每行拆成 8 个 16 字节的 cell；行内 cell 索引：x = 0..7，行索引：y = 0, 1, 2, ...
如果共享内存基地址对齐到 swizzle 的完整模式边界，逻辑 cell (y, x) 的物理目的地是：phys_addr = base + 128*y + 16*x
我们像这样应用 swizzle 公式：
a[6:4]' = a[6:4] ^ a[9:7]
整个 128B 行被视为 8 个可 swizzle 的 16B cell。行号对 8 取模来选择置换，模式每 8 行重复一次。总重复大小 = 8 * 128B = 1024B

### 幻灯片 24：64B swizzle

64B swizzle 作用于由 4 个 16B cell 组成的组。
行内 cell 索引：x = 0..3
行索引：y = 0, 1, 2, ...
逻辑 cell (y, x) 的物理目的地是：
phys_addr = base + 128*y + 16*x'
x' = x ^ (y & 3)
a[5:4]' = a[5:4] ^ a[8:7]
行的左 64B 半边按行号 mod 4 做置换，右 64B 半边以同样方式置换。模式每 4 行重复一次，总重复大小 = 4 * 128B = 512B

### 幻灯片 25：32B swizzle

32B swizzle 作用于成对的 16B cell。
行内 cell 索引：x = 0-1
行索引：y = 0, 1, 2, ...
逻辑 cell (y, x) 的物理目的地是：
phys_addr = base + 128*y + 16*x'
x' = x ^ (y & 1)
a4' = a[4] ^ a[7]
在每个 32B 区域内，两个 16B cell 每隔一个 128B 行交换一次。完整模式每 2 行重复一次。由于每行是 128B，重复周期为 256B

### 幻灯片 26：设置 swizzle

### 幻灯片 27：交错

全局内存中的数据并不总是线性的。像 cuDNN 这样的库主要使用 NCHW 内存布局，以在卷积和其他操作中获得最佳 GPU 性能。

CUtensorMapInterleave 参数告知 TMA 如何解码地址空间。它把 HBM 中物理上交错的排布映射为共享内存中逻辑上线性的重建。

Hopper TMA 支持为 NC/xHWCx 布局设计的特定交错模式：
CU_TENSOR_MAP_INTERLEAVE_16B

CU_TENSOR_MAP_INTERLEAVE_32B

### 幻灯片 28：两种主要模式

布局：NC/8HWC8（8 通道的向量）。

数学：8 通道 x 2 字节（FP16）= 16 字节。

行为：数据以 16 字节交错的块访问。

布局：NC/16HWC16（16 通道的向量）。

数学：16 通道 x 2 字节 = 32 字节。

行为：数据以 32 字节交错的块访问。

注意：如果通道数不是分片（slice）的倍数，最后一个分片必须零填充，以维持交错粒度。

### 幻灯片 29：交错与 swizzle

耦合：交错与 swizzle 并非相互独立。

要求：文档声明：

"当 interleave 为 CU_TENSOR_MAP_INTERLEAVE_32B 时，swizzle 参数必须设置为 CU_TENSOR_MAP_SWIZZLE_32B。"

理由：从全局内存去交错 32B 块的硬件路径，直接接入共享内存中 32B atom 的 swizzle 逻辑。两者同步联动，把复杂的全局布局映射为无 bank 冲突的共享内存布局。

### 幻灯片 30：几个要点

如果使用 32B 交错，你的全局地址应当 32B 对齐；如果使用 16B 交错，全局地址应当 16B 对齐。

此外，如果使用 16B 交错，全局步长应是 16 的倍数；如果使用 32B 交错，全局步长应是 32 的倍数。

此外，如果你使用交错，维度数必须大于或等于 3。

### 幻灯片 31：用法

### 幻灯片 32：L2 提升

与从 L2 缓存取数据相比，从 HBM 取数据很慢。

预取（prefetch）通过发出从 HBM 到 L2 缓存的非阻塞传输，来预判未来的数据需求。

不提升时（CU_TENSOR_MAP_L2_PROMOTION_NONE），内存控制器使用标准的缓存行抓取大小（通常 32 字节）。这效率不高，因为 L2 缓存以 128 字节行为单位管理数据。

把抓取大小提升到 128B 或 256B 后，单个 TMA 请求就能保证一次把更大的连续数据块带入 L2 缓存。

这对 GEMM 这类可预测模式最大化带宽效率，确保数据恰好在计算单元需要时驻留在 L2 中。

### 幻灯片 33：L2 提升枚举及其硬件含义

### 幻灯片 34：CU_TENSOR_MAP_L2_PROMOTION_L2_64B 与 CU_TENSOR_MAP_L2_PROMOTION_NONE

CU_TENSOR_MAP_L2_PROMOTION_NONE：使用默认的 32 字节 sector 抓取。如果你的张量非常稀疏或步长巨大，抓取邻居数据就是浪费。用它来避免"过度抓取"（把带宽浪费在无用的数据上）。

CU_TENSOR_MAP_L2_PROMOTION_L2_64B：抓取两个相邻的 32B sector（共 64 字节）。

对内层维度恰好为 64 字节（32 个元素）的特定半精度（FP16）张量形状有用。

### 幻灯片 35：CU_TENSOR_MAP_L2_PROMOTION_L2_128B

一次抓取完整的 128 字节缓存行。

把 DRAM 命令开销减少 75%（1 条命令代替 4 条）。

对齐：与 CUDA 的标准向量加载大小匹配（ld.global.v4 = 16B x 32 线程 = 512B，通常拆成 128B 块）。

建议：这是大多数稠密 FP16/BF16 GEMM 操作的标准默认选择。

### 幻灯片 36：CU_TENSOR_MAP_L2_PROMOTION_L2_256B

在单个逻辑事务中抓取两条完整缓存行（256 字节）。

H100 的内存控制器足够健壮，可以处理多行预取。

要求数据密度高。你的应用程序必须立即消费这些数据，才配得上占用的缓存空间。

如果数据没被使用，缓存污染的风险最高。

### 幻灯片 37：一个小而重要的提示

经验法则：

稠密 GEMM/卷积：始终使用 L2_128B 或 L2_256B。最大化总线效率。

嵌入查找 / 稀疏：使用 NONE（32B）。不要抓取你不会触碰的数据。

调优变量：选择取决于 Tensor_Inner_Dim_Bytes。

如果内层维度 < 64B，L2_128B 可能是浪费。让提升大小匹配你的连续数据宽度。

### 幻灯片 38：oobFill

oobFill 帮助处理 H100 GPU 上张量拷贝操作中的越界情况。

在目的地自动用零或 NaN 填充越界区域，而不是在 HBM 中。

HBM 中的源数据在整个过程中保持不变。

免除 kernel 中的手动边界检查，提升代码清晰度与性能。

对 tiling 操作很有用，尤其是 tile 延伸到张量边缘之外时。

填充的选择取决于应用需求；padding 用零，调试用 NaN。

### 幻灯片 39：用法

第二条语句使用 NaN，但在 FMA 操作期间它被当作零处理。
