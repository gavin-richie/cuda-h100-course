import { cp, mkdir, readFile, rm, stat, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.resolve(__dirname, "..");
const outDir = path.join(rootDir, "docs");

const site = {
  name: "CUDA Programming for NVIDIA H100s",
  author: "Prateek Shukla",
  // Official home is the org user site; override with SITE_URL when deploying
  // the same bundle from a project site (owner.github.io/<repo>/).
  url: process.env.SITE_URL || "https://cudacourseh100.github.io/",
  description:
    "Advanced CUDA course on NVIDIA Hopper and H100 by Prateek Shukla, covering TMA, cuTensorMap, cp.async.bulk, mbarrier, WGMMA, kernel design, and multi-GPU orchestration.",
  socialImagePath: "/social-card.png",
};

const courseTopics = [
  "Hopper architecture",
  "asynchronous execution",
  "thread block clusters",
  "distributed shared memory",
  "mbarrier",
  "cuTensorMap",
  "cp.async.bulk",
  "WGMMA",
  "warp-specialized kernel design",
  "kernel launch control",
  "multi-GPU orchestration",
];

const homeFaqs = [
  {
    question: "Is this a beginner CUDA course?",
    answer:
      "No. The course assumes you already know C or C++, can read CUDA kernels, and want the Hopper-specific execution model rather than beginner CUDA onboarding.",
  },
  {
    question: "What does the course actually teach?",
    answer:
      "It teaches Hopper and H100 mechanisms directly: asynchronous execution, mbarrier, thread block clusters, distributed shared memory, cuTensorMap, cp.async.bulk, WGMMA, warp-specialized kernel design, and multi-GPU orchestration.",
  },
  {
    question: "Does the course cover WGMMA and Tensor Memory Accelerator?",
    answer:
      "Yes. WGMMA is covered across lessons 6 and 7, while TMA and cuTensorMap are covered in lessons 4 and 5 as part of Hopper's descriptor-driven asynchronous data movement model.",
  },
  {
    question: "Does the course include multi-GPU topics?",
    answer:
      "Yes. Lessons 9 and 10 cover NVLink, NVSwitch, topology, Slurm, PMIx, NCCL communicators, collectives, and distributed training patterns.",
  },
  {
    question: "Why is H100 different from older CUDA mental models?",
    answer:
      "Because Hopper shifts the programming model toward overlap-first execution: asynchronous copies, barriers, wait logic, descriptor-backed movement, and warpgroup tensor-core issue become central instead of secondary details.",
  },
];

const lessonPages = [
  {
    file: "pages/lesson-1.html",
    path: "/pages/lesson-1.html",
    lessonNumber: 1,
    shortTitle: "Introduction to H100s",
    title: "H100 Architecture and Async Execution | Lesson 1 | CUDA Programming for NVIDIA H100s",
    description:
      "Learn the H100 architecture, memory hierarchy, Tensor Cores, and the asynchronous Hopper execution model that changes how modern CUDA kernels are designed.",
    section: "Hopper Architecture",
    tags: ["H100", "Hopper architecture", "memory hierarchy", "Tensor Cores", "asynchronous execution"],
  },
  {
    file: "pages/lesson-2.html",
    path: "/pages/lesson-2.html",
    lessonNumber: 2,
    shortTitle: "Clusters, Data Types, Inline PTX, and Pointers",
    title: "Clusters, DSMEM, Inline PTX, and Pointers | Lesson 2 | CUDA Programming for NVIDIA H100s",
    description:
      "Thread block clusters, distributed shared memory, inline PTX, state spaces, and pointer conversion on Hopper.",
    section: "Clusters and PTX",
    tags: ["thread block clusters", "distributed shared memory", "inline PTX", "state spaces", "pointers"],
  },
  {
    file: "pages/lesson-3.html",
    path: "/pages/lesson-3.html",
    lessonNumber: 3,
    shortTitle: "Asynchronicity and Barriers",
    title: "Asynchronicity, mbarrier, and Fences | Lesson 3 | CUDA Programming for NVIDIA H100s",
    description:
      "mbarrier, producer-consumer coordination, latency hiding, RAW and WAR hazards, fences, and correctness under overlap on Hopper.",
    section: "Barriers and Synchronization",
    tags: ["mbarrier", "barriers", "latency hiding", "RAW hazards", "proxy fences"],
  },
  {
    file: "pages/lesson-4.html",
    path: "/pages/lesson-4.html",
    lessonNumber: 4,
    shortTitle: "cuTensorMap",
    title: "cuTensorMap and TMA Descriptors | Lesson 4 | CUDA Programming for NVIDIA H100s",
    description:
      "cuTensorMap descriptors for TMA on Hopper: tensor shapes, strides, swizzle, interleave, L2 promotion, and descriptor-driven async movement.",
    section: "TMA and Descriptors",
    tags: ["cuTensorMap", "TMA", "tensor descriptors", "swizzle", "L2 promotion"],
  },
  {
    file: "pages/lesson-5.html",
    path: "/pages/lesson-5.html",
    lessonNumber: 5,
    shortTitle: "cp.async.bulk",
    title: "cp.async.bulk on Hopper | Lesson 5 | CUDA Programming for NVIDIA H100s",
    description:
      "Hopper cp.async.bulk instructions, barrier-based completion, structured and unstructured transfers, multicast, prefetch, and async reductions.",
    section: "Async Bulk Copy",
    tags: ["cp.async.bulk", "async copy", "barrier completion", "multicast", "prefetch"],
  },
  {
    file: "pages/lesson-6.html",
    path: "/pages/lesson-6.html",
    lessonNumber: 6,
    shortTitle: "WGMMA Part 1",
    title: "WGMMA Part 1 | Lesson 6 | CUDA Programming for NVIDIA H100s",
    description:
      "WGMMA fundamentals on Hopper: warpgroups, wgmma.mma_async, ldmatrix, shared-memory descriptors, and tensor core dataflow.",
    section: "WGMMA Fundamentals",
    tags: ["WGMMA", "warpgroups", "wgmma.mma_async", "ldmatrix", "tensor cores"],
  },
  {
    file: "pages/lesson-7.html",
    path: "/pages/lesson-7.html",
    lessonNumber: 7,
    shortTitle: "WGMMA Part 2",
    title: "WGMMA Part 2, FP8, and Sparse Kernels | Lesson 7 | CUDA Programming for NVIDIA H100s",
    description:
      "Group commit and wait, stmatrix, FP8 packing, K-major constraints, and sparse WGMMA for Hopper tensor core kernels.",
    section: "Advanced WGMMA",
    tags: ["WGMMA", "FP8", "stmatrix", "K-major", "sparse tensor cores"],
  },
  {
    file: "pages/lesson-8.html",
    path: "/pages/lesson-8.html",
    lessonNumber: 8,
    shortTitle: "Kernel Design",
    title: "Kernel Design for Hopper | Lesson 8 | CUDA Programming for NVIDIA H100s",
    description:
      "Warp specialization, pipelining, circular buffers, persistent scheduling, and epilogues for compute-bound Hopper kernels.",
    section: "Kernel Design",
    tags: ["warp specialization", "kernel design", "persistent scheduling", "circular buffering", "epilogues"],
  },
  {
    file: "pages/lesson-8.1.html",
    path: "/pages/lesson-8.1.html",
    lessonNumber: "8.1",
    shortTitle: "Stream-K",
    title: "Stream-K Scheduling on Hopper | Lesson 8.1 | CUDA Programming for NVIDIA H100s",
    description:
      "Stream-K scheduling on Hopper: work decomposition, fixup, scheduler state, and utilization tradeoffs for GEMM kernels.",
    section: "Stream-K Scheduling",
    tags: ["Stream-K", "tile scheduling", "scheduler state", "fixup", "GEMM utilization"],
  },
  {
    file: "pages/lesson-8.2.html",
    path: "/pages/lesson-8.2.html",
    lessonNumber: "8.2",
    shortTitle: "Kernel Launch",
    title: "Kernel Launch Control on Hopper | Lesson 8.2 | CUDA Programming for NVIDIA H100s",
    description:
      "Kernel launch control on Hopper: launch bounds, grid constants, dependent grids, programmatic stream serialization, and overlap tuning between producer and consumer kernels.",
    section: "Kernel Launch Control",
    tags: ["kernel launch", "dependent grids", "programmatic stream serialization", "griddepcontrol", "launch bounds"],
  },
  {
    file: "pages/lesson-9.html",
    path: "/pages/lesson-9.html",
    lessonNumber: 9,
    shortTitle: "Multi GPU Part 1",
    title: "Multi-GPU Topology on H100 | Lesson 9 | CUDA Programming for NVIDIA H100s",
    description:
      "NVLink, NVSwitch, DGX H100 topology, peer-to-peer movement, and the interconnect constraints behind multi-GPU scaling.",
    section: "Multi-GPU Topology",
    tags: ["NVLink", "NVSwitch", "DGX H100", "peer-to-peer", "multi-GPU scaling"],
  },
  {
    file: "pages/lesson-10.html",
    path: "/pages/lesson-10.html",
    lessonNumber: 10,
    shortTitle: "Multi GPU Part 2",
    title: "NCCL, PMIx, and Distributed Orchestration | Lesson 10 | CUDA Programming for NVIDIA H100s",
    description:
      "Slurm, PMIx, NCCL communicators, collectives, and data, tensor, pipeline, and expert parallel orchestration on H100 systems.",
    section: "Distributed Orchestration",
    tags: ["Slurm", "PMIx", "NCCL", "collectives", "distributed orchestration"],
  },
];

const rootFiles = ["index.html", "slides.html", "styles.css", "script.js", "favicon.svg", "social-card.svg", "social-card.png"];

const zhLessonPages = [
  {
    file: "pages/zh/lesson-1.html",
    path: "/pages/zh/lesson-1.html",
    lessonNumber: 1,
    shortTitle: "H100 简介",
    title: "第 1 课 - H100 简介 | CUDA Programming for NVIDIA H100s",
    description:
      "学习 H100 架构、内存层级、张量核心，以及改变现代 CUDA kernel 设计方式的 Hopper 异步执行模型。",
    section: "Hopper 架构",
    tags: ["H100", "Hopper 架构", "内存层级", "张量核心", "异步执行"],
    lang: "zh-CN",
    locale: "zh_CN",
  },
  {
    file: "pages/zh/lesson-2.html",
    path: "/pages/zh/lesson-2.html",
    lessonNumber: 2,
    shortTitle: "集群、数据类型、内联 PTX 与指针",
    title: "第 2 课 - 集群、数据类型、内联 PTX 与指针 | CUDA Programming for NVIDIA H100s",
    description:
      "Hopper 上的线程块集群、分布式共享内存、内联 PTX、状态空间与指针转换。",
    section: "集群与 PTX",
    tags: ["线程块集群", "分布式共享内存", "内联 PTX", "状态空间", "指针"],
    lang: "zh-CN",
    locale: "zh_CN",
  },
  {
    file: "pages/zh/lesson-3.html",
    path: "/pages/zh/lesson-3.html",
    lessonNumber: 3,
    shortTitle: "异步与屏障",
    title: "第 3 课 - 异步与屏障 | CUDA Programming for NVIDIA H100s",
    description:
      "mbarrier、生产者/消费者协调、延迟隐藏、RAW 与 WAR 冒险、fence，以及 Hopper 上重叠执行下的正确性。",
    section: "屏障与同步",
    tags: ["mbarrier", "屏障", "延迟隐藏", "RAW 冒险", "proxy fence"],
    lang: "zh-CN",
    locale: "zh_CN",
  },
  {
    file: "pages/zh/lesson-4.html",
    path: "/pages/zh/lesson-4.html",
    lessonNumber: 4,
    shortTitle: "cuTensorMap",
    title: "第 4 课 - cuTensorMap | CUDA Programming for NVIDIA H100s",
    description:
      "Hopper 上 TMA 的 cuTensorMap 描述符：张量形状、stride、swizzle、interleave、L2 promotion 与描述符驱动的异步搬运。",
    section: "TMA 与描述符",
    tags: ["cuTensorMap", "TMA", "张量描述符", "swizzle", "L2 promotion"],
    lang: "zh-CN",
    locale: "zh_CN",
  },
  {
    file: "pages/zh/lesson-5.html",
    path: "/pages/zh/lesson-5.html",
    lessonNumber: 5,
    shortTitle: "cp.async.bulk",
    title: "第 5 课 - cp.async.bulk | CUDA Programming for NVIDIA H100s",
    description:
      "Hopper 的 cp.async.bulk 指令、基于屏障的完成机制、结构化与非结构化传输、multicast、预取与异步规约。",
    section: "异步批量拷贝",
    tags: ["cp.async.bulk", "异步拷贝", "屏障完成", "multicast", "预取"],
    lang: "zh-CN",
    locale: "zh_CN",
  },
  {
    file: "pages/zh/lesson-6.html",
    path: "/pages/zh/lesson-6.html",
    lessonNumber: 6,
    shortTitle: "WGMMA Part 1",
    title: "第 6 课 - WGMMA Part 1 | CUDA Programming for NVIDIA H100s",
    description:
      "Hopper 上的 WGMMA 基础：warpgroup、wgmma.mma_async、ldmatrix、共享内存描述符与张量核心数据流。",
    section: "WGMMA 基础",
    tags: ["WGMMA", "warpgroup", "wgmma.mma_async", "ldmatrix", "张量核心"],
    lang: "zh-CN",
    locale: "zh_CN",
  },
  {
    file: "pages/zh/lesson-7.html",
    path: "/pages/zh/lesson-7.html",
    lessonNumber: 7,
    shortTitle: "WGMMA Part 2",
    title: "第 7 课 - WGMMA Part 2 | CUDA Programming for NVIDIA H100s",
    description:
      "组提交与等待、stmatrix、FP8 打包、K-major 约束，以及面向 Hopper 张量核心 kernel 的稀疏 WGMMA。",
    section: "进阶 WGMMA",
    tags: ["WGMMA", "FP8", "stmatrix", "K-major", "稀疏张量核心"],
    lang: "zh-CN",
    locale: "zh_CN",
  },
  {
    file: "pages/zh/lesson-8.html",
    path: "/pages/zh/lesson-8.html",
    lessonNumber: 8,
    shortTitle: "内核设计",
    title: "第 8 课 - 内核设计 | CUDA Programming for NVIDIA H100s",
    description:
      "面向计算受限 Hopper kernel 的 warp 专用化、流水线、循环缓冲、持久调度与 epilogue。",
    section: "内核设计",
    tags: ["warp 专用化", "内核设计", "持久调度", "循环缓冲", "epilogue"],
    lang: "zh-CN",
    locale: "zh_CN",
  },
  {
    file: "pages/zh/lesson-8.1.html",
    path: "/pages/zh/lesson-8.1.html",
    lessonNumber: "8.1",
    shortTitle: "Stream-K",
    title: "第 8.1 课 - Stream-K | CUDA Programming for NVIDIA H100s",
    description:
      "Hopper 上的 Stream-K 调度：工作分解、fixup、调度器状态，以及 GEMM kernel 的利用率权衡。",
    section: "Stream-K 调度",
    tags: ["Stream-K", "tile 调度", "调度器状态", "fixup", "GEMM 利用率"],
    lang: "zh-CN",
    locale: "zh_CN",
  },
  {
    file: "pages/zh/lesson-8.2.html",
    path: "/pages/zh/lesson-8.2.html",
    lessonNumber: "8.2",
    shortTitle: "Kernel 启动",
    title: "第 8.2 课 - Kernel 启动 | CUDA Programming for NVIDIA H100s",
    description:
      "Hopper 上的 kernel 启动控制：launch bounds、grid 常量、依赖网格、程序化流串行，以及生产者与消费者 kernel 之间的重叠调优。",
    section: "Kernel 启动控制",
    tags: ["kernel 启动", "依赖网格", "程序化流串行", "griddepcontrol", "launch bounds"],
    lang: "zh-CN",
    locale: "zh_CN",
  },
  {
    file: "pages/zh/lesson-9.html",
    path: "/pages/zh/lesson-9.html",
    lessonNumber: 9,
    shortTitle: "多 GPU 第 1 部分",
    title: "第 9 课 - 多 GPU 第 1 部分 | CUDA Programming for NVIDIA H100s",
    description:
      "NVLink、NVSwitch、DGX H100 拓扑、点对点传输，以及多 GPU 扩展背后的互连约束。",
    section: "多 GPU 拓扑",
    tags: ["NVLink", "NVSwitch", "DGX H100", "点对点", "多 GPU 扩展"],
    lang: "zh-CN",
    locale: "zh_CN",
  },
  {
    file: "pages/zh/lesson-10.html",
    path: "/pages/zh/lesson-10.html",
    lessonNumber: 10,
    shortTitle: "多 GPU 第 2 部分",
    title: "第 10 课 - 多 GPU 第 2 部分 | CUDA Programming for NVIDIA H100s",
    description:
      "Slurm、PMIx、NCCL 通信器与集合通信，以及 H100 系统上的数据、张量、流水线与专家并行编排。",
    section: "分布式编排",
    tags: ["Slurm", "PMIx", "NCCL", "集合通信", "分布式编排"],
    lang: "zh-CN",
    locale: "zh_CN",
  },
];

// Wire hreflang alternates between the English and Chinese version of each lesson.
const allLessonPages = [...lessonPages, ...zhLessonPages];
for (const lesson of allLessonPages) {
  const base = lesson.file.replace(/^pages\/(zh\/)?/, "").replace(/\.html$/, "");
  lesson.alternates = [
    { lang: "en", path: `/pages/${base}.html` },
    { lang: "zh-CN", path: `/pages/zh/${base}.html` },
  ];
}

const pageFiles = allLessonPages.map(({ file }) => file);

const slideFiles = [
  "0. CUDA Programming for NVIDIA H100 GPUs.pdf",
  "1. Introduction to H100.pdf",
  "2. Clusters, Data types, inline PTX, State Spaces.pdf",
  "3. Asynchronicity and barriers.pdf",
  "4. cuTensorMap.pdf",
  "5. cp.async.bulk.pdf",
  "6. WGMMA-1.pdf",
  "7. Wgmma part 2.pdf",
  "8. Kernel Design.pdf",
  "8.1 Stream-K.pdf",
  "8.2 Kernel Launch.pdf",
  "9. Multi GPU.pdf",
  "10. Multi GPU  Part 2.pdf",
];
const requiredSlideFiles = slideFiles.filter((file) => file !== "0. CUDA Programming for NVIDIA H100 GPUs.pdf");

const codeFiles = [
  "sm90_gemm_tma_warpspecialized_pingpong.hpp",
  "sm90_gemm_tma_warpspecialized_cooperative.hpp",
  "sm90_epilogue_tma_warpspecialized.hpp",
  "sm90_mma_tma_gmma_rs_warpspecialized.hpp",
  "sm90_mma_tma_gmma_ss_warpspecialized.hpp",
  "sm90_tile_scheduler.hpp",
  "sm90_tile_scheduler_group.hpp",
  "sm90_tile_scheduler_stream_k.hpp",
  "fast.cu/README.md",
  "fast.cu/LICENSE",
  "fast.cu/examples/matmul/pingpong_experimental.cuh",
  "fast.cu/examples/matmul/matmul_12.cuh",
];

const pageSeo = new Map([
  [
    "index.html",
    {
      kind: "home",
      path: "/",
      title: "CUDA Programming for NVIDIA H100s | Hopper Course",
      socialTitle: "CUDA Programming for NVIDIA H100s",
      description: site.description,
      openGraphType: "website",
      tags: courseTopics,
    },
  ],
  [
    "slides.html",
    {
      kind: "collection",
      path: "/slides.html",
      title: "Course Slides | CUDA Programming for NVIDIA H100s",
      socialTitle: "Course Slides | CUDA Programming for NVIDIA H100s",
      description:
        "Download the full slide decks for CUDA Programming for NVIDIA H100s, covering Hopper architecture, TMA, barriers, cp.async.bulk, WGMMA, kernel design, and multi-GPU systems.",
      openGraphType: "website",
      tags: ["course slides", "Hopper slides", "WGMMA", "TMA", "multi-GPU systems"],
    },
  ],
  ...lessonPages.map((lesson) => [
    lesson.file,
    {
      ...lesson,
      kind: "lesson",
      socialTitle: `${lesson.shortTitle} | Lesson ${lesson.lessonNumber}`,
      openGraphType: "article",
    },
  ]),
  ...zhLessonPages.map((lesson) => [
    lesson.file,
    {
      ...lesson,
      kind: "lesson",
      socialTitle: `${lesson.shortTitle} | 第 ${lesson.lessonNumber} 课`,
      openGraphType: "article",
    },
  ]),
  [
    "404.html",
    {
      kind: "error",
      path: "/404.html",
      title: "404 | CUDA Programming for NVIDIA H100s",
      socialTitle: "Page not found | CUDA Programming for NVIDIA H100s",
      description: "The page you requested could not be found on the CUDA Programming for NVIDIA H100s site.",
      openGraphType: "website",
      robots: "noindex, nofollow, noarchive",
      tags: ["404"],
    },
  ],
]);

const localAttrPattern = /\b(?:href|src)="([^"]+)"/g;
const idPattern = /\sid="([^"]+)"/g;
const titlePattern = /<title>[\s\S]*?<\/title>/i;
const descriptionPattern = /<meta\b[^>]*name="description"[^>]*content="[^"]*"[^>]*>\s*/i;
const viewportPattern = /<meta\b[^>]*name="viewport"[^>]*>\s*/i;
const seoBlockPattern = /\s*<!-- SEO:BEGIN -->[\s\S]*?<!-- SEO:END -->/g;

async function ensureDir(targetPath) {
  await mkdir(path.dirname(targetPath), { recursive: true });
}

async function copyRelative(fromRelative, toRelative = fromRelative, { optional = false } = {}) {
  const sourcePath = path.join(rootDir, fromRelative);
  const targetPath = path.join(outDir, toRelative);
  if (optional && !(await fileExists(sourcePath))) {
    console.warn(`  Skipping (not found): ${fromRelative}`);
    return;
  }
  await ensureDir(targetPath);
  await cp(sourcePath, targetPath, { recursive: true });
}

async function fileExists(targetPath) {
  try {
    await stat(targetPath);
    return true;
  } catch {
    return false;
  }
}

async function snapshotExistingSlides() {
  const snapshot = new Map();

  for (const file of slideFiles) {
    const existingPath = path.join(outDir, "H100-Course", "slides", file);

    if (await fileExists(existingPath)) {
      snapshot.set(file, await readFile(existingPath));
    }
  }

  return snapshot;
}

async function restoreSlideFallbacks(slideSnapshot) {
  for (const [file, contents] of slideSnapshot.entries()) {
    const targetPath = path.join(outDir, "H100-Course", "slides", file);

    if (await fileExists(targetPath)) {
      continue;
    }

    await ensureDir(targetPath);
    await writeFile(targetPath, contents);
    console.warn(`  Restored fallback slide snapshot: H100-Course/slides/${file}`);
  }
}

async function ensureRequiredSlidesPublished() {
  const missingSlides = [];

  for (const file of requiredSlideFiles) {
    const publishedPath = path.join(outDir, "H100-Course", "slides", file);

    if (!(await fileExists(publishedPath))) {
      missingSlides.push(`H100-Course/slides/${file}`);
    }
  }

  if (missingSlides.length > 0) {
    throw new Error(`Missing required slide decks in published bundle:\n${missingSlides.join("\n")}`);
  }
}

function absoluteUrl(sitePath = "/") {
  const cleanPath = sitePath === "/" ? "" : sitePath.replace(/^\//, "");
  return new URL(cleanPath, site.url).toString();
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function escapeXml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;");
}

function stripSeoBlock(html) {
  return html.replace(seoBlockPattern, "");
}

function setTitle(html, title) {
  const titleTag = `    <title>${escapeHtml(title)}</title>`;
  if (titlePattern.test(html)) {
    return html.replace(titlePattern, titleTag);
  }
  return html.replace("</head>", `${titleTag}\n  </head>`);
}

function setDescription(html, description) {
  const descriptionTag = `    <meta name="description" content="${escapeHtml(description)}">\n`;
  if (descriptionPattern.test(html)) {
    return html.replace(descriptionPattern, descriptionTag);
  }
  if (viewportPattern.test(html)) {
    return html.replace(viewportPattern, (match) => `${match}${descriptionTag}`);
  }
  return html.replace("</head>", `${descriptionTag}  </head>`);
}

function buildBreadcrumbSchema(items) {
  return {
    "@type": "BreadcrumbList",
    itemListElement: items.map((item, index) => ({
      "@type": "ListItem",
      position: index + 1,
      name: item.name,
      item: item.url,
    })),
  };
}

function buildStructuredData(meta) {
  const homeUrl = absoluteUrl("/");
  const pageUrl = absoluteUrl(meta.path);
  const person = {
    "@type": "Person",
    "@id": `${homeUrl}#person`,
    name: site.author,
    url: homeUrl,
  };

  if (meta.kind === "home") {
    return {
      "@context": "https://schema.org",
      "@graph": [
        person,
        {
          "@type": "WebSite",
          "@id": `${homeUrl}#website`,
          url: homeUrl,
          name: site.name,
          description: site.description,
          inLanguage: "en",
          author: { "@id": `${homeUrl}#person` },
        },
        {
          "@type": "Course",
          "@id": `${homeUrl}#course`,
          name: site.name,
          description: site.description,
          url: homeUrl,
          provider: { "@id": `${homeUrl}#person` },
          inLanguage: "en",
          educationalLevel: "Advanced",
          courseMode: "online",
          teaches: courseTopics,
        },
        {
          "@type": "ItemList",
          "@id": `${homeUrl}#lessons`,
          name: "Course lessons",
          itemListElement: lessonPages.map((lesson, index) => ({
            "@type": "ListItem",
            position: index + 1,
            url: absoluteUrl(lesson.path),
            name: lesson.shortTitle,
          })),
        },
        {
          "@type": "FAQPage",
          "@id": `${homeUrl}#faq`,
          url: homeUrl,
          mainEntity: homeFaqs.map((entry) => ({
            "@type": "Question",
            name: entry.question,
            acceptedAnswer: {
              "@type": "Answer",
              text: entry.answer,
            },
          })),
        },
      ],
    };
  }

  if (meta.kind === "collection") {
    return {
      "@context": "https://schema.org",
      "@graph": [
        {
          "@type": "CollectionPage",
          "@id": `${pageUrl}#page`,
          url: pageUrl,
          name: meta.title,
          description: meta.description,
          inLanguage: meta.lang ?? "en",
          author: person,
          about: meta.tags,
          isPartOf: {
            "@type": "WebSite",
            name: site.name,
            url: homeUrl,
          },
        },
        buildBreadcrumbSchema([
          { name: "Home", url: homeUrl },
          { name: "Slides", url: pageUrl },
        ]),
      ],
    };
  }

  if (meta.kind === "lesson") {
    return {
      "@context": "https://schema.org",
      "@graph": [
        {
          "@type": "TechArticle",
          "@id": `${pageUrl}#article`,
          url: pageUrl,
          headline: meta.shortTitle,
          name: meta.shortTitle,
          description: meta.description,
          inLanguage: meta.lang ?? "en",
          author: person,
          mainEntityOfPage: pageUrl,
          articleSection: meta.section,
          about: meta.tags,
          isPartOf: {
            "@type": "Course",
            name: site.name,
            url: homeUrl,
          },
        },
        buildBreadcrumbSchema([
          { name: "Home", url: homeUrl },
          { name: meta.shortTitle, url: pageUrl },
        ]),
      ],
    };
  }

  return null;
}

function renderStructuredData(meta) {
  const structuredData = buildStructuredData(meta);
  if (!structuredData) {
    return "";
  }

  const payload = JSON.stringify(structuredData, null, 2).replace(/</g, "\\u003c");
  return [
    '    <script type="application/ld+json">',
    payload
      .split("\n")
      .map((line) => `    ${line}`)
      .join("\n"),
    "    </script>",
  ].join("\n");
}

function renderSeoBlock(meta) {
  const pageUrl = absoluteUrl(meta.path);
  const socialImageUrl = absoluteUrl(site.socialImagePath);
  const robots = meta.robots ?? "index, follow, max-image-preview:large";

  const tags = [
    '<!-- SEO:BEGIN -->',
    `    <meta name="author" content="${escapeHtml(site.author)}">`,
    `    <meta name="robots" content="${escapeHtml(robots)}">`,
    `    <link rel="canonical" href="${escapeHtml(pageUrl)}">`,
    `    <meta property="og:locale" content="${escapeHtml(meta.locale ?? "en_US")}">`,
  ];

  for (const alt of meta.alternates ?? []) {
    tags.push(`    <link rel="alternate" hreflang="${escapeHtml(alt.lang)}" href="${escapeHtml(absoluteUrl(alt.path))}">`);
  }
  if (meta.alternates?.length) {
    tags.push(`    <link rel="alternate" hreflang="x-default" href="${escapeHtml(absoluteUrl(meta.path))}">`);
  }
  if (meta.locale && meta.locale !== "en_US") {
    tags.push('    <meta property="og:locale:alternate" content="en_US">');
  }

  tags.push(
    `    <meta property="og:site_name" content="${escapeHtml(site.name)}">`,
    `    <meta property="og:type" content="${escapeHtml(meta.openGraphType)}">`,
    `    <meta property="og:title" content="${escapeHtml(meta.socialTitle ?? meta.title)}">`,
    `    <meta property="og:description" content="${escapeHtml(meta.description)}">`,
    `    <meta property="og:url" content="${escapeHtml(pageUrl)}">`,
    `    <meta property="og:image" content="${escapeHtml(socialImageUrl)}">`,
    '    <meta property="og:image:type" content="image/png">',
    '    <meta property="og:image:width" content="1200">',
    '    <meta property="og:image:height" content="630">',
    `    <meta property="og:image:alt" content="${escapeHtml(site.name)}">`,
    '    <meta name="twitter:card" content="summary_large_image">',
    `    <meta name="twitter:title" content="${escapeHtml(meta.socialTitle ?? meta.title)}">`,
    `    <meta name="twitter:description" content="${escapeHtml(meta.description)}">`,
    `    <meta name="twitter:image" content="${escapeHtml(socialImageUrl)}">`,
    `    <meta name="twitter:image:alt" content="${escapeHtml(site.name)}">`,
  );

  if (meta.openGraphType === "article") {
    tags.push(`    <meta property="article:author" content="${escapeHtml(site.author)}">`);
    if (meta.section) {
      tags.push(`    <meta property="article:section" content="${escapeHtml(meta.section)}">`);
    }
    for (const tag of meta.tags ?? []) {
      tags.push(`    <meta property="article:tag" content="${escapeHtml(tag)}">`);
    }
  }

  const structuredData = renderStructuredData(meta);
  if (structuredData) {
    tags.push(structuredData);
  }

  tags.push("    <!-- SEO:END -->");
  return tags.join("\n");
}

function applySeo(html, meta) {
  let nextHtml = stripSeoBlock(html);
  nextHtml = setTitle(nextHtml, meta.title);
  nextHtml = setDescription(nextHtml, meta.description);
  return nextHtml.replace("</head>", `${renderSeoBlock(meta)}\n  </head>`);
}

async function optimizeHtml(relativeHtmlPath, htmlOverride) {
  const meta = pageSeo.get(relativeHtmlPath);
  if (!meta) {
    return;
  }

  const htmlPath = path.join(outDir, relativeHtmlPath);
  const html = htmlOverride ?? (await readFile(htmlPath, "utf8"));
  const optimizedHtml = applySeo(html, meta);
  await writeFile(htmlPath, optimizedHtml);
}

async function generateSitemap() {
  const entries = [];

  for (const [relativePath, meta] of pageSeo.entries()) {
    if (meta.kind === "error") {
      continue;
    }

    const sourcePath = path.join(rootDir, relativePath);
    const stats = await stat(sourcePath).catch(() => null);
    const lastmod = (stats?.mtime ?? new Date()).toISOString().slice(0, 10);

    entries.push({
      url: absoluteUrl(meta.path),
      lastmod,
    });
  }

  const sitemap = [
    '<?xml version="1.0" encoding="UTF-8"?>',
    '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
    ...entries.map(
      (entry) => [
        "  <url>",
        `    <loc>${escapeXml(entry.url)}</loc>`,
        `    <lastmod>${entry.lastmod}</lastmod>`,
        "  </url>",
      ].join("\n")
    ),
    "</urlset>",
    "",
  ].join("\n");

  await writeFile(path.join(outDir, "sitemap.xml"), sitemap);
}

async function generateRobots() {
  const robots = ["User-agent: *", "Allow: /", "", `Sitemap: ${absoluteUrl("/sitemap.xml")}`, ""].join("\n");
  await writeFile(path.join(outDir, "robots.txt"), robots);
}

async function validateHtml(relativeHtmlPath) {
  const htmlPath = path.join(outDir, relativeHtmlPath);
  const html = await readFile(htmlPath, "utf8");

  const ids = new Set([...html.matchAll(idPattern)].map((match) => match[1]));
  const missing = [];

  for (const match of html.matchAll(localAttrPattern)) {
    const value = match[1];

    if (
      value.startsWith("http://") ||
      value.startsWith("https://") ||
      value.startsWith("mailto:") ||
      value.startsWith("tel:")
    ) {
      continue;
    }

    if (value.startsWith("#")) {
      if (!ids.has(value.slice(1))) {
        missing.push(`Missing in-page anchor target: ${value}`);
      }
      continue;
    }

    const cleanValue = value.split("#")[0].split("?")[0];
    const targetPath = path.resolve(path.dirname(htmlPath), cleanValue);

    if (!(await fileExists(targetPath))) {
      if (cleanValue.endsWith(".pdf") || cleanValue.startsWith("files/") || cleanValue.startsWith("../files/") || cleanValue.startsWith("../../files/") || cleanValue.startsWith("preview-snaps/") || cleanValue.startsWith("../../preview-snaps/")) {
        console.warn(`  Warning: optional asset missing from ${relativeHtmlPath}: ${cleanValue}`);
      } else {
        missing.push(`Missing local asset from ${relativeHtmlPath}: ${cleanValue}`);
      }
    }
  }

  if (missing.length > 0) {
    throw new Error(missing.join("\n"));
  }
}

async function build() {
  const slideSnapshot = await snapshotExistingSlides();

  await rm(outDir, { recursive: true, force: true });
  await mkdir(outDir, { recursive: true });

  for (const file of rootFiles) {
    await copyRelative(file);
  }

  await copyRelative("markdown", "markdown", { optional: true });
  await copyRelative("preview-snaps", "preview-snaps", { optional: true });

  for (const file of pageFiles) {
    await copyRelative(file);
  }

  for (const file of slideFiles) {
    await copyRelative(path.join("H100-Course", "slides", file), undefined, { optional: true });
  }

  await restoreSlideFallbacks(slideSnapshot);
  await ensureRequiredSlidesPublished();

  for (const file of codeFiles) {
    await copyRelative(path.join("files", file), undefined, { optional: true });
  }

  const htmlFilesToOptimize = ["index.html", "slides.html", ...pageFiles];

  for (const htmlFile of htmlFilesToOptimize) {
    await optimizeHtml(htmlFile);
  }

  const rawIndexHtml = await readFile(path.join(rootDir, "index.html"), "utf8");
  await optimizeHtml("404.html", rawIndexHtml);

  await generateRobots();
  await generateSitemap();
  await writeFile(path.join(outDir, ".nojekyll"), "");

  const htmlFilesToValidate = [...htmlFilesToOptimize, "404.html"];

  for (const htmlFile of htmlFilesToValidate) {
    await validateHtml(htmlFile);
  }

  console.log("Built GitHub Pages bundle in docs/");
}

build().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
