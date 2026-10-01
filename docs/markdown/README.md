# AI-Fetchable Lesson Markdown

These files combine the published lesson pages with full extracted slide text so an AI agent can pull searchable lesson content with `curl` or `wget`.

## Fetch Examples

```sh
curl -L https://cudacourseh100.github.io/markdown/lessons/lesson-01-introduction-to-h100s.md
wget -O lesson-06-wgmma-part-1.md https://cudacourseh100.github.io/markdown/lessons/lesson-06-wgmma-part-1.md
```

## Files

- `lessons/lesson-01-introduction-to-h100s.md` - Lesson 1: Introduction to H100s
- `lessons/lesson-02-clusters-data-types-inline-ptx-pointers.md` - Lesson 2: Clusters, Data Types, Inline PTX, Pointers
- `lessons/lesson-03-asynchronicity-and-barriers.md` - Lesson 3: Asynchronicity and Barriers
- `lessons/lesson-04-cutensormap.md` - Lesson 4: cuTensorMap
- `lessons/lesson-05-cp-async-bulk.md` - Lesson 5: `cp.async.bulk`
- `lessons/lesson-06-wgmma-part-1.md` - Lesson 6: WGMMA Part 1
- `lessons/lesson-07-wgmma-part-2.md` - Lesson 7: WGMMA Part 2
- `lessons/lesson-08-kernel-design.md` - Lesson 8: Kernel Design
- `lessons/lesson-08-1-stream-k.md` - Lesson 8.1: Stream-K
- `lessons/lesson-08-2-kernel-launch.md` - Lesson 8.2: Kernel Launch
- `lessons/lesson-09-multi-gpu-part-1.md` - Lesson 9: Multi GPU Part 1
- `lessons/lesson-10-multi-gpu-part-2.md` - Lesson 10: Multi GPU Part 2

## Simplified Chinese Versions (`lessons_zh/`)

Each file under `lessons/` has a Simplified Chinese translation under `lessons_zh/` with the same filename. Translations preserve all code fences, inline code, file paths, URLs, and frontmatter path values verbatim, and add two frontmatter fields: `language: "zh-CN"` and `original_markdown` (pointing back to the English source file).

```sh
curl -L https://cudacourseh100.github.io/markdown/lessons_zh/lesson-06-wgmma-part-1.md
```

- `lessons_zh/lesson-01-introduction-to-h100s.md` - 第 1 课：H100 简介
- `lessons_zh/lesson-02-clusters-data-types-inline-ptx-pointers.md` - 第 2 课：集群、数据类型、内联 PTX 与指针
- `lessons_zh/lesson-03-asynchronicity-and-barriers.md` - 第 3 课：异步与屏障
- `lessons_zh/lesson-04-cutensormap.md` - 第 4 课：cuTensorMap
- `lessons_zh/lesson-05-cp-async-bulk.md` - 第 5 课：`cp.async.bulk`
- `lessons_zh/lesson-06-wgmma-part-1.md` - 第 6 课：WGMMA Part 1
- `lessons_zh/lesson-07-wgmma-part-2.md` - 第 7 课：WGMMA Part 2
- `lessons_zh/lesson-08-kernel-design.md` - 第 8 课：内核设计
- `lessons_zh/lesson-08-1-stream-k.md` - 第 8.1 课：Stream-K
- `lessons_zh/lesson-08-2-kernel-launch.md` - 第 8.2 课：Kernel 启动
- `lessons_zh/lesson-09-multi-gpu-part-1.md` - 第 9 课：多 GPU 第 1 部分
- `lessons_zh/lesson-10-multi-gpu-part-2.md` - 第 10 课：多 GPU 第 2 部分

Note: `lessons_zh/` is maintained as a parallel translation of `lessons/`. Running `scripts/export-lesson-markdown.py` regenerates only the English files; update `lessons_zh/` separately when the English content changes.

## Strategy Docs

- `seo/seo-strategy.md` - SEO strategy
- `seo/indexing-checklist.md` - SEO indexing checklist

## Regeneration

```sh
python3 scripts/export-lesson-markdown.py
```
