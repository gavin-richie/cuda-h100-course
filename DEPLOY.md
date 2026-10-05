# 部署指南（GitHub Pages）

本仓库通过 GitHub Actions 自动构建并部署 GitHub Pages，站点同时提供英文版与中文版课程页面。

- 英文版：`https://cudacourseh100.github.io/pages/lesson-1.html`
- 中文版：`https://cudacourseh100.github.io/pages/zh/lesson-1.html`
- 中文主页：`https://cudacourseh100.github.io/zh/`（英文主页右上角语言下拉框切换）
- 中文幻灯片页：`https://cudacourseh100.github.io/zh/slides.html`（链接中文 PDF 讲义与中文课程页）
- 中文 PDF 讲义：`https://cudacourseh100.github.io/H100-Course/slides-zh/1. Introduction to H100.pdf` 等 12 份，
  由 `python3 scripts/build-zh-slides.py` 从 `markdown/lessons_zh/` 重新生成（需要 `.fonts/NotoSansSC.ttf`，
  缺失时脚本会自动下载）。
- 每个课程页导航栏右侧有 `Simplified Chinese` / `English` 切换按钮；主页与幻灯片页使用语言下拉框。

## 部署架构

```
push main ──> GitHub Actions (.github/workflows/pages.yml)
                ├─ 克隆 H100-Course 幻灯片与 fast.cu 示例源码
                ├─ node scripts/build-pages.mjs   # 构建产物输出到 docs/
                └─ actions/deploy-pages           # 以 workflow 方式发布 docs/
```

构建脚本会把英文页（`pages/`）与中文页（`pages/zh/`）一并复制到 `docs/`，并注入 canonical、
hreflang（`en` / `zh-CN` / `x-default`）、`og:locale` 与结构化数据；`sitemap.xml` 同时包含两种语言的 URL。
`SITE_URL` 环境变量可覆盖站点根地址：组织用户站保持默认值，项目站（`owner.github.io/<repo>/`）由工作流自动计算。

## 用 gh CLI 从零创建仓库并部署

前提：`gh auth login` 已完成，且本目录已提交。

```bash
# 1. 登录（如未登录）
gh auth login

# 2. 创建公开仓库（GitHub Pages 免费计划要求公开仓库；--source 会把当前目录推送上去）
gh repo create cuda-h100-course --public \
  --description "CUDA Programming for NVIDIA H100s — course site (EN/中文)" \
  --homepage-url "https://<你的用户名>.github.io/cuda-h100-course/" \
  --source=. --remote=deploy --push
# 若当前 origin 已指向其他远端，可改用手动方式：
#   gh repo create cuda-h100-course --public --description "..."
#   git remote add deploy git@github.com:<你的用户名>/cuda-h100-course.git
#   git push deploy main

# 3. 启用 GitHub Pages，来源为 GitHub Actions（build_type=workflow）
gh api -X POST repos/<你的用户名>/cuda-h100-course/pages -f "build_type=workflow"

# 4. 触发 / 观察部署
gh workflow run "Deploy GitHub Pages" --repo <你的用户名>/cuda-h100-course   # 可选手动触发
gh run watch $(gh run list --repo <你的用户名>/cuda-h100-course -L 1 --json databaseId -q '.[0].databaseId') \
  --repo <你的用户名>/cuda-h100-course

# 5. 验证
curl -I https://<你的用户名>.github.io/cuda-h100-course/pages/zh/lesson-1.html
```

部署成功后：

- 英文首页：`https://<你的用户名>.github.io/cuda-h100-course/`
- 中文课程页：`https://<你的用户名>.github.io/cuda-h100-course/pages/zh/lesson-1.html`

## 在官方仓库（cudacourseh100.github.io）更新部署

```bash
git add <改动文件>
git commit -m "Add Simplified Chinese lesson pages with language switch"
git push origin main          # Actions 会自动构建并发布
gh run list --repo cudacourseh100/cudacourseh100.github.io -L 3
gh run watch <run-id> --repo cudacourseh100/cudacourseh100.github.io
```

## 本地构建与预览

```bash
node scripts/build-pages.mjs          # 重建 docs/（含链接校验、SEO 注入、sitemap）
python3 -m http.server 8099 -d docs   # 预览 http://127.0.0.1:8099/pages/zh/lesson-1.html
```

注意：本地缺少 `H100-Course/` 幻灯片与 `files/fast.cu/` 源码时，构建会自动回退到 `docs/` 中的
既有快照并将缺失的可选资源降级为警告；CI 中这两个仓库会先被克隆，因此线上产物是完整的。

## 常见问题

- **部署页面 404**：确认 Pages 已按上面第 3 步启用（Source = GitHub Actions），而不是 "deploy from branch"。
- **首次运行失败**：`gh run rerun <run-id>` 重跑；Actions 需要仓库设置中允许工作流创建 Pages 部署
  （Settings → Actions → General → Workflow permissions 保持默认即可，workflow 文件已声明 `pages: write`）。
- **canonical / sitemap 域名不对**：项目站部署时确保工作流的 `SITE_URL` 计算逻辑未被移除，或手动设置
  `SITE_URL` 环境变量后再构建。
