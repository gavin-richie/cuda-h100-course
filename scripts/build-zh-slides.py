#!/usr/bin/env python3
"""Generate Simplified Chinese slide-deck PDFs from markdown/lessons_zh/.

Each lesson's "## 完整幻灯片文本" section is rendered back into a 16:9 dark
deck (one page per slide) that matches the site's visual language. Output
goes to slides-zh/<original english deck filename>.pdf so the zh pages can
link them as ../H100-Course/slides-zh/... after build-pages.mjs copies the
directory.

Usage: python3 scripts/build-zh-slides.py [--out slides-zh]
Font: downloads Noto Sans SC (variable TTF) into .fonts/ when missing.
"""

from __future__ import annotations

import re
import sys
import urllib.request
from pathlib import Path

from reportlab.lib.colors import HexColor, Color
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parent.parent
FONT_URL = (
    "https://raw.githubusercontent.com/google/fonts/main/ofl/notosanssc/"
    "NotoSansSC%5Bwght%5D.ttf"
)
FONT_PATH = ROOT / ".fonts" / "NotoSansSC.ttf"

PAGE_W, PAGE_H = 960.0, 540.0
BG = HexColor("#08111d")
PANEL = HexColor("#0f1c29")
TEXT = HexColor("#f4f1eb")
TEXT_SOFT = HexColor("#c9c5bb")
KICKER = HexColor("#607182")

# lesson number -> (markdown file stem, english deck filename, english title, accent)
DECKS = [
    ("01", "lesson-01-introduction-to-h100s.md", "1. Introduction to H100.pdf",
     "Introduction to H100s", "#79d3ff"),
    ("02", "lesson-02-clusters-data-types-inline-ptx-pointers.md",
     "2. Clusters, Data types, inline PTX, State Spaces.pdf",
     "Clusters, Data Types, Inline PTX, and Pointers", "#8ed6ff"),
    ("03", "lesson-03-asynchronicity-and-barriers.md",
     "3. Asynchronicity and barriers.pdf", "Asynchronicity and Barriers", "#ffd166"),
    ("04", "lesson-04-cutensormap.md", "4. cuTensorMap.pdf", "cuTensorMap", "#79d3ff"),
    ("05", "lesson-05-cp-async-bulk.md", "5. cp.async.bulk.pdf", "cp.async.bulk", "#7fffd4"),
    ("06", "lesson-06-wgmma-part-1.md", "6. WGMMA-1.pdf", "WGMMA Part 1", "#c5ff74"),
    ("07", "lesson-07-wgmma-part-2.md", "7. Wgmma part 2.pdf", "WGMMA Part 2", "#ffb86c"),
    ("08", "lesson-08-kernel-design.md", "8. Kernel Design.pdf", "Kernel Design", "#79d3ff"),
    ("08.1", "lesson-08-1-stream-k.md", "8.1 Stream-K.pdf", "Stream-K", "#94f0ff"),
    ("08.2", "lesson-08-2-kernel-launch.md", "8.2 Kernel Launch.pdf", "Kernel Launch", "#c8ff7a"),
    ("09", "lesson-09-multi-gpu-part-1.md", "9. Multi GPU.pdf", "Multi GPU Part 1", "#ffd166"),
    ("10", "lesson-10-multi-gpu-part-2.md", "10. Multi GPU  Part 2.pdf",
     "Multi GPU Part 2", "#ff9b61"),
]

SLIDE_RE = re.compile(r"^### 幻灯片 (\d+)：(.*)$")
TOKEN_RE = re.compile(r"[A-Za-z0-9_$@.:/+\-]+|.")


def ensure_font() -> str:
    if not FONT_PATH.exists():
        FONT_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading font -> {FONT_PATH}")
        urllib.request.urlretrieve(FONT_URL, FONT_PATH)
    pdfmetrics.registerFont(TTFont("NotoSansSC", str(FONT_PATH)))
    return "NotoSansSC"


def parse_deck(md_path: Path):
    """Return (zh_lesson_title, [(slide_no, slide_title, blocks)]).

    blocks: list of ("p" | "li", text) in source order.
    """
    text = md_path.read_text(encoding="utf-8")
    fm = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    fm_text = fm.group(1) if fm else ""
    title_match = re.search(r'^title:\s*"(.+?)"', fm_text, re.M)
    zh_title = title_match.group(1) if title_match else md_path.stem
    zh_title = re.sub(r"^第 [\d.]+ 课\s*-\s*", "", zh_title)

    slides: list[tuple[int, str, list[tuple[str, str]]]] = []
    in_slides = False
    current = None  # (no, title, blocks)
    paragraph: list[str] = []

    def flush_paragraph():
        nonlocal paragraph
        if current is not None and paragraph:
            joined = " ".join(part.strip() for part in paragraph).strip()
            if joined:
                current[2].append(("p", joined))
            paragraph = []

    for line in text.splitlines():
        if line.startswith("## "):
            flush_paragraph()
            in_slides = line.startswith("## 完整幻灯片文本")
            if in_slides and current is not None:
                slides.append(current)
                current = None
            continue
        if not in_slides:
            continue
        m = SLIDE_RE.match(line)
        if m:
            flush_paragraph()
            if current is not None:
                slides.append(current)
            current = (int(m.group(1)), m.group(2).strip(), [])
            continue
        if current is None:
            continue
        stripped = line.strip()
        if not stripped:
            flush_paragraph()
            continue
        if stripped.startswith("- "):
            flush_paragraph()
            current[2].append(("li", clean(stripped[2:])))
        else:
            paragraph.append(clean(stripped))
    flush_paragraph()
    if current is not None:
        slides.append(current)
    return zh_title, slides


def clean(text: str) -> str:
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\*\*([^*]*)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]*)\*", r"\1", text)
    return text


def wrap(text: str, font: str, size: float, max_w: float) -> list[str]:
    """Greedy wrap that keeps Latin words intact and breaks CJK per glyph."""
    lines: list[str] = []
    current = ""
    current_w = 0.0
    for token in TOKEN_RE.findall(text):
        w = pdfmetrics.stringWidth(token, font, size)
        if current_w + w > max_w and current:
            lines.append(current)
            current = token.lstrip() if token.isspace() is False else token
            current_w = pdfmetrics.stringWidth(current, font, size)
            continue
        current += token
        current_w += w
    if current:
        lines.append(current)
    return lines


def draw_cover(c: canvas.Canvas, font: str, number: str, zh_title: str,
               en_title: str, accent: HexColor):
    c.setFillColor(BG)
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    c.setStrokeColor(accent)
    c.setLineWidth(3)
    c.line(70, 430, 160, 430)
    c.setFillColor(KICKER)
    c.setFont(font, 13)
    c.drawString(70, 452, "CUDA PROGRAMMING FOR NVIDIA H100s")
    c.setFillColor(accent)
    c.setFont(font, 15)
    c.drawString(70, 402, f"第 {number} 课")
    c.setFillColor(TEXT)
    size = 44 if pdfmetrics.stringWidth(zh_title, font, 44) < 780 else 32
    c.setFont(font, size)
    c.drawString(70, 340, zh_title)
    c.setFillColor(TEXT_SOFT)
    c.setFont(font, 15)
    c.drawString(70, 300, en_title)
    c.setFillColor(Color(1, 1, 1, alpha=0.05))
    c.setFont(font, 210)
    c.drawRightString(PAGE_W - 40, 90, number)
    c.setFillColor(KICKER)
    c.setFont(font, 11)
    c.drawString(70, 96, "简体中文研读版 · 由英文原版讲义翻译 · Prateek Shukla")
    c.setFillColor(accent)
    c.rect(70, 70, 46, 4, fill=1, stroke=0)


def draw_slide(c: canvas.Canvas, font: str, number: str, zh_title: str,
               index: int, total: int, slide_title: str,
               blocks: list[tuple[str, str]], accent: HexColor):
    c.setFillColor(BG)
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    c.setFillColor(PANEL)
    round_rect(c, 46, 96, PAGE_W - 92, 396, 14)
    c.setFillColor(accent)
    c.rect(46, 96, 4, 396, fill=1, stroke=0)

    c.setFillColor(KICKER)
    c.setFont(font, 10.5)
    c.drawString(78, 452, f"第 {number} 课 · {zh_title}")
    c.setFillColor(TEXT)
    c.setFont(font, 21)
    c.drawString(78, 414, slide_title or f"幻灯片 {index}")

    body_w = PAGE_W - 92 - 64
    body_top = 380
    body_bottom = 122

    size = 13.0
    while size >= 8.5:
        leading = size * 1.42
        para_gap = size * 0.75
        height = 0.0
        laid: list[tuple[str, list[str], float, float]] = []
        for kind, text in blocks:
            indent = 26 if kind == "li" else 0
            lines = wrap(text, font, size, body_w - indent)
            laid.append((kind, lines, leading, indent))
            height += len(lines) * leading + para_gap
        if height <= body_top - body_bottom or size <= 8.5:
            y = body_top
            for kind, lines, leading, indent in laid:
                for i, line in enumerate(lines):
                    if kind == "li" and i == 0:
                        c.setFillColor(accent)
                        c.setFont(font, size)
                        c.drawString(78 + 2, y - size, "—")
                    c.setFillColor(TEXT if kind == "li" else TEXT_SOFT)
                    c.setFont(font, size)
                    c.drawString(78 + indent, y - size, line)
                    y -= leading
                y -= para_gap
            break
        size -= 0.5

    c.setFillColor(KICKER)
    c.setFont(font, 8.5)
    c.drawString(46, 58, "CUDA Programming for NVIDIA H100s · 简体中文讲义")
    c.drawRightString(PAGE_W - 46, 58, f"幻灯片 {index} / {total}")


def round_rect(c: canvas.Canvas, x, y, w, h, r):
    c.roundRect(x, y, w, h, r, fill=1, stroke=0)


def build_deck(out_dir: Path, md_name: str, deck_name: str, en_title: str,
               number: str, accent_hex: str, font: str) -> tuple[int, int]:
    zh_title, slides = parse_deck(ROOT / "markdown" / "lessons_zh" / md_name)
    accent = HexColor(accent_hex)
    out_path = out_dir / deck_name
    c = canvas.Canvas(str(out_path), pagesize=(PAGE_W, PAGE_H))
    c.setTitle(f"第 {number} 课 {zh_title} — 简体中文讲义")
    c.setAuthor("Prateek Shukla")
    c.setSubject(f"CUDA Programming for NVIDIA H100s — {en_title} (Simplified Chinese reading deck)")
    draw_cover(c, font, number, zh_title, en_title, accent)
    c.showPage()
    total = len(slides)
    for _, (no, title, blocks) in enumerate(slides):
        draw_slide(c, font, number, zh_title, no, total, title, blocks, accent)
        c.showPage()
    c.save()
    return total + 1, total


def main() -> int:
    out_dir = ROOT / "slides-zh"
    out_dir.mkdir(exist_ok=True)
    font = ensure_font()
    pages_sum = 0
    for number, md_name, deck_name, en_title, accent in DECKS:
        pages, slides = build_deck(out_dir, md_name, deck_name, en_title, number, accent, font)
        pages_sum += pages
        print(f"  {deck_name}: {slides} slides, {pages} pages")
    print(f"wrote {len(DECKS)} decks ({pages_sum} pages) to {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
