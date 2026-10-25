# pdf2epub

Convert image-scanned PDF books into clean **EPUB 3** ebooks using a local-first OCR + NLP pipeline, per-page Markdown files for side-by-side review, and targeted cropped-region LLM repair only where confidence is low.

## How It Works

1. **Tier 1 — Local OCR & Layout Extraction (Free)**
   - **macOS (Primary):** Uses a native Swift helper (`swift/vision-ocr.swift`) powered by `PDFKit` + `Vision` (`VNRecognizeTextRequest` with `.accurate` recognition and language correction)—the same engine used by macOS Preview.
   - **Cross-Platform Fallback:** Uses `tesseract.js` (`--engine tesseract`).
2. **Tier 2 — Local NLP & Multi-Signal Confidence Scoring (Free)**
   - Strips repeating running headers, footers, and standalone page numbers.
   - Reconstructs paragraphs, detects headings and list items, de-hyphenates wrapped words across line breaks, and flags cross-page paragraph continuations (`continuesFromPrev` / `continuesToNext`).
   - Scores every block `[0..1]` using:
     - **OCR Line Confidence (45%)**
     - **Lexical Validity (30%):** Local Hunspell (`nspell`) dictionary check + a book-wide token frequency index (so recurring author names, character names, or technical terms aren't penalized as typos).
     - **Garble Heuristics (25%):** Detects alphanumeric mixing (`l1brary`), stray OCR noise symbols (`|`, `~`, `^`), and impossible consonant clusters.
3. **Tier 3 — Targeted Cropped-Region LLM Repair (Optional & Budget-Capped)**
   - Only paragraphs scoring below the threshold (default `0.82`) are wrapped in `<!-- @flagged ... -->` comments inside `pages/page-NNNN.md`.
   - When triggered via `fix-flagged` or the Web UI, crops **only** the flagged block's bounding box from the PDF page image and sends that strip + 1 sentence of surrounding context to a multimodal LLM (`gpt-4o-mini` or `claude-3-5-haiku-latest`).

---

## Requirements

- **Node.js** `>=24.0.0`
- **pnpm** `>=12`
- **macOS** with Xcode Command Line Tools (`swiftc`) for native Apple Vision OCR (automatically compiled on first run to `.cache/bin/vision-ocr`).

## Installation

```bash
pnpm install
```

---

## Quick Start

### 1. Scan a PDF into Per-Page Markdown Files

Scan a page range (or omit `--pages` to scan the entire book):

```bash
pnpm start scan ./path/to/book.pdf -o ./books/my-book --pages 1-25 --title "Book Title" --author "Author Name"
```

This creates:
- `./books/my-book/book.json` — Book metadata, OCR settings, and review statistics.
- `./books/my-book/pages/page-0001.md` ... `page-0025.md` — One human-editable Markdown file per PDF page.

### 2. Check Review Status & Flagged Pages

Inspect which pages are clean and which pages have low-confidence blocks:

```bash
pnpm start status ./books/my-book
```

### 3. Review Pages (Editor or Local Web UI)

You can edit `pages/page-NNNN.md` directly in Cursor/VS Code, or launch the local split-pane review UI to compare the rendered PDF page (with highlighted bounding boxes) alongside the Markdown editor:

```bash
pnpm start review ./books/my-book --port 4321
```

Then open `http://127.0.0.1:4321` in your browser.

### 4. Repair Flagged Blocks with an LLM (Optional)

First, run with `--dry-run` to see the exact token and USD cost estimate without making any API calls:

```bash
pnpm start fix-flagged ./books/my-book --dry-run
```

To repair flagged blocks using OpenAI (`OPENAI_API_KEY`) or Anthropic (`ANTHROPIC_API_KEY`):

```bash
OPENAI_API_KEY="sk-..." pnpm start fix-flagged ./books/my-book --max-blocks 25
```

### 5. Build the EPUB 3 Book

Once all pages are clean or reviewed (or pass `--allow-unreviewed` to build immediately), compile the workspace into an `.epub` file:

```bash
pnpm start build ./books/my-book -o ./books/my-book/my-book.epub
```

---

## CLI Reference

| Command | Description | Key Options |
| --- | --- | --- |
| `pnpm start scan <pdf> -o <dir>` | Run OCR + local NLP scoring and write `pages/page-NNNN.md` | `--pages <range>`, `--threshold <0..1>`, `--dpi <num>`, `--engine <macos-vision\|tesseract>`, `--title`, `--author`, `--auto-llm`, `--max-llm-blocks <n>` |
| `pnpm start status <bookDir>` | Show summary of clean, reviewed, and `needs-review` pages | — |
| `pnpm start fix-flagged <bookDir>` | Crop flagged bounding boxes and repair via multimodal LLM | `--dry-run`, `--page <n>`, `--max-blocks <n>`, `--model <name>` |
| `pnpm start review <bookDir>` | Launch local side-by-side PDF + Markdown review web UI | `-p, --port <port>` (default `4321`) |
| `pnpm start build <bookDir> -o <file.epub>` | Stitch cross-page paragraphs, split chapters, and package EPUB 3 | `--allow-unreviewed` |

---

## Development Scripts

```bash
pnpm lint        # Format and lint with Biome
pnpm typecheck   # TypeScript strict type-checking
pnpm test        # Run Vitest unit & integration test suite
```
