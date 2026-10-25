import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import { join } from 'node:path';
import { z } from 'zod';
import { repairFlaggedBlocks } from '../llm/repair.ts';
import {
	formatPageFilename,
	loadAllPageDocuments,
	parsePageDocument,
	readBookManifest,
	syncManifestStats,
	writePageFile,
} from '../markdown/page-serializer.ts';
import { createOcrProvider } from '../ocr/index.ts';
import { PAGE_STATUSES } from '../types/domain.ts';

const SavePageBodySchema = z.object({
	markdown: z.string(),
	status: z.enum(PAGE_STATUSES).optional(),
});

const RepairBlockBodySchema = z.object({
	blockId: z.string().optional(),
});

function sendJson(res: ServerResponse, statusCode: number, data: unknown): void {
	const payload = JSON.stringify(data);
	res.writeHead(statusCode, {
		'content-type': 'application/json; charset=utf-8',
		'content-length': Buffer.byteLength(payload),
	});
	res.end(payload);
}

async function readJsonBody(req: IncomingMessage): Promise<unknown> {
	const chunks: Buffer[] = [];
	for await (const chunk of req) {
		chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
	}
	const raw = Buffer.concat(chunks).toString('utf8');
	return raw ? JSON.parse(raw) : {};
}

const REVIEW_APP_HTML = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>pdf2epub Review Inspector</title>
<style>
	:root {
		--bg: #0f172a;
		--panel: #1e293b;
		--border: #334155;
		--text: #f8fafc;
		--muted: #94a3b8;
		--accent: #38bdf8;
		--warn: #f59e0b;
		--danger: #ef4444;
		--ok: #22c55e;
	}
	* { box-sizing: border-box; }
	body {
		margin: 0;
		font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
		background: var(--bg);
		color: var(--text);
		height: 100vh;
		display: flex;
		flex-direction: column;
	}
	header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: 10px 16px;
		background: var(--panel);
		border-bottom: 1px solid var(--border);
		gap: 12px;
	}
	.toolbar {
		display: flex;
		align-items: center;
		gap: 8px;
		flex-wrap: wrap;
	}
	button, select {
		background: #0f172a;
		color: var(--text);
		border: 1px solid var(--border);
		border-radius: 6px;
		padding: 6px 12px;
		font-size: 13px;
		cursor: pointer;
	}
	button:hover { border-color: var(--accent); }
	button.primary { background: #0284c7; border-color: #0284c7; font-weight: 500; }
	button.success { background: #16a34a; border-color: #16a34a; font-weight: 500; }
	.badge {
		font-size: 12px;
		padding: 3px 8px;
		border-radius: 999px;
		background: #334155;
	}
	.badge.needs-review { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid #f59e0b; }
	.badge.clean, .badge.reviewed, .badge.llm-fixed { background: rgba(34, 197, 94, 0.2); color: #4ade80; border: 1px solid #22c55e; }
	main {
		flex: 1;
		display: grid;
		grid-template-columns: 1fr 1fr;
		overflow: hidden;
	}
	.pane {
		height: 100%;
		overflow: auto;
		padding: 16px;
		display: flex;
		flex-direction: column;
		gap: 12px;
	}
	.pane + .pane { border-left: 1px solid var(--border); }
	.canvas-wrap {
		position: relative;
		display: inline-block;
		width: 100%;
		background: #090d16;
		border: 1px solid var(--border);
		border-radius: 8px;
		overflow: hidden;
	}
	.canvas-wrap img {
		display: block;
		width: 100%;
		height: auto;
	}
	.canvas-wrap svg {
		position: absolute;
		inset: 0;
		width: 100%;
		height: 100%;
		pointer-events: none;
	}
	textarea {
		flex: 1;
		width: 100%;
		background: #090d16;
		color: var(--text);
		border: 1px solid var(--border);
		border-radius: 8px;
		padding: 12px;
		font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
		font-size: 13px;
		line-height: 1.55;
		resize: none;
	}
	.flagged-list {
		display: flex;
		flex-direction: column;
		gap: 6px;
	}
	.flagged-item {
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: 8px 10px;
		background: rgba(245, 158, 11, 0.1);
		border: 1px solid rgba(245, 158, 11, 0.35);
		border-radius: 6px;
		font-size: 12px;
	}
	.status-msg { font-size: 12px; color: var(--muted); }
</style>
</head>
<body>
<header>
	<div class="toolbar">
		<strong id="book-title">pdf2epub</strong>
		<span id="stats-summary" class="status-msg"></span>
	</div>
	<div class="toolbar">
		<button id="prev-btn">Prev Page</button>
		<select id="page-select" aria-label="Select page"></select>
		<button id="next-btn">Next Page</button>
		<button id="next-flagged-btn">Next Flagged</button>
	</div>
	<div class="toolbar">
		<span id="page-badge" class="badge"></span>
		<button id="save-btn" class="primary">Save Markdown</button>
		<button id="mark-reviewed-btn" class="success">Mark Reviewed</button>
		<span id="toast" class="status-msg"></span>
	</div>
</header>
<main>
	<section class="pane">
		<div class="canvas-wrap">
			<img id="pdf-img" alt="Scanned PDF page" />
			<svg id="bbox-overlay" viewBox="0 0 1000 1000" preserveAspectRatio="none"></svg>
		</div>
	</section>
	<section class="pane">
		<div id="flagged-container" class="flagged-list"></div>
		<textarea id="md-editor" spellcheck="true" aria-label="Page Markdown editor"></textarea>
	</section>
</main>
<script>
	let pages = [];
	let currentPage = null;

	const pageSelect = document.getElementById('page-select');
	const pdfImg = document.getElementById('pdf-img');
	const bboxOverlay = document.getElementById('bbox-overlay');
	const mdEditor = document.getElementById('md-editor');
	const pageBadge = document.getElementById('page-badge');
	const flaggedContainer = document.getElementById('flagged-container');
	const toast = document.getElementById('toast');

	function showToast(msg) {
		toast.textContent = msg;
		setTimeout(() => { if (toast.textContent === msg) toast.textContent = ''; }, 3000);
	}

	async function loadBookOverview() {
		const res = await fetch('/api/book');
		const data = await res.json();
		document.getElementById('book-title').textContent = data.manifest.title;
		pages = data.pages;
		const flaggedCount = pages.filter(p => p.status === 'needs-review').length;
		document.getElementById('stats-summary').textContent =
			pages.length + ' scanned pages (' + flaggedCount + ' needs-review)';

		pageSelect.innerHTML = '';
		for (const p of pages) {
			const opt = document.createElement('option');
			opt.value = String(p.page);
			const marker = p.status === 'needs-review' ? ' [!] ' : ' ';
			opt.textContent = 'Page ' + p.page + marker + '(' + (p.confidence * 100).toFixed(0) + '%)';
			pageSelect.appendChild(opt);
		}

		if (currentPage === null && pages.length > 0) {
			const firstFlagged = pages.find(p => p.status === 'needs-review');
			await selectPage((firstFlagged || pages[0]).page);
		}
	}

	async function selectPage(pageNum) {
		currentPage = Number(pageNum);
		pageSelect.value = String(currentPage);
		pdfImg.src = '/api/pages/' + currentPage + '/image';

		const res = await fetch('/api/pages/' + currentPage);
		const data = await res.json();
		mdEditor.value = data.markdown;

		const fm = data.document.frontmatter;
		pageBadge.textContent = fm.status + ' (' + (fm.confidence * 100).toFixed(0) + '%)';
		pageBadge.className = 'badge ' + fm.status;

		bboxOverlay.innerHTML = '';
		flaggedContainer.innerHTML = '';

		for (const block of data.document.blocks) {
			if (!block.confidence.flagged) continue;
			const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
			rect.setAttribute('x', String(block.bbox.x * 1000));
			rect.setAttribute('y', String(block.bbox.y * 1000));
			rect.setAttribute('width', String(block.bbox.width * 1000));
			rect.setAttribute('height', String(block.bbox.height * 1000));
			rect.setAttribute('fill', 'rgba(245, 158, 11, 0.22)');
			rect.setAttribute('stroke', '#f59e0b');
			rect.setAttribute('stroke-width', '3');
			bboxOverlay.appendChild(rect);

			const row = document.createElement('div');
			row.className = 'flagged-item';
			const info = document.createElement('span');
			info.textContent = block.id + ' — score ' + block.confidence.composite.toFixed(2) + ' [' + block.confidence.reasons.join(', ') + ']';
			const btn = document.createElement('button');
			btn.textContent = 'Repair Block with LLM';
			btn.onclick = () => repairBlock(block.id);
			row.appendChild(info);
			row.appendChild(btn);
			flaggedContainer.appendChild(row);
		}
	}

	async function savePage(statusOverride) {
		if (currentPage === null) return;
		const res = await fetch('/api/pages/' + currentPage, {
			method: 'PUT',
			headers: { 'content-type': 'application/json' },
			body: JSON.stringify({ markdown: mdEditor.value, status: statusOverride }),
		});
		if (res.ok) {
			showToast('Saved page ' + currentPage);
			await loadBookOverview();
			await selectPage(currentPage);
		} else {
			const err = await res.json();
			showToast('Error: ' + err.error);
		}
	}

	async function repairBlock(blockId) {
		if (currentPage === null) return;
		showToast('Calling LLM repair for ' + blockId + '...');
		const res = await fetch('/api/pages/' + currentPage + '/repair', {
			method: 'POST',
			headers: { 'content-type': 'application/json' },
			body: JSON.stringify({ blockId }),
		});
		const data = await res.json();
		if (res.ok) {
			showToast('Repaired ' + data.repairedBlocks + ' block(s)');
			await loadBookOverview();
			await selectPage(currentPage);
		} else {
			showToast('LLM Error: ' + data.error);
		}
	}

	pageSelect.onchange = () => selectPage(Number(pageSelect.value));
	document.getElementById('prev-btn').onclick = () => {
		const idx = pages.findIndex(p => p.page === currentPage);
		if (idx > 0) selectPage(pages[idx - 1].page);
	};
	document.getElementById('next-btn').onclick = () => {
		const idx = pages.findIndex(p => p.page === currentPage);
		if (idx >= 0 && idx < pages.length - 1) selectPage(pages[idx + 1].page);
	};
	document.getElementById('next-flagged-btn').onclick = () => {
		const next = pages.find(p => p.page > currentPage && p.status === 'needs-review')
			|| pages.find(p => p.status === 'needs-review');
		if (next) selectPage(next.page);
	};
	document.getElementById('save-btn').onclick = () => savePage();
	document.getElementById('mark-reviewed-btn').onclick = () => savePage('reviewed');

	loadBookOverview();
</script>
</body>
</html>`;

export interface ReviewServerHandle {
	url: string;
	close: () => Promise<void>;
}

export async function startReviewServer(options: {
	bookDir: string;
	port?: number;
}): Promise<ReviewServerHandle> {
	const port = options.port ?? 4321;
	const { bookDir } = options;

	const server = createServer(async (req, res) => {
		try {
			const url = new URL(req.url ?? '/', `http://localhost:${port}`);
			const pathname = url.pathname;

			if (req.method === 'GET' && pathname === '/') {
				res.writeHead(200, { 'content-type': 'text/html; charset=utf-8' });
				res.end(REVIEW_APP_HTML);
				return;
			}

			if (req.method === 'GET' && pathname === '/api/book') {
				const manifest = readBookManifest(bookDir);
				const docs = loadAllPageDocuments(bookDir);
				sendJson(res, 200, {
					manifest,
					pages: docs.map(d => d.frontmatter),
				});
				return;
			}

			const pageImageMatch = pathname.match(/^\/api\/pages\/(\d+)\/image$/);
			if (req.method === 'GET' && pageImageMatch?.[1]) {
				const pageNum = Number.parseInt(pageImageMatch[1], 10);
				const manifest = readBookManifest(bookDir);
				const cacheDir = join(bookDir, '.cache/pages');
				mkdirSync(cacheDir, { recursive: true });
				const imgPath = join(cacheDir, `page-${String(pageNum).padStart(4, '0')}.png`);

				if (!existsSync(imgPath)) {
					const provider = createOcrProvider(manifest.ocrEngine);
					await provider.renderPageImage({
						pdfPath: manifest.sourcePdf,
						page: pageNum,
						outputPath: imgPath,
						dpi: 160,
					});
				}

				const imgBuffer = readFileSync(imgPath);
				res.writeHead(200, {
					'content-type': 'image/png',
					'content-length': imgBuffer.length,
				});
				res.end(imgBuffer);
				return;
			}

			const pageRepairMatch = pathname.match(/^\/api\/pages\/(\d+)\/repair$/);
			if (req.method === 'POST' && pageRepairMatch?.[1]) {
				const pageNum = Number.parseInt(pageRepairMatch[1], 10);
				const body = RepairBlockBodySchema.parse(await readJsonBody(req));
				const repairOpts: {
					bookDir: string;
					page: number;
					maxBlocks: number;
					blockId?: string;
				} = {
					bookDir,
					page: pageNum,
					maxBlocks: 5,
				};
				if (body.blockId !== undefined) {
					repairOpts.blockId = body.blockId;
				}
				const result = await repairFlaggedBlocks(repairOpts);
				sendJson(res, 200, result);
				return;
			}

			const pageMatch = pathname.match(/^\/api\/pages\/(\d+)$/);
			if (pageMatch?.[1]) {
				const pageNum = Number.parseInt(pageMatch[1], 10);
				const mdPath = join(bookDir, 'pages', formatPageFilename(pageNum));

				if (req.method === 'GET') {
					const rawMd = readFileSync(mdPath, 'utf8');
					const doc = parsePageDocument(rawMd);
					sendJson(res, 200, { markdown: rawMd, document: doc });
					return;
				}

				if (req.method === 'PUT') {
					const body = SavePageBodySchema.parse(await readJsonBody(req));
					const parsedDoc = parsePageDocument(body.markdown);
					if (body.status) {
						parsedDoc.frontmatter.status = body.status;
						if (body.status === 'reviewed') {
							for (const block of parsedDoc.blocks) {
								block.confidence.flagged = false;
							}
							parsedDoc.frontmatter.flaggedBlocks = 0;
						}
						writePageFile(bookDir, parsedDoc);
					} else {
						writeFileSync(mdPath, body.markdown, 'utf8');
					}
					syncManifestStats(bookDir);
					sendJson(res, 200, { ok: true });
					return;
				}
			}

			sendJson(res, 404, { error: 'Not found' });
		} catch (error) {
			console.error('[pdf2epub] Review server error:', error);
			sendJson(res, 500, { error: 'Internal server error' });
		}
	});

	await new Promise<void>(resolvePromise => {
		server.listen(port, '127.0.0.1', () => resolvePromise());
	});

	return {
		url: `http://127.0.0.1:${port}`,
		close: () =>
			new Promise<void>((resolveClose, rejectClose) => {
				server.close(err => (err ? rejectClose(err) : resolveClose()));
			}),
	};
}
