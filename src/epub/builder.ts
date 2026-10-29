import { randomUUID } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import JSZip from 'jszip';
import { loadAllPageDocuments, readBookManifest } from '../markdown/page-serializer.ts';
import {
	BOOK_COVER_ASSET_PATH,
	figureAssetPathFromMarkdown,
	isBookCoverMarkdown,
} from '../nlp/figure-detector.ts';
import { isTocEntryLine, TOC_ENTRY_LINE_REGEX } from '../nlp/layout-reconstructor.ts';
import type { BookManifest, PageDocument } from '../types/domain.ts';

export interface EpubParagraphItem {
	kind: 'heading' | 'paragraph' | 'code' | 'table' | 'figure';
	text: string;
	sourcePages: number[];
}

export interface EpubChapter {
	id: string;
	filename: string;
	title: string;
	items: EpubParagraphItem[];
}

export interface BuildEpubOptions {
	bookDir: string;
	outputPath: string;
	allowUnreviewed?: boolean;
}

export interface BuildEpubResult {
	outputPath: string;
	chapterCount: number;
	pageCount: number;
	stitchedCrossPageParagraphs: number;
}

function escapeXml(raw: string): string {
	return raw
		.replace(/&/g, '&amp;')
		.replace(/</g, '&lt;')
		.replace(/>/g, '&gt;')
		.replace(/"/g, '&quot;')
		.replace(/'/g, '&apos;');
}

export function stitchPagesAndGroupChapters(
	pages: readonly PageDocument[],
	fallbackTitle = 'Book',
): {
	chapters: EpubChapter[];
	stitchedCount: number;
} {
	const sortedPages = [...pages].sort((a, b) => a.frontmatter.page - b.frontmatter.page);
	const mergedItems: EpubParagraphItem[] = [];
	let stitchedCount = 0;

	for (let pIdx = 0; pIdx < sortedPages.length; pIdx++) {
		const pageDoc = sortedPages[pIdx];
		if (!pageDoc) {
			continue;
		}
		const prevPageDoc = sortedPages[pIdx - 1];

		for (let bIdx = 0; bIdx < pageDoc.blocks.length; bIdx++) {
			const block = pageDoc.blocks[bIdx];
			if (!block?.text.trim()) {
				continue;
			}
			if (pageDoc.frontmatter.page === 1 && isBookCoverMarkdown(block.text)) {
				continue;
			}

			const isFirstBlockOnPage = bIdx === 0;
			const prevItem = mergedItems[mergedItems.length - 1];
			const figureAssetPath = figureAssetPathFromMarkdown(block.text);
			const itemKind = figureAssetPath ? 'figure' : block.kind;

			const startsLikeContinuation = /^[a-z,;—–-]/.test(block.text.trim());
			const prevEndsWithHyphen = prevItem ? /[-‐‑]$/.test(prevItem.text.trim()) : false;
			const isToc =
				isTocEntryLine(block.text) || (prevItem ? isTocEntryLine(prevItem.text) : false);
			const shouldStitchWithPrevious =
				!isToc &&
				isFirstBlockOnPage &&
				itemKind === 'paragraph' &&
				prevItem?.kind === 'paragraph' &&
				(pageDoc.frontmatter.continuesFromPrev ||
					prevEndsWithHyphen ||
					(Boolean(prevPageDoc?.frontmatter.continuesToNext) && startsLikeContinuation));

			if (shouldStitchWithPrevious && prevItem) {
				const endsWithHyphen = /[-‐‑]$/.test(prevItem.text.trim());
				if (endsWithHyphen && /^[a-z]/.test(block.text.trim())) {
					prevItem.text = `${prevItem.text.trim().replace(/[-‐‑]$/, '')}${block.text.trim()}`;
				} else {
					prevItem.text = `${prevItem.text.trim()} ${block.text.trim()}`;
				}
				if (!prevItem.sourcePages.includes(pageDoc.frontmatter.page)) {
					prevItem.sourcePages.push(pageDoc.frontmatter.page);
				}
				stitchedCount += 1;
			} else {
				mergedItems.push({
					kind: itemKind,
					text: block.text.trim(),
					sourcePages: [pageDoc.frontmatter.page],
				});
			}
		}
	}

	const chapters: EpubChapter[] = [];
	let currentItems: EpubParagraphItem[] = [];
	let currentTitle = fallbackTitle;

	const flushChapter = () => {
		if (currentItems.length === 0) {
			return;
		}
		const index = chapters.length + 1;
		chapters.push({
			id: `chapter-${String(index).padStart(3, '0')}`,
			filename: `chapter-${String(index).padStart(3, '0')}.xhtml`,
			title: currentTitle,
			items: currentItems,
		});
		currentItems = [];
	};

	for (const item of mergedItems) {
		if (item.kind === 'heading') {
			if (currentItems.length > 0) {
				flushChapter();
			}
			currentTitle = item.text;
			currentItems.push(item);
		} else {
			currentItems.push(item);
		}
	}
	flushChapter();

	if (chapters.length === 0) {
		chapters.push({
			id: 'chapter-001',
			filename: 'chapter-001.xhtml',
			title: fallbackTitle,
			items: [],
		});
	}

	return { chapters, stitchedCount };
}

function renderFigureXhtml(markdown: string): string {
	const match = markdown.match(/^!\[([^\]]+)\]\((figures\/page-\d{4}-figure-\d+\.png)\)$/);
	if (!match?.[1] || !match[2]) {
		return `\t\t<p>${escapeXml(markdown)}</p>`;
	}
	return `\t\t<figure><img src="${escapeXml(match[2])}" alt="${escapeXml(match[1])}" /></figure>`;
}

function renderParagraphOrTocXhtml(text: string): string {
	const lines = text
		.split(/\r?\n/)
		.map(l => l.trim())
		.filter(Boolean);

	if (lines.length > 0 && lines.every(l => TOC_ENTRY_LINE_REGEX.test(l))) {
		return lines
			.map(line => {
				const match = line.match(TOC_ENTRY_LINE_REGEX);
				const label = match?.[1]?.trim() ?? line;
				const pageNum = match?.[2]?.trim() ?? '';
				return `\t\t<p class="toc-row"><span class="toc-label">${escapeXml(label)}</span><span class="toc-leader" aria-hidden="true"></span><span class="toc-page">${escapeXml(pageNum)}</span></p>`;
			})
			.join('\n');
	}

	return `\t\t<p>${escapeXml(text)}</p>`;
}

function renderTableXhtml(markdown: string): string {
	const comparisonTable = markdown.match(
		/^<table>\s*<thead><tr><th>([^<]*)<\/th><th>([^<]*)<\/th><\/tr><\/thead>\s*<tbody><tr>\s*<td><pre><code>([\s\S]*?)<\/code><\/pre><\/td>\s*<td><pre><code>([\s\S]*?)<\/code><\/pre><\/td>\s*<\/tr><\/tbody>\s*<\/table>$/,
	);
	if (comparisonTable?.slice(1).every(value => value !== undefined && !/[<>]/.test(value))) {
		return `\t\t${markdown.trim().replace(/\n/g, '\n\t\t')}`;
	}

	const rows = markdown
		.split(/\r?\n/)
		.map(line => line.trim())
		.filter(line => line.startsWith('|') && line.endsWith('|'))
		.map(line =>
			line
				.slice(1, -1)
				.split('|')
				.map(cell => cell.trim()),
		);
	if (rows.length < 2) {
		return `\t\t<p>${escapeXml(markdown)}</p>`;
	}

	const renderCell = (cell: string) =>
		cell
			.split(/<br\s*\/?\s*>/i)
			.map(value => {
				const code = value.match(/^`([^`]*)`$/);
				return code?.[1] !== undefined ? `<code>${escapeXml(code[1])}</code>` : escapeXml(value);
			})
			.join('<br />');
	const headerCells = rows[0] ?? [];
	const bodyRows = rows.slice(2);
	return [
		'\t\t<table>',
		`\t\t\t<thead><tr>${headerCells.map(cell => `<th>${renderCell(cell)}</th>`).join('')}</tr></thead>`,
		'\t\t\t<tbody>',
		...bodyRows.map(
			row => `\t\t\t\t<tr>${row.map(cell => `<td>${renderCell(cell)}</td>`).join('')}</tr>`,
		),
		'\t\t\t</tbody>',
		'\t\t</table>',
	].join('\n');
}

function renderChapterXhtml(chapter: EpubChapter, language: string): string {
	const bodyHtml = chapter.items
		.map(item => {
			if (item.kind === 'heading') {
				return `\t\t<h1>${escapeXml(item.text)}</h1>`;
			}
			if (item.kind === 'figure') {
				return renderFigureXhtml(item.text);
			}
			if (item.kind === 'code') {
				return `\t\t<pre><code>${escapeXml(item.text)}</code></pre>`;
			}
			if (item.kind === 'table') {
				return renderTableXhtml(item.text);
			}
			return renderParagraphOrTocXhtml(item.text);
		})
		.join('\n');

	return [
		'<?xml version="1.0" encoding="UTF-8"?>',
		'<!DOCTYPE html>',
		`<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" xml:lang="${escapeXml(language)}">`,
		'<head>',
		`\t<meta charset="UTF-8" />`,
		`\t<title>${escapeXml(chapter.title)}</title>`,
		`\t<link rel="stylesheet" type="text/css" href="styles.css" />`,
		'</head>',
		'<body>',
		`\t<section epub:type="chapter">`,
		bodyHtml,
		'\t</section>',
		'</body>',
		'</html>',
		'',
	].join('\n');
}

function renderCoverXhtml(language: string): string {
	return [
		'<?xml version="1.0" encoding="UTF-8"?>',
		'<!DOCTYPE html>',
		`<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" xml:lang="${escapeXml(language)}">`,
		'<head>',
		'\t<meta charset="UTF-8" />',
		'\t<title>Cover</title>',
		'\t<link rel="stylesheet" type="text/css" href="styles.css" />',
		'</head>',
		'<body class="cover-page">',
		'\t<section epub:type="cover">',
		'\t\t<img src="images/cover.png" alt="Book cover" />',
		'\t</section>',
		'</body>',
		'</html>',
		'',
	].join('\n');
}

function renderNavXhtml(title: string, chapters: readonly EpubChapter[], language: string): string {
	const links = chapters
		.map(ch => `\t\t\t\t<li><a href="${escapeXml(ch.filename)}">${escapeXml(ch.title)}</a></li>`)
		.join('\n');

	return [
		'<?xml version="1.0" encoding="UTF-8"?>',
		'<!DOCTYPE html>',
		`<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" xml:lang="${escapeXml(language)}">`,
		'<head>',
		'\t<meta charset="UTF-8" />',
		`\t<title>${escapeXml(title)} - Table of Contents</title>`,
		'\t<link rel="stylesheet" type="text/css" href="styles.css" />',
		'</head>',
		'<body>',
		'\t<nav epub:type="toc" id="toc">',
		`\t\t<h1>${escapeXml(title)}</h1>`,
		'\t\t<ol>',
		links,
		'\t\t</ol>',
		'\t</nav>',
		'</body>',
		'</html>',
		'',
	].join('\n');
}

function renderContentOpf(
	manifest: BookManifest,
	chapters: readonly EpubChapter[],
	figurePaths: readonly string[],
	hasBookCover: boolean,
): string {
	const bookUuid = `urn:uuid:${randomUUID()}`;
	const modifiedIso = new Date().toISOString().replace(/\.\d{3}Z$/, 'Z');

	const manifestItems = [
		'\t\t<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav" />',
		'\t\t<item id="css" href="styles.css" media-type="text/css" />',
		...(hasBookCover
			? [
					'\t\t<item id="cover" href="cover.xhtml" media-type="application/xhtml+xml" />',
					'\t\t<item id="cover-image" href="images/cover.png" media-type="image/png" properties="cover-image" />',
				]
			: []),
		...chapters.map(
			ch =>
				`\t\t<item id="${escapeXml(ch.id)}" href="${escapeXml(ch.filename)}" media-type="application/xhtml+xml" />`,
		),
		...figurePaths.map(
			(path, index) =>
				`\t\t<item id="figure-${String(index + 1).padStart(3, '0')}" href="${escapeXml(path)}" media-type="image/png" />`,
		),
	].join('\n');

	const spineItems = [
		...(hasBookCover ? ['\t\t<itemref idref="cover" />'] : []),
		...chapters.map(ch => `\t\t<itemref idref="${escapeXml(ch.id)}" />`),
	].join('\n');

	return [
		'<?xml version="1.0" encoding="UTF-8"?>',
		'<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="book-id">',
		'\t<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">',
		`\t\t<dc:identifier id="book-id">${escapeXml(bookUuid)}</dc:identifier>`,
		`\t\t<dc:title>${escapeXml(manifest.title)}</dc:title>`,
		`\t\t<dc:creator>${escapeXml(manifest.author)}</dc:creator>`,
		`\t\t<dc:language>${escapeXml(manifest.language)}</dc:language>`,
		`\t\t<meta property="dcterms:modified">${modifiedIso}</meta>`,
		'\t</metadata>',
		'\t<manifest>',
		manifestItems,
		'\t</manifest>',
		'\t<spine>',
		spineItems,
		'\t</spine>',
		'</package>',
		'',
	].join('\n');
}

const CONTAINER_XML = [
	'<?xml version="1.0" encoding="UTF-8"?>',
	'<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">',
	'\t<rootfiles>',
	'\t\t<rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml" />',
	'\t</rootfiles>',
	'</container>',
	'',
].join('\n');

const DEFAULT_STYLES_CSS = [
	'body {',
	'\tfont-family: Georgia, "Times New Roman", serif;',
	'\tline-height: 1.6;',
	'\tmargin: 1.5em;',
	'}',
	'h1 {',
	'\tfont-size: 1.5em;',
	'\tmargin-top: 1.5em;',
	'\tmargin-bottom: 0.75em;',
	'\tline-height: 1.25;',
	'}',
	'p {',
	'\tmargin: 0 0 0.85em 0;',
	'\ttext-align: justify;',
	'}',
	'p.toc-row {',
	'\tdisplay: flex;',
	'\talign-items: baseline;',
	'\tgap: 0.4em;',
	'\tmargin: 0 0 0.35em 0;',
	'\ttext-align: left;',
	'}',
	'.toc-label {',
	'\tflex: 0 1 auto;',
	'}',
	'.toc-leader {',
	'\tflex: 1 1 1.5em;',
	'\tborder-bottom: 1px dotted currentColor;',
	'\topacity: 0.65;',
	'\ttransform: translateY(-0.2em);',
	'}',
	'.toc-page {',
	'\tflex: 0 0 auto;',
	'\tfont-variant-numeric: tabular-nums;',
	'}',
	'figure {',
	'\tmargin: 1.25em auto;',
	'\ttext-align: center;',
	'}',
	'figure img {',
	'\tmax-width: 100%;',
	'\theight: auto;',
	'}',
	'.cover-page {',
	'\tmargin: 0;',
	'\tpadding: 0;',
	'}',
	'.cover-page img {',
	'\tmax-width: 100%;',
	'\tmax-height: 100vh;',
	'\twidth: auto;',
	'\tdisplay: block;',
	'\tmargin: 0 auto;',
	'}',
	'',
].join('\n');

export async function buildEpubFromWorkspace(options: BuildEpubOptions): Promise<BuildEpubResult> {
	const manifest = readBookManifest(options.bookDir);
	const pages = loadAllPageDocuments(options.bookDir);

	if (pages.length === 0) {
		throw new Error(`No page Markdown files found in ${options.bookDir}/pages`);
	}

	const unreviewedPages = pages.filter(p => p.frontmatter.status === 'needs-review');
	if (unreviewedPages.length > 0 && !options.allowUnreviewed) {
		const samplePages = unreviewedPages
			.slice(0, 8)
			.map(p => p.frontmatter.page)
			.join(', ');
		throw new Error(
			`${unreviewedPages.length} page(s) still have status "needs-review" (e.g. pages ${samplePages}). Review them, run "pdf2epub fix-flagged", or pass --allow-unreviewed to build anyway.`,
		);
	}

	const { chapters, stitchedCount } = stitchPagesAndGroupChapters(pages, manifest.title);
	const hasBookCover =
		pages
			.find(page => page.frontmatter.page === 1)
			?.blocks.some(block => isBookCoverMarkdown(block.text)) ?? false;
	const figurePaths = [
		...new Set(
			chapters.flatMap(chapter =>
				chapter.items.flatMap(item => {
					if (item.kind !== 'figure') {
						return [];
					}
					const assetPath = figureAssetPathFromMarkdown(item.text);
					return assetPath ? [assetPath] : [];
				}),
			),
		),
	];

	const zip = new JSZip();
	zip.file('mimetype', 'application/epub+zip', { compression: 'STORE' });
	zip.file('META-INF/container.xml', CONTAINER_XML, { compression: 'DEFLATE' });
	zip.file('OEBPS/styles.css', DEFAULT_STYLES_CSS, { compression: 'DEFLATE' });
	zip.file('OEBPS/nav.xhtml', renderNavXhtml(manifest.title, chapters, manifest.language), {
		compression: 'DEFLATE',
	});
	zip.file('OEBPS/content.opf', renderContentOpf(manifest, chapters, figurePaths, hasBookCover), {
		compression: 'DEFLATE',
	});
	if (hasBookCover) {
		zip.file('OEBPS/cover.xhtml', renderCoverXhtml(manifest.language), { compression: 'DEFLATE' });
		zip.file(
			'OEBPS/images/cover.png',
			readFileSync(join(options.bookDir, 'pages', BOOK_COVER_ASSET_PATH)),
			{ compression: 'STORE' },
		);
	}
	for (const figurePath of figurePaths) {
		zip.file(`OEBPS/${figurePath}`, readFileSync(join(options.bookDir, 'pages', figurePath)), {
			compression: 'STORE',
		});
	}

	for (const chapter of chapters) {
		zip.file(`OEBPS/${chapter.filename}`, renderChapterXhtml(chapter, manifest.language), {
			compression: 'DEFLATE',
		});
	}

	const buffer = await zip.generateAsync({
		type: 'nodebuffer',
		mimeType: 'application/epub+zip',
	});

	const resolvedOutput = resolve(options.outputPath);
	mkdirSync(dirname(resolvedOutput), { recursive: true });
	writeFileSync(resolvedOutput, buffer);

	return {
		outputPath: resolvedOutput,
		chapterCount: chapters.length,
		pageCount: pages.length,
		stitchedCrossPageParagraphs: stitchedCount,
	};
}
