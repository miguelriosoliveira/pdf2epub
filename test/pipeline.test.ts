import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { createServer as createTcpServer } from 'node:net';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import JSZip from 'jszip';
import { afterEach, describe, expect, it } from 'vitest';
import { parsePagesOption } from '../src/cli/index.ts';
import { buildEpubFromWorkspace, stitchPagesAndGroupChapters } from '../src/epub/builder.ts';
import { collectFlaggedCandidates, estimateLlmRepairCost } from '../src/llm/repair.ts';
import {
	parsePageDocument,
	serializePageDocument,
	writeBookManifest,
	writePageFile,
} from '../src/markdown/page-serializer.ts';
import {
	analyzeGarbleHeuristics,
	buildBookVocabulary,
	detectRepeatingMarginKeys,
	joinLinesWithDehyphenation,
	processOcrPagesToDocuments,
	reconstructPageLayout,
	scoreBlockConfidence,
} from '../src/nlp/index.ts';
import type { OcrProvider, RenderPageImageOptions } from '../src/ocr/provider.ts';
import { writeOcrPagesToWorkspace } from '../src/pipeline/scan-writer.ts';
import type { OcrPageResult, PageDocument } from '../src/types/domain.ts';
import { startReviewServer } from '../src/web/server.ts';

const tempDirs: string[] = [];

function createTempDir(): string {
	const dir = mkdtempSync(join(tmpdir(), 'pdf2epub-test-'));
	tempDirs.push(dir);
	return dir;
}

async function findAvailablePort(): Promise<number> {
	const server = createTcpServer();
	await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve));
	const address = server.address();
	if (!address || typeof address === 'string') {
		throw new Error('Could not determine an available test port');
	}
	await new Promise<void>((resolve, reject) =>
		server.close(error => (error ? reject(error) : resolve())),
	);
	return address.port;
}

afterEach(() => {
	while (tempDirs.length > 0) {
		const dir = tempDirs.pop();
		if (dir) {
			rmSync(dir, { recursive: true, force: true });
		}
	}
});

describe('layout reconstruction & header/footer stripping', () => {
	it('strips repeating headers, footers, and page numbers and de-hyphenates wrapped words', () => {
		const mockPages: OcrPageResult[] = [1, 2, 3, 4].map(pageNum => ({
			page: pageNum,
			widthPx: 1200,
			heightPx: 1800,
			lines: [
				{
					text: 'SYSTEM DESIGN FUNDAMENTALS',
					confidence: 0.99,
					bbox: { x: 0.1, y: 0.03, width: 0.5, height: 0.015 },
				},
				{
					text: `Chapter ${pageNum}`,
					confidence: 0.98,
					bbox: { x: 0.1, y: 0.12, width: 0.3, height: 0.032 },
				},
				{
					text: 'This is a clean opening paragraph with an algorithm-',
					confidence: 0.97,
					bbox: { x: 0.1, y: 0.2, width: 0.78, height: 0.018 },
				},
				{
					text: 'ic complexity discussion and self-',
					confidence: 0.96,
					bbox: { x: 0.1, y: 0.222, width: 0.78, height: 0.018 },
				},
				{
					text: 'balancing trees that continues onto the next page',
					confidence: 0.97,
					bbox: { x: 0.1, y: 0.244, width: 0.78, height: 0.018 },
				},
				{
					text: `${pageNum * 10}`,
					confidence: 0.99,
					bbox: { x: 0.48, y: 0.94, width: 0.04, height: 0.015 },
				},
			],
		}));

		const repeatingKeys = detectRepeatingMarginKeys(mockPages);
		expect(repeatingKeys.has('system design fundamentals')).toBe(true);

		const firstPage = mockPages[0];
		expect(firstPage).toBeDefined();
		if (!firstPage) return;

		const layout = reconstructPageLayout(firstPage, repeatingKeys);
		expect(layout.strippedHeadersFooters).toEqual(['SYSTEM DESIGN FUNDAMENTALS', '10']);
		expect(layout.blocks).toHaveLength(2);
		expect(layout.blocks[0]?.kind).toBe('heading');
		expect(layout.blocks[0]?.text).toBe('Chapter 1');
		expect(layout.blocks[1]?.kind).toBe('paragraph');
		expect(layout.blocks[1]?.text).toBe(
			'This is a clean opening paragraph with an algorithmic complexity discussion and self-balancing trees that continues onto the next page',
		);
		expect(layout.continuesToNext).toBe(true);
	});

	it('handles empty line join gracefully', () => {
		expect(joinLinesWithDehyphenation([])).toBe('');
	});
});

describe('multi-signal NLP confidence scoring', () => {
	it('scores clean dictionary prose high and flags garbled/low-OCR blocks', () => {
		const cleanBlock = {
			id: 'p1-b1',
			kind: 'paragraph' as const,
			text: 'Binary search trees allow fast lookup, insertion, and deletion of elements in logarithmic time.',
			bbox: { x: 0.1, y: 0.2, width: 0.8, height: 0.1 },
			lines: [
				{
					text: 'Binary search trees allow fast lookup, insertion, and deletion of elements in logarithmic time.',
					confidence: 0.98,
					bbox: { x: 0.1, y: 0.2, width: 0.8, height: 0.1 },
				},
			],
		};

		const cleanScore = scoreBlockConfidence(cleanBlock, { threshold: 0.82 });
		expect(cleanScore.flagged).toBe(false);
		expect(cleanScore.composite).toBeGreaterThan(0.9);

		const noisyBlock = {
			id: 'p1-b2',
			kind: 'paragraph' as const,
			text: 'The mansc1pt was discovcred in the l1brary ~vault~ beneath the c4thedral | wall.',
			bbox: { x: 0.1, y: 0.4, width: 0.8, height: 0.1 },
			lines: [
				{
					text: 'The mansc1pt was discovcred in the l1brary ~vault~ beneath the c4thedral | wall.',
					confidence: 0.62,
					bbox: { x: 0.1, y: 0.4, width: 0.8, height: 0.1 },
				},
			],
		};

		const noisyScore = scoreBlockConfidence(noisyBlock, { threshold: 0.82 });
		expect(noisyScore.flagged).toBe(true);
		expect(noisyScore.composite).toBeLessThan(0.75);
		expect(noisyScore.reasons.some(r => r.startsWith('ocr:'))).toBe(true);
		expect(noisyScore.reasons.some(r => r.startsWith('oov:'))).toBe(true);
		expect(noisyScore.reasons.some(r => r.startsWith('garble:'))).toBe(true);
	});

	it('exempts recurring book-specific proper nouns via buildBookVocabulary', () => {
		const pages: OcrPageResult[] = [1, 2, 3].map(p => ({
			page: p,
			widthPx: 1000,
			heightPx: 1500,
			lines: [
				{
					text: 'Sedgewick and Knuth explain asymptotic runtime clearly.',
					confidence: 0.96,
					bbox: { x: 0.1, y: 0.2, width: 0.8, height: 0.02 },
				},
			],
		}));

		const vocab = buildBookVocabulary(pages, 3);
		expect(vocab.has('sedgewick')).toBe(true);
		expect(vocab.has('knuth')).toBe(true);

		const garble = analyzeGarbleHeuristics('Normal clean text without OCR artifacts.');
		expect(garble.anomalyCount).toBe(0);
		expect(garble.score).toBe(1);
	});
});

describe('OCR asymptotic-notation recovery', () => {
	it('uses nearby theta and omega labels to repair corrupted Greek symbols', () => {
		const page: OcrPageResult = {
			page: 51,
			widthPx: 1600,
			heightPx: 2200,
			lines: [
				{
					text: 'Academics use big O, big O (theta), and big 22 (omega) to describe runtimes.',
					confidence: 1,
					bbox: { x: 0.1, y: 0.3, width: 0.7, height: 0.02 },
				},
				{
					text: '• © (big theta): In academia, © means both 0 and S. © gives a tight bound.',
					confidence: 1,
					bbox: { x: 0.1, y: 0.45, width: 0.7, height: 0.02 },
				},
				{
					text: '• 2 (big omega): In academia, Q is the equivalent concept but for lower bound. It is Ω(N), Q(log N), and Q(1).',
					confidence: 1,
					bbox: { x: 0.1, y: 0.6, width: 0.7, height: 0.02 },
				},
			],
		};

		const document = processOcrPagesToDocuments([page])[0];
		const text = document?.blocks.map(block => block.text).join('\n') ?? '';

		expect(text).toContain('Academics use big O, big Θ (theta), and big Ω (omega)');
		expect(text).toContain(
			'Θ (big theta): In academia, Θ means both O and Ω. Θ gives a tight bound.',
		);
		expect(text).toContain('Ω (big omega): In academia, Ω is the equivalent concept');
		expect(text).toContain('It is Ω(N), Ω(log N), and Ω(1).');
		const greekBlocks = document?.blocks.filter(block => /[ΘΩ]/u.test(block.text)) ?? [];
		expect(
			greekBlocks.every(
				block => !block.confidence.reasons.some(reason => reason.startsWith('oov:')),
			),
		).toBe(true);
	});

	it('turns isolated graph labels into an inline crop covering the chart', () => {
		const page: OcrPageResult = {
			page: 51,
			widthPx: 1600,
			heightPx: 2200,
			lines: [
				{
					text: 'No matter how big the constant is and how slow the linear increase is.',
					confidence: 1,
					bbox: { x: 0.1, y: 0.07, width: 0.8, height: 0.016 },
				},
				{
					text: '0(1)',
					confidence: 1,
					bbox: { x: 0.417, y: 0.147, width: 0.045, height: 0.022 },
				},
				{
					text: 'o(5)',
					confidence: 1,
					bbox: { x: 0.424, y: 0.214, width: 0.043, height: 0.022 },
				},
				{
					text: 'There are many more runtimes than this.',
					confidence: 1,
					bbox: { x: 0.097, y: 0.289, width: 0.819, height: 0.017 },
				},
			],
		};

		const document = processOcrPagesToDocuments([page])[0];
		const figure = document?.blocks.find(block => block.text.startsWith('![Figure from page 51]'));

		expect(figure).toBeDefined();
		expect(figure?.text).toBe('![Figure from page 51](figures/page-0051-figure-01.png)');
		expect(figure?.bbox.x).toBeLessThan(0.39);
		expect((figure?.bbox.x ?? 1) + (figure?.bbox.width ?? 0)).toBeGreaterThan(0.63);
		expect(figure?.bbox.y).toBeLessThan(0.12);
		expect((figure?.bbox.y ?? 0) + (figure?.bbox.height ?? 0)).toBeGreaterThan(0.27);
		expect(document?.blocks.some(block => block.text === '0(1)' || block.text === 'o(5)')).toBe(
			false,
		);
	});
});

describe('scan output with detected figures', () => {
	it('writes the figure crop asset referenced by the page Markdown', async () => {
		const bookDir = createTempDir();
		const renderCalls: RenderPageImageOptions[] = [];
		const provider: OcrProvider = {
			engine: 'macos-vision',
			getPdfPageCount: async () => 51,
			recognizePages: async () => ({ totalPages: 51, pages: [] }),
			renderPageImage: async options => {
				renderCalls.push(options);
				mkdirSync(dirname(options.outputPath), { recursive: true });
				writeFileSync(options.outputPath, 'test image');
				return options.outputPath;
			},
		};
		const page: OcrPageResult = {
			page: 51,
			widthPx: 1600,
			heightPx: 2200,
			lines: [
				{
					text: 'No matter how big the constant is and how slow the linear increase is.',
					confidence: 1,
					bbox: { x: 0.1, y: 0.07, width: 0.8, height: 0.016 },
				},
				{
					text: '0(1)',
					confidence: 1,
					bbox: { x: 0.417, y: 0.147, width: 0.045, height: 0.022 },
				},
				{
					text: 'o(5)',
					confidence: 1,
					bbox: { x: 0.424, y: 0.214, width: 0.043, height: 0.022 },
				},
				{
					text: 'There are many more runtimes than this.',
					confidence: 1,
					bbox: { x: 0.097, y: 0.289, width: 0.819, height: 0.017 },
				},
			],
		};

		await writeOcrPagesToWorkspace({
			bookDir,
			pdfPath: '/source/interview-book.pdf',
			pages: [page],
			provider,
			dpi: 220,
		});

		const markdown = readFileSync(join(bookDir, 'pages', 'page-0051.md'), 'utf8');
		const figurePath = join(bookDir, 'pages', 'figures', 'page-0051-figure-01.png');
		expect(markdown).toContain('![Figure from page 51](figures/page-0051-figure-01.png)');
		expect(existsSync(figurePath)).toBe(true);
		expect(renderCalls).toHaveLength(1);
		expect(renderCalls[0]?.pdfPath).toBe('/source/interview-book.pdf');
		expect(renderCalls[0]?.page).toBe(51);
		expect(renderCalls[0]?.crop?.x).toBeLessThan(0.39);
	});
});

describe('Markdown round-trip serialization & LLM cost estimation', () => {
	it('serializes and parses page-NNNN.md preserving frontmatter and @flagged annotations', () => {
		const sampleDoc: PageDocument = {
			frontmatter: {
				page: 42,
				status: 'needs-review',
				confidence: 0.64,
				flaggedBlocks: 1,
				continuesFromPrev: true,
				continuesToNext: false,
			},
			blocks: [
				{
					id: 'p42-b1',
					kind: 'paragraph',
					text: 'carried over from page 41 cleanly.',
					bbox: { x: 0.1, y: 0.12, width: 0.8, height: 0.04 },
					confidence: {
						composite: 0.96,
						ocr: 0.97,
						dictionary: 0.95,
						garble: 1,
						flagged: false,
						reasons: [],
					},
				},
				{
					id: 'p42-b2',
					kind: 'paragraph',
					text: 'The mansc1pt was discovcred in the l1brary.',
					bbox: { x: 0.11, y: 0.34, width: 0.78, height: 0.12 },
					confidence: {
						composite: 0.64,
						ocr: 0.68,
						dictionary: 0.55,
						garble: 0.7,
						flagged: true,
						reasons: ['ocr:0.68', 'oov:3(mansc1pt,discovcred,l1brary)', 'garble:2'],
					},
				},
			],
		};

		const markdown = serializePageDocument(sampleDoc);
		expect(markdown).toContain('page: 42');
		expect(markdown).toContain('<!-- @flagged id="p42-b2"');
		expect(markdown).toContain('<!-- /@flagged -->');

		const parsed = parsePageDocument(markdown);
		expect(parsed.frontmatter).toEqual(sampleDoc.frontmatter);
		expect(parsed.blocks).toHaveLength(2);
		expect(parsed.blocks[0]?.confidence.flagged).toBe(false);
		expect(parsed.blocks[1]?.confidence.flagged).toBe(true);
		expect(parsed.blocks[1]?.id).toBe('p42-b2');
		expect(parsed.blocks[1]?.bbox).toEqual(sampleDoc.blocks[1]?.bbox);

		const candidates = collectFlaggedCandidates([parsed]);
		expect(candidates).toHaveLength(1);
		const estimate = estimateLlmRepairCost(candidates, 10);
		expect(estimate.selectedBlocks).toBe(1);
		expect(estimate.estimatedInputTokens).toBeGreaterThan(250);
	});
});

describe('EPUB 3 stitching and archive builder', () => {
	it('stitches cross-page paragraphs, splits chapters at headings, and outputs valid EPUB zip', async () => {
		const workspaceDir = createTempDir();
		const now = new Date().toISOString();

		const page1: PageDocument = {
			frontmatter: {
				page: 1,
				status: 'clean',
				confidence: 0.97,
				flaggedBlocks: 0,
				continuesFromPrev: false,
				continuesToNext: true,
			},
			blocks: [
				{
					id: 'p1-b1',
					kind: 'heading',
					text: 'Chapter 1: Arrays and Strings',
					bbox: { x: 0.1, y: 0.1, width: 0.6, height: 0.04 },
					confidence: {
						composite: 0.98,
						ocr: 0.98,
						dictionary: 1,
						garble: 1,
						flagged: false,
						reasons: [],
					},
				},
				{
					id: 'p1-b2',
					kind: 'paragraph',
					text: 'Hash tables are one of the most important data struc-',
					bbox: { x: 0.1, y: 0.2, width: 0.8, height: 0.1 },
					confidence: {
						composite: 0.97,
						ocr: 0.97,
						dictionary: 1,
						garble: 1,
						flagged: false,
						reasons: [],
					},
				},
			],
		};

		const page2: PageDocument = {
			frontmatter: {
				page: 2,
				status: 'clean',
				confidence: 0.98,
				flaggedBlocks: 0,
				continuesFromPrev: true,
				continuesToNext: false,
			},
			blocks: [
				{
					id: 'p2-b1',
					kind: 'paragraph',
					text: 'tures for coding interviews because they provide constant time lookup.',
					bbox: { x: 0.1, y: 0.1, width: 0.8, height: 0.1 },
					confidence: {
						composite: 0.98,
						ocr: 0.98,
						dictionary: 1,
						garble: 1,
						flagged: false,
						reasons: [],
					},
				},
				{
					id: 'p2-b2',
					kind: 'heading',
					text: 'Chapter 2: Linked Lists',
					bbox: { x: 0.1, y: 0.3, width: 0.5, height: 0.04 },
					confidence: {
						composite: 0.98,
						ocr: 0.98,
						dictionary: 1,
						garble: 1,
						flagged: false,
						reasons: [],
					},
				},
				{
					id: 'p2-b3',
					kind: 'paragraph',
					text: 'A linked list represents a sequence of nodes.',
					bbox: { x: 0.1, y: 0.4, width: 0.8, height: 0.08 },
					confidence: {
						composite: 0.98,
						ocr: 0.98,
						dictionary: 1,
						garble: 1,
						flagged: false,
						reasons: [],
					},
				},
			],
		};

		writePageFile(workspaceDir, page1);
		writePageFile(workspaceDir, page2);
		writeBookManifest(workspaceDir, {
			title: 'Interview Guide',
			author: 'Test Author',
			language: 'en',
			sourcePdf: '/tmp/fake.pdf',
			totalPages: 2,
			scannedPages: [1, 2],
			threshold: 0.82,
			ocrEngine: 'macos-vision',
			createdAt: now,
			updatedAt: now,
			stats: {
				cleanPages: 2,
				flaggedPages: 0,
				reviewedPages: 0,
				totalFlaggedBlocks: 0,
			},
		});

		const { chapters, stitchedCount } = stitchPagesAndGroupChapters([page1, page2]);
		expect(stitchedCount).toBe(1);
		expect(chapters).toHaveLength(2);
		expect(chapters[0]?.title).toBe('Chapter 1: Arrays and Strings');
		expect(chapters[0]?.items[1]?.text).toBe(
			'Hash tables are one of the most important data structures for coding interviews because they provide constant time lookup.',
		);

		const epubPath = join(workspaceDir, 'output.epub');
		const buildRes = await buildEpubFromWorkspace({
			bookDir: workspaceDir,
			outputPath: epubPath,
		});
		expect(buildRes.chapterCount).toBe(2);
		expect(existsSync(epubPath)).toBe(true);

		const zip = await JSZip.loadAsync(
			await import('node:fs/promises').then(fs => fs.readFile(epubPath)),
		);
		expect(await zip.file('mimetype')?.async('string')).toBe('application/epub+zip');
		const ch1Xml = await zip.file('OEBPS/chapter-001.xhtml')?.async('string');
		expect(ch1Xml).toContain('data structures for coding interviews');
	});

	it('consolidates fragmented TOC dot-leader rows and renders responsive .toc-leader XHTML in EPUB', async () => {
		const tocPage: OcrPageResult = {
			page: 8,
			widthPx: 1492,
			heightPx: 2196,
			lines: [
				{
					text: 'Introduction.',
					confidence: 1,
					bbox: { x: 0.089, y: 0.0712, width: 0.115, height: 0.0102 },
				},
				{
					text: '. 2',
					confidence: 0.5,
					bbox: { x: 0.88, y: 0.071, width: 0.021, height: 0.0104 },
				},
				{
					text: 'I.',
					confidence: 1,
					bbox: { x: 0.091, y: 0.0944, width: 0.012, height: 0.0101 },
				},
				{
					text: 'The Interview Process',
					confidence: 1,
					bbox: { x: 0.132, y: 0.0942, width: 0.179, height: 0.0104 },
				},
				{
					text: '4',
					confidence: 1,
					bbox: { x: 0.888, y: 0.0944, width: 0.012, height: 0.0101 },
				},
				{
					text: '....',
					confidence: 1,
					bbox: { x: 0.316, y: 0.1002, width: 0.108, height: 0.0043 },
				},
				{
					text: 'Why?',
					confidence: 1,
					bbox: { x: 0.13, y: 0.1177, width: 0.047, height: 0.013 },
				},
				{
					text: '4',
					confidence: 1,
					bbox: { x: 0.888, y: 0.1177, width: 0.012, height: 0.0101 },
				},
				{
					text: '....aaa',
					confidence: 0.3,
					bbox: { x: 0.384, y: 0.125, width: 0.128, height: 0.0029 },
				},
			],
		};

		const layout = reconstructPageLayout(tocPage);
		expect(layout.blocks.map(b => b.text)).toEqual([
			'Introduction ... 2',
			'I. The Interview Process ... 4',
			'Why? ... 4',
		]);

		const workspaceDir = createTempDir();
		const now = new Date().toISOString();
		writePageFile(workspaceDir, {
			frontmatter: {
				page: 8,
				status: 'clean',
				confidence: 0.99,
				flaggedBlocks: 0,
				continuesFromPrev: false,
				continuesToNext: false,
			},
			blocks: layout.blocks.map(b => ({
				id: b.id,
				kind: b.kind,
				text: b.text,
				bbox: b.bbox,
				confidence: {
					composite: 0.99,
					ocr: 0.99,
					dictionary: 1,
					garble: 1,
					flagged: false,
					reasons: [],
				},
			})),
		});
		writeBookManifest(workspaceDir, {
			title: 'Sample TOC Book',
			author: 'Test Author',
			language: 'en',
			sourcePdf: '/tmp/fake.pdf',
			totalPages: 1,
			scannedPages: [8],
			threshold: 0.82,
			ocrEngine: 'macos-vision',
			createdAt: now,
			updatedAt: now,
			stats: {
				cleanPages: 1,
				flaggedPages: 0,
				reviewedPages: 0,
				totalFlaggedBlocks: 0,
			},
		});

		const epubPath = join(workspaceDir, 'toc.epub');
		await buildEpubFromWorkspace({ bookDir: workspaceDir, outputPath: epubPath });
		const zip = await JSZip.loadAsync(
			await import('node:fs/promises').then(fs => fs.readFile(epubPath)),
		);
		const ch1Xml = await zip.file('OEBPS/chapter-001.xhtml')?.async('string');
		expect(ch1Xml).toContain(
			'<p class="toc-row"><span class="toc-label">I. The Interview Process</span><span class="toc-leader" aria-hidden="true"></span><span class="toc-page">4</span></p>',
		);
		const css = await zip.file('OEBPS/styles.css')?.async('string');
		expect(css).toContain('.toc-leader');
		expect(css).toContain('border-bottom: 1px dotted currentColor;');
	});

	it('packages detected Markdown figure crops as EPUB images', async () => {
		const workspaceDir = createTempDir();
		const now = new Date().toISOString();
		const imagePath = 'figures/page-0051-figure-01.png';
		mkdirSync(join(workspaceDir, 'pages', 'figures'), { recursive: true });
		writeFileSync(join(workspaceDir, 'pages', imagePath), Buffer.from('test figure image'));
		writePageFile(workspaceDir, {
			frontmatter: {
				page: 1,
				status: 'clean',
				confidence: 1,
				flaggedBlocks: 0,
				continuesFromPrev: false,
				continuesToNext: false,
			},
			blocks: [
				{
					id: 'p1-b1',
					kind: 'heading',
					text: 'Chapter 1: Big O',
					bbox: { x: 0.1, y: 0.1, width: 0.5, height: 0.03 },
					confidence: {
						composite: 1,
						ocr: 1,
						dictionary: 1,
						garble: 1,
						flagged: false,
						reasons: [],
					},
				},
				{
					id: 'p1-figure-1',
					kind: 'paragraph',
					text: `![Figure from page 51](${imagePath})`,
					bbox: { x: 0.2, y: 0.2, width: 0.4, height: 0.2 },
					confidence: {
						composite: 1,
						ocr: 1,
						dictionary: 1,
						garble: 1,
						flagged: false,
						reasons: [],
					},
				},
			],
		});
		writeBookManifest(workspaceDir, {
			title: 'Interview Guide',
			author: 'Test Author',
			language: 'en',
			sourcePdf: '/tmp/fake.pdf',
			totalPages: 1,
			scannedPages: [1],
			threshold: 0.82,
			ocrEngine: 'macos-vision',
			createdAt: now,
			updatedAt: now,
			stats: {
				cleanPages: 1,
				flaggedPages: 0,
				reviewedPages: 0,
				totalFlaggedBlocks: 0,
			},
		});

		const epubPath = join(workspaceDir, 'figures.epub');
		await buildEpubFromWorkspace({ bookDir: workspaceDir, outputPath: epubPath });
		const zip = await JSZip.loadAsync(
			await import('node:fs/promises').then(fs => fs.readFile(epubPath)),
		);
		const chapter = await zip.file('OEBPS/chapter-001.xhtml')?.async('string');
		const packageXml = await zip.file('OEBPS/content.opf')?.async('string');

		expect(chapter).toContain(
			'<img src="figures/page-0051-figure-01.png" alt="Figure from page 51" />',
		);
		expect(packageXml).toContain('href="figures/page-0051-figure-01.png" media-type="image/png"');
		expect(await zip.file(`OEBPS/${imagePath}`)?.async('string')).toBe('test figure image');
	});
});

describe('CLI page option parser', () => {
	it('parses page range strings accurately', () => {
		expect(parsePagesOption('all')).toBe('all');
		expect(parsePagesOption('1-3, 7, 10-11')).toEqual([1, 2, 3, 7, 10, 11]);
	});
});

describe('review page preview layout', () => {
	it('keeps the overlay aligned with the full image in a scrollable preview', async () => {
		const reviewServer = await startReviewServer({
			bookDir: createTempDir(),
			port: await findAvailablePort(),
		});

		try {
			const response = await fetch(reviewServer.url);
			const html = await response.text();
			const paneStyles = html.match(/\.pane\s*\{([^}]*)\}/)?.[1] ?? '';
			const canvasStyles = html.match(/\.canvas-wrap\s*\{([^}]*)\}/)?.[1] ?? '';

			expect(response.ok).toBe(true);
			expect(paneStyles).toMatch(/overflow:\s*auto/);
			expect(canvasStyles).toMatch(/flex-shrink:\s*0/);
		} finally {
			await reviewServer.close();
		}
	});
});

describe('manual flagged-block review', () => {
	it('accepts one block at a time and completes the page after the last block', async () => {
		const bookDir = createTempDir();
		const now = new Date().toISOString();
		writePageFile(bookDir, {
			frontmatter: {
				page: 12,
				status: 'needs-review',
				confidence: 0.7,
				flaggedBlocks: 2,
				continuesFromPrev: false,
				continuesToNext: false,
			},
			blocks: [
				{
					id: 'p12-b15',
					kind: 'paragraph',
					text: 'Rabin-Karp Substring Search ... 636',
					bbox: { x: 0.13, y: 0.37, width: 0.77, height: 0.015 },
					confidence: {
						composite: 0.76,
						ocr: 1,
						dictionary: 0.2,
						garble: 1,
						flagged: true,
						reasons: ['oov:2(Rabin-Karp,Substring)'],
					},
				},
				{
					id: 'p12-b24',
					kind: 'paragraph',
					text: 'Trie & TrieNode ... 649',
					bbox: { x: 0.13, y: 0.58, width: 0.77, height: 0.012 },
					confidence: {
						composite: 0.7,
						ocr: 1,
						dictionary: 0,
						garble: 1,
						flagged: true,
						reasons: ['oov:2(Trie,TrieNode)'],
					},
				},
			],
		});
		writeBookManifest(bookDir, {
			title: 'Interview Guide',
			author: 'Test Author',
			language: 'en',
			sourcePdf: '/tmp/fake.pdf',
			totalPages: 12,
			scannedPages: [12],
			threshold: 0.82,
			ocrEngine: 'macos-vision',
			createdAt: now,
			updatedAt: now,
			stats: {
				cleanPages: 0,
				flaggedPages: 1,
				reviewedPages: 0,
				totalFlaggedBlocks: 2,
			},
		});

		const reviewServer = await startReviewServer({ bookDir, port: await findAvailablePort() });
		try {
			const htmlResponse = await fetch(reviewServer.url);
			expect(await htmlResponse.text()).toContain('Accept Block');

			const acceptBlock = (blockId: string) =>
				fetch(`${reviewServer.url}/api/pages/12/accept`, {
					method: 'POST',
					headers: { 'content-type': 'application/json' },
					body: JSON.stringify({ blockId }),
				});
			const loadPage = async () => {
				const response = await fetch(`${reviewServer.url}/api/pages/12`);
				return (await response.json()) as { document: PageDocument; markdown: string };
			};

			const firstAcceptance = await acceptBlock('p12-b15');
			expect(firstAcceptance.ok).toBe(true);
			const afterFirstAcceptance = await loadPage();
			expect(afterFirstAcceptance.document.frontmatter.status).toBe('needs-review');
			expect(afterFirstAcceptance.document.frontmatter.flaggedBlocks).toBe(1);
			expect(afterFirstAcceptance.document.blocks[0]?.confidence.flagged).toBe(false);
			expect(afterFirstAcceptance.document.blocks[1]?.confidence.flagged).toBe(true);
			expect(afterFirstAcceptance.markdown).not.toContain('id="p12-b15"');
			expect(afterFirstAcceptance.markdown).toContain('id="p12-b24"');

			const lastAcceptance = await acceptBlock('p12-b24');
			expect(lastAcceptance.ok).toBe(true);
			const afterLastAcceptance = await loadPage();
			expect(afterLastAcceptance.document.frontmatter.status).toBe('reviewed');
			expect(afterLastAcceptance.document.frontmatter.flaggedBlocks).toBe(0);
			expect(afterLastAcceptance.document.blocks.every(block => !block.confidence.flagged)).toBe(
				true,
			);
		} finally {
			await reviewServer.close();
		}
	});
});
