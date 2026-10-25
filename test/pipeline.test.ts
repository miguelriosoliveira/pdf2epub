import { existsSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
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
	reconstructPageLayout,
	scoreBlockConfidence,
} from '../src/nlp/index.ts';
import type { OcrPageResult, PageDocument } from '../src/types/domain.ts';

const tempDirs: string[] = [];

function createTempDir(): string {
	const dir = mkdtempSync(join(tmpdir(), 'pdf2epub-test-'));
	tempDirs.push(dir);
	return dir;
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
});

describe('CLI page option parser', () => {
	it('parses page range strings accurately', () => {
		expect(parsePagesOption('all')).toBe('all');
		expect(parsePagesOption('1-3, 7, 10-11')).toEqual([1, 2, 3, 7, 10, 11]);
	});
});
