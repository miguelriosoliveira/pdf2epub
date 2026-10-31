import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { createServer as createTcpServer } from 'node:net';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { createContext, runInContext } from 'node:vm';
import { inflateSync } from 'node:zlib';
import JSZip from 'jszip';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { parsePagesOption } from '../src/cli/index.ts';
import { buildEpubFromWorkspace, stitchPagesAndGroupChapters } from '../src/epub/builder.ts';
import {
	collectFlaggedCandidates,
	estimateLlmRepairCost,
	repairFlaggedBlocks,
} from '../src/llm/repair.ts';
import {
	parsePageDocument,
	readBookManifest,
	readPageFile,
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
import { LocalTesseractAlternative } from '../src/ocr/alternative-ocr.ts';
import type { OcrProvider, RenderPageImageOptions } from '../src/ocr/provider.ts';
import { TesseractOcrProvider } from '../src/ocr/tesseract-provider.ts';
import { VisionOcrProvider } from '../src/ocr/vision-provider.ts';
import { writeOcrPagesToWorkspace } from '../src/pipeline/scan-writer.ts';
import type { OcrPageResult, PageDocument } from '../src/types/domain.ts';
import { startReviewServer } from '../src/web/server.ts';

const tempDirs: string[] = [];

function createTempDir(): string {
	const dir = mkdtempSync(join(tmpdir(), 'pdf2epub-test-'));
	tempDirs.push(dir);
	return dir;
}

function paethPredictor(a: number, b: number, c: number): number {
	const p = a + b - c;
	const pa = Math.abs(p - a);
	const pb = Math.abs(p - b);
	const pc = Math.abs(p - c);
	return pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
}

function readPngInkMargins(imagePath: string): number[] {
	const png = readFileSync(imagePath);
	let width = 0;
	let height = 0;
	let bitDepth = 0;
	let colorType = 0;
	let offset = 8;
	const imageData: Buffer[] = [];

	while (offset + 12 <= png.length) {
		const chunkLength = png.readUInt32BE(offset);
		const chunkType = png.toString('ascii', offset + 4, offset + 8);
		const chunkDataStart = offset + 8;
		if (chunkType === 'IHDR') {
			width = png.readUInt32BE(chunkDataStart);
			height = png.readUInt32BE(chunkDataStart + 4);
			bitDepth = png[chunkDataStart + 8] ?? 0;
			colorType = png[chunkDataStart + 9] ?? 0;
		} else if (chunkType === 'IDAT') {
			imageData.push(png.subarray(chunkDataStart, chunkDataStart + chunkLength));
		} else if (chunkType === 'IEND') {
			break;
		}
		offset = chunkDataStart + chunkLength + 4;
	}

	if (bitDepth !== 8 || colorType !== 6) {
		throw new Error(`Unexpected rendered PNG format: bitDepth=${bitDepth}, colorType=${colorType}`);
	}

	const bytesPerPixel = 4;
	const rowBytes = width * bytesPerPixel;
	const inflated = inflateSync(Buffer.concat(imageData));
	let previousRow = Buffer.alloc(rowBytes);
	let minX = width;
	let minY = height;
	let maxX = -1;
	let maxY = -1;
	let sourceOffset = 0;

	for (let y = 0; y < height; y++) {
		const filter = inflated[sourceOffset++];
		const row = Buffer.alloc(rowBytes);
		for (let x = 0; x < rowBytes; x++) {
			const raw = inflated[sourceOffset++] ?? 0;
			const left = x >= bytesPerPixel ? (row[x - bytesPerPixel] ?? 0) : 0;
			const above = previousRow[x] ?? 0;
			const upperLeft = x >= bytesPerPixel ? (previousRow[x - bytesPerPixel] ?? 0) : 0;
			const predictor =
				filter === 1
					? left
					: filter === 2
						? above
						: filter === 3
							? Math.floor((left + above) / 2)
							: filter === 4
								? paethPredictor(left, above, upperLeft)
								: 0;
			row[x] = (raw + predictor) & 0xff;
		}

		for (let x = 0; x < width; x++) {
			const pixel = x * bytesPerPixel;
			const red = row[pixel] ?? 255;
			const green = row[pixel + 1] ?? 255;
			const blue = row[pixel + 2] ?? 255;
			const luminance = (299 * red + 587 * green + 114 * blue) / 1000;
			if (luminance < 245) {
				minX = Math.min(minX, x);
				minY = Math.min(minY, y);
				maxX = Math.max(maxX, x);
				maxY = Math.max(maxY, y);
			}
		}
		previousRow = row;
	}

	return [minX, width - maxX - 1, minY, height - maxY - 1];
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
	it('stores repeated running headers, footers and printed page numbers as page metadata', () => {
		const pages: OcrPageResult[] = [51, 52].map((page, index) => ({
			page,
			widthPx: 1600,
			heightPx: 2200,
			lines: [
				{
					text: index === 0 ? 'VI | Big O' : 'Vl Big O',
					confidence: 1,
					bbox: { x: 0.8, y: 0.03, width: 0.1, height: 0.015 },
				},
				{
					text: 'The chapter discussion continues here.',
					confidence: 1,
					bbox: { x: 0.1, y: 0.1, width: 0.7, height: 0.02 },
				},
				{
					text:
						index === 0
							? 'CrackingTheCodingInterview.com | 6th Edition'
							: 'CrackingTheCodinglnterview.com | 6th Edition',
					confidence: 1,
					bbox: { x: 0.5, y: 0.94, width: 0.32, height: 0.015 },
				},
				{
					text: String(39 + index),
					confidence: 1,
					bbox: { x: 0.86, y: 0.94, width: 0.03, height: 0.015 },
				},
			],
		}));

		const docs = processOcrPagesToDocuments(pages, { profile: 'narrative' });
		for (const [index, document] of docs.entries()) {
			const markdown = serializePageDocument(document);
			const parsed = parsePageDocument(markdown);
			expect(parsed.frontmatter).toEqual(
				expect.objectContaining({
					runningHeaders: [index === 0 ? 'VI | Big O' : 'Vl Big O'],
					runningFooters: [
						index === 0
							? 'CrackingTheCodingInterview.com | 6th Edition'
							: 'CrackingTheCodinglnterview.com | 6th Edition',
					],
					printedPage: String(39 + index),
				}),
			);
			expect(parsed.blocks.map(block => block.text)).toEqual([
				'The chapter discussion continues here.',
			]);
		}
		const { chapters } = stitchPagesAndGroupChapters(docs, 'Sample Book');
		expect(chapters).toHaveLength(1);
		expect(chapters[0]?.title).toBe('Sample Book');
		expect(chapters[0]?.items.map(item => item.text)).not.toContain('VI | Big O');
	});

	it('detects local running headers without stripping unique chapter headings', () => {
		const pages: OcrPageResult[] = Array.from({ length: 10 }, (_, index) => {
			const page = 51 + index;
			return {
				page,
				widthPx: 1600,
				heightPx: 2200,
				lines: [
					...(page === 53 || page === 54
						? [
								{
									text: 'Local Section',
									confidence: 1,
									bbox: { x: 0.7, y: 0.03, width: 0.2, height: 0.015 },
								},
							]
						: []),
					...(page === 53
						? [
								{
									text: 'Chapter 3: A New Topic',
									confidence: 1,
									bbox: { x: 0.1, y: 0.08, width: 0.5, height: 0.025 },
								},
							]
						: []),
					{
						text: 'Body text remains here.',
						confidence: 1,
						bbox: { x: 0.1, y: 0.2, width: 0.7, height: 0.02 },
					},
				],
			};
		});
		const docs = processOcrPagesToDocuments(pages, { profile: 'narrative' });
		const first = docs.find(doc => doc.frontmatter.page === 53);
		const second = docs.find(doc => doc.frontmatter.page === 54);
		expect(first?.frontmatter).toEqual(
			expect.objectContaining({ runningHeaders: ['Local Section'] }),
		);
		expect(second?.frontmatter).toEqual(
			expect.objectContaining({ runningHeaders: ['Local Section'] }),
		);
		expect(first?.blocks.some(block => block.text === 'Chapter 3: A New Topic')).toBe(true);
		expect(first?.blocks.some(block => block.text === 'Local Section')).toBe(false);
	});

	it('prefers a footer page number over a standalone roman header', () => {
		const page: OcrPageResult = {
			page: 51,
			widthPx: 1600,
			heightPx: 2200,
			lines: [
				{ text: 'VI', confidence: 1, bbox: { x: 0.8, y: 0.03, width: 0.03, height: 0.015 } },
				{ text: 'Content.', confidence: 1, bbox: { x: 0.1, y: 0.1, width: 0.7, height: 0.02 } },
				{ text: '39', confidence: 1, bbox: { x: 0.86, y: 0.94, width: 0.03, height: 0.015 } },
			],
		};
		const document = processOcrPagesToDocuments([page], { profile: 'narrative' })[0];
		expect(document?.frontmatter).toEqual(
			expect.objectContaining({
				runningHeaders: ['VI'],
				printedPage: '39',
			}),
		);
	});

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
	it('flags ambiguous logarithm-like notation without silently changing it', () => {
		const page: OcrPageResult = {
			page: 10,
			widthPx: 1600,
			heightPx: 2200,
			lines: [
				{
					text: 'The runtime is O(1og m) or O(m 10g m).',
					confidence: 1,
					bbox: { x: 0.1, y: 0.2, width: 0.7, height: 0.02 },
				},
				{
					text: 'The experiment used 10g of salt.',
					confidence: 1,
					bbox: { x: 0.1, y: 0.4, width: 0.7, height: 0.02 },
				},
			],
		};
		const document = processOcrPagesToDocuments([page], { profile: 'technical' })[0];
		const math = document?.blocks.find(block => block.text.startsWith('The runtime'));
		expect(math?.text).toBe('The runtime is O(1og m) or O(m 10g m).');
		expect(math?.confidence.flagged).toBe(true);
		expect(document?.blocks.some(block => block.text === 'The experiment used 10g of salt.')).toBe(
			true,
		);
	});

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

	it('uses isolated graph labels to define a roomy raster-crop region', () => {
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
		expect(figure?.bbox.x).toBeCloseTo(0.237, 2);
		expect((figure?.bbox.x ?? 1) + (figure?.bbox.width ?? 0)).toBeGreaterThan(0.63);
		expect(figure?.bbox.y).toBeLessThan(0.12);
		expect((figure?.bbox.y ?? 0) + (figure?.bbox.height ?? 0)).toBeGreaterThan(0.27);
		expect(document?.blocks.some(block => block.text === '0(1)' || block.text === 'o(5)')).toBe(
			false,
		);
	});

	it('detects a graph whose OCR labels are spread across the plot', () => {
		const page: OcrPageResult = {
			page: 54,
			widthPx: 1600,
			heightPx: 2200,
			lines: [
				...[0.05, 0.1, 0.15, 0.2, 0.25].map(y => ({
					text: 'Ordinary surrounding prose line',
					confidence: 1,
					bbox: { x: 0.1, y, width: 0.8, height: 0.015 },
				})),
				{
					text: 'The following graph depicts the rate of increase for common runtimes.',
					confidence: 1,
					bbox: { x: 0.094, y: 0.323, width: 0.662, height: 0.016 },
				},
				{
					text: 'O(x!)',
					confidence: 1,
					bbox: { x: 0.31, y: 0.35, width: 0.019, height: 0.029 },
				},
				{
					text: '0(x log x)',
					confidence: 1,
					bbox: { x: 0.521, y: 0.345, width: 0.08, height: 0.064 },
				},
				{
					text: 'O(log x)',
					confidence: 1,
					bbox: { x: 0.653, y: 0.575, width: 0.073, height: 0.019 },
				},
				{
					text: 'As you can see, the runtime grows at different rates.',
					confidence: 1,
					bbox: { x: 0.099, y: 0.634, width: 0.816, height: 0.015 },
				},
			],
		};

		const document = processOcrPagesToDocuments([page])[0];
		const figure = document?.blocks.find(block => block.text.startsWith('![Figure from page 54]'));

		expect(figure).toBeDefined();
		expect(figure?.bbox.x).toBeLessThan(0.31);
		expect((figure?.bbox.x ?? 1) + (figure?.bbox.width ?? 0)).toBeGreaterThan(0.726);
		expect(figure?.bbox.y).toBeCloseTo(0.345, 2);
		expect((figure?.bbox.y ?? 0) + (figure?.bbox.height ?? 0)).toBeLessThan(0.634);
		expect(document?.blocks.some(block => block.text.includes('graph depicts'))).toBe(true);
		expect(
			document?.blocks.some(block => block.text === 'O(x!)' || block.text === 'O(log x)'),
		).toBe(false);
	});
});

describe('technical layout reconstruction', () => {
	it('flags possible section headings without silently converting ordinary list items', () => {
		const line = (text: string, y: number, width: number) => ({
			text,
			confidence: 1,
			bbox: { x: 0.1, y, width, height: 0.014 },
		});
		const page: OcrPageResult = {
			page: 10,
			widthPx: 1600,
			heightPx: 2200,
			lines: [
				line('• Complexity and Memory', 0.2, 0.25),
				line('The next paragraph explains the algorithm.', 0.24, 0.7),
				line('• Best case: This example uses one iteration.', 0.35, 0.7),
				line('Asymptotic Limits', 0.45, 0.25),
				line('A detailed explanation follows this title.', 0.469, 0.7),
			],
		};
		const document = processOcrPagesToDocuments([page], { profile: 'technical' })[0];
		const possibleHeading = document?.blocks.find(block =>
			block.text.includes('Complexity and Memory'),
		);
		const listItem = document?.blocks.find(block => block.text.includes('Best case:'));
		const mergedHeading = document?.blocks.find(block => block.text.includes('Asymptotic Limits'));
		expect(possibleHeading?.confidence.flagged).toBe(true);
		expect(possibleHeading?.confidence.reasons).toContain('layout:verify');
		expect(listItem?.confidence.reasons).not.toContain('layout:verify');
		expect(mergedHeading?.confidence.reasons).toContain('layout:verify');
	});

	it('uses a corroborated text-layer row to recover a missing table cell for review', () => {
		const line = (text: string, x: number, y: number, width: number) => ({
			text,
			confidence: 1,
			bbox: { x, y, width, height: 0.012 },
		});
		const page: OcrPageResult = {
			page: 10,
			widthPx: 1600,
			heightPx: 2200,
			lines: [
				line('Level', 0.1, 0.2, 0.08),
				line('Nodes', 0.25, 0.2, 0.08),
				line('Expression', 0.5, 0.2, 0.15),
				line('2', 0.25, 0.22, 0.02),
				line('2 times previous', 0.5, 0.22, 0.2),
				line('2', 0.1, 0.24, 0.02),
				line('4', 0.25, 0.24, 0.02),
				line('4 total', 0.5, 0.24, 0.1),
				line('3', 0.1, 0.26, 0.02),
				line('8', 0.25, 0.26, 0.02),
				line('8 total', 0.5, 0.26, 0.1),
			],
			textLayerLines: [line('1 2 2 times previous', 0.1, 0.22, 0.6)],
		};
		const document = processOcrPagesToDocuments([page], { profile: 'technical' })[0];
		const table = document?.blocks.find(block => block.kind === 'table');
		expect(table?.text).toContain('| 1 | 2 | 2 times previous |');
		expect(table?.confidence.flagged).toBe(true);
		expect(table?.confidence.reasons).toContain('table:text-layer');

		const conflicting = { ...page, textLayerLines: [line('1 9', 0.1, 0.22, 0.17)] };
		const unmatched = processOcrPagesToDocuments([conflicting], { profile: 'technical' })[0];
		expect(unmatched?.blocks.find(block => block.kind === 'table')?.text).not.toContain(
			'| 1 | 2 |',
		);
	});

	it('keeps corroborated trailing braces inside a code block instead of prose', () => {
		const line = (text: string, x: number, y: number, width = 0.1) => ({
			text,
			confidence: 1,
			bbox: { x, y, width, height: 0.012 },
		});
		const page: OcrPageResult = {
			page: 10,
			widthPx: 1600,
			heightPx: 2200,
			lines: [
				line('void run() {', 0.13, 0.1),
				line('for (int a : first) {', 0.16, 0.12),
				line('use(a);', 0.18, 0.14),
				line('4', 0.1, 0.16),
				line('5', 0.1, 0.18),
				line('The algorithm is complete.', 0.1, 0.21, 0.75),
			],
			textLayerLines: [line('4 }', 0.1, 0.16), line('5 }', 0.1, 0.18)],
		};
		const document = processOcrPagesToDocuments([page], { profile: 'technical' })[0];
		const code = document?.blocks.find(block => block.kind === 'code');
		expect(code?.text).toBe('void run() {\n  for (int a : first) {\n    use(a);\n  }\n}');
		expect(code?.confidence.reasons).toContain('code:text-layer');
		expect(document?.blocks.some(block => block.text.includes('The algorithm is complete.'))).toBe(
			true,
		);
	});

	it('does not indent subsequent code because of braces inside string literals', () => {
		const line = (text: string, y: number) => ({
			text,
			confidence: 1,
			bbox: { x: 0.1, y, width: 0.3, height: 0.014 },
		});
		const page: OcrPageResult = {
			page: 10,
			widthPx: 1600,
			heightPx: 2200,
			lines: [
				line('void run() {', 0.1),
				line('print("{");', 0.12),
				line('print("done");', 0.14),
				line('}', 0.16),
			],
		};
		const document = processOcrPagesToDocuments([page], { profile: 'technical' })[0];
		expect(document?.blocks.find(block => block.kind === 'code')?.text).toBe(
			'void run() {\n  print("{");\n  print("done");\n}',
		);
	});

	it('does not discard a numeric literal at the start of a code line', () => {
		const line = (text: string, y: number) => ({
			text,
			confidence: 1,
			bbox: { x: 0.1, y, width: 0.3, height: 0.014 },
		});
		const page: OcrPageResult = {
			page: 10,
			widthPx: 1600,
			heightPx: 2200,
			lines: [line('void run() {', 0.1), line('2 + 3;', 0.12), line('}', 0.14)],
		};
		const document = processOcrPagesToDocuments([page], { profile: 'technical' })[0];
		expect(document?.blocks.find(block => block.kind === 'code')?.text).toContain('2 + 3;');
	});

	it('joins adjacent OCR fragments belonging to the same code line', () => {
		const line = (text: string, x: number, y: number, width: number) => ({
			text,
			confidence: 1,
			bbox: { x, y, width, height: 0.014 },
		});
		const page: OcrPageResult = {
			page: 10,
			widthPx: 1600,
			heightPx: 2200,
			lines: [
				line('void check(int x) {', 0.1, 0.1, 0.3),
				line('if', 0.13, 0.12, 0.02),
				line('(x > 0) {', 0.15, 0.12, 0.2),
				line('print(x);', 0.16, 0.14, 0.2),
				line('}', 0.13, 0.16, 0.02),
				line('}', 0.1, 0.18, 0.02),
			],
		};
		const document = processOcrPagesToDocuments([page], { profile: 'technical' })[0];
		const code = document?.blocks.find(block => block.kind === 'code');
		expect(code?.text).toContain('if (x > 0) {');
		expect(code?.confidence.flagged).toBe(true);
		expect(document?.blocks.some(block => block.kind === 'paragraph' && block.text === 'if')).toBe(
			false,
		);
	});

	it('does not classify side-by-side narrative sentences as code just because they contain semicolons', () => {
		const line = (text: string, x: number, y: number) => ({
			text,
			confidence: 1,
			bbox: { x, y, width: 0.25, height: 0.014 },
		});
		const page: OcrPageResult = {
			page: 10,
			widthPx: 1600,
			heightPx: 2200,
			lines: [
				line('At home', 0.1, 0.1),
				line('At work', 0.52, 0.1),
				line('She waited; the sun rose.', 0.1, 0.13),
				line('He waited; the rain fell.', 0.52, 0.13),
				line('The wind blew; she smiled.', 0.1, 0.15),
				line('The rain stopped; he smiled.', 0.52, 0.15),
				line('She returned; dinner was ready.', 0.1, 0.17),
				line('He returned; the day was over.', 0.52, 0.17),
			],
		};
		const document = processOcrPagesToDocuments([page], { profile: 'auto' })[0];
		expect(document?.blocks.some(block => block.kind === 'code' || block.kind === 'table')).toBe(
			false,
		);
	});

	it('uses a matching text-layer line to recover missing numbered code lines for review', () => {
		const line = (text: string, x: number, y: number) => ({
			text,
			confidence: 1,
			bbox: { x, y, width: 0.1, height: 0.012 },
		});
		const page: OcrPageResult = {
			page: 10,
			widthPx: 1600,
			heightPx: 2200,
			lines: [
				line('void run() {', 0.13, 0.1),
				line('for (int a : first) {', 0.16, 0.12),
				line('use(a);', 0.18, 0.14),
				line('for (int b : second) {', 0.16, 0.18),
				line('use(b);', 0.18, 0.2),
				line('7', 0.1, 0.22),
				line('}', 0.13, 0.24),
			],
			textLayerLines: [
				line('3 use(a);', 0.1, 0.14),
				line('4 }', 0.1, 0.16),
				line('5 for (int b : second) {', 0.1, 0.18),
				line('7 }', 0.1, 0.22),
			],
		};
		const document = processOcrPagesToDocuments([page], { profile: 'technical' })[0];
		const code = document?.blocks.find(block => block.kind === 'code');
		expect(code?.text).toBe(
			'void run() {\n  for (int a : first) {\n    use(a);\n  }\n  for (int b : second) {\n    use(b);\n  }\n}',
		);
		expect(code?.confidence.flagged).toBe(true);
		expect(code?.confidence.reasons).toContain('code:verify');
		expect(code?.confidence.reasons).toContain('code:text-layer');

		const conflictingLayer = {
			...page,
			textLayerLines: [line('9 }', 0.1, 0.16), line('8 }', 0.1, 0.22)],
		};
		const uncorroborated = processOcrPagesToDocuments([conflictingLayer], {
			profile: 'technical',
		})[0];
		const uncorroboratedCode = uncorroborated?.blocks.find(block => block.kind === 'code');
		expect(uncorroboratedCode?.text).not.toContain('use(a);\n  }\n  for');
		expect(uncorroboratedCode?.confidence.flagged).toBe(true);
		expect(uncorroboratedCode?.confidence.reasons).not.toContain('code:text-layer');
	});

	it('does not apply mathematical spelling normalization to code identifiers', () => {
		const line = (text: string, y: number) => ({
			text,
			confidence: 1,
			bbox: { x: 0.1, y, width: 0.3, height: 0.014 },
		});
		const page: OcrPageResult = {
			page: 10,
			widthPx: 1600,
			heightPx: 2200,
			lines: [line('void run() {', 0.1), line('o(value);', 0.12), line('}', 0.14)],
		};
		const document = processOcrPagesToDocuments([page], { profile: 'technical' })[0];
		const code = document?.blocks.find(block => block.kind === 'code');
		expect(code?.text).toContain('o(value);');
		expect(code?.text).not.toContain('O(value);');
	});

	it('keeps a function split across consecutive pages as one code block in the EPUB', () => {
		const line = (text: string, y: number) => ({
			text,
			confidence: 1,
			bbox: { x: 0.1, y, width: 0.4, height: 0.014 },
		});
		const pages: OcrPageResult[] = [
			{
				page: 10,
				widthPx: 1600,
				heightPx: 2200,
				lines: [
					line('The following function calculates a result.', 0.72),
					line('1 int factorial(int n) {', 0.89),
				],
			},
			{
				page: 11,
				widthPx: 1600,
				heightPx: 2200,
				lines: [
					line('if (n <= 1) {', 0.08),
					line('return 1;', 0.1),
					line('}', 0.12),
					line('return factorial(n - 1);', 0.14),
					line('}', 0.16),
					line('The algorithm completes.', 0.22),
				],
			},
		];
		const documents = processOcrPagesToDocuments(pages, { profile: 'technical' });
		const [first, second] = documents;
		expect(first?.blocks.at(-1)?.kind).toBe('code');
		expect(first?.blocks.at(-1)?.confidence.flagged).toBe(true);
		expect(first?.frontmatter).toEqual(expect.objectContaining({ codeContinuesToNext: true }));
		expect(second?.frontmatter).toEqual(expect.objectContaining({ codeContinuesFromPrev: true }));
		const roundTripped = documents.map(document =>
			parsePageDocument(serializePageDocument(document)),
		);
		const { chapters } = stitchPagesAndGroupChapters(roundTripped, 'Sample');
		const codeItems = chapters
			.flatMap(chapter => chapter.items)
			.filter(item => item.kind === 'code');
		expect(codeItems).toHaveLength(1);
		expect(codeItems[0]?.sourcePages).toEqual([10, 11]);
		expect(codeItems[0]?.text).toBe(
			'int factorial(int n) {\n  if (n <= 1) {\n    return 1;\n  }\n  return factorial(n - 1);\n}',
		);
		const nonconsecutive = processOcrPagesToDocuments(
			pages.map((page, index) => (index === 1 ? { ...page, page: 12 } : page)),
			{ profile: 'technical' },
		);
		expect(nonconsecutive[0]?.frontmatter.codeContinuesToNext).toBeUndefined();
		expect(nonconsecutive[1]?.frontmatter.codeContinuesFromPrev).toBeUndefined();
	});

	it('keeps a sparse continued code fragment with a standalone comment together', () => {
		const line = (text: string, y: number) => ({
			text,
			confidence: 1,
			bbox: { x: 0.1, y, width: 0.4, height: 0.014 },
		});
		const pages: OcrPageResult[] = [
			{
				page: 10,
				widthPx: 1600,
				heightPx: 2200,
				lines: [
					line('The following function calculates a result.', 0.72),
					line('1 void printPairs(int[] a) {', 0.89),
				],
			},
			{
				page: 11,
				widthPx: 1600,
				heightPx: 2200,
				lines: [
					line('for (int i = 0; i < a.length; i++) {', 0.075),
					line('for (int j = 0; j < a.length; j++) {', 0.09),
					line('/* constant work */', 0.105),
					line('}', 0.15),
					line('The function performs constant work.', 0.2),
				],
			},
		];
		const documents = processOcrPagesToDocuments(pages, { profile: 'auto' });
		expect(documents[0]?.frontmatter).toEqual(
			expect.objectContaining({ codeContinuesToNext: true }),
		);
		expect(documents[1]?.frontmatter).toEqual(
			expect.objectContaining({ codeContinuesFromPrev: true }),
		);
		const code = documents[1]?.blocks.find(block => block.kind === 'code');
		expect(code?.text).toContain('/* constant work */');
		expect(code?.confidence.flagged).toBe(true);
	});

	it('flags OCR-derived code and mathematical expressions even when OCR confidence is high', () => {
		const line = (text: string, y: number) => ({
			text,
			confidence: 1,
			bbox: { x: 0.1, y, width: 0.7, height: 0.014 },
		});
		const page: OcrPageResult = {
			page: 10,
			widthPx: 1600,
			heightPx: 2200,
			lines: [
				line('She read the book and walked home.', 0.1),
				line('The runtime of this loop is O(N?).', 0.2),
				line('void process(int[] values) {', 0.35),
				line('for (int i = 0; i ‹ values.length; i++) {', 0.37),
				line('print(values[i]);', 0.39),
				line('}', 0.41),
				line('}', 0.43),
			],
		};
		const document = processOcrPagesToDocuments([page], { profile: 'technical' })[0];
		if (!document) throw new Error('Expected one processed page');
		const math = document?.blocks.find(block => block.text.includes('O(N?)'));
		const code = document?.blocks.find(block => block.kind === 'code');
		const prose = document?.blocks.find(block => block.text.includes('walked home'));

		expect(document?.frontmatter.status).toBe('needs-review');
		expect(math?.confidence.flagged).toBe(true);
		expect(math?.confidence.reasons).toContain('math:verify');
		expect(code?.confidence.flagged).toBe(true);
		expect(code?.confidence.reasons).toContain('code:verify');
		expect(prose?.confidence.flagged).toBe(false);
		const parsed = parsePageDocument(serializePageDocument(document));
		expect(parsed.blocks.find(block => block.kind === 'code')?.confidence.reasons).toContain(
			'code:verify',
		);
	});

	it('marks ambiguous fractions and numeric pair diagrams for review', () => {
		const line = (text: string, y: number) => ({
			text,
			confidence: 1,
			bbox: { x: 0.1, y, width: 0.7, height: 0.014 },
		});
		const page: OcrPageResult = {
			page: 10,
			widthPx: 1600,
			heightPx: 2200,
			lines: [
				line('After a step there are ½ elements remaining.', 0.1),
				line('The pairs are (0, 1) (0, 2) (8, 5).', 0.2),
			],
		};
		const document = processOcrPagesToDocuments([page], { profile: 'technical' })[0];
		expect(
			document?.blocks.filter(block => block.confidence.reasons.includes('math:verify')),
		).toHaveLength(2);
	});

	it('keeps independently titled side-by-side code examples separate, including fragmented lines', () => {
		const line = (text: string, x: number, y: number, width: number) => ({
			text,
			confidence: 1,
			bbox: { x, y, width, height: 0.012 },
		});
		const page: OcrPageResult = {
			page: 10,
			widthPx: 1600,
			heightPx: 2200,
			lines: [
				line('Sequential work', 0.14, 0.1, 0.2),
				line('Nested work', 0.52, 0.1, 0.2),
				line('for (int a : first) {', 0.18, 0.13, 0.2),
				line('for (int a : first) {', 0.56, 0.13, 0.2),
				line('consume(a);', 0.2, 0.15, 0.12),
				line('for (int b : second) {', 0.58, 0.15, 0.2),
				line('}', 0.18, 0.17, 0.01),
				line('print(a +', 0.6, 0.17, 0.1),
				line('"," + b);', 0.7, 0.17, 0.1),
				line('for (int b : second) {', 0.18, 0.19, 0.2),
				line('}', 0.58, 0.19, 0.01),
				line('consume(b);', 0.2, 0.21, 0.12),
				line('}', 0.56, 0.21, 0.01),
				line('}', 0.18, 0.23, 0.01),
				line('These examples have different runtimes.', 0.09, 0.27, 0.75),
			],
		};
		const document = processOcrPagesToDocuments([page], { profile: 'technical' })[0];
		const markdown = document ? serializePageDocument(document) : '';
		const comparison = document?.blocks.find(block => block.kind === 'table');
		expect(comparison?.confidence.flagged).toBe(true);
		expect(comparison?.confidence.reasons).toContain('table:verify');
		expect(markdown).toContain('<th>Sequential work</th><th>Nested work</th>');
		expect(markdown).toContain('for (int a : first) {\n  consume(a);\n}\nfor (int b : second) {');
		expect(markdown).toContain(
			'for (int a : first) {\n  for (int b : second) {\n    print(a + "," + b);',
		);
		expect(markdown).toContain('These examples have different runtimes.');
	});

	it('keeps code snippets and side-by-side code comparisons structured', () => {
		const page: OcrPageResult = {
			page: 53,
			widthPx: 1600,
			heightPx: 2200,
			lines: [
				{
					text: 'int pairSumSequence(int n) { /* Ex 2.*/',
					confidence: 1,
					bbox: { x: 0.122, y: 0.107, width: 0.373, height: 0.013 },
				},
				{ text: '2', confidence: 1, bbox: { x: 0.09, y: 0.124, width: 0.011, height: 0.009 } },
				{
					text: 'int sum = 0;',
					confidence: 1,
					bbox: { x: 0.146, y: 0.124, width: 0.116, height: 0.01 },
				},
				{ text: '3', confidence: 1, bbox: { x: 0.09, y: 0.138, width: 0.011, height: 0.009 } },
				{
					text: 'for (int i = 0; i < n; i++) {',
					confidence: 1,
					bbox: { x: 0.141, y: 0.136, width: 0.281, height: 0.012 },
				},
				{
					text: 'sum += pairSum(i, i + 1);',
					confidence: 1,
					bbox: { x: 0.171, y: 0.15, width: 0.236, height: 0.015 },
				},
				{ text: '}', confidence: 1, bbox: { x: 0.146, y: 0.166, width: 0.013, height: 0.013 } },
				{
					text: 'return sum;',
					confidence: 1,
					bbox: { x: 0.147, y: 0.181, width: 0.105, height: 0.012 },
				},
				{ text: '7 }', confidence: 1, bbox: { x: 0.09, y: 0.196, width: 0.045, height: 0.012 } },
				{ text: '8', confidence: 1, bbox: { x: 0.09, y: 0.211, width: 0.011, height: 0.01 } },
				{
					text: 'int pairSum(int a, int b) {',
					confidence: 1,
					bbox: { x: 0.124, y: 0.225, width: 0.255, height: 0.012 },
				},
				{
					text: '10 return a + b;',
					confidence: 1,
					bbox: { x: 0.146, y: 0.24, width: 0.124, height: 0.012 },
				},
				{ text: '11 }', confidence: 1, bbox: { x: 0.09, y: 0.254, width: 0.045, height: 0.012 } },
				{
					text: 'Consider the below code:',
					confidence: 1,
					bbox: { x: 0.09, y: 0.493, width: 0.199, height: 0.012 },
				},
				{
					text: 'Min and Max 1',
					confidence: 1,
					bbox: { x: 0.171, y: 0.522, width: 0.122, height: 0.013 },
				},
				{
					text: 'Min and Max 2',
					confidence: 1,
					bbox: { x: 0.501, y: 0.522, width: 0.124, height: 0.012 },
				},
				{
					text: 'int min = Integer.MAX_VALUE;',
					confidence: 1,
					bbox: { x: 0.197, y: 0.542, width: 0.274, height: 0.013 },
				},
				{
					text: '1 int min = Integer.MAX_VALUE;',
					confidence: 1,
					bbox: { x: 0.503, y: 0.542, width: 0.3, height: 0.013 },
				},
				{
					text: 'int max = Integer.MIN_VALUE;',
					confidence: 1,
					bbox: { x: 0.208, y: 0.557, width: 0.266, height: 0.013 },
				},
				{
					text: '2 int max = Integer.MIN_VALUE;',
					confidence: 1,
					bbox: { x: 0.537, y: 0.557, width: 0.266, height: 0.015 },
				},
				{
					text: 'for (int x : array) {',
					confidence: 1,
					bbox: { x: 0.208, y: 0.571, width: 0.199, height: 0.013 },
				},
				{
					text: '3 for (int x : array) {',
					confidence: 1,
					bbox: { x: 0.503, y: 0.571, width: 0.238, height: 0.013 },
				},
				{
					text: 'if (x < min) min = x;',
					confidence: 1,
					bbox: { x: 0.229, y: 0.586, width: 0.199, height: 0.013 },
				},
				{
					text: '4 if (x < min) min = x;',
					confidence: 1,
					bbox: { x: 0.503, y: 0.587, width: 0.203, height: 0.013 },
				},
				{
					text: 'if (x > max) max = x;',
					confidence: 1,
					bbox: { x: 0.229, y: 0.6, width: 0.199, height: 0.013 },
				},
				{ text: '}', confidence: 1, bbox: { x: 0.537, y: 0.615, width: 0.013, height: 0.013 } },
				{
					text: 'for (int x : array) {',
					confidence: 1,
					bbox: { x: 0.537, y: 0.615, width: 0.201, height: 0.013 },
				},
				{
					text: 'if (x > max) max = x;',
					confidence: 1,
					bbox: { x: 0.561, y: 0.631, width: 0.199, height: 0.013 },
				},
				{ text: '}', confidence: 1, bbox: { x: 0.533, y: 0.645, width: 0.017, height: 0.013 } },
				{
					text: 'Which one is faster?',
					confidence: 1,
					bbox: { x: 0.086, y: 0.669, width: 0.3, height: 0.015 },
				},
			],
		};

		const document = processOcrPagesToDocuments([page])[0];
		const markdown = document ? serializePageDocument(document) : '';
		const parsed = parsePageDocument(markdown);

		expect(markdown).toContain('```');
		expect(markdown).toContain('int pairSumSequence(int n)');
		expect(markdown).toContain('return a + b;\n}');
		expect(markdown).not.toContain('11 }');
		expect(markdown).toContain('<table>');
		expect(markdown).toContain('<td><pre><code>');
		expect(markdown).not.toContain('\n\n8\n\n');
		expect(parsed.blocks.some(block => block.kind === 'code')).toBe(true);
		expect(parsed.blocks.some(block => block.kind === 'table')).toBe(true);

		const narrative = processOcrPagesToDocuments([page], { profile: 'narrative' })[0];
		expect(narrative?.blocks.some(block => block.kind === 'code')).toBe(false);
		expect(narrative?.blocks.some(block => block.text.startsWith('![Figure'))).toBe(false);
		const technical = processOcrPagesToDocuments([page], { profile: 'technical' })[0];
		expect(technical?.blocks.some(block => block.kind === 'code')).toBe(true);
		expect(markdown).toContain('  int sum = 0;');
		expect(markdown).toContain('    sum += pairSum(i, i + 1);');
		expect(markdown).toContain('<pre><code>');
		expect(markdown).toContain('<th>Min and Max 1</th><th>Min and Max 2</th>');
		expect(markdown).not.toContain('<br>');
	});

	it('preserves a recursion tree as a figure and its neighboring grid as a table', () => {
		const page: OcrPageResult = {
			page: 57,
			widthPx: 1600,
			heightPx: 2200,
			lines: [
				{
					text: 'if (n <= 1) {',
					confidence: 1,
					bbox: { x: 0.148, y: 0.074, width: 0.126, height: 0.013 },
				},
				{
					text: 'return 1;',
					confidence: 1,
					bbox: { x: 0.173, y: 0.088, width: 0.086, height: 0.014 },
				},
				{
					text: 'return f(n - 1) + f(n - 1);',
					confidence: 1,
					bbox: { x: 0.15, y: 0.118, width: 0.255, height: 0.013 },
				},
				{ text: '}', confidence: 1, bbox: { x: 0.126, y: 0.132, width: 0.011, height: 0.012 } },
				{
					text: 'Suppose we call f (4). This calls f (3) twice and then f (2).',
					confidence: 1,
					bbox: { x: 0.09, y: 0.192, width: 0.73, height: 0.019 },
				},
				{ text: 'f (4)', confidence: 1, bbox: { x: 0.473, y: 0.216, width: 0.047, height: 0.016 } },
				{ text: 'f (3)', confidence: 1, bbox: { x: 0.287, y: 0.243, width: 0.045, height: 0.013 } },
				{ text: 'f (3)', confidence: 1, bbox: { x: 0.66, y: 0.244, width: 0.043, height: 0.013 } },
				{ text: 'f (1)', confidence: 1, bbox: { x: 0.165, y: 0.294, width: 0.034, height: 0.012 } },
				{ text: 'f (1)', confidence: 1, bbox: { x: 0.242, y: 0.294, width: 0.036, height: 0.012 } },
				{ text: 'f (1)', confidence: 1, bbox: { x: 0.426, y: 0.295, width: 0.047, height: 0.013 } },
				{ text: 'f (1)', confidence: 1, bbox: { x: 0.516, y: 0.292, width: 0.051, height: 0.016 } },
				{ text: 'f (1)', confidence: 1, bbox: { x: 0.61, y: 0.291, width: 0.047, height: 0.016 } },
				{ text: 'f (1)', confidence: 1, bbox: { x: 0.708, y: 0.292, width: 0.045, height: 0.016 } },
				{ text: 'f (1)', confidence: 1, bbox: { x: 0.782, y: 0.292, width: 0.047, height: 0.016 } },
				{
					text: 'How many calls are in this tree? (Don’t count!)',
					confidence: 1,
					bbox: { x: 0.092, y: 0.331, width: 0.351, height: 0.013 },
				},
				{
					text: 'The number of nodes on each level is:',
					confidence: 1,
					bbox: { x: 0.092, y: 0.374, width: 0.593, height: 0.013 },
				},
				{ text: 'Level', confidence: 1, bbox: { x: 0.218, y: 0.4, width: 0.034, height: 0.012 } },
				{ text: '# Nodes', confidence: 1, bbox: { x: 0.268, y: 0.4, width: 0.069, height: 0.012 } },
				{
					text: 'Also expressed as...',
					confidence: 1,
					bbox: { x: 0.358, y: 0.4, width: 0.152, height: 0.013 },
				},
				{ text: 'Or...', confidence: 1, bbox: { x: 0.694, y: 0.4, width: 0.036, height: 0.013 } },
				{ text: '2^0', confidence: 1, bbox: { x: 0.694, y: 0.419, width: 0.024, height: 0.013 } },
				{ text: '2', confidence: 1, bbox: { x: 0.27, y: 0.44, width: 0.013, height: 0.01 } },
				{
					text: '2 * previous level = 2',
					confidence: 1,
					bbox: { x: 0.358, y: 0.44, width: 0.163, height: 0.012 },
				},
				{ text: '2^1', confidence: 1, bbox: { x: 0.694, y: 0.439, width: 0.026, height: 0.015 } },
				{ text: '2', confidence: 1, bbox: { x: 0.21, y: 0.461, width: 0.013, height: 0.012 } },
				{ text: '4', confidence: 1, bbox: { x: 0.268, y: 0.462, width: 0.015, height: 0.012 } },
				{
					text: '2 * previous level = 2 * 2^1 = 2^2',
					confidence: 1,
					bbox: { x: 0.355, y: 0.459, width: 0.266, height: 0.018 },
				},
				{ text: '2^2', confidence: 1, bbox: { x: 0.694, y: 0.459, width: 0.026, height: 0.015 } },
				{ text: '8', confidence: 1, bbox: { x: 0.27, y: 0.483, width: 0.013, height: 0.01 } },
				{
					text: '2 * previous level = 2 * 2^2 = 2^3',
					confidence: 1,
					bbox: { x: 0.355, y: 0.48, width: 0.266, height: 0.018 },
				},
				{ text: '2^3', confidence: 1, bbox: { x: 0.694, y: 0.481, width: 0.024, height: 0.015 } },
				{ text: '16', confidence: 1, bbox: { x: 0.268, y: 0.503, width: 0.024, height: 0.013 } },
				{
					text: '2 * previous level = 2 * 2^3 = 2^4',
					confidence: 1,
					bbox: { x: 0.355, y: 0.5, width: 0.266, height: 0.017 },
				},
				{ text: '2^4', confidence: 1, bbox: { x: 0.694, y: 0.503, width: 0.024, height: 0.013 } },
			],
		};

		const document = processOcrPagesToDocuments([page])[0];
		const markdown = document ? serializePageDocument(document) : '';

		expect(markdown).toContain('![Figure from page 57](figures/page-0057-figure-01.png)');
		expect(markdown).toContain('| Level | # Nodes | Also expressed as... | Or... |');
		expect(markdown).toContain('| 4 | 2 * previous level = 2 * 2^1 = 2^2 |');
		expect(markdown).toContain('Suppose we call f (4). This calls f (3) twice');
		expect(markdown).not.toContain('\n\nf (3)\n\n');
		expect(markdown).not.toContain('\n\n+(2)\n\n');
		expect(markdown).not.toContain('\n\n(2)\n\n');
		const parsed = parsePageDocument(markdown);
		expect(parsed.blocks.some(block => block.kind === 'table')).toBe(true);
	});
});

describe('scan output with detected figures', () => {
	it.skipIf(process.platform !== 'darwin' || !existsSync(join(process.cwd(), 'test/ctci.pdf')))(
		'keeps code comparisons separate and recovers corroborated missing braces from the sample PDF',
		async () => {
			const result = await new VisionOcrProvider().recognizePages({
				pdfPath: join(process.cwd(), 'test/ctci.pdf'),
				pages: Array.from({ length: 10 }, (_, index) => 51 + index),
				dpi: 220,
				language: 'en',
			});
			const documents = processOcrPagesToDocuments(result.pages, { profile: 'auto' });
			expect(documents).toHaveLength(10);
			const paint = documents
				.find(doc => doc.frontmatter.page === 51)
				?.blocks.find(block => block.text.includes('of paint'));
			if (paint?.text.includes('players of paint')) expect(paint.confidence.flagged).toBe(true);
			const comparison = documents
				.find(doc => doc.frontmatter.page === 55)
				?.blocks.find(block => block.kind === 'table');
			const code = documents
				.find(doc => doc.frontmatter.page === 58)
				?.blocks.find(block => block.kind === 'code');
			const nodeTable = documents
				.find(doc => doc.frontmatter.page === 57)
				?.blocks.find(block => block.kind === 'table');
			expect(comparison?.text).toContain('<th>Add the Runtimes:');
			expect(comparison?.text).toContain('<th>Multiply the Runtimes:');
			expect(comparison?.text).toContain('print(a + "," + b);');
			expect(code?.text).toMatch(/sum\s*\+=\s*array\[i\];\n\s*}\n\s*for\s*\(/);
			expect(code?.confidence.flagged).toBe(true);
			expect(nodeTable?.text).toContain('| 1 | 2 |');
			expect(nodeTable?.text).toContain('| 3 | 8 |');
			expect(nodeTable?.text).toContain('| 4 | 16 |');
			expect(nodeTable?.confidence.flagged).toBe(true);
			expect(
				documents.find(doc => doc.frontmatter.page === 56)?.frontmatter.codeContinuesToNext,
			).toBe(true);
			expect(
				documents.find(doc => doc.frontmatter.page === 57)?.frontmatter.codeContinuesFromPrev,
			).toBe(true);
			expect(
				documents.find(doc => doc.frontmatter.page === 59)?.frontmatter.codeContinuesToNext,
			).toBe(true);
			expect(
				documents.find(doc => doc.frontmatter.page === 60)?.frontmatter.codeContinuesFromPrev,
			).toBe(true);
			const { chapters } = stitchPagesAndGroupChapters(documents, 'Sample');
			const joinedCode = chapters
				.flatMap(chapter => chapter.items)
				.filter(item => item.kind === 'code' && item.sourcePages.length > 1);
			expect(joinedCode.map(item => item.sourcePages)).toContainEqual([56, 57]);
			expect(joinedCode.map(item => item.sourcePages)).toContainEqual([59, 60]);
		},
		30_000,
	);

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
		const coverPage: OcrPageResult = {
			page: 1,
			widthPx: 1600,
			heightPx: 2200,
			lines: [
				{
					text: 'OCR cover title should not be retained',
					confidence: 1,
					bbox: { x: 0.1, y: 0.2, width: 0.6, height: 0.03 },
				},
			],
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
			pages: [coverPage, page],
			provider,
			dpi: 220,
		});

		const coverMarkdown = readFileSync(join(bookDir, 'pages', 'page-0001.md'), 'utf8');
		const markdown = readFileSync(join(bookDir, 'pages', 'page-0051.md'), 'utf8');
		const coverPath = join(bookDir, 'pages', 'figures', 'book-cover.png');
		const figurePath = join(bookDir, 'pages', 'figures', 'page-0051-figure-01.png');
		expect(coverMarkdown).toContain('![Book cover](figures/book-cover.png)');
		expect(coverMarkdown).not.toContain('OCR cover title');
		expect(markdown).toContain('![Figure from page 51](figures/page-0051-figure-01.png)');
		expect(existsSync(coverPath)).toBe(true);
		expect(existsSync(figurePath)).toBe(true);
		expect(renderCalls).toHaveLength(2);
		const coverRender = renderCalls.find(call => call.page === 1);
		expect(coverRender?.pdfPath).toBe('/source/interview-book.pdf');
		expect(coverRender?.crop).toBeUndefined();
		const figureRender = renderCalls.find(call => call.page === 51);
		expect(figureRender?.crop?.x).toBeLessThan(0.39);
		expect(figureRender?.trimWhitespace).toBe(true);
	});

	it.skipIf(process.platform !== 'darwin')(
		'trims a graph to its ink bounds and applies equal pixel margins',
		async () => {
			const imagePath = join(createTempDir(), 'tight-graph.png');
			await new VisionOcrProvider().renderPageImage({
				pdfPath: join(process.cwd(), 'test/ctci.pdf'),
				page: 51,
				outputPath: imagePath,
				dpi: 160,
				crop: { x: 0.237, y: 0.105, width: 0.41, height: 0.182 },
				trimWhitespace: true,
			});

			const margins = readPngInkMargins(imagePath);
			expect(margins).toHaveLength(4);
			expect(Math.max(...margins) - Math.min(...margins)).toBeLessThanOrEqual(2);
		},
		30_000,
	);

	it.skipIf(process.platform !== 'darwin')(
		'keeps the neighboring caption and paragraph outside the graph crop',
		async () => {
			const imagePath = join(createTempDir(), 'captioned-graph.png');
			await new VisionOcrProvider().renderPageImage({
				pdfPath: join(process.cwd(), 'test/ctci.pdf'),
				page: 54,
				outputPath: imagePath,
				dpi: 220,
				crop: {
					x: 0.1305,
					y: 0.345,
					width: 0.7757,
					height: 0.283,
				},
				trimWhitespace: true,
			});

			const png = readFileSync(imagePath);
			const width = png.readUInt32BE(16);
			const height = png.readUInt32BE(20);
			const margins = readPngInkMargins(imagePath);
			expect(height).toBeLessThan(670);
			expect(width / height).toBeLessThan(1.25);
			expect(Math.max(...margins) - Math.min(...margins)).toBeLessThanOrEqual(2);
		},
		30_000,
	);
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

	it('round-trips technical code blocks with blank lines and Markdown tables', () => {
		const page: PageDocument = {
			frontmatter: {
				page: 44,
				status: 'clean',
				confidence: 1,
				flaggedBlocks: 0,
				continuesFromPrev: false,
				continuesToNext: false,
			},
			blocks: [
				{
					id: 'p44-code-1',
					kind: 'code',
					text: 'int value = 1;\n\nreturn value;',
					bbox: { x: 0.1, y: 0.1, width: 0.5, height: 0.1 },
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
					id: 'p44-table-1',
					kind: 'table',
					text: '| A | B |\n| --- | --- |\n| 1 | 2 |',
					bbox: { x: 0.1, y: 0.3, width: 0.8, height: 0.1 },
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
		};

		const markdown = serializePageDocument(page);
		const parsed = parsePageDocument(markdown);
		expect(parsed.blocks.map(block => block.kind)).toEqual(['code', 'table']);
		expect(parsed.blocks[0]?.text).toBe('int value = 1;\n\nreturn value;');
		expect(parsed.blocks[1]?.text).toBe('| A | B |\n| --- | --- |\n| 1 | 2 |');
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
		const coverImagePath = 'figures/book-cover.png';
		mkdirSync(join(workspaceDir, 'pages', 'figures'), { recursive: true });
		writeFileSync(join(workspaceDir, 'pages', imagePath), Buffer.from('test figure image'));
		writeFileSync(join(workspaceDir, 'pages', coverImagePath), Buffer.from('test cover image'));
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
					id: 'p1-cover',
					kind: 'paragraph',
					text: '![Book cover](figures/book-cover.png)',
					bbox: { x: 0, y: 0, width: 1, height: 1 },
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
		writePageFile(workspaceDir, {
			frontmatter: {
				page: 2,
				status: 'clean',
				confidence: 1,
				flaggedBlocks: 0,
				continuesFromPrev: false,
				continuesToNext: false,
			},
			blocks: [
				{
					id: 'p2-b1',
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
					id: 'p2-code-1',
					kind: 'code',
					text: 'if (n < 2) {\n  n++;\n\n}',
					bbox: { x: 0.1, y: 0.16, width: 0.3, height: 0.04 },
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
					id: 'p2-table-1',
					kind: 'table',
					text: '<table>\n<thead><tr><th>Min and Max 1</th><th>Min and Max 2</th></tr></thead>\n<tbody><tr>\n<td><pre><code>if (x &lt; min) min = x;\n  return value;</code></pre></td>\n<td><pre><code>int max = 0;</code></pre></td>\n</tr></tbody>\n</table>',
					bbox: { x: 0.1, y: 0.21, width: 0.5, height: 0.08 },
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
					id: 'p2-figure-1',
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

		const epubPath = join(workspaceDir, 'figures.epub');
		const markdownBeforeBuild = readFileSync(join(workspaceDir, 'pages', 'page-0002.md'), 'utf8');
		await buildEpubFromWorkspace({ bookDir: workspaceDir, outputPath: epubPath });
		const zip = await JSZip.loadAsync(
			await import('node:fs/promises').then(fs => fs.readFile(epubPath)),
		);
		const cover = await zip.file('OEBPS/cover.xhtml')?.async('string');
		const chapter = await zip.file('OEBPS/chapter-001.xhtml')?.async('string');
		const packageXml = await zip.file('OEBPS/content.opf')?.async('string');

		expect(cover).toContain('<img src="images/cover.png" alt="Book cover" />');
		expect(chapter).not.toContain('Book cover');
		expect(chapter).toContain(
			'<img src="figures/page-0051-figure-01.png" alt="Figure from page 51" />',
		);
		expect(chapter).toContain(
			'<pre class="numbered-code"><code><span class="code-line">if (n &lt; 2) {</span>\n<span class="code-line">  n++;</span>\n<span class="code-line"></span>\n<span class="code-line">}</span></code></pre>',
		);
		expect(chapter).toContain('<table>');
		expect(chapter).toContain(
			'<td><pre class="numbered-code"><code><span class="code-line">if (x &lt; min) min = x;</span>\n<span class="code-line">  return value;</span></code></pre></td>',
		);
		expect(chapter).toContain(
			'<td><pre class="numbered-code"><code><span class="code-line">int max = 0;</span></code></pre></td>',
		);
		const css = await zip.file('OEBPS/styles.css')?.async('string');
		expect(css).toContain('counter-reset: code-line');
		expect(css).toContain('counter-increment: code-line');
		expect(css).toContain('content: counter(code-line)');
		const codeText = [
			...(chapter ?? '').matchAll(/<pre class="numbered-code"><code>([\s\S]*?)<\/code><\/pre>/g),
		].map(match =>
			(match[1] ?? '')
				.replace(/<span class="code-line">|<\/span>/g, '')
				.replace(/&lt;/g, '<')
				.replace(/&gt;/g, '>')
				.replace(/&amp;/g, '&'),
		);
		expect(codeText).toEqual([
			'if (n < 2) {\n  n++;\n\n}',
			'if (x < min) min = x;\n  return value;',
			'int max = 0;',
		]);
		expect(readFileSync(join(workspaceDir, 'pages', 'page-0002.md'), 'utf8')).toBe(
			markdownBeforeBuild,
		);
		expect(packageXml).toContain(
			'href="images/cover.png" media-type="image/png" properties="cover-image"',
		);
		expect(packageXml).toContain('href="figures/page-0051-figure-01.png" media-type="image/png"');
		expect(await zip.file(`OEBPS/${imagePath}`)?.async('string')).toBe('test figure image');
		expect(await zip.file('OEBPS/images/cover.png')?.async('string')).toBe('test cover image');
	});
});

describe('CLI page option parser', () => {
	it('parses page range strings accurately', () => {
		expect(parsePagesOption('all')).toBe('all');
		expect(parsePagesOption('1-3, 7, 10-11')).toEqual([1, 2, 3, 7, 10, 11]);
	});
});

describe('review page preview layout', () => {
	it('shows local alternative OCR for a flagged region without changing the page', async () => {
		const bookDir = createTempDir();
		const now = new Date().toISOString();
		writePageFile(bookDir, {
			frontmatter: {
				page: 51,
				status: 'needs-review',
				confidence: 0.81,
				flaggedBlocks: 1,
				continuesFromPrev: false,
				continuesToNext: false,
			},
			blocks: [
				{
					id: 'p51-b1',
					kind: 'paragraph',
					text: 'The runtime is O(N?).',
					bbox: { x: 0.1, y: 0.2, width: 0.7, height: 0.03 },
					confidence: {
						composite: 0.81,
						ocr: 1,
						dictionary: 1,
						garble: 1,
						flagged: true,
						reasons: ['math:verify'],
					},
				},
				{
					id: 'p51-b2',
					kind: 'paragraph',
					text: 'A separate unflagged paragraph.',
					bbox: { x: 0.1, y: 0.4, width: 0.7, height: 0.03 },
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
		writeBookManifest(bookDir, {
			title: 'Test Book',
			author: 'Test Author',
			language: 'en',
			sourcePdf: '/tmp/fake.pdf',
			totalPages: 51,
			scannedPages: [51],
			threshold: 0.82,
			ocrEngine: 'macos-vision',
			createdAt: now,
			updatedAt: now,
			stats: { cleanPages: 0, flaggedPages: 1, reviewedPages: 0, totalFlaggedBlocks: 1 },
		});
		const cachedImage = join(bookDir, '.cache', 'pages', 'page-0051.png');
		mkdirSync(dirname(cachedImage), { recursive: true });
		const image = Buffer.alloc(24);
		Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]).copy(image);
		image.write('IHDR', 12, 'ascii');
		image.writeUInt32BE(1000, 16);
		image.writeUInt32BE(2000, 20);
		writeFileSync(cachedImage, image);
		const requests: Array<{
			imagePath: string;
			rectangle: { left: number; top: number; width: number; height: number };
		}> = [];
		const workerStarts: string[] = [];
		let closed = false;
		let failNext = false;
		const alternativeOcr = new LocalTesseractAlternative(
			join(bookDir, '.cache', 'tesseract-test'),
			async language => {
				workerStarts.push(language);
				return {
					recognize: async (imagePath, options) => {
						requests.push({ imagePath, rectangle: options.rectangle });
						if (failNext) {
							failNext = false;
							throw new Error('Simulated local OCR failure');
						}
						return { data: { text: 'The runtime is O(N²).' } };
					},
					terminate: async () => {
						closed = true;
					},
				};
			},
		);
		const pagePath = join(bookDir, 'pages', 'page-0051.md');
		const original = readFileSync(pagePath, 'utf8');
		const reviewServer = await startReviewServer({
			bookDir,
			port: await findAvailablePort(),
			alternativeOcr,
		});
		try {
			const html = await (await fetch(reviewServer.url)).text();
			expect(html).toContain('Compare OCR');
			expect(requests).toHaveLength(0);
			expect(workerStarts).toHaveLength(0);
			const response = await fetch(`${reviewServer.url}/api/pages/51/ocr-alternative`, {
				method: 'POST',
				headers: { 'content-type': 'application/json' },
				body: JSON.stringify({ blockId: 'p51-b1' }),
			});
			expect(response.ok).toBe(true);
			expect(await response.json()).toMatchObject({
				blockId: 'p51-b1',
				engine: 'tesseract',
				text: 'The runtime is O(N²).',
			});
			expect(requests).toEqual([
				{ imagePath: cachedImage, rectangle: { left: 90, top: 376, width: 720, height: 109 } },
			]);
			expect(workerStarts).toEqual(['eng']);
			expect(readFileSync(pagePath, 'utf8')).toBe(original);
			expect(readPageFile(bookDir, 51).frontmatter.status).toBe('needs-review');
			const missing = await fetch(`${reviewServer.url}/api/pages/51/ocr-alternative`, {
				method: 'POST',
				headers: { 'content-type': 'application/json' },
				body: JSON.stringify({ blockId: 'missing' }),
			});
			expect(missing.status).toBe(404);
			const unflagged = await fetch(`${reviewServer.url}/api/pages/51/ocr-alternative`, {
				method: 'POST',
				headers: { 'content-type': 'application/json' },
				body: JSON.stringify({ blockId: 'p51-b2' }),
			});
			expect(unflagged.status).toBe(409);
			const invalid = await fetch(`${reviewServer.url}/api/pages/51/ocr-alternative`, {
				method: 'POST',
				headers: { 'content-type': 'application/json' },
				body: '{}',
			});
			expect(invalid.status).toBe(400);
			expect(requests).toHaveLength(1);

			const script = html.match(/<script>([\s\S]*?)<\/script>/)?.[1];
			if (!script) throw new Error('Review page script not found');
			type ElementStub = {
				value: string;
				textContent: string;
				innerHTML: string;
				className: string;
				hidden: boolean;
				disabled: boolean;
				children: ElementStub[];
				onclick: (() => Promise<void>) | null;
				appendChild: (element: ElementStub) => void;
				setAttribute: (name: string, value: string) => void;
			};
			const makeElement = (): ElementStub => {
				const children: ElementStub[] = [];
				return {
					value: '',
					textContent: '',
					innerHTML: '',
					className: '',
					hidden: false,
					disabled: false,
					children,
					onclick: null,
					appendChild: element => {
						children.push(element);
					},
					setAttribute: () => {},
				};
			};
			const elements = new Map<string, ElementStub>();
			const documentStub = {
				getElementById: (id: string) => {
					const element = elements.get(id) ?? makeElement();
					elements.set(id, element);
					return element;
				},
				createElement: () => makeElement(),
				createElementNS: () => makeElement(),
			};
			const context = createContext({
				document: documentStub,
				fetch: (url: string, options?: RequestInit) => fetch(`${reviewServer.url}${url}`, options),
				setTimeout: () => 0,
			});
			runInContext(script, context);
			await runInContext('selectPage(51)', context);
			const editor = documentStub.getElementById('md-editor');
			editor.value = 'Unsaved user edits';
			const row = documentStub.getElementById('flagged-container').children.at(-1);
			const actions = row?.children.find(element => element.className === 'flagged-actions');
			const compare = actions?.children.find(element => element.textContent === 'Compare OCR');
			const output = row?.children.find(element => element.className === 'alternative-result');
			expect(compare?.onclick).toBeTypeOf('function');
			await compare?.onclick?.();
			expect(output?.hidden).toBe(false);
			expect(output?.textContent).toContain('The runtime is O(N²).');
			expect(editor.value).toBe('Unsaved user edits');
			expect(requests).toHaveLength(2);
			expect(workerStarts).toEqual(['eng']);
			expect(readFileSync(pagePath, 'utf8')).toBe(original);
			failNext = true;
			const errorLog = vi.spyOn(console, 'error').mockImplementation(() => {});
			try {
				await compare?.onclick?.();
				expect(output?.textContent).toContain('local Tesseract');
				expect(editor.value).toBe('Unsaved user edits');
				await compare?.onclick?.();
				expect(output?.textContent).toContain('The runtime is O(N²).');
			} finally {
				errorLog.mockRestore();
			}
			const manifest = readBookManifest(bookDir);
			writeBookManifest(bookDir, { ...manifest, language: 'zh-TW' });
			const traditional = await fetch(`${reviewServer.url}/api/pages/51/ocr-alternative`, {
				method: 'POST',
				headers: { 'content-type': 'application/json' },
				body: JSON.stringify({ blockId: 'p51-b1' }),
			});
			expect(traditional.status).toBe(200);
			expect(workerStarts).toEqual(['eng', 'chi_tra']);
			writeBookManifest(bookDir, { ...manifest, language: 'zz' });
			const unsupported = await fetch(`${reviewServer.url}/api/pages/51/ocr-alternative`, {
				method: 'POST',
				headers: { 'content-type': 'application/json' },
				body: JSON.stringify({ blockId: 'p51-b1' }),
			});
			expect(unsupported.status).toBe(422);
			expect(requests).toHaveLength(5);
		} finally {
			await reviewServer.close();
		}
		expect(closed).toBe(true);
	});

	it('does not discard unsaved Markdown edits when accepting or repairing a block', async () => {
		const reviewServer = await startReviewServer({
			bookDir: createTempDir(),
			port: await findAvailablePort(),
		});
		try {
			const html = await (await fetch(reviewServer.url)).text();
			const script = html.match(/<script>([\s\S]*?)<\/script>/)?.[1];
			if (!script) throw new Error('Review page script not found');
			type ElementStub = {
				value: string;
				textContent: string;
				innerHTML: string;
				className: string;
				onclick: (() => void) | null;
				appendChild: (element: ElementStub) => void;
				setAttribute: (name: string, value: string) => void;
			};
			const makeElement = (): ElementStub => ({
				value: '',
				textContent: '',
				innerHTML: '',
				className: '',
				onclick: null,
				appendChild: () => {},
				setAttribute: () => {},
			});
			const elements = new Map<string, ElementStub>();
			const documentStub = {
				getElementById: (id: string) => {
					const element = elements.get(id) ?? makeElement();
					elements.set(id, element);
					return element;
				},
				createElement: () => makeElement(),
				createElementNS: () => makeElement(),
			};
			let markdown = 'Original Markdown';
			const calls: string[] = [];
			const fakeFetch = async (url: string, options?: { method?: string; body?: string }) => {
				calls.push(`${options?.method ?? 'GET'} ${url}`);
				if (url === '/api/book')
					return {
						ok: true,
						json: async () => ({
							manifest: { title: 'Book' },
							pages: [{ page: 5, status: 'needs-review', confidence: 0.8 }],
						}),
					};
				if (url === '/api/pages/5' && options?.method === 'PUT') {
					const body: unknown = JSON.parse(options.body ?? '{}');
					if (
						typeof body !== 'object' ||
						body === null ||
						!('markdown' in body) ||
						typeof body.markdown !== 'string'
					)
						throw new Error('Invalid save');
					markdown = body.markdown;
					return { ok: true, json: async () => ({ ok: true }) };
				}
				if (url === '/api/pages/5')
					return {
						ok: true,
						json: async () => ({
							markdown,
							document: { frontmatter: { status: 'needs-review', confidence: 0.8 }, blocks: [] },
						}),
					};
				return {
					ok: true,
					json: async () => ({ repairedBlocks: 1, reviewRequiredBlocks: 1, rejectedBlocks: 0 }),
				};
			};
			const context = createContext({
				document: documentStub,
				fetch: fakeFetch,
				setTimeout: () => 0,
			});
			runInContext(script, context);
			await runInContext('selectPage(5)', context);
			const editor = documentStub.getElementById('md-editor');
			editor.value = 'Unsaved corrected Markdown';
			await runInContext('acceptBlock("p5-b1")', context);
			await runInContext('repairBlock("p5-b1")', context);
			await runInContext('selectPage(6)', context);
			expect(calls).not.toContain('POST /api/pages/5/accept');
			expect(calls).not.toContain('POST /api/pages/5/repair');
			expect(calls).not.toContain('GET /api/pages/6');
			expect(editor.value).toBe('Unsaved corrected Markdown');
			expect(documentStub.getElementById('toast').textContent).toContain('Save');
			await runInContext('savePage()', context);
			await runInContext('acceptBlock("p5-b1")', context);
			expect(calls).toContain('POST /api/pages/5/accept');
			expect(markdown).toBe('Unsaved corrected Markdown');
		} finally {
			await reviewServer.close();
		}
	});

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
	it('keeps LLM suggestions for unverified code and math flagged for human review', async () => {
		const bookDir = createTempDir();
		const now = new Date().toISOString();
		const confidence = (composite: number, reasons: string[]) => ({
			composite,
			ocr: 1,
			dictionary: 1,
			garble: 1,
			flagged: true,
			reasons,
		});
		writePageFile(bookDir, {
			frontmatter: {
				page: 5,
				status: 'needs-review',
				confidence: 0.6,
				flaggedBlocks: 4,
				continuesFromPrev: false,
				continuesToNext: false,
			},
			blocks: [
				{
					id: 'p5-prose',
					kind: 'paragraph',
					text: 'A noisy sentnce.',
					bbox: { x: 0.1, y: 0.1, width: 0.7, height: 0.03 },
					confidence: confidence(0.6, ['ocr:0.60']),
				},
				{
					id: 'p5-code',
					kind: 'code',
					text: 'if (x ‹ y) { return x; }',
					bbox: { x: 0.1, y: 0.3, width: 0.7, height: 0.03 },
					confidence: confidence(0.81, ['ocr:0.70']),
				},
				{
					id: 'p5-math',
					kind: 'paragraph',
					text: 'The runtime is O(N?).',
					bbox: { x: 0.1, y: 0.5, width: 0.7, height: 0.03 },
					confidence: confidence(0.81, ['ocr:0.70']),
				},
				{
					id: 'p5-table',
					kind: 'table',
					text: '| A | B |\n| --- | --- |\n| 1 | 2 |',
					bbox: { x: 0.1, y: 0.7, width: 0.7, height: 0.03 },
					confidence: confidence(0.81, ['table:verify']),
				},
			],
		});
		writeBookManifest(bookDir, {
			title: 'Test Book',
			author: 'Test Author',
			language: 'en',
			sourcePdf: '/tmp/fake.pdf',
			totalPages: 5,
			scannedPages: [5],
			threshold: 0.82,
			ocrEngine: 'macos-vision',
			createdAt: now,
			updatedAt: now,
			stats: { cleanPages: 0, flaggedPages: 1, reviewedPages: 0, totalFlaggedBlocks: 4 },
		});
		const mockRender = async (options: RenderPageImageOptions) => {
			mkdirSync(dirname(options.outputPath), { recursive: true });
			writeFileSync(options.outputPath, 'mock image');
			return options.outputPath;
		};
		vi.spyOn(VisionOcrProvider.prototype, 'renderPageImage').mockImplementation(mockRender);
		vi.spyOn(TesseractOcrProvider.prototype, 'renderPageImage').mockImplementation(mockRender);
		vi.stubEnv('OPENAI_API_KEY', 'local-test-key');
		const response = (content: string) =>
			new Response(JSON.stringify({ choices: [{ message: { content } }] }), {
				status: 200,
				headers: { 'content-type': 'application/json' },
			});
		vi.stubGlobal(
			'fetch',
			vi
				.fn()
				.mockResolvedValueOnce(response('A corrected sentence.'))
				.mockResolvedValueOnce(response('if (x < y) { return x; }'))
				.mockResolvedValueOnce(response('The runtime is O(N²).'))
				.mockResolvedValueOnce(response('This response is not a table.')),
		);
		try {
			const result = await repairFlaggedBlocks({
				bookDir,
				apiKey: 'local-test-key',
				model: 'gpt-test',
				maxBlocks: 4,
			});
			expect(result.repairedBlocks).toBe(3);
			expect(result.reviewRequiredBlocks).toBe(2);
			expect(result.rejectedBlocks).toBe(1);
			const page = readPageFile(bookDir, 5);
			expect(page.frontmatter.status).toBe('needs-review');
			expect(page.frontmatter.flaggedBlocks).toBe(3);
			expect(page.frontmatter.confidence).toBe(0.81);
			expect(page.blocks[0]?.text).toBe('A corrected sentence.');
			expect(page.blocks[0]?.confidence.flagged).toBe(false);
			expect(page.blocks[1]?.text).toBe('if (x < y) { return x; }');
			expect(page.blocks[1]?.confidence.flagged).toBe(true);
			expect(page.blocks[1]?.confidence.reasons).toContain('code:verify');
			expect(page.blocks[2]?.text).toBe('The runtime is O(N²).');
			expect(page.blocks[2]?.confidence.flagged).toBe(true);
			expect(page.blocks[2]?.confidence.reasons).toContain('math:verify');
			expect(page.blocks[3]?.kind).toBe('table');
			expect(page.blocks[3]?.text).toBe('| A | B |\n| --- | --- |\n| 1 | 2 |');
			expect(page.blocks[3]?.confidence.flagged).toBe(true);
			const epubPath = join(bookDir, 'unreviewed.epub');
			await expect(buildEpubFromWorkspace({ bookDir, outputPath: epubPath })).rejects.toThrow(
				'still have status "needs-review"',
			);
			expect(existsSync(epubPath)).toBe(false);
		} finally {
			vi.unstubAllGlobals();
			vi.unstubAllEnvs();
			vi.restoreAllMocks();
		}
	});

	it('accepts one block at a time and completes the page after the last block', async () => {
		const bookDir = createTempDir();
		const now = new Date().toISOString();
		writePageFile(bookDir, {
			frontmatter: {
				page: 12,
				runningHeaders: ['VI | Big O'],
				runningFooters: ['Cracking the Coding Interview'],
				printedPage: 'x',
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
			expect(afterLastAcceptance.document.frontmatter).toEqual(
				expect.objectContaining({
					runningHeaders: ['VI | Big O'],
					runningFooters: ['Cracking the Coding Interview'],
					printedPage: 'x',
				}),
			);
			expect(afterLastAcceptance.document.blocks.every(block => !block.confidence.flagged)).toBe(
				true,
			);
		} finally {
			await reviewServer.close();
		}
	});
});
