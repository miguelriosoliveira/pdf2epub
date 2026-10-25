import { execFile } from 'node:child_process';
import { mkdirSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { promisify } from 'node:util';
import { createWorker } from 'tesseract.js';
import type { OcrBatchOutput, OcrLine, OcrPageResult } from '../types/domain.ts';
import type { OcrProvider, OcrRecognizeOptions, RenderPageImageOptions } from './provider.ts';
import { VisionOcrProvider } from './vision-provider.ts';

const execFileAsync = promisify(execFile);

function clamp01(val: number): number {
	return Math.min(1, Math.max(0, val));
}

function normalizeTesseractLang(lang = 'en'): string {
	const cleaned = lang.toLowerCase().split('-')[0] ?? 'en';
	const map: Record<string, string> = {
		en: 'eng',
		pt: 'por',
		es: 'spa',
		fr: 'fra',
		de: 'deu',
		it: 'ita',
	};
	return map[cleaned] ?? 'eng';
}

export class TesseractOcrProvider implements OcrProvider {
	readonly engine = 'tesseract' as const;
	private readonly macHelper = process.platform === 'darwin' ? new VisionOcrProvider() : null;

	async getPdfPageCount(pdfPath: string): Promise<number> {
		if (this.macHelper) {
			return this.macHelper.getPdfPageCount(pdfPath);
		}
		const { stdout } = await execFileAsync('pdfinfo', [resolve(pdfPath)]);
		const match = stdout.match(/Pages:\s+(\d+)/);
		if (!match?.[1]) {
			throw new Error(`Could not determine page count from pdfinfo for ${pdfPath}`);
		}
		return Number.parseInt(match[1], 10);
	}

	async renderPageImage(options: RenderPageImageOptions): Promise<string> {
		if (this.macHelper) {
			return this.macHelper.renderPageImage(options);
		}
		const resolvedPdf = resolve(options.pdfPath);
		const resolvedOut = resolve(options.outputPath);
		mkdirSync(dirname(resolvedOut), { recursive: true });
		const prefix = resolvedOut.replace(/\.png$/i, '');
		await execFileAsync('pdftoppm', [
			'-png',
			'-r',
			String(options.dpi ?? 200),
			'-f',
			String(options.page),
			'-l',
			String(options.page),
			'-singlefile',
			resolvedPdf,
			prefix,
		]);
		return `${prefix}.png`;
	}

	async recognizePages(options: OcrRecognizeOptions): Promise<OcrBatchOutput> {
		const totalPages = await this.getPdfPageCount(options.pdfPath);
		const targetPages =
			options.pages === 'all'
				? Array.from({ length: totalPages }, (_, idx) => idx + 1)
				: options.pages.filter(p => p >= 1 && p <= totalPages);

		const tempDir = mkdtempSync(join(tmpdir(), 'pdf2epub-tess-'));
		const worker = await createWorker(normalizeTesseractLang(options.language));

		try {
			const pages: OcrPageResult[] = [];
			for (let idx = 0; idx < targetPages.length; idx++) {
				const pageNumber = targetPages[idx];
				if (pageNumber === undefined) {
					continue;
				}
				const imagePath = join(tempDir, `page-${pageNumber}.png`);
				await this.renderPageImage({
					pdfPath: options.pdfPath,
					page: pageNumber,
					outputPath: imagePath,
					dpi: options.dpi ?? 200,
				});

				const response = await worker.recognize(imagePath, {}, { blocks: true });
				const lines: OcrLine[] = [];
				let widthPx = 1600;
				let heightPx = 2200;

				const blocks = response.data.blocks ?? [];
				for (const block of blocks) {
					for (const para of block.paragraphs) {
						for (const line of para.lines) {
							widthPx = Math.max(widthPx, line.bbox.x1);
							heightPx = Math.max(heightPx, line.bbox.y1);
						}
					}
				}

				for (const block of blocks) {
					for (const para of block.paragraphs) {
						for (const line of para.lines) {
							const text = line.text.trim();
							if (!text) {
								continue;
							}
							lines.push({
								text,
								confidence: clamp01(line.confidence / 100),
								bbox: {
									x: clamp01(line.bbox.x0 / widthPx),
									y: clamp01(line.bbox.y0 / heightPx),
									width: clamp01((line.bbox.x1 - line.bbox.x0) / widthPx),
									height: clamp01((line.bbox.y1 - line.bbox.y0) / heightPx),
								},
							});
						}
					}
				}

				pages.push({
					page: pageNumber,
					widthPx,
					heightPx,
					lines,
				});
				options.onPageProgress?.(idx + 1, targetPages.length);
			}

			return {
				totalPages,
				pages,
			};
		} finally {
			await worker.terminate();
			rmSync(tempDir, { recursive: true, force: true });
		}
	}
}
