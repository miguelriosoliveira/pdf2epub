import { execFile } from 'node:child_process';
import { existsSync, mkdirSync, statSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';
import {
	formatBBox,
	type OcrBatchOutput,
	OcrBatchOutputSchema,
	type OcrPageResult,
} from '../types/domain.ts';
import type { OcrProvider, OcrRecognizeOptions, RenderPageImageOptions } from './provider.ts';

const execFileAsync = promisify(execFile);

const CURRENT_DIR = dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = resolve(CURRENT_DIR, '../..');
const SWIFT_SOURCE_PATH = resolve(PROJECT_ROOT, 'swift/vision-ocr.swift');
const SWIFT_BIN_PATH = resolve(PROJECT_ROOT, '.cache/bin/vision-ocr');

export function normalizeVisionLanguage(lang = 'en'): string {
	const cleaned = lang.trim();
	if (cleaned === 'en') {
		return 'en-US';
	}
	if (cleaned === 'pt') {
		return 'pt-BR';
	}
	if (cleaned === 'es') {
		return 'es-ES';
	}
	return cleaned;
}

export class VisionOcrProvider implements OcrProvider {
	readonly engine = 'macos-vision' as const;

	private async ensureBinary(): Promise<string> {
		if (!existsSync(SWIFT_SOURCE_PATH)) {
			throw new Error(`Swift OCR source not found at ${SWIFT_SOURCE_PATH}`);
		}

		let needsBuild = !existsSync(SWIFT_BIN_PATH);
		if (!needsBuild) {
			const srcStat = statSync(SWIFT_SOURCE_PATH);
			const binStat = statSync(SWIFT_BIN_PATH);
			if (srcStat.mtimeMs > binStat.mtimeMs || binStat.size === 0) {
				needsBuild = true;
			}
		}

		if (needsBuild) {
			mkdirSync(dirname(SWIFT_BIN_PATH), { recursive: true });
			await execFileAsync('swiftc', ['-O', SWIFT_SOURCE_PATH, '-o', SWIFT_BIN_PATH], {
				maxBuffer: 32 * 1024 * 1024,
			});
		}

		return SWIFT_BIN_PATH;
	}

	async getPdfPageCount(pdfPath: string): Promise<number> {
		const bin = await this.ensureBinary();
		const resolvedPdf = resolve(pdfPath);
		const { stdout } = await execFileAsync(bin, ['info', '--pdf', resolvedPdf], {
			maxBuffer: 16 * 1024 * 1024,
		});
		const parsed = OcrBatchOutputSchema.parse(JSON.parse(stdout));
		return parsed.totalPages;
	}

	async recognizePages(options: OcrRecognizeOptions): Promise<OcrBatchOutput> {
		const bin = await this.ensureBinary();
		const resolvedPdf = resolve(options.pdfPath);
		const lang = normalizeVisionLanguage(options.language);
		const dpi = String(options.dpi ?? 220);

		const totalPages = await this.getPdfPageCount(resolvedPdf);
		const targetPages =
			options.pages === 'all'
				? Array.from({ length: totalPages }, (_, idx) => idx + 1)
				: options.pages.filter(p => p >= 1 && p <= totalPages);

		const batchSize = 10;
		const collectedPages: OcrPageResult[] = [];

		for (let i = 0; i < targetPages.length; i += batchSize) {
			const slice = targetPages.slice(i, i + batchSize);
			const pagesSpec = slice.join(',');
			const { stdout } = await execFileAsync(
				bin,
				['ocr', '--pdf', resolvedPdf, '--pages', pagesSpec, '--lang', lang, '--dpi', dpi],
				{
					maxBuffer: 64 * 1024 * 1024,
				},
			);
			const batchResult = OcrBatchOutputSchema.parse(JSON.parse(stdout));
			collectedPages.push(...batchResult.pages);
			options.onPageProgress?.(collectedPages.length, targetPages.length);
		}

		return {
			totalPages,
			pages: collectedPages,
		};
	}

	async renderPageImage(options: RenderPageImageOptions): Promise<string> {
		const bin = await this.ensureBinary();
		const resolvedPdf = resolve(options.pdfPath);
		const resolvedOut = resolve(options.outputPath);
		const args = [
			'render-page',
			'--pdf',
			resolvedPdf,
			'--page',
			String(options.page),
			'--dpi',
			String(options.dpi ?? 200),
			'--out',
			resolvedOut,
		];
		if (options.crop) {
			args.push('--crop', formatBBox(options.crop));
		}
		await execFileAsync(bin, args, {
			maxBuffer: 16 * 1024 * 1024,
		});
		return resolvedOut;
	}
}
