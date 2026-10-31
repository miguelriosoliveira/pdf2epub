import { mkdirSync, readFileSync } from 'node:fs';
import { createWorker } from 'tesseract.js';
import type { NormalizedBBox } from '../types/domain.ts';
import { resolveTesseractLang } from './tesseract-provider.ts';

export interface AlternativeOcr {
	recognize(options: {
		imagePath: string;
		bbox: NormalizedBBox;
		language: string;
	}): Promise<string>;
	close(): Promise<void>;
}

interface AlternativeWorker {
	recognize(
		imagePath: string,
		options: {
			rectangle: { left: number; top: number; width: number; height: number };
		},
	): Promise<{ data: { text: string } }>;
	terminate(): Promise<unknown>;
}

type AlternativeWorkerFactory = (language: string, cacheDir: string) => Promise<AlternativeWorker>;

function pngDimensions(path: string): { width: number; height: number } {
	const image = readFileSync(path);
	const signature = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
	if (
		image.length < 24 ||
		!image.subarray(0, 8).equals(signature) ||
		image.toString('ascii', 12, 16) !== 'IHDR'
	) {
		throw new Error('Alternative OCR requires a valid PNG page image');
	}
	const width = image.readUInt32BE(16);
	const height = image.readUInt32BE(20);
	if (!width || !height) throw new Error('PNG page image has invalid dimensions');
	return { width, height };
}

export class LocalTesseractAlternative implements AlternativeOcr {
	private readonly cacheDir: string;
	private readonly workerFactory: AlternativeWorkerFactory;
	private readonly workers = new Map<string, Promise<AlternativeWorker>>();
	private queue: Promise<void> = Promise.resolve();
	private closed = false;

	constructor(
		cacheDir: string,
		workerFactory: AlternativeWorkerFactory = (language, path) =>
			createWorker(language, 1, { cachePath: path }),
	) {
		this.cacheDir = cacheDir;
		this.workerFactory = workerFactory;
	}

	private getWorker(language: string): Promise<AlternativeWorker> {
		const lang = resolveTesseractLang(language);
		if (!lang) throw new Error(`Unsupported Tesseract language code: ${language}`);
		const existing = this.workers.get(lang);
		if (existing) return existing;
		mkdirSync(this.cacheDir, { recursive: true });
		const worker = this.workerFactory(lang, this.cacheDir).catch(error => {
			this.workers.delete(lang);
			throw error;
		});
		this.workers.set(lang, worker);
		return worker;
	}

	async recognize(options: {
		imagePath: string;
		bbox: NormalizedBBox;
		language: string;
	}): Promise<string> {
		const work = this.queue.then(async () => {
			if (this.closed) throw new Error('Alternative OCR is closed');
			const { width, height } = pngDimensions(options.imagePath);
			const left = Math.max(0, Math.floor((options.bbox.x - 0.01) * width));
			const top = Math.max(0, Math.floor((options.bbox.y - 0.012) * height));
			const right = Math.min(
				width,
				Math.ceil((options.bbox.x + options.bbox.width + 0.01) * width),
			);
			const bottom = Math.min(
				height,
				Math.ceil((options.bbox.y + options.bbox.height + 0.012) * height),
			);
			if (right <= left || bottom <= top) throw new Error('Invalid alternative OCR region');
			const worker = await this.getWorker(options.language);
			const result = await worker.recognize(options.imagePath, {
				rectangle: { left, top, width: right - left, height: bottom - top },
			});
			return result.data.text.trim();
		});
		this.queue = work.then(
			() => undefined,
			() => undefined,
		);
		return work;
	}

	async close(): Promise<void> {
		this.closed = true;
		await this.queue;
		const workers = await Promise.all(this.workers.values());
		await Promise.all(workers.map(worker => worker.terminate()));
		this.workers.clear();
	}
}
