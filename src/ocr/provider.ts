import type { NormalizedBBox, OcrBatchOutput, OcrEngine } from '../types/domain.ts';

export interface OcrRecognizeOptions {
	pdfPath: string;
	pages: number[] | 'all';
	language?: string;
	dpi?: number;
	/** Pages read by one Vision process. `all` reads the whole selection in one process. */
	batchSize?: number | 'all';
	onPageProgress?: (completedPages: number, totalTargetPages: number) => void;
}

export interface RenderPageImageOptions {
	pdfPath: string;
	page: number;
	outputPath: string;
	dpi?: number;
	crop?: NormalizedBBox;
	/** Tighten a coarse crop to visible ink and add an equal pixel margin. */
	trimWhitespace?: boolean;
}

export interface OcrProvider {
	readonly engine: OcrEngine;
	getPdfPageCount(pdfPath: string): Promise<number>;
	recognizePages(options: OcrRecognizeOptions): Promise<OcrBatchOutput>;
	renderPageImage(options: RenderPageImageOptions): Promise<string>;
}
