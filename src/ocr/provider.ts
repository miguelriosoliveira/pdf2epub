import type { NormalizedBBox, OcrBatchOutput, OcrEngine } from '../types/domain.ts';

export interface OcrRecognizeOptions {
	pdfPath: string;
	pages: number[] | 'all';
	language?: string;
	dpi?: number;
	onPageProgress?: (completedPages: number, totalTargetPages: number) => void;
}

export interface RenderPageImageOptions {
	pdfPath: string;
	page: number;
	outputPath: string;
	dpi?: number;
	crop?: NormalizedBBox;
}

export interface OcrProvider {
	readonly engine: OcrEngine;
	getPdfPageCount(pdfPath: string): Promise<number>;
	recognizePages(options: OcrRecognizeOptions): Promise<OcrBatchOutput>;
	renderPageImage(options: RenderPageImageOptions): Promise<string>;
}
