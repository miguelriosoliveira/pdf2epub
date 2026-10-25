import type { OcrEngine } from '../types/domain.ts';
import type { OcrProvider } from './provider.ts';
import { TesseractOcrProvider } from './tesseract-provider.ts';
import { VisionOcrProvider } from './vision-provider.ts';

export function createOcrProvider(preferred?: OcrEngine): OcrProvider {
	if (preferred === 'tesseract') {
		return new TesseractOcrProvider();
	}
	if (process.platform === 'darwin') {
		return new VisionOcrProvider();
	}
	return new TesseractOcrProvider();
}

export * from './provider.ts';
export * from './tesseract-provider.ts';
export * from './vision-provider.ts';
