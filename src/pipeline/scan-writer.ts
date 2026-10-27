import { join } from 'node:path';
import { writePageFile } from '../markdown/page-serializer.ts';
import { figureAssetPathFromMarkdown } from '../nlp/figure-detector.ts';
import { processOcrPagesToDocuments } from '../nlp/index.ts';
import type { OcrProvider } from '../ocr/provider.ts';
import type { OcrPageResult, PageDocument } from '../types/domain.ts';

export async function writeOcrPagesToWorkspace(options: {
	bookDir: string;
	pdfPath: string;
	pages: readonly OcrPageResult[];
	provider: OcrProvider;
	dpi?: number;
	threshold?: number;
}): Promise<PageDocument[]> {
	const documents = processOcrPagesToDocuments(
		options.pages,
		options.threshold === undefined ? {} : { threshold: options.threshold },
	);

	for (const document of documents) {
		for (const block of document.blocks) {
			const figurePath = figureAssetPathFromMarkdown(block.text);
			if (!figurePath) {
				continue;
			}

			await options.provider.renderPageImage({
				pdfPath: options.pdfPath,
				page: document.frontmatter.page,
				outputPath: join(options.bookDir, 'pages', figurePath),
				...(options.dpi === undefined ? {} : { dpi: options.dpi }),
				crop: block.bbox,
			});
		}

		writePageFile(options.bookDir, document);
	}

	return documents;
}
