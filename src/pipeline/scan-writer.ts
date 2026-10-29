import { join } from 'node:path';
import { writePageFile } from '../markdown/page-serializer.ts';
import {
	BOOK_COVER_ASSET_PATH,
	BOOK_COVER_MARKDOWN,
	figureAssetPathFromMarkdown,
} from '../nlp/figure-detector.ts';
import { processOcrPagesToDocuments } from '../nlp/index.ts';
import type { OcrProvider } from '../ocr/provider.ts';
import type { LayoutProfile, OcrPageResult, PageDocument } from '../types/domain.ts';

export async function writeOcrPagesToWorkspace(options: {
	bookDir: string;
	pdfPath: string;
	pages: readonly OcrPageResult[];
	provider: OcrProvider;
	dpi?: number;
	threshold?: number;
	profile?: LayoutProfile;
}): Promise<PageDocument[]> {
	const documents = processOcrPagesToDocuments(options.pages, {
		...(options.threshold === undefined ? {} : { threshold: options.threshold }),
		...(options.profile === undefined ? {} : { profile: options.profile }),
	});

	for (const document of documents) {
		if (document.frontmatter.page === 1) {
			document.blocks = [
				{
					id: 'p1-cover',
					kind: 'paragraph',
					text: BOOK_COVER_MARKDOWN,
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
			];
			document.frontmatter.status = 'clean';
			document.frontmatter.confidence = 1;
			document.frontmatter.flaggedBlocks = 0;
			document.frontmatter.continuesFromPrev = false;
			document.frontmatter.continuesToNext = false;
			await options.provider.renderPageImage({
				pdfPath: options.pdfPath,
				page: 1,
				outputPath: join(options.bookDir, 'pages', BOOK_COVER_ASSET_PATH),
				...(options.dpi === undefined ? {} : { dpi: options.dpi }),
			});
		}

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
				trimWhitespace: true,
			});
		}

		writePageFile(options.bookDir, document);
	}

	return documents;
}
