import type { OcrPageResult, PageDocument, PageStatus } from '../types/domain.ts';
import { buildBookVocabulary, scorePageBlocks } from './confidence-scorer.ts';
import { detectRepeatingMarginKeys, reconstructPageLayout } from './layout-reconstructor.ts';

export function processOcrPagesToDocuments(
	pages: readonly OcrPageResult[],
	options: {
		threshold?: number;
	} = {},
): PageDocument[] {
	const threshold = options.threshold ?? 0.82;
	const repeatingMarginKeys = detectRepeatingMarginKeys(pages);
	const bookVocab = buildBookVocabulary(pages);

	return pages.map(ocrPage => {
		const layout = reconstructPageLayout(ocrPage, repeatingMarginKeys);
		const blocks = scorePageBlocks(layout.blocks, { threshold, bookVocab });

		const flaggedBlocks = blocks.filter(b => b.confidence.flagged).length;
		const minBlockConfidence =
			blocks.length > 0 ? Math.min(...blocks.map(b => b.confidence.composite)) : 1;
		const status: PageStatus = flaggedBlocks > 0 ? 'needs-review' : 'clean';

		return {
			frontmatter: {
				page: ocrPage.page,
				status,
				confidence: Math.round(minBlockConfidence * 10000) / 10000,
				flaggedBlocks,
				continuesFromPrev: layout.continuesFromPrev,
				continuesToNext: layout.continuesToNext,
			},
			blocks,
		};
	});
}

export * from './confidence-scorer.ts';
export * from './layout-reconstructor.ts';
