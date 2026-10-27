import type { OcrPageResult, PageDocument, PageStatus } from '../types/domain.ts';
import { buildBookVocabulary, scorePageBlocks } from './confidence-scorer.ts';
import { detectGraphFigureRegions, figureAssetPath } from './figure-detector.ts';
import { detectRepeatingMarginKeys, reconstructPageLayout } from './layout-reconstructor.ts';
import { normalizeAsymptoticNotation } from './math-normalizer.ts';

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
		const figureRegions = detectGraphFigureRegions(ocrPage.lines);
		const figureLabelLines = new Set(figureRegions.flatMap(region => region.labelLines));
		const layoutPage = {
			...ocrPage,
			lines: ocrPage.lines.filter(line => !figureLabelLines.has(line)),
		};
		const layout = reconstructPageLayout(layoutPage, repeatingMarginKeys);
		const normalizedLayoutBlocks = layout.blocks.map(block => ({
			...block,
			text: normalizeAsymptoticNotation(block.text),
		}));
		const textBlocks = scorePageBlocks(normalizedLayoutBlocks, { threshold, bookVocab });
		const figureBlocks = figureRegions.map((region, index) => ({
			id: `p${ocrPage.page}-figure-${index + 1}`,
			kind: 'paragraph' as const,
			text: `![Figure from page ${ocrPage.page}](${figureAssetPath(ocrPage.page, index)})`,
			bbox: region.bbox,
			confidence: {
				composite: 1,
				ocr: 1,
				dictionary: 1,
				garble: 1,
				flagged: false,
				reasons: [],
			},
		}));
		const blocks = [...textBlocks, ...figureBlocks].sort((a, b) => a.bbox.y - b.bbox.y);

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
export * from './figure-detector.ts';
export * from './layout-reconstructor.ts';
export * from './math-normalizer.ts';
