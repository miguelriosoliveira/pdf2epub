import type { LayoutProfile, OcrPageResult, PageDocument, PageStatus } from '../types/domain.ts';
import { unclosedBraceDepth } from './code-structure.ts';
import { buildBookVocabulary, scorePageBlocks } from './confidence-scorer.ts';
import { detectGraphFigureRegions, figureAssetPath } from './figure-detector.ts';
import {
	detectRepeatingMarginKeys,
	isRunningHeaderOrFooter,
	reconstructPageLayout,
} from './layout-reconstructor.ts';
import { normalizeAsymptoticNotation } from './math-normalizer.ts';
import { extractTechnicalRegions } from './technical-layout.ts';

export function processOcrPagesToDocuments(
	pages: readonly OcrPageResult[],
	options: {
		threshold?: number;
		profile?: LayoutProfile;
	} = {},
): PageDocument[] {
	const threshold = options.threshold ?? 0.82;
	const profile = options.profile ?? 'auto';
	const repeatingMarginKeys = detectRepeatingMarginKeys(pages);
	const bookVocab = buildBookVocabulary(pages);

	const documents: PageDocument[] = pages.map(ocrPage => {
		const contentPage = {
			...ocrPage,
			lines: ocrPage.lines.filter(line => !isRunningHeaderOrFooter(line, repeatingMarginKeys)),
		};
		const technical =
			profile === 'narrative' ? null : extractTechnicalRegions(contentPage, profile);
		const figureRegions = [
			...(profile === 'narrative' ? [] : detectGraphFigureRegions(contentPage.lines)),
			...(technical?.figureRegions ?? []),
		];
		const figureLabelLines = new Set(figureRegions.flatMap(region => region.labelLines));
		const consumedLines = new Set([...figureLabelLines, ...(technical?.consumedLines ?? [])]);
		const layoutPage = {
			...ocrPage,
			lines: ocrPage.lines.filter(line => !consumedLines.has(line)),
		};
		const layout = reconstructPageLayout(layoutPage, repeatingMarginKeys);
		const normalizedLayoutBlocks = layout.blocks.map(block => ({
			...block,
			text: normalizeAsymptoticNotation(block.text),
		}));
		const scoringOptions = {
			threshold,
			bookVocab,
			verifyTechnicalContent: profile !== 'narrative',
		};
		const textBlocks = scorePageBlocks(normalizedLayoutBlocks, scoringOptions);
		// Technical blocks contain code and structured tables: prose substitutions can change
		// identifiers, operators, and markup, so preserve the recognized text verbatim.
		const technicalBlocks = scorePageBlocks(technical?.blocks ?? [], scoringOptions);
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
		const blocks = [...textBlocks, ...technicalBlocks, ...figureBlocks].sort(
			(a, b) => a.bbox.y - b.bbox.y,
		);

		const flaggedBlocks = blocks.filter(b => b.confidence.flagged).length;
		const minBlockConfidence =
			blocks.length > 0 ? Math.min(...blocks.map(b => b.confidence.composite)) : 1;
		const status: PageStatus = flaggedBlocks > 0 ? 'needs-review' : 'clean';

		return {
			frontmatter: {
				page: ocrPage.page,
				...(layout.runningHeaders.length ? { runningHeaders: layout.runningHeaders } : {}),
				...(layout.runningFooters.length ? { runningFooters: layout.runningFooters } : {}),
				...(layout.printedPage === undefined ? {} : { printedPage: layout.printedPage }),
				status,
				confidence: Math.round(minBlockConfidence * 10000) / 10000,
				flaggedBlocks,
				continuesFromPrev: layout.continuesFromPrev,
				continuesToNext: layout.continuesToNext,
			},
			blocks,
		};
	});
	const documentsByPage = new Map(documents.map(document => [document.frontmatter.page, document]));
	for (const current of documents) {
		const previous = documentsByPage.get(current.frontmatter.page - 1);
		const previousCode = previous?.blocks.at(-1);
		const currentCode = current.blocks[0];
		if (
			previous &&
			previousCode?.kind === 'code' &&
			currentCode?.kind === 'code' &&
			previousCode.bbox.y + previousCode.bbox.height >= 0.85 &&
			currentCode.bbox.y <= 0.16 &&
			unclosedBraceDepth(previousCode.text) > 0
		) {
			previous.frontmatter.codeContinuesToNext = true;
			current.frontmatter.codeContinuesFromPrev = true;
		}
	}
	return documents;
}

export * from './confidence-scorer.ts';
export * from './figure-detector.ts';
export * from './layout-reconstructor.ts';
export * from './math-normalizer.ts';
export * from './technical-layout.ts';
