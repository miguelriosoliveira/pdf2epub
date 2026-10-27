import type { BlockKind, NormalizedBBox, OcrLine, OcrPageResult } from '../types/domain.ts';

export interface RawReconstructedBlock {
	id: string;
	kind: BlockKind;
	text: string;
	bbox: NormalizedBBox;
	lines: OcrLine[];
}

export interface ReconstructedPageLayout {
	page: number;
	blocks: RawReconstructedBlock[];
	continuesFromPrev: boolean;
	continuesToNext: boolean;
	strippedHeadersFooters: string[];
}

const COMPOUND_PREFIXES = new Set([
	'self',
	'well',
	'non',
	'cross',
	'multi',
	'pre',
	'post',
	're',
	'co',
	'anti',
	'sub',
	'inter',
	'over',
	'under',
	'semi',
	'quasi',
	'all',
	'high',
	'low',
	'full',
	'half',
	'part',
	'real',
	'long',
	'short',
	'single',
	'double',
	'big',
]);

const EXPLICIT_HEADING_REGEX =
	/^(?:(?:chapter|part|section|appendix|prologue|epilogue|preface|introduction|foreword)\b|(?:[IVXLCDM]{1,6}|\d{1,2})\s*[.|•—-]\s+[A-Z])/i;

const LIST_ITEM_START_REGEX = /^(?:[•◦▪▸*-]|\d{1,2}[.)])\s+/;

const STANDALONE_PAGE_NUMBER_REGEX =
	/^(?:page\s+)?(?:\d{1,4}|[ivxlcdm]{1,8})(?:\s*(?:of|\/)\s*\d{1,4})?$/i;

const RUNNING_HEADER_PAGE_NUM_REGEX = /^(?:\d{1,4}\s*[|•—-]\s*.+|.+\s*[|•—-]\s*\d{1,4})$/;

export const TOC_ENTRY_LINE_REGEX = /^(.+?)\s+\.{3,}\s+(\d{1,4}|[ivxlcdm]{1,8})$/i;

const PURE_DOT_LEADER_REGEX = /^[.•·…\s_-]+$/;
const NOISY_DOT_LEADER_REGEX = /^[.•·…\s]{3,}[a-z0-9.•·…\s]{0,4}$/i;
const RIGHT_PAGE_TOKEN_REGEX = /^[:\-•·….\s]*(\d{1,4}|[ivxlcdm]{1,8})[.•·…\s]*$/i;
const TRAILING_DOT_LEADER_IN_TEXT_REGEX = /(?:\s*[.•·…]{2,}[\s.•·…]*)+$/;
const INLINE_TOC_WITH_DOTS_REGEX =
	/^(.+?)\s*[.•·…]{2,}[\s:.\-•·…]*(\d{1,4}|[ivxlcdm]{1,8})[.•·…\s]*$/i;

function normalizeHeaderFooterKey(text: string): string {
	return text
		.toLowerCase()
		.replace(/\d+/g, '#')
		.replace(/[^\p{L}#]+/gu, ' ')
		.trim();
}

function isTopOrBottomMargin(line: OcrLine): boolean {
	const top = line.bbox.y;
	const bottom = line.bbox.y + line.bbox.height;
	return top < 0.058 || bottom > 0.915;
}

export function detectRepeatingMarginKeys(pages: readonly OcrPageResult[]): Set<string> {
	const keyCounts = new Map<string, number>();
	for (const page of pages) {
		const seenOnPage = new Set<string>();
		for (const line of page.lines) {
			if (!isTopOrBottomMargin(line)) {
				continue;
			}
			const key = normalizeHeaderFooterKey(line.text);
			if (key.length >= 3 && !seenOnPage.has(key)) {
				seenOnPage.add(key);
				keyCounts.set(key, (keyCounts.get(key) ?? 0) + 1);
			}
		}
	}

	const minOccurrences = pages.length >= 4 ? Math.max(2, Math.floor(pages.length * 0.35)) : 3;
	const repeating = new Set<string>();
	for (const [key, count] of keyCounts) {
		if (count >= minOccurrences) {
			repeating.add(key);
		}
	}
	return repeating;
}

export function isRunningHeaderOrFooter(
	line: OcrLine,
	repeatingMarginKeys: ReadonlySet<string>,
): boolean {
	if (!isTopOrBottomMargin(line)) {
		return false;
	}
	const trimmed = line.text.trim();
	if (STANDALONE_PAGE_NUMBER_REGEX.test(trimmed)) {
		return true;
	}
	if (RUNNING_HEADER_PAGE_NUM_REGEX.test(trimmed)) {
		return true;
	}
	const key = normalizeHeaderFooterKey(trimmed);
	return repeatingMarginKeys.has(key);
}

function computeMedian(values: readonly number[]): number {
	if (values.length === 0) {
		return 0.012;
	}
	const sorted = [...values].sort((a, b) => a - b);
	const mid = Math.floor(sorted.length / 2);
	return sorted[mid] ?? 0.012;
}

export function mergeBBoxes(lines: readonly OcrLine[]): NormalizedBBox {
	const first = lines[0];
	if (!first) {
		return { x: 0, y: 0, width: 0, height: 0 };
	}
	let minX = first.bbox.x;
	let minY = first.bbox.y;
	let maxX = first.bbox.x + first.bbox.width;
	let maxY = first.bbox.y + first.bbox.height;

	for (const line of lines) {
		minX = Math.min(minX, line.bbox.x);
		minY = Math.min(minY, line.bbox.y);
		maxX = Math.max(maxX, line.bbox.x + line.bbox.width);
		maxY = Math.max(maxY, line.bbox.y + line.bbox.height);
	}

	return {
		x: Math.min(1, Math.max(0, minX)),
		y: Math.min(1, Math.max(0, minY)),
		width: Math.min(1, Math.max(0, maxX - minX)),
		height: Math.min(1, Math.max(0, maxY - minY)),
	};
}

function isDotLeaderSegment(line: OcrLine, medianHeight: number): boolean {
	const trimmed = line.text.trim();
	if (!trimmed) {
		return true;
	}
	if (PURE_DOT_LEADER_REGEX.test(trimmed)) {
		return true;
	}
	if (line.bbox.height <= medianHeight * 0.6 && NOISY_DOT_LEADER_REGEX.test(trimmed)) {
		return true;
	}
	return false;
}

function normalizeRomanNumeralPrefix(text: string): string {
	return text.replace(/^([IVXLCDMixvxlcdml]+)(\.\s+)/, (full, prefix: string, suffix: string) => {
		const normalized = prefix.replace(/l/g, 'I').toUpperCase();
		return /^[IVXLCDM]+$/.test(normalized) ? `${normalized}${suffix}` : full;
	});
}

function cleanTocLabel(rawLabel: string): string {
	const strippedDots = rawLabel
		.replace(TRAILING_DOT_LEADER_IN_TEXT_REGEX, '')
		.replace(/[\s.•·…]+$/, '')
		.replace(/\s+/g, ' ')
		.trim();
	return normalizeRomanNumeralPrefix(strippedDots);
}

function areLinesOnSameHorizontalRow(
	rowLines: readonly OcrLine[],
	candidate: OcrLine,
	medianHeight: number,
): boolean {
	const textLines = rowLines.filter(l => l.bbox.height > medianHeight * 0.55);
	const referenceLines = textLines.length > 0 ? textLines : rowLines;

	const refTop = Math.min(...referenceLines.map(l => l.bbox.y));
	const refBottom = Math.max(...referenceLines.map(l => l.bbox.y + l.bbox.height));
	const refCenter = (refTop + refBottom) * 0.5;

	const candTop = candidate.bbox.y;
	const candBottom = candidate.bbox.y + candidate.bbox.height;
	const candCenter = (candTop + candBottom) * 0.5;

	const overlap = Math.min(refBottom, candBottom) - Math.max(refTop, candTop);
	const minHeight = Math.min(refBottom - refTop, candidate.bbox.height);
	if (minHeight > 0 && overlap / minHeight >= 0.35) {
		return true;
	}

	if (Math.abs(candCenter - refCenter) <= medianHeight * 0.55) {
		return true;
	}

	// Dot leaders sit along the bottom baseline of the row
	if (Math.abs(candBottom - refBottom) <= medianHeight * 0.45) {
		return true;
	}

	return false;
}

export function consolidateHorizontalRows(
	lines: readonly OcrLine[],
	medianHeight: number,
): OcrLine[] {
	if (lines.length === 0) {
		return [];
	}

	const sortedByBaseline = [...lines].sort((a, b) => {
		const aBottom = a.bbox.y + a.bbox.height;
		const bBottom = b.bbox.y + b.bbox.height;
		if (Math.abs(aBottom - bBottom) > medianHeight * 0.45) {
			return a.bbox.y - b.bbox.y;
		}
		return a.bbox.x - b.bbox.x;
	});

	const rows: OcrLine[][] = [];
	for (const line of sortedByBaseline) {
		const currentRow = rows[rows.length - 1];
		if (!currentRow || !areLinesOnSameHorizontalRow(currentRow, line, medianHeight)) {
			rows.push([line]);
		} else {
			currentRow.push(line);
		}
	}

	const consolidated: OcrLine[] = [];

	for (const row of rows) {
		row.sort((a, b) => a.bbox.x - b.bbox.x);

		let hasDotLeader = false;
		const meaningfulSegments: OcrLine[] = [];

		for (const seg of row) {
			if (isDotLeaderSegment(seg, medianHeight)) {
				hasDotLeader = true;
				continue;
			}
			if (TRAILING_DOT_LEADER_IN_TEXT_REGEX.test(seg.text)) {
				hasDotLeader = true;
			}
			meaningfulSegments.push(seg);
		}

		if (meaningfulSegments.length === 0) {
			continue;
		}

		if (meaningfulSegments.length === 1) {
			const single = meaningfulSegments[0];
			if (!single) continue;
			const inlineMatch = single.text.trim().match(INLINE_TOC_WITH_DOTS_REGEX);
			if (inlineMatch?.[1] && inlineMatch[2]) {
				const label = cleanTocLabel(inlineMatch[1]);
				const pageNum = inlineMatch[2].trim();
				if (label && pageNum) {
					consolidated.push({
						text: `${label} ... ${pageNum}`,
						confidence: single.confidence,
						bbox: mergeBBoxes(row),
					});
					continue;
				}
			}

			const cleanedSingleText = hasDotLeader
				? cleanTocLabel(single.text)
				: normalizeRomanNumeralPrefix(single.text.trim());
			if (cleanedSingleText) {
				consolidated.push({
					text: cleanedSingleText,
					confidence: single.confidence,
					bbox: single.bbox,
				});
			}
			continue;
		}

		const lastSeg = meaningfulSegments[meaningfulSegments.length - 1];
		const prevSeg = meaningfulSegments[meaningfulSegments.length - 2];
		const rightPageMatch = lastSeg?.text.trim().match(RIGHT_PAGE_TOKEN_REGEX);

		const horizontalGap =
			lastSeg && prevSeg ? lastSeg.bbox.x - (prevSeg.bbox.x + prevSeg.bbox.width) : 0;
		const isRightAlignedPageNum =
			Boolean(rightPageMatch?.[1]) &&
			lastSeg !== undefined &&
			(lastSeg.bbox.x >= 0.72 || hasDotLeader || horizontalGap >= 0.08);

		if (isRightAlignedPageNum && rightPageMatch?.[1] && lastSeg) {
			const pageToken = rightPageMatch[1];
			const leftSegments = meaningfulSegments.slice(0, -1);
			const rawLabel = leftSegments.map(s => s.text.trim()).join(' ');
			const label = cleanTocLabel(rawLabel);

			if (label) {
				// Weight confidence primarily by the label segments so a leading dot before a page number (. 2) doesn't false-flag
				const labelWeight = leftSegments.reduce((acc, s) => acc + Math.max(1, s.text.length), 0);
				const labelWeightedConf =
					leftSegments.reduce((acc, s) => acc + s.confidence * Math.max(1, s.text.length), 0) /
					Math.max(1, labelWeight);
				const pageConf = Math.max(0.88, lastSeg.confidence);
				const rowConfidence = Math.min(1, labelWeightedConf * 0.9 + pageConf * 0.1);

				consolidated.push({
					text: `${label} ... ${pageToken}`,
					confidence: rowConfidence,
					bbox: mergeBBoxes(row),
				});
				continue;
			}
		}

		// Standard multi-segment horizontal line (e.g. "I." + "The Interview Process")
		const combinedText = normalizeRomanNumeralPrefix(
			meaningfulSegments
				.map(s => s.text.trim())
				.join(' ')
				.replace(/\s+/g, ' ')
				.trim(),
		);
		const totalChars = meaningfulSegments.reduce((acc, s) => acc + Math.max(1, s.text.length), 0);
		const weightedConf =
			meaningfulSegments.reduce((acc, s) => acc + s.confidence * Math.max(1, s.text.length), 0) /
			Math.max(1, totalChars);

		consolidated.push({
			text: combinedText,
			confidence: weightedConf,
			bbox: mergeBBoxes(meaningfulSegments),
		});
	}

	return consolidated;
}

export function isTocEntryLine(text: string): boolean {
	return TOC_ENTRY_LINE_REGEX.test(text.trim());
}

export function joinLinesWithDehyphenation(lines: readonly OcrLine[]): string {
	if (lines.length === 0) {
		return '';
	}
	let result = lines[0]?.text.trim() ?? '';

	for (let i = 1; i < lines.length; i++) {
		const nextText = lines[i]?.text.trim() ?? '';
		if (!nextText) {
			continue;
		}

		const hyphenMatch = result.match(/([A-Za-z]{2,})[-‐‑]\s*$/);
		const nextStartsLower = /^[a-z]{2,}/.test(nextText);

		if (hyphenMatch?.[1] && nextStartsLower) {
			const prefix = hyphenMatch[1].toLowerCase();
			if (COMPOUND_PREFIXES.has(prefix)) {
				result = `${result.replace(/[-‐‑]\s*$/, '-')}${nextText}`;
			} else {
				result = `${result.replace(/[-‐‑]\s*$/, '')}${nextText}`;
			}
		} else {
			result = `${result} ${nextText}`;
		}
	}

	return result.replace(/\s+/g, ' ').trim();
}

function isHeadingLine(line: OcrLine, medianHeight: number, bodyWidth = 0.8): boolean {
	const text = line.text.trim();
	if (!text || text.length > 95) {
		return false;
	}
	if (isTocEntryLine(text)) {
		return false;
	}
	if (/[,;.!?]$/.test(text) || LIST_ITEM_START_REGEX.test(text)) {
		return false;
	}
	if (EXPLICIT_HEADING_REGEX.test(text)) {
		return true;
	}
	if (line.bbox.height >= medianHeight * 1.18 && /^[A-Z0-9]/.test(text)) {
		return true;
	}
	// Short standalone title line near top of page
	if (
		line.bbox.y < 0.28 &&
		line.bbox.width < bodyWidth * 0.55 &&
		/^[A-Z][A-Za-z0-9\s,&:'’-]{2,55}$/.test(text)
	) {
		return true;
	}
	return false;
}

function shouldStartNewBlock(
	prevLine: OcrLine,
	currLine: OcrLine,
	medianHeight: number,
	leftMargin: number,
	rightMargin: number,
): boolean {
	if (isTocEntryLine(prevLine.text) || isTocEntryLine(currLine.text)) {
		return true;
	}

	const bodyWidth = Math.max(0.2, rightMargin - leftMargin);
	if (
		isHeadingLine(prevLine, medianHeight, bodyWidth) ||
		isHeadingLine(currLine, medianHeight, bodyWidth)
	) {
		return true;
	}

	if (LIST_ITEM_START_REGEX.test(currLine.text.trim())) {
		return true;
	}

	const prevBottom = prevLine.bbox.y + prevLine.bbox.height;
	const verticalGap = currLine.bbox.y - prevBottom;
	if (verticalGap > medianHeight * 0.72) {
		return true;
	}

	const currIndent = currLine.bbox.x - leftMargin;
	const prevIndent = prevLine.bbox.x - leftMargin;
	if (
		currIndent > 0.025 &&
		currIndent - prevIndent > 0.018 &&
		/[.!?:"”’)]$/.test(prevLine.text.trim())
	) {
		return true;
	}

	const prevRight = prevLine.bbox.x + prevLine.bbox.width;
	if (rightMargin - prevRight > bodyWidth * 0.2 && /[.!?:"”’)]$/.test(prevLine.text.trim())) {
		return true;
	}

	return false;
}

export function detectContinuesFromPrev(firstBlock: RawReconstructedBlock | undefined): boolean {
	if (firstBlock?.kind !== 'paragraph') {
		return false;
	}
	const trimmed = firstBlock.text.trim();
	if (isTocEntryLine(trimmed)) {
		return false;
	}
	return /^[a-z,;—–-]/.test(trimmed);
}

export function detectContinuesToNext(lastBlock: RawReconstructedBlock | undefined): boolean {
	if (lastBlock?.kind !== 'paragraph') {
		return false;
	}
	const trimmed = lastBlock.text.trim();
	if (!trimmed || isTocEntryLine(trimmed)) {
		return false;
	}
	// Short trailing signature or caption lines at the bottom of a page do not continue
	if (trimmed.length < 45 && lastBlock.lines.length === 1 && lastBlock.bbox.width < 0.45) {
		return false;
	}
	return !/[.!?:"”’)\]]$/.test(trimmed);
}

export function reconstructPageLayout(
	pageResult: OcrPageResult,
	repeatingMarginKeys: ReadonlySet<string> = new Set(),
): ReconstructedPageLayout {
	const strippedHeadersFooters: string[] = [];
	const rawContentLines: OcrLine[] = [];

	for (const line of pageResult.lines) {
		if (isRunningHeaderOrFooter(line, repeatingMarginKeys)) {
			strippedHeadersFooters.push(line.text.trim());
		} else {
			rawContentLines.push(line);
		}
	}

	if (rawContentLines.length === 0) {
		return {
			page: pageResult.page,
			blocks: [],
			continuesFromPrev: false,
			continuesToNext: false,
			strippedHeadersFooters,
		};
	}

	const rawMedianHeight = computeMedian(
		rawContentLines.filter(l => l.bbox.height > 0.006).map(l => l.bbox.height),
	);
	const contentLines = consolidateHorizontalRows(rawContentLines, rawMedianHeight);

	if (contentLines.length === 0) {
		return {
			page: pageResult.page,
			blocks: [],
			continuesFromPrev: false,
			continuesToNext: false,
			strippedHeadersFooters,
		};
	}

	const medianHeight = computeMedian(contentLines.map(l => l.bbox.height));
	const leftMargin = Math.min(...contentLines.map(l => l.bbox.x));
	const rightMargin = Math.max(...contentLines.map(l => l.bbox.x + l.bbox.width));

	const lineGroups: OcrLine[][] = [];
	for (const line of contentLines) {
		const currentGroup = lineGroups[lineGroups.length - 1];
		const prevLine = currentGroup?.[currentGroup.length - 1];

		if (
			!currentGroup ||
			!prevLine ||
			shouldStartNewBlock(prevLine, line, medianHeight, leftMargin, rightMargin)
		) {
			lineGroups.push([line]);
		} else {
			currentGroup.push(line);
		}
	}

	const bodyWidth = Math.max(0.2, rightMargin - leftMargin);
	const blocks: RawReconstructedBlock[] = lineGroups.map((group, idx) => {
		const firstLine = group[0];
		const kind: BlockKind =
			group.length === 1 && firstLine && isHeadingLine(firstLine, medianHeight, bodyWidth)
				? 'heading'
				: 'paragraph';

		return {
			id: `p${pageResult.page}-b${idx + 1}`,
			kind,
			text: joinLinesWithDehyphenation(group),
			bbox: mergeBBoxes(group),
			lines: group,
		};
	});

	return {
		page: pageResult.page,
		blocks,
		continuesFromPrev: detectContinuesFromPrev(blocks[0]),
		continuesToNext: detectContinuesToNext(blocks[blocks.length - 1]),
		strippedHeadersFooters,
	};
}
