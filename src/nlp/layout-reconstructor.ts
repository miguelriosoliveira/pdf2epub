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
	return top < 0.085 || bottom > 0.915;
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
		return 0.018;
	}
	const sorted = [...values].sort((a, b) => a - b);
	const mid = Math.floor(sorted.length / 2);
	return sorted[mid] ?? 0.018;
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
	return /^[a-z,;—–-]/.test(trimmed);
}

export function detectContinuesToNext(lastBlock: RawReconstructedBlock | undefined): boolean {
	if (lastBlock?.kind !== 'paragraph') {
		return false;
	}
	const trimmed = lastBlock.text.trim();
	if (!trimmed) {
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
	const contentLines: OcrLine[] = [];

	for (const line of pageResult.lines) {
		if (isRunningHeaderOrFooter(line, repeatingMarginKeys)) {
			strippedHeadersFooters.push(line.text.trim());
		} else {
			contentLines.push(line);
		}
	}

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
