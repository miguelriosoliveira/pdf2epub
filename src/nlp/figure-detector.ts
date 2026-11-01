import type { NormalizedBBox, OcrLine } from '../types/domain.ts';

export interface DetectedFigureRegion {
	bbox: NormalizedBBox;
	labelLines: OcrLine[];
}

const GRAPH_LABEL_REGEX = /^[o0]\s*\(\s*[^()]{1,14}\s*\)$/i;
// This is a roomy search area; the renderer trims to ink and applies equal pixel padding.
const FIGURE_HORIZONTAL_PADDING = 0.18;
const FIGURE_VERTICAL_PADDING = 0.035;
const GRAPH_CAPTION_REGEX =
	/\b(?:graph|chart|diagram|figure)\b.*\b(?:depicts|shows|illustrates|plots)\b/i;
export const BOOK_COVER_ASSET_PATH = 'figures/book-cover.png';
export const BOOK_COVER_MARKDOWN = `![Book cover](${BOOK_COVER_ASSET_PATH})`;

function median(values: readonly number[]): number {
	if (values.length === 0) {
		return 0.015;
	}
	const sorted = [...values].sort((a, b) => a - b);
	const middle = Math.floor(sorted.length / 2);
	if (sorted.length % 2 === 0) {
		return ((sorted[middle - 1] ?? 0) + (sorted[middle] ?? 0)) / 2;
	}
	return sorted[middle] ?? 0.015;
}

function padBBox(
	box: NormalizedBBox,
	lines: readonly OcrLine[],
	labelLines: readonly OcrLine[],
): NormalizedBBox {
	const left = Math.max(0, box.x - FIGURE_HORIZONTAL_PADDING);
	const right = Math.min(1, box.x + box.width + FIGURE_HORIZONTAL_PADDING);
	let top = Math.max(0, box.y - FIGURE_VERTICAL_PADDING);
	let bottom = Math.min(1, box.y + box.height + FIGURE_VERTICAL_PADDING);
	const labelSet = new Set(labelLines);
	const precedingText = lines
		.filter(
			line =>
				!labelSet.has(line) &&
				line.bbox.width > 0.15 &&
				line.bbox.y + line.bbox.height <= box.y &&
				box.y - (line.bbox.y + line.bbox.height) <= 0.08,
		)
		.sort((a, b) => b.bbox.y + b.bbox.height - (a.bbox.y + a.bbox.height))[0];
	const precedingCaption = lines
		.filter(
			line =>
				!labelSet.has(line) &&
				GRAPH_CAPTION_REGEX.test(line.text) &&
				line.bbox.y + line.bbox.height <= box.y &&
				box.y - (line.bbox.y + line.bbox.height) <= 0.08,
		)
		.sort((a, b) => b.bbox.y - a.bbox.y)[0];
	if (precedingCaption) {
		top = precedingCaption.bbox.y + precedingCaption.bbox.height + 0.006;
	} else if (precedingText) {
		top = Math.min(top, precedingText.bbox.y + precedingText.bbox.height + 0.003);
	}
	const followingText = lines
		.filter(
			line =>
				!labelSet.has(line) &&
				line.bbox.width > 0.15 &&
				line.bbox.y >= box.y + box.height &&
				line.bbox.y - (box.y + box.height) <= 0.08,
		)
		.sort((a, b) => a.bbox.y - b.bbox.y)[0];
	if (followingText) {
		bottom = Math.max(bottom, followingText.bbox.y - 0.006);
	}
	return { x: left, y: top, width: right - left, height: bottom - top };
}

export function detectGraphFigureRegions(lines: readonly OcrLine[]): DetectedFigureRegion[] {
	const medianLineHeight = median(lines.map(line => line.bbox.height).filter(height => height > 0));
	const candidates = lines.filter(
		line =>
			GRAPH_LABEL_REGEX.test(line.text.trim()) &&
			line.bbox.width <= 0.16 &&
			line.bbox.height >= medianLineHeight * 1.08,
	);
	const consumed = new Set<OcrLine>();
	const regions: DetectedFigureRegion[] = [];

	for (const seed of candidates) {
		if (consumed.has(seed)) {
			continue;
		}
		const seedCenterX = seed.bbox.x + seed.bbox.width / 2;
		const seedCenterY = seed.bbox.y + seed.bbox.height / 2;
		const group = candidates.filter(line => {
			const centerX = line.bbox.x + line.bbox.width / 2;
			const centerY = line.bbox.y + line.bbox.height / 2;
			return (
				!consumed.has(line) &&
				Math.abs(centerX - seedCenterX) <= 0.4 &&
				Math.abs(centerY - seedCenterY) <= 0.3
			);
		});

		if (group.length < 2) {
			continue;
		}

		const centerYs = group.map(line => line.bbox.y + line.bbox.height / 2);
		const verticalSpan = Math.max(...centerYs) - Math.min(...centerYs);
		if (verticalSpan < Math.max(0.035, medianLineHeight * 1.8)) {
			continue;
		}

		const minX = Math.min(...group.map(line => line.bbox.x));
		const minY = Math.min(...group.map(line => line.bbox.y));
		const maxX = Math.max(...group.map(line => line.bbox.x + line.bbox.width));
		const maxY = Math.max(...group.map(line => line.bbox.y + line.bbox.height));
		const labelBounds = {
			x: minX,
			y: minY,
			width: maxX - minX,
			height: maxY - minY,
		};
		const needsCaptionAnchor = labelBounds.width > 0.15 || verticalSpan > 0.14;
		const hasNearbyCaption = lines.some(
			line =>
				!group.includes(line) &&
				GRAPH_CAPTION_REGEX.test(line.text) &&
				line.bbox.y + line.bbox.height <= minY &&
				minY - (line.bbox.y + line.bbox.height) <= 0.08,
		);
		if (needsCaptionAnchor && !hasNearbyCaption) {
			continue;
		}
		regions.push({
			bbox: padBBox(labelBounds, lines, group),
			labelLines: group,
		});
		for (const line of group) {
			consumed.add(line);
		}
	}

	return regions.sort((a, b) => a.bbox.y - b.bbox.y);
}

/** A line this wide is running text. A drawing that contains one is a text block with a border. */
const RUNNING_TEXT_MIN_WIDTH = 0.4;
/** Labels whose centres sit this close outside a drawing still belong to it. */
const INK_LABEL_MARGIN = 0.006;

function isCenterInside(line: OcrLine, box: NormalizedBBox, margin = 0): boolean {
	const centerX = line.bbox.x + line.bbox.width / 2;
	const centerY = line.bbox.y + line.bbox.height / 2;
	return (
		centerX >= box.x - margin &&
		centerX <= box.x + box.width + margin &&
		centerY >= box.y - margin &&
		centerY <= box.y + box.height + margin
	);
}

function unionBox(boxes: readonly NormalizedBBox[]): NormalizedBBox {
	const left = Math.min(...boxes.map(box => box.x));
	const top = Math.min(...boxes.map(box => box.y));
	const right = Math.max(...boxes.map(box => box.x + box.width));
	const bottom = Math.max(...boxes.map(box => box.y + box.height));
	return { x: left, y: top, width: right - left, height: bottom - top };
}

function containsBox(outer: NormalizedBBox, inner: NormalizedBBox): boolean {
	return (
		inner.x >= outer.x &&
		inner.y >= outer.y &&
		inner.x + inner.width <= outer.x + outer.width &&
		inner.y + inner.height <= outer.y + outer.height
	);
}

/** Drawings further apart than this (as a page fraction) are never parts of one figure. */
const MAX_DRAWING_GAP = 0.12;

/**
 * The space between two drawings that share a column (stacked) or a row (side by side), or
 * undefined when they share neither. `band` is that space across the shared extent.
 */
function bandBetween(
	a: NormalizedBBox,
	b: NormalizedBBox,
): { gap: number; band: NormalizedBBox } | undefined {
	const sharedColumn =
		Math.min(a.x + a.width, b.x + b.width) - Math.max(a.x, b.x) >= 0.5 * Math.min(a.width, b.width);
	const sharedRow =
		Math.min(a.y + a.height, b.y + b.height) - Math.max(a.y, b.y) >=
		0.5 * Math.min(a.height, b.height);
	const left = Math.min(a.x, b.x);
	const right = Math.max(a.x + a.width, b.x + b.width);
	const top = Math.min(a.y, b.y);
	const bottom = Math.max(a.y + a.height, b.y + b.height);
	if (sharedColumn) {
		const upper = a.y <= b.y ? a : b;
		const lower = upper === a ? b : a;
		const gap = lower.y - (upper.y + upper.height);
		return {
			gap,
			band: { x: left, y: upper.y + upper.height, width: right - left, height: gap },
		};
	}
	if (sharedRow) {
		const leftmost = a.x <= b.x ? a : b;
		const rightmost = leftmost === a ? b : a;
		const gap = rightmost.x - (leftmost.x + leftmost.width);
		return {
			gap,
			band: { x: leftmost.x + leftmost.width, y: top, width: gap, height: bottom - top },
		};
	}
	return undefined;
}

/**
 * Two drawings that share a column or a row, with only short labels between them, are one
 * figure. A flowchart's top row can lead into a row of plain-text steps, and text erased from a
 * tree's root can split it in two; both read as one picture.
 */
function joinsDrawings(a: NormalizedBBox, b: NormalizedBBox, lines: readonly OcrLine[]): boolean {
	const between = bandBetween(a, b);
	if (!between) {
		return false;
	}
	if (between.gap <= 0) {
		return true;
	}
	if (between.gap > MAX_DRAWING_GAP) {
		return false;
	}
	const labels = lines.filter(line => isCenterInside(line, between.band));
	if (labels.length === 0 || !labels.every(line => line.bbox.width < RUNNING_TEXT_MIN_WIDTH)) {
		return false;
	}
	// A paragraph that runs past the two drawings lies in the gap without its centre in it, and
	// it is running text, so it keeps the drawings apart.
	return !lines.some(
		line => line.bbox.width >= RUNNING_TEXT_MIN_WIDTH && touchesBand(line, between.band),
	);
}

/** A line whose centre sits in the band's rows and that overlaps the band's columns at all. */
function touchesBand(line: OcrLine, band: NormalizedBBox): boolean {
	const centreY = line.bbox.y + line.bbox.height / 2;
	return (
		centreY >= band.y &&
		centreY <= band.y + band.height &&
		line.bbox.x < band.x + band.width &&
		line.bbox.x + line.bbox.width > band.x
	);
}

function joinNearbyDrawings(
	candidates: readonly NormalizedBBox[],
	lines: readonly OcrLine[],
): NormalizedBBox[] {
	let drawings = [...candidates];
	let joined = true;
	while (joined) {
		joined = false;
		for (let i = 0; i < drawings.length && !joined; i++) {
			for (let j = i + 1; j < drawings.length && !joined; j++) {
				const a = drawings[i];
				const b = drawings[j];
				if (a && b && joinsDrawings(a, b, lines)) {
					drawings = [
						...drawings.filter((_, index) => index !== i && index !== j),
						unionBox([a, b]),
					];
					joined = true;
				}
			}
		}
	}
	return drawings;
}

/**
 * Turns drawings found by the OCR helper into figure regions. The helper has already erased
 * recognized text, so the short labels inside a drawing reach this function as ordinary lines
 * and are claimed as part of the image. Candidates that hold running text are skipped.
 */
export function detectInkFigureRegions(
	lines: readonly OcrLine[],
	candidates: readonly NormalizedBBox[],
): DetectedFigureRegion[] {
	const claimed = new Set<OcrLine>();
	const regions: DetectedFigureRegion[] = [];
	const largestFirst = joinNearbyDrawings(candidates, lines).sort(
		(a, b) => b.width * b.height - a.width * a.height,
	);

	for (const candidate of largestFirst) {
		if (regions.some(region => containsBox(region.bbox, candidate))) {
			continue;
		}
		const nearby = lines.filter(
			line => !claimed.has(line) && isCenterInside(line, candidate, INK_LABEL_MARGIN),
		);
		if (nearby.some(line => line.bbox.width >= RUNNING_TEXT_MIN_WIDTH)) {
			continue;
		}
		const box = unionBox([candidate, ...nearby.map(line => line.bbox)]);
		const labels = lines.filter(line => !claimed.has(line) && isCenterInside(line, box));
		if (labels.some(line => line.bbox.width >= RUNNING_TEXT_MIN_WIDTH)) {
			continue;
		}
		for (const line of labels) {
			claimed.add(line);
		}
		regions.push({
			bbox: unionBox([box, ...labels.map(line => line.bbox)]),
			labelLines: labels,
		});
	}
	return regions.sort((a, b) => a.bbox.y - b.bbox.y);
}

export function figureAssetPath(page: number, figureIndex: number): string {
	return `figures/page-${String(page).padStart(4, '0')}-figure-${String(figureIndex + 1).padStart(2, '0')}.png`;
}

export function figureAssetPathFromMarkdown(text: string): string | undefined {
	return text.match(/^!\[Figure from page \d+\]\((figures\/page-\d{4}-figure-\d+\.png)\)$/)?.[1];
}

export function isBookCoverMarkdown(text: string): boolean {
	return text === BOOK_COVER_MARKDOWN;
}
