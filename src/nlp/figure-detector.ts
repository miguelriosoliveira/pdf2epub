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

export function figureAssetPath(page: number, figureIndex: number): string {
	return `figures/page-${String(page).padStart(4, '0')}-figure-${String(figureIndex + 1).padStart(2, '0')}.png`;
}

export function figureAssetPathFromMarkdown(text: string): string | undefined {
	return text.match(/^!\[Figure from page \d+\]\((figures\/page-\d{4}-figure-\d+\.png)\)$/)?.[1];
}

export function isBookCoverMarkdown(text: string): boolean {
	return text === BOOK_COVER_MARKDOWN;
}
