import type { BlockKind, LayoutProfile, OcrLine, OcrPageResult } from '../types/domain.ts';
import type { DetectedFigureRegion } from './figure-detector.ts';
import { mergeBBoxes, type RawReconstructedBlock } from './layout-reconstructor.ts';

export interface TechnicalLayoutExtraction {
	blocks: RawReconstructedBlock[];
	consumedLines: Set<OcrLine>;
	figureRegions: DetectedFigureRegion[];
}

interface OcrRow {
	centerY: number;
	lines: OcrLine[];
}

const CODE_MARKER_REGEX = /[{};]|(?:^|\s)(?:int|return|for\s*\(|if\s*\(|while\s*\(|void)\b/i;
const LINE_NUMBER_REGEX = /^\s*\d{1,2}\s*$/;
const TREE_LABEL_REGEX = /^f\s*\(\s*\d+\s*\)?$/i;

function median(values: readonly number[]): number {
	if (values.length === 0) {
		return 0.015;
	}
	const sorted = [...values].sort((a, b) => a - b);
	const middle = Math.floor(sorted.length / 2);
	return sorted.length % 2 === 0
		? ((sorted[middle - 1] ?? 0) + (sorted[middle] ?? 0)) / 2
		: (sorted[middle] ?? 0.015);
}

function lineCenterY(line: OcrLine): number {
	return line.bbox.y + line.bbox.height / 2;
}

function groupRows(lines: readonly OcrLine[], medianHeight: number): OcrRow[] {
	const sorted = [...lines].sort((a, b) => lineCenterY(a) - lineCenterY(b));
	const rows: OcrRow[] = [];
	for (const line of sorted) {
		const centerY = lineCenterY(line);
		const previous = rows[rows.length - 1];
		if (previous && Math.abs(centerY - previous.centerY) <= medianHeight * 0.65) {
			previous.lines.push(line);
			previous.centerY =
				previous.lines.reduce((sum, item) => sum + lineCenterY(item), 0) / previous.lines.length;
		} else {
			rows.push({ centerY, lines: [line] });
		}
	}
	return rows;
}

function makeStructuredBlock(
	page: number,
	index: number,
	kind: Extract<BlockKind, 'code' | 'table'>,
	text: string,
	lines: OcrLine[],
): RawReconstructedBlock {
	return {
		id: `p${page}-technical-${index}`,
		kind,
		text,
		bbox: mergeBBoxes(lines),
		lines,
	};
}

function isCodeLine(text: string): boolean {
	return CODE_MARKER_REGEX.test(text.trim());
}

function stripCodeLineNumber(text: string): string {
	return text.trim().replace(/^\d{1,2}\s+(?=\S)/, '');
}

function indentCodeLines(lines: readonly string[]): string[] {
	let depth = 0;
	return lines.map(rawLine => {
		const line = rawLine.trim();
		const leadingClosers = line.match(/^}+/)?.[0].length ?? 0;
		depth = Math.max(0, depth - leadingClosers);
		const formatted = `${'  '.repeat(depth)}${line}`;
		const openingBraces = line.match(/{/g)?.length ?? 0;
		const closingBraces = (line.match(/}/g)?.length ?? 0) - leadingClosers;
		depth = Math.max(0, depth + openingBraces - closingBraces);
		return formatted;
	});
}

function escapeHtml(text: string): string {
	return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function createComparisonTable(
	page: OcrPageResult,
	lines: readonly OcrLine[],
	consumed: Set<OcrLine>,
	index: number,
): RawReconstructedBlock | undefined {
	const headings = lines
		.filter(line => /^Min\s+and\s+Max\s+[12]$/i.test(line.text.trim()))
		.sort((a, b) => a.bbox.x - b.bbox.x);
	const leftHeading = headings.find(line => /\b1$/.test(line.text.trim()));
	const rightHeading = headings.find(line => /\b2$/.test(line.text.trim()));
	if (!leftHeading || !rightHeading) {
		return undefined;
	}
	if (Math.abs(lineCenterY(leftHeading) - lineCenterY(rightHeading)) > 0.02) {
		return undefined;
	}

	const headerBottom = Math.max(
		leftHeading.bbox.y + leftHeading.bbox.height,
		rightHeading.bbox.y + rightHeading.bbox.height,
	);
	const midpoint = (leftHeading.bbox.x + rightHeading.bbox.x) / 2;
	const codeLines = lines.filter(
		line =>
			line.bbox.y > headerBottom &&
			line.bbox.y < headerBottom + 0.15 &&
			isCodeLine(line.text) &&
			line.bbox.x > leftHeading.bbox.x - 0.02 &&
			line.bbox.x < rightHeading.bbox.x + rightHeading.bbox.width + 0.02,
	);
	const leftCode = codeLines
		.filter(line => line.bbox.x < midpoint)
		.sort((a, b) => lineCenterY(a) - lineCenterY(b) || a.bbox.x - b.bbox.x)
		.map(line => stripCodeLineNumber(line.text));
	const rightCode = codeLines
		.filter(line => line.bbox.x >= midpoint)
		.sort((a, b) => lineCenterY(a) - lineCenterY(b) || a.bbox.x - b.bbox.x)
		.map(line => stripCodeLineNumber(line.text));
	if (leftCode.length < 3 || rightCode.length < 3) {
		return undefined;
	}

	const consumedLines = [leftHeading, rightHeading, ...codeLines];
	for (const numberLine of lines.filter(line => LINE_NUMBER_REGEX.test(line.text))) {
		if (
			codeLines.some(
				codeLine =>
					Math.abs(lineCenterY(codeLine) - lineCenterY(numberLine)) < 0.009 &&
					numberLine.bbox.x < codeLine.bbox.x,
			)
		) {
			consumedLines.push(numberLine);
		}
	}
	for (const line of consumedLines) {
		consumed.add(line);
	}

	const leftCodeBlock = escapeHtml(indentCodeLines(leftCode).join('\n'));
	const rightCodeBlock = escapeHtml(indentCodeLines(rightCode).join('\n'));
	const tableText = [
		'<table>',
		`<thead><tr><th>${escapeHtml(leftHeading.text.trim())}</th><th>${escapeHtml(rightHeading.text.trim())}</th></tr></thead>`,
		'<tbody><tr>',
		`<td><pre><code>${leftCodeBlock}</code></pre></td>`,
		`<td><pre><code>${rightCodeBlock}</code></pre></td>`,
		'</tr></tbody>',
		'</table>',
	].join('\n');
	return makeStructuredBlock(page.page, index, 'table', tableText, consumedLines);
}

function createTableBlock(
	page: OcrPageResult,
	lines: readonly OcrLine[],
	consumed: Set<OcrLine>,
	index: number,
	profile: LayoutProfile,
): RawReconstructedBlock | undefined {
	const availableLines = lines.filter(line => !consumed.has(line));
	const lineHeight = median(availableLines.map(line => line.bbox.height));
	const rows = groupRows(availableLines, lineHeight);
	const minRows = profile === 'technical' ? 4 : 5;

	for (let headerIndex = 0; headerIndex < rows.length; headerIndex++) {
		const header = rows[headerIndex];
		if (!header || header.lines.length < 3) {
			continue;
		}
		const headerLines = [...header.lines].sort((a, b) => a.bbox.x - b.bbox.x);
		const headerSpan =
			(headerLines[headerLines.length - 1]?.bbox.x ?? 0) - (headerLines[0]?.bbox.x ?? 0);
		if (headerSpan < 0.3 || headerLines.some(line => line.bbox.width > 0.25)) {
			continue;
		}

		const rowsInTable: OcrRow[] = [header];
		let previousY = header.centerY;
		for (let rowIndex = headerIndex + 1; rowIndex < rows.length; rowIndex++) {
			const row = rows[rowIndex];
			if (!row) {
				break;
			}
			const gap = row.centerY - previousY;
			if (gap < 0.012 || gap > 0.035 || row.lines.some(line => line.bbox.width > 0.42)) {
				break;
			}
			const rowAnchored = row.lines.every(line =>
				headerLines.some(headerLine => Math.abs(line.bbox.x - headerLine.bbox.x) <= 0.075),
			);
			if (!rowAnchored) {
				break;
			}
			rowsInTable.push(row);
			previousY = row.centerY;
		}
		if (rowsInTable.length < minRows) {
			continue;
		}

		const anchors = headerLines.map(line => line.bbox.x);
		const normalizeHeader = (text: string, column: number) =>
			column === 0 && /^evel$/i.test(text.trim()) ? 'Level' : text.trim();
		const headerCells = headerLines.map((line, column) => normalizeHeader(line.text, column));
		const tableRows = rowsInTable.slice(1).map(row => {
			const cells = anchors.map(() => '');
			for (const line of [...row.lines].sort((a, b) => a.bbox.x - b.bbox.x)) {
				let nearest = 0;
				for (let column = 1; column < anchors.length; column++) {
					if (
						Math.abs(line.bbox.x - (anchors[column] ?? 0)) <
						Math.abs(line.bbox.x - (anchors[nearest] ?? 0))
					) {
						nearest = column;
					}
				}
				const value = (cells[nearest] ? `${cells[nearest]} ` : '') + line.text.trim();
				cells[nearest] = value.replace(/\|/g, '\\|');
			}
			return cells;
		});
		const markdown = [
			`| ${headerCells.join(' | ')} |`,
			`| ${headerCells.map(() => '---').join(' | ')} |`,
			...tableRows.map(row => `| ${row.join(' | ')} |`),
		].join('\n');
		const tableLines = rowsInTable.flatMap(row => row.lines);
		for (const line of tableLines) {
			consumed.add(line);
		}
		return makeStructuredBlock(page.page, index, 'table', markdown, tableLines);
	}

	return undefined;
}

function createCodeBlocks(
	page: OcrPageResult,
	lines: readonly OcrLine[],
	consumed: Set<OcrLine>,
	startIndex: number,
	profile: LayoutProfile,
): RawReconstructedBlock[] {
	const availableLines = lines.filter(line => !consumed.has(line));
	const lineHeight = median(availableLines.map(line => line.bbox.height));
	const codeLines = availableLines.filter(line => isCodeLine(line.text));
	const groups: OcrLine[][] = [];
	for (const line of [...codeLines].sort((a, b) => lineCenterY(a) - lineCenterY(b))) {
		const group = groups[groups.length - 1];
		const previous = group?.[group.length - 1];
		const verticalGap = previous
			? line.bbox.y - (previous.bbox.y + previous.bbox.height)
			: Number.POSITIVE_INFINITY;
		if (!group || verticalGap > Math.max(0.035, lineHeight * 2.5)) {
			groups.push([line]);
		} else {
			group.push(line);
		}
	}

	const minimumLines = profile === 'technical' ? 2 : 3;
	const blocks: RawReconstructedBlock[] = [];
	for (const group of groups) {
		if (group.length < minimumLines) {
			continue;
		}
		const codeLines = [...group]
			.sort((a, b) => lineCenterY(a) - lineCenterY(b) || a.bbox.x - b.bbox.x)
			.map(line => stripCodeLineNumber(line.text))
			.filter(Boolean);
		if (codeLines.length === 0) {
			continue;
		}

		const lineNumbers = availableLines.filter(
			line =>
				LINE_NUMBER_REGEX.test(line.text) &&
				group.some(
					codeLine =>
						Math.abs(lineCenterY(codeLine) - lineCenterY(line)) <
							Math.max(0.02, lineHeight * 1.5) &&
						line.bbox.x < codeLine.bbox.x &&
						codeLine.bbox.x - line.bbox.x <= 0.12,
				),
		);
		const blockLines = [...group, ...lineNumbers];
		for (const line of blockLines) {
			consumed.add(line);
		}
		blocks.push(
			makeStructuredBlock(
				page.page,
				startIndex + blocks.length,
				'code',
				indentCodeLines(codeLines).join('\n'),
				blockLines,
			),
		);
	}
	return blocks;
}

function detectRecursionTree(lines: readonly OcrLine[]): DetectedFigureRegion[] {
	const candidates = lines.filter(
		line => TREE_LABEL_REGEX.test(line.text.trim()) && line.bbox.width <= 0.08,
	);
	if (candidates.length < 7) {
		return [];
	}

	const minX = Math.min(...candidates.map(line => line.bbox.x));
	const minY = Math.min(...candidates.map(line => line.bbox.y));
	const maxX = Math.max(...candidates.map(line => line.bbox.x + line.bbox.width));
	const maxY = Math.max(...candidates.map(line => line.bbox.y + line.bbox.height));
	if (maxX - minX < 0.35 || maxY - minY < 0.05) {
		return [];
	}

	const hasTreeContext = lines.some(
		line =>
			!candidates.includes(line) &&
			/\b(?:tree|recursive|recursion)\b/i.test(line.text) &&
			line.bbox.y >= maxY &&
			line.bbox.y - maxY <= 0.08,
	);
	if (!hasTreeContext) {
		return [];
	}

	const previousText = lines
		.filter(
			line =>
				!candidates.includes(line) &&
				line.bbox.width > 0.2 &&
				line.bbox.y + line.bbox.height <= minY &&
				minY - (line.bbox.y + line.bbox.height) <= 0.08,
		)
		.sort((a, b) => b.bbox.y + b.bbox.height - (a.bbox.y + a.bbox.height))[0];
	const treeQuestion = lines.find(
		line =>
			!candidates.includes(line) &&
			/\b(?:tree|recursive|recursion)\b/i.test(line.text) &&
			line.bbox.y >= maxY,
	);
	const x = Math.max(0, minX - 0.02);
	const right = Math.min(1, maxX + 0.02);
	const top = previousText ? previousText.bbox.y + previousText.bbox.height + 0.003 : minY - 0.02;
	const bottom = treeQuestion ? treeQuestion.bbox.y - 0.003 : maxY + 0.02;
	const region = {
		x,
		y: Math.max(0, top),
		width: right - x,
		height: Math.max(0, bottom - Math.max(0, top)),
	};
	const associatedLabels = lines.filter(line => {
		const text = line.text.trim();
		const centerX = line.bbox.x + line.bbox.width / 2;
		const centerY = line.bbox.y + line.bbox.height / 2;
		return (
			!candidates.includes(line) &&
			text.length <= 8 &&
			/^[\p{L}\p{N}\s()+*^!?-]+$/u.test(text) &&
			centerX >= region.x &&
			centerX <= region.x + region.width &&
			centerY >= region.y &&
			centerY <= region.y + region.height
		);
	});
	return [
		{
			bbox: region,
			labelLines: [...candidates, ...associatedLabels],
		},
	];
}

export function extractTechnicalRegions(
	page: OcrPageResult,
	profile: LayoutProfile,
): TechnicalLayoutExtraction {
	const consumedLines = new Set<OcrLine>();
	const blocks: RawReconstructedBlock[] = [];
	const comparisonBlock = createComparisonTable(page, page.lines, consumedLines, 1);
	if (comparisonBlock) {
		blocks.push(comparisonBlock);
	}

	const tableBlock = createTableBlock(page, page.lines, consumedLines, blocks.length + 1, profile);
	if (tableBlock) {
		blocks.push(tableBlock);
	}
	blocks.push(...createCodeBlocks(page, page.lines, consumedLines, blocks.length + 1, profile));

	return {
		blocks,
		consumedLines,
		figureRegions: detectRecursionTree(page.lines),
	};
}
