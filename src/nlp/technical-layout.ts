import type { BlockKind, LayoutProfile, OcrLine, OcrPageResult } from '../types/domain.ts';
import { unclosedBraceDepth } from './code-structure.ts';
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
const CODE_STRUCTURE_REGEX =
	/[{}]|\b(?:for|if|while|switch)\s*\(|\b(?:def|class|function|const|let|var|int|void)\s+\w+/i;
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
	const trimmed = text.trim();
	return CODE_MARKER_REGEX.test(trimmed) || /^\/\*.*\*\/$/.test(trimmed) || /^\/\//.test(trimmed);
}

function stripCodeLineNumber(text: string): string {
	return text.trim().replace(/^\d{1,2}\s+(?=[A-Za-z_$}{]|\/\/|\/\*)/, '');
}

function indentCodeLines(lines: readonly string[]): string[] {
	let precedingText = '';
	return lines.map(rawLine => {
		const line = rawLine.trim();
		const leadingClosers = line.match(/^}+/)?.[0].length ?? 0;
		const depth = Math.max(0, unclosedBraceDepth(precedingText) - leadingClosers);
		const formatted = `${'  '.repeat(depth)}${line}`;
		precedingText += `${line}\n`;
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
	const available = lines.filter(line => !consumed.has(line));
	const rows = groupRows(available, median(available.map(line => line.bbox.height)));

	for (let headingIndex = 0; headingIndex < rows.length; headingIndex++) {
		const headingRow = rows[headingIndex];
		if (headingRow?.lines.length !== 2) continue;
		const [leftHeading, rightHeading] = [...headingRow.lines].sort((a, b) => a.bbox.x - b.bbox.x);
		if (
			!leftHeading ||
			!rightHeading ||
			rightHeading.bbox.x - leftHeading.bbox.x < 0.2 ||
			leftHeading.bbox.x + leftHeading.bbox.width > rightHeading.bbox.x ||
			[leftHeading, rightHeading].some(
				line => line.text.length > 100 || !/\p{L}/u.test(line.text) || isCodeLine(line.text),
			)
		)
			continue;

		const headerBottom = Math.max(
			leftHeading.bbox.y + leftHeading.bbox.height,
			rightHeading.bbox.y + rightHeading.bbox.height,
		);
		const midpoint = (leftHeading.bbox.x + rightHeading.bbox.x) / 2;
		const leftCode: string[] = [];
		const rightCode: string[] = [];
		const comparisonLines: OcrLine[] = [leftHeading, rightHeading];

		for (const row of rows.slice(headingIndex + 1)) {
			if (row.centerY - headerBottom > 0.2) break;
			if (row.lines.some(line => line.bbox.width > 0.5)) break;
			const segments = row.lines.filter(
				line =>
					line.bbox.x >= leftHeading.bbox.x - 0.025 && line.bbox.x < rightHeading.bbox.x + 0.38,
			);
			if (segments.length === 0) continue;
			const left = segments.filter(
				line => line.bbox.x < midpoint && !LINE_NUMBER_REGEX.test(line.text),
			);
			const right = segments.filter(
				line => line.bbox.x >= midpoint && !LINE_NUMBER_REGEX.test(line.text),
			);
			const rowHasCode = [...left, ...right].some(line => isCodeLine(line.text));
			if (!rowHasCode && (left.length > 0 || right.length > 0)) break;
			for (const [parts, output] of [
				[left, leftCode],
				[right, rightCode],
			] as const) {
				const text = parts
					.sort((a, b) => a.bbox.x - b.bbox.x)
					.map(line => stripCodeLineNumber(line.text))
					.join(' ')
					.trim();
				if (text) output.push(text);
			}
			comparisonLines.push(...segments);
		}

		if (
			leftCode.filter(isCodeLine).length < 3 ||
			rightCode.filter(isCodeLine).length < 3 ||
			!leftCode.some(line => CODE_STRUCTURE_REGEX.test(line)) ||
			!rightCode.some(line => CODE_STRUCTURE_REGEX.test(line))
		)
			continue;
		for (const line of comparisonLines) consumed.add(line);

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
		return makeStructuredBlock(page.page, index, 'table', tableText, comparisonLines);
	}
	return undefined;
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
		const textLayerEvidence: OcrLine[] = [];
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
			if (cells.length >= 2 && !cells[0] && /^\d+$/.test(cells[1] ?? '')) {
				const layerLine = page.textLayerLines?.find(line => {
					const numbers = line.text.trim().match(/^(\d+)\s+(\d+)(?:\s|$)/);
					return (
						numbers?.[2] === cells[1] &&
						Math.abs(lineCenterY(line) - row.centerY) <= 0.007 &&
						Math.abs(line.bbox.x - (anchors[0] ?? 0)) <= 0.04 &&
						line.bbox.x + line.bbox.width >= (anchors[1] ?? 1) - 0.02
					);
				});
				const missingCell = layerLine?.text.trim().match(/^(\d+)\s+(\d+)(?:\s|$)/)?.[1];
				if (layerLine && missingCell !== undefined) {
					cells[0] = missingCell;
					textLayerEvidence.push({
						...layerLine,
						confidence: Math.min(layerLine.confidence, 0.8),
						source: 'text-layer',
					});
				}
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
		return makeStructuredBlock(page.page, index, 'table', markdown, [
			...tableLines,
			...textLayerEvidence,
		]);
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
	const visibleLines = lines.filter(line => !consumed.has(line));
	const recoveredClosers: OcrLine[] = [];
	const textLayerLines = page.textLayerLines ?? [];
	for (const layerLine of textLayerLines) {
		const match = layerLine.text.trim().match(/^(\d{1,2})\s+}\s*$/);
		if (!match?.[1]) continue;
		const sameRow = (line: OcrLine) =>
			Math.abs(lineCenterY(line) - lineCenterY(layerLine)) <= 0.007;
		const marker = visibleLines.find(line => line.text.trim() === match[1] && sameRow(line));
		if (
			visibleLines.some(
				line =>
					!LINE_NUMBER_REGEX.test(line.text) &&
					sameRow(line) &&
					line.bbox.x > layerLine.bbox.x + 0.02,
			)
		)
			continue;
		const lineNumber = Number.parseInt(match[1], 10);
		const corroboratesNeighbor = (number: number): boolean =>
			textLayerLines.some(candidate => {
				const neighbor = candidate.text.trim().match(/^(\d{1,2})\s+(.+)$/);
				if (
					!neighbor?.[1] ||
					!neighbor[2] ||
					Number.parseInt(neighbor[1], 10) !== number ||
					Math.abs(lineCenterY(candidate) - lineCenterY(layerLine)) > 0.04
				)
					return false;
				const expected = neighbor[2].replace(/\s+/g, '');
				return visibleLines.some(
					line =>
						Math.abs(lineCenterY(line) - lineCenterY(candidate)) <= 0.007 &&
						stripCodeLineNumber(line.text).replace(/\s+/g, '') === expected,
				);
			});
		if (!marker && !(corroboratesNeighbor(lineNumber - 1) && corroboratesNeighbor(lineNumber + 1)))
			continue;
		const neighboringCode = visibleLines.filter(
			line =>
				isCodeLine(line.text) &&
				line.bbox.x > layerLine.bbox.x &&
				Math.abs(lineCenterY(line) - lineCenterY(layerLine)) <= 0.04,
		);
		const hasPreviousCode = neighboringCode.some(
			line => lineCenterY(line) < lineCenterY(layerLine),
		);
		const hasFollowingCode = neighboringCode.some(
			line => lineCenterY(line) > lineCenterY(layerLine),
		);
		if (!hasPreviousCode || !hasFollowingCode) {
			const priorCode = [...visibleLines, ...recoveredClosers]
				.filter(
					line =>
						isCodeLine(line.text) &&
						lineCenterY(line) < lineCenterY(layerLine) &&
						lineCenterY(layerLine) - lineCenterY(line) <= 0.12,
				)
				.sort((a, b) => lineCenterY(a) - lineCenterY(b));
			const lastCode = priorCode.at(-1);
			const interruptedByProse =
				lastCode &&
				visibleLines.some(
					line =>
						line.bbox.width > 0.5 &&
						lineCenterY(line) > lineCenterY(lastCode) &&
						lineCenterY(line) < lineCenterY(layerLine),
				);
			if (
				!marker ||
				priorCode.length < 2 ||
				interruptedByProse ||
				unclosedBraceDepth(priorCode.map(line => line.text).join('\n')) === 0
			)
				continue;
		}
		recoveredClosers.push({
			text: '}',
			confidence: Math.min(marker?.confidence ?? 0.8, 0.8),
			bbox: layerLine.bbox,
			source: 'text-layer',
		});
	}
	const availableLines = [...visibleLines, ...recoveredClosers];
	const lineHeight = median(availableLines.map(line => line.bbox.height));
	const fragments = new Map<OcrLine, OcrLine[]>();
	const usedPrefixes = new Set<OcrLine>();
	const codeLines = availableLines
		.filter(line => isCodeLine(line.text))
		.map(line => {
			const prefix = visibleLines.find(
				candidate =>
					!usedPrefixes.has(candidate) &&
					/^(?:if|else|for|while|return|throw)$/i.test(candidate.text.trim()) &&
					Math.abs(lineCenterY(candidate) - lineCenterY(line)) <= 0.007 &&
					candidate.bbox.x < line.bbox.x &&
					Math.abs(line.bbox.x - (candidate.bbox.x + candidate.bbox.width)) <= 0.025,
			);
			if (!prefix) return line;
			usedPrefixes.add(prefix);
			const merged: OcrLine = {
				text: `${prefix.text.trim()} ${line.text.trim()}`,
				confidence: Math.min(prefix.confidence, line.confidence),
				bbox: mergeBBoxes([prefix, line]),
			};
			fragments.set(merged, [prefix, line]);
			return merged;
		});
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
		const possiblePageEnd =
			group.length === 1 &&
			group.some(line => line.bbox.y >= 0.85 && /\b\w+\s+\w+\s*\([^)]*\)\s*\{/.test(line.text));
		if (
			(group.length < minimumLines && !possiblePageEnd) ||
			!group.some(line => CODE_STRUCTURE_REGEX.test(line.text))
		) {
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
		const blockLines = [...group.flatMap(line => fragments.get(line) ?? [line]), ...lineNumbers];
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
