import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { parse as parseYaml, stringify as stringifyYaml } from 'yaml';
import {
	type BookManifest,
	BookManifestSchema,
	type BookStats,
	formatBBox,
	type PageDocument,
	PageFrontmatterSchema,
	parseBBox,
	type TextBlock,
} from '../types/domain.ts';

const FRONTMATTER_REGEX = /^---\r?\n([\s\S]*?)\r?\n---\r?\n?([\s\S]*)$/;
const FLAGGED_ANNOTATION_REGEX = /^<!--\s*@flagged\s+([^>]*?)\s*-->$/;
const FLAGGED_CLOSE_REGEX = /^<!--\s*\/@flagged\s*-->$/;
const ATTR_PAIR_REGEX = /([a-zA-Z]+)="([^"]*)"/g;

export function formatPageFilename(page: number): string {
	return `page-${String(page).padStart(4, '0')}.md`;
}

function escapeAttrValue(value: string): string {
	return value.replace(/"/g, "'").replace(/-->/g, '--');
}

function renderBlockMarkdown(block: TextBlock): string {
	const content = block.kind === 'heading' ? `# ${block.text.trim()}` : block.text.trim();
	if (!block.confidence.flagged) {
		return content;
	}

	const reasons = escapeAttrValue(block.confidence.reasons.join(';'));
	const header = [
		'<!-- @flagged',
		`id="${escapeAttrValue(block.id)}"`,
		`score="${block.confidence.composite.toFixed(2)}"`,
		`ocr="${block.confidence.ocr.toFixed(2)}"`,
		`dict="${block.confidence.dictionary.toFixed(2)}"`,
		`garble="${block.confidence.garble.toFixed(2)}"`,
		`reasons="${reasons}"`,
		`bbox="${formatBBox(block.bbox)}"`,
		'-->',
	].join(' ');

	return `${header}\n${content}\n<!-- /@flagged -->`;
}

export function serializePageDocument(doc: PageDocument): string {
	const yamlFrontmatter = stringifyYaml(doc.frontmatter).trimEnd();
	const bodyBlocks = doc.blocks.map(renderBlockMarkdown).join('\n\n');
	return bodyBlocks
		? `---\n${yamlFrontmatter}\n---\n\n${bodyBlocks}\n`
		: `---\n${yamlFrontmatter}\n---\n`;
}

function parseAttributes(attrString: string): Record<string, string> {
	const attrs: Record<string, string> = {};
	for (const match of attrString.matchAll(ATTR_PAIR_REGEX)) {
		const key = match[1];
		const val = match[2];
		if (key !== undefined && val !== undefined) {
			attrs[key] = val;
		}
	}
	return attrs;
}

function parseTextContentToBlock(
	rawContent: string,
	fallbackId: string,
	attrs?: Record<string, string>,
): TextBlock {
	const trimmed = rawContent.trim();
	const isHeading = /^#{1,3}\s+/.test(trimmed);
	const text = isHeading ? trimmed.replace(/^#{1,3}\s+/, '').trim() : trimmed;

	if (!attrs) {
		return {
			id: fallbackId,
			kind: isHeading ? 'heading' : 'paragraph',
			text,
			bbox: { x: 0, y: 0, width: 1, height: 0.05 },
			confidence: {
				composite: 1,
				ocr: 1,
				dictionary: 1,
				garble: 1,
				flagged: false,
				reasons: [],
			},
		};
	}

	const score = Number.parseFloat(attrs.score ?? '0.5');
	const ocr = Number.parseFloat(attrs.ocr ?? String(score));
	const dict = Number.parseFloat(attrs.dict ?? String(score));
	const garble = Number.parseFloat(attrs.garble ?? '1');
	const reasons = (attrs.reasons ?? '')
		.split(/[;,]/)
		.map(s => s.trim())
		.filter(Boolean);
	const bbox = attrs.bbox ? parseBBox(attrs.bbox) : { x: 0, y: 0, width: 1, height: 0.05 };

	return {
		id: attrs.id || fallbackId,
		kind: isHeading ? 'heading' : 'paragraph',
		text,
		bbox,
		confidence: {
			composite: Number.isNaN(score) ? 0.5 : score,
			ocr: Number.isNaN(ocr) ? 0.5 : ocr,
			dictionary: Number.isNaN(dict) ? 0.5 : dict,
			garble: Number.isNaN(garble) ? 1 : garble,
			flagged: true,
			reasons,
		},
	};
}

export function parsePageDocument(markdown: string): PageDocument {
	const fmMatch = markdown.match(FRONTMATTER_REGEX);
	if (!fmMatch?.[1]) {
		throw new Error('Markdown file is missing valid YAML frontmatter');
	}

	const rawYaml: unknown = parseYaml(fmMatch[1]);
	const frontmatter = PageFrontmatterSchema.parse(rawYaml);
	const body = (fmMatch[2] ?? '').trim();

	if (!body) {
		return { frontmatter, blocks: [] };
	}

	const lines = body.split(/\r?\n/);
	const blocks: TextBlock[] = [];

	let index = 0;
	let blockCounter = 1;

	while (index < lines.length) {
		const currentLine = lines[index]?.trim() ?? '';
		if (!currentLine) {
			index += 1;
			continue;
		}

		const flaggedOpen = currentLine.match(FLAGGED_ANNOTATION_REGEX);
		if (flaggedOpen?.[1] !== undefined) {
			const attrs = parseAttributes(flaggedOpen[1]);
			index += 1;
			const innerLines: string[] = [];
			while (index < lines.length) {
				const candidate = lines[index] ?? '';
				if (FLAGGED_CLOSE_REGEX.test(candidate.trim())) {
					index += 1;
					break;
				}
				innerLines.push(candidate);
				index += 1;
			}
			const content = innerLines.join('\n').trim();
			if (content) {
				blocks.push(
					parseTextContentToBlock(content, `p${frontmatter.page}-b${blockCounter}`, attrs),
				);
				blockCounter += 1;
			}
			continue;
		}

		const paraLines: string[] = [];
		while (index < lines.length) {
			const line = lines[index] ?? '';
			const trimmed = line.trim();
			if (!trimmed || FLAGGED_ANNOTATION_REGEX.test(trimmed)) {
				break;
			}
			paraLines.push(line);
			index += 1;
		}
		const content = paraLines.join('\n').trim();
		if (content) {
			blocks.push(parseTextContentToBlock(content, `p${frontmatter.page}-b${blockCounter}`));
			blockCounter += 1;
		}
	}

	return {
		frontmatter: {
			...frontmatter,
			flaggedBlocks: blocks.filter(b => b.confidence.flagged).length,
		},
		blocks,
	};
}

export function writePageFile(bookDir: string, doc: PageDocument): string {
	const pagesDir = join(bookDir, 'pages');
	mkdirSync(pagesDir, { recursive: true });
	const filePath = join(pagesDir, formatPageFilename(doc.frontmatter.page));
	writeFileSync(filePath, serializePageDocument(doc), 'utf8');
	return filePath;
}

export function readPageFile(bookDir: string, page: number): PageDocument {
	const filePath = join(bookDir, 'pages', formatPageFilename(page));
	const raw = readFileSync(filePath, 'utf8');
	return parsePageDocument(raw);
}

export function loadAllPageDocuments(bookDir: string): PageDocument[] {
	const pagesDir = join(bookDir, 'pages');
	if (!existsSync(pagesDir)) {
		return [];
	}
	const files = readdirSync(pagesDir)
		.filter(f => /^page-\d+\.md$/.test(f))
		.sort();

	return files.map(file => {
		const raw = readFileSync(join(pagesDir, file), 'utf8');
		return parsePageDocument(raw);
	});
}

export function computeBookStats(docs: readonly PageDocument[]): BookStats {
	let cleanPages = 0;
	let flaggedPages = 0;
	let reviewedPages = 0;
	let totalFlaggedBlocks = 0;

	for (const doc of docs) {
		const flaggedCount = doc.blocks.filter(b => b.confidence.flagged).length;
		totalFlaggedBlocks += flaggedCount;
		if (doc.frontmatter.status === 'reviewed' || doc.frontmatter.status === 'llm-fixed') {
			reviewedPages += 1;
		} else if (doc.frontmatter.status === 'needs-review' || flaggedCount > 0) {
			flaggedPages += 1;
		} else {
			cleanPages += 1;
		}
	}

	return {
		cleanPages,
		flaggedPages,
		reviewedPages,
		totalFlaggedBlocks,
	};
}

export function writeBookManifest(bookDir: string, manifest: BookManifest): string {
	const manifestPath = join(bookDir, 'book.json');
	mkdirSync(dirname(manifestPath), { recursive: true });
	writeFileSync(manifestPath, `${JSON.stringify(manifest, null, '\t')}\n`, 'utf8');
	return manifestPath;
}

export function readBookManifest(bookDir: string): BookManifest {
	const manifestPath = join(bookDir, 'book.json');
	const raw: unknown = JSON.parse(readFileSync(manifestPath, 'utf8'));
	return BookManifestSchema.parse(raw);
}

export function syncManifestStats(bookDir: string): BookManifest {
	const manifest = readBookManifest(bookDir);
	const docs = loadAllPageDocuments(bookDir);
	const updated: BookManifest = {
		...manifest,
		scannedPages: docs.map(d => d.frontmatter.page),
		updatedAt: new Date().toISOString(),
		stats: computeBookStats(docs),
	};
	writeBookManifest(bookDir, updated);
	return updated;
}
