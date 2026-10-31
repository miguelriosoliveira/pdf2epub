import { z } from 'zod';

export const PAGE_STATUSES = ['clean', 'needs-review', 'reviewed', 'llm-fixed'] as const;
export type PageStatus = (typeof PAGE_STATUSES)[number];

export const BLOCK_KINDS = ['heading', 'paragraph', 'code', 'table'] as const;
export type BlockKind = (typeof BLOCK_KINDS)[number];

export const LAYOUT_PROFILES = ['auto', 'narrative', 'technical'] as const;
export type LayoutProfile = (typeof LAYOUT_PROFILES)[number];
export const LayoutProfileSchema = z.enum(LAYOUT_PROFILES);

export const OCR_ENGINES = ['macos-vision', 'tesseract'] as const;
export type OcrEngine = (typeof OCR_ENGINES)[number];

export const NormalizedBBoxSchema = z.object({
	x: z.number().min(0).max(1),
	y: z.number().min(0).max(1),
	width: z.number().min(0).max(1),
	height: z.number().min(0).max(1),
});
export type NormalizedBBox = z.infer<typeof NormalizedBBoxSchema>;

export const OcrLineSchema = z.object({
	text: z.string(),
	confidence: z.number().min(0).max(1),
	bbox: NormalizedBBoxSchema,
	source: z.enum(['text-layer']).optional(),
});
export type OcrLine = z.infer<typeof OcrLineSchema>;

export const OcrPageResultSchema = z.object({
	page: z.number().int().positive(),
	widthPx: z.number().positive(),
	heightPx: z.number().positive(),
	lines: z.array(OcrLineSchema),
	textLayerLines: z.array(OcrLineSchema).optional(),
});
export type OcrPageResult = z.infer<typeof OcrPageResultSchema>;

export const OcrBatchOutputSchema = z.object({
	totalPages: z.number().int().nonnegative(),
	pages: z.array(OcrPageResultSchema),
});
export type OcrBatchOutput = z.infer<typeof OcrBatchOutputSchema>;

export const ConfidenceBreakdownSchema = z.object({
	composite: z.number().min(0).max(1),
	ocr: z.number().min(0).max(1),
	dictionary: z.number().min(0).max(1),
	garble: z.number().min(0).max(1),
	flagged: z.boolean(),
	reasons: z.array(z.string()),
});
export type ConfidenceBreakdown = z.infer<typeof ConfidenceBreakdownSchema>;

export const TextBlockSchema = z.object({
	id: z.string().min(1),
	kind: z.enum(BLOCK_KINDS),
	text: z.string(),
	bbox: NormalizedBBoxSchema,
	confidence: ConfidenceBreakdownSchema,
	repairedByLlm: z.boolean().optional(),
});
export type TextBlock = z.infer<typeof TextBlockSchema>;

export const PageFrontmatterSchema = z.object({
	page: z.number().int().positive(),
	runningHeaders: z.array(z.string()).optional(),
	runningFooters: z.array(z.string()).optional(),
	printedPage: z.string().optional(),
	codeContinuesFromPrev: z.boolean().optional(),
	codeContinuesToNext: z.boolean().optional(),
	status: z.enum(PAGE_STATUSES),
	confidence: z.number().min(0).max(1),
	flaggedBlocks: z.number().int().nonnegative(),
	continuesFromPrev: z.boolean(),
	continuesToNext: z.boolean(),
});
export type PageFrontmatter = z.infer<typeof PageFrontmatterSchema>;

export const PageDocumentSchema = z.object({
	frontmatter: PageFrontmatterSchema,
	blocks: z.array(TextBlockSchema),
});
export type PageDocument = z.infer<typeof PageDocumentSchema>;

export const BookStatsSchema = z.object({
	cleanPages: z.number().int().nonnegative(),
	flaggedPages: z.number().int().nonnegative(),
	reviewedPages: z.number().int().nonnegative(),
	totalFlaggedBlocks: z.number().int().nonnegative(),
});
export type BookStats = z.infer<typeof BookStatsSchema>;

export const BookManifestSchema = z.object({
	title: z.string().min(1),
	author: z.string().min(1),
	language: z.string().min(2),
	sourcePdf: z.string().min(1),
	totalPages: z.number().int().positive(),
	scannedPages: z.array(z.number().int().positive()),
	threshold: z.number().min(0).max(1),
	ocrEngine: z.enum(OCR_ENGINES),
	layoutProfile: LayoutProfileSchema.optional(),
	createdAt: z.string(),
	updatedAt: z.string(),
	stats: BookStatsSchema,
});
export type BookManifest = z.infer<typeof BookManifestSchema>;

export function formatBBox(bbox: NormalizedBBox): string {
	return [bbox.x, bbox.y, bbox.width, bbox.height].map(v => v.toFixed(4)).join(',');
}

export function parseBBox(raw: string): NormalizedBBox {
	const parts = raw.split(',').map(part => Number.parseFloat(part.trim()));
	if (parts.length !== 4 || parts.some(val => Number.isNaN(val))) {
		throw new Error(`Invalid bbox string: "${raw}"`);
	}
	return NormalizedBBoxSchema.parse({
		x: Math.min(1, Math.max(0, parts[0] ?? 0)),
		y: Math.min(1, Math.max(0, parts[1] ?? 0)),
		width: Math.min(1, Math.max(0, parts[2] ?? 0)),
		height: Math.min(1, Math.max(0, parts[3] ?? 0)),
	});
}
