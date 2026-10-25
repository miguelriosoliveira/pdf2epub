import { mkdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { z } from 'zod';
import {
	loadAllPageDocuments,
	readBookManifest,
	syncManifestStats,
	writePageFile,
} from '../markdown/page-serializer.ts';
import { createOcrProvider } from '../ocr/index.ts';
import type { PageDocument, TextBlock } from '../types/domain.ts';

export interface FlaggedBlockCandidate {
	page: number;
	blockIndex: number;
	block: TextBlock;
	prevContext: string;
	nextContext: string;
}

export interface LlmCostEstimate {
	candidateBlocks: number;
	selectedBlocks: number;
	estimatedInputTokens: number;
	estimatedOutputTokens: number;
	estimatedCostUsd: number;
}

export interface LlmRepairOptions {
	bookDir: string;
	page?: number;
	blockId?: string;
	maxBlocks?: number;
	dryRun?: boolean;
	model?: string;
	apiKey?: string;
	baseUrl?: string;
}

export interface LlmRepairResult {
	estimate: LlmCostEstimate;
	repairedBlocks: number;
	touchedPages: number[];
}

const OpenAiChatResponseSchema = z.object({
	choices: z.array(
		z.object({
			message: z.object({
				content: z.string().nullable().optional(),
			}),
		}),
	),
});

const AnthropicMessageResponseSchema = z.object({
	content: z.array(
		z.object({
			type: z.string(),
			text: z.string().optional(),
		}),
	),
});

export function collectFlaggedCandidates(
	docs: readonly PageDocument[],
	filter: { page?: number; blockId?: string } = {},
): FlaggedBlockCandidate[] {
	const candidates: FlaggedBlockCandidate[] = [];

	for (const doc of docs) {
		if (filter.page !== undefined && doc.frontmatter.page !== filter.page) {
			continue;
		}
		for (let idx = 0; idx < doc.blocks.length; idx++) {
			const block = doc.blocks[idx];
			if (!block?.confidence.flagged) {
				continue;
			}
			if (filter.blockId !== undefined && block.id !== filter.blockId) {
				continue;
			}
			const prevContext = doc.blocks[idx - 1]?.text.slice(-180) ?? '';
			const nextContext = doc.blocks[idx + 1]?.text.slice(0, 180) ?? '';
			candidates.push({
				page: doc.frontmatter.page,
				blockIndex: idx,
				block,
				prevContext,
				nextContext,
			});
		}
	}

	candidates.sort((a, b) => a.block.confidence.composite - b.block.confidence.composite);
	return candidates;
}

export function estimateLlmRepairCost(
	candidates: readonly FlaggedBlockCandidate[],
	maxBlocks = 25,
): LlmCostEstimate {
	const selected = candidates.slice(0, maxBlocks);
	let inputTokens = 0;
	let outputTokens = 0;

	for (const item of selected) {
		const textTokens = Math.ceil(
			(item.block.text.length + item.prevContext.length + item.nextContext.length) / 4,
		);
		const croppedImageTokens = 260;
		const promptOverhead = 95;
		inputTokens += textTokens + croppedImageTokens + promptOverhead;
		outputTokens += Math.ceil(item.block.text.length / 4) + 15;
	}

	// Default estimate using gpt-4o-mini / claude-haiku pricing tier ($0.15 / 1M input, $0.60 / 1M output)
	const costUsd = (inputTokens * 0.15 + outputTokens * 0.6) / 1_000_000;

	return {
		candidateBlocks: candidates.length,
		selectedBlocks: selected.length,
		estimatedInputTokens: inputTokens,
		estimatedOutputTokens: outputTokens,
		estimatedCostUsd: Math.round(costUsd * 100000) / 100000,
	};
}

async function callMultimodalRepairModel(params: {
	croppedPngBase64: string;
	noisyText: string;
	prevContext: string;
	nextContext: string;
	model: string;
	apiKey: string;
	baseUrl?: string;
}): Promise<string> {
	const systemPrompt = [
		'You are an exact OCR transcription repair assistant for a scanned book.',
		'Fix only OCR typos, garbled characters, and broken word splits in the target paragraph using the cropped scan image and surrounding context.',
		'Return ONLY the corrected target paragraph text with no markdown fences, commentary, or explanations.',
	].join(' ');

	const userPrompt = [
		params.prevContext ? `Previous context: "${params.prevContext}"` : '',
		`Noisy OCR target block: "${params.noisyText}"`,
		params.nextContext ? `Following context: "${params.nextContext}"` : '',
	]
		.filter(Boolean)
		.join('\n\n');

	const isAnthropic =
		params.model.startsWith('claude-') ||
		(!process.env.OPENAI_API_KEY && Boolean(process.env.ANTHROPIC_API_KEY));

	if (isAnthropic) {
		const endpoint = `${params.baseUrl ?? 'https://api.anthropic.com'}/v1/messages`;
		const response = await fetch(endpoint, {
			method: 'POST',
			headers: {
				'content-type': 'application/json',
				'x-api-key': params.apiKey,
				'anthropic-version': '2023-06-01',
			},
			body: JSON.stringify({
				model: params.model,
				max_tokens: 1024,
				system: systemPrompt,
				messages: [
					{
						role: 'user',
						content: [
							{
								type: 'image',
								source: {
									type: 'base64',
									media_type: 'image/png',
									data: params.croppedPngBase64,
								},
							},
							{
								type: 'text',
								text: userPrompt,
							},
						],
					},
				],
			}),
		});

		if (!response.ok) {
			throw new Error(`Anthropic API error (${response.status}): ${await response.text()}`);
		}
		const parsed = AnthropicMessageResponseSchema.parse(await response.json());
		const text = parsed.content.find(c => c.type === 'text')?.text?.trim();
		if (!text) {
			throw new Error('Anthropic API returned empty content');
		}
		return text;
	}

	const endpoint = `${params.baseUrl ?? 'https://api.openai.com/v1'}/chat/completions`;
	const response = await fetch(endpoint, {
		method: 'POST',
		headers: {
			'content-type': 'application/json',
			authorization: `Bearer ${params.apiKey}`,
		},
		body: JSON.stringify({
			model: params.model,
			temperature: 0,
			messages: [
				{ role: 'system', content: systemPrompt },
				{
					role: 'user',
					content: [
						{ type: 'text', text: userPrompt },
						{
							type: 'image_url',
							image_url: {
								url: `data:image/png;base64,${params.croppedPngBase64}`,
								detail: 'low',
							},
						},
					],
				},
			],
		}),
	});

	if (!response.ok) {
		throw new Error(`OpenAI API error (${response.status}): ${await response.text()}`);
	}
	const parsed = OpenAiChatResponseSchema.parse(await response.json());
	const text = parsed.choices[0]?.message.content?.trim();
	if (!text) {
		throw new Error('OpenAI API returned empty content');
	}
	return text;
}

export async function repairFlaggedBlocks(options: LlmRepairOptions): Promise<LlmRepairResult> {
	const manifest = readBookManifest(options.bookDir);
	const docs = loadAllPageDocuments(options.bookDir);
	const docByPage = new Map<number, PageDocument>(docs.map(d => [d.frontmatter.page, d]));

	const filter: { page?: number; blockId?: string } = {};
	if (options.page !== undefined) {
		filter.page = options.page;
	}
	if (options.blockId !== undefined) {
		filter.blockId = options.blockId;
	}

	const candidates = collectFlaggedCandidates(docs, filter);
	const maxBlocks = options.maxBlocks ?? 25;
	const estimate = estimateLlmRepairCost(candidates, maxBlocks);

	if (options.dryRun || estimate.selectedBlocks === 0) {
		return {
			estimate,
			repairedBlocks: 0,
			touchedPages: [],
		};
	}

	const apiKey = options.apiKey ?? process.env.OPENAI_API_KEY ?? process.env.ANTHROPIC_API_KEY;
	if (!apiKey) {
		throw new Error(
			'No LLM API key found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY, or run with --dry-run to inspect estimated cost.',
		);
	}

	const defaultModel = process.env.OPENAI_API_KEY ? 'gpt-4o-mini' : 'claude-3-5-haiku-latest';
	const model = options.model ?? process.env.PDF2EPUB_LLM_MODEL ?? defaultModel;

	const ocrProvider = createOcrProvider(manifest.ocrEngine);
	const cropsDir = join(options.bookDir, '.cache/crops');
	mkdirSync(cropsDir, { recursive: true });

	const selected = candidates.slice(0, maxBlocks);
	const touchedPages = new Set<number>();
	let repairedBlocks = 0;

	for (const candidate of selected) {
		const cropPath = join(cropsDir, `${candidate.block.id}.png`);
		await ocrProvider.renderPageImage({
			pdfPath: manifest.sourcePdf,
			page: candidate.page,
			outputPath: cropPath,
			dpi: 220,
			crop: candidate.block.bbox,
		});

		const croppedPngBase64 = readFileSync(cropPath).toString('base64');
		const callParams: {
			croppedPngBase64: string;
			noisyText: string;
			prevContext: string;
			nextContext: string;
			model: string;
			apiKey: string;
			baseUrl?: string;
		} = {
			croppedPngBase64,
			noisyText: candidate.block.text,
			prevContext: candidate.prevContext,
			nextContext: candidate.nextContext,
			model,
			apiKey,
		};
		if (options.baseUrl !== undefined) {
			callParams.baseUrl = options.baseUrl;
		}

		const repairedText = await callMultimodalRepairModel(callParams);

		const pageDoc = docByPage.get(candidate.page);
		const targetBlock = pageDoc?.blocks[candidate.blockIndex];
		if (pageDoc && targetBlock) {
			targetBlock.text = repairedText;
			targetBlock.repairedByLlm = true;
			targetBlock.confidence = {
				...targetBlock.confidence,
				composite: 0.99,
				flagged: false,
				reasons: ['llm-repaired'],
			};
			touchedPages.add(candidate.page);
			repairedBlocks += 1;
		}
	}

	for (const pageNum of touchedPages) {
		const pageDoc = docByPage.get(pageNum);
		if (!pageDoc) {
			continue;
		}
		const remainingFlagged = pageDoc.blocks.filter(b => b.confidence.flagged).length;
		pageDoc.frontmatter.flaggedBlocks = remainingFlagged;
		if (remainingFlagged === 0) {
			pageDoc.frontmatter.status = 'llm-fixed';
			pageDoc.frontmatter.confidence = 0.99;
		}
		writePageFile(options.bookDir, pageDoc);
	}

	syncManifestStats(options.bookDir);

	return {
		estimate,
		repairedBlocks,
		touchedPages: [...touchedPages].sort((a, b) => a - b),
	};
}
