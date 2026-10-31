#!/usr/bin/env node
import { basename, extname, resolve } from 'node:path';
import { Command } from 'commander';
import { buildEpubFromWorkspace } from '../epub/builder.ts';
import {
	collectFlaggedCandidates,
	estimateLlmRepairCost,
	repairFlaggedBlocks,
} from '../llm/repair.ts';
import {
	computeBookStats,
	loadAllPageDocuments,
	readBookManifest,
	writeBookManifest,
} from '../markdown/page-serializer.ts';
import { createOcrProvider } from '../ocr/index.ts';
import { writeOcrPagesToWorkspace } from '../pipeline/scan-writer.ts';
import { type BookManifest, LayoutProfileSchema, type OcrEngine } from '../types/domain.ts';
import { startReviewServer } from '../web/server.ts';

export function parsePagesOption(raw?: string): number[] | 'all' {
	if (!raw || raw.trim() === 'all') {
		return 'all';
	}
	const pageSet = new Set<number>();
	for (const part of raw.split(',')) {
		const trimmed = part.trim();
		if (!trimmed) {
			continue;
		}
		if (trimmed.includes('-')) {
			const [startStr, endStr] = trimmed.split('-');
			const start = Number.parseInt(startStr ?? '', 10);
			const end = Number.parseInt(endStr ?? '', 10);
			if (Number.isNaN(start) || Number.isNaN(end) || start < 1 || end < start) {
				throw new Error(`Invalid page range: "${trimmed}"`);
			}
			for (let p = start; p <= end; p++) {
				pageSet.add(p);
			}
		} else {
			const page = Number.parseInt(trimmed, 10);
			if (Number.isNaN(page) || page < 1) {
				throw new Error(`Invalid page number: "${trimmed}"`);
			}
			pageSet.add(page);
		}
	}
	return [...pageSet].sort((a, b) => a - b);
}

function inferBookMetadataFromFilename(pdfPath: string): { title: string; author: string } {
	const name = basename(pdfPath, extname(pdfPath));
	const parts = name.split(/\s+-\s+/);
	if (parts.length >= 2) {
		const author = parts[parts.length - 1]?.trim() || 'Unknown Author';
		const title = parts.slice(0, -1).join(' - ').trim() || name;
		return { title, author };
	}
	return { title: name, author: 'Unknown Author' };
}

export function createCliProgram(): Command {
	const program = new Command();

	program
		.name('pdf2epub')
		.description(
			'Convert scanned PDF books into EPUB via local OCR, NLP confidence scoring, and per-page Markdown review',
		)
		.version('0.1.0');

	program
		.command('scan')
		.description('Run local OCR + NLP confidence scoring and generate per-page Markdown files')
		.argument('<pdf>', 'Path to scanned PDF file')
		.requiredOption('-o, --out <dir>', 'Output book workspace directory')
		.option('--pages <range>', 'Page range to scan, e.g. "1-25" or "1,5,10-15"', 'all')
		.option('--lang <lang>', 'OCR language code', 'en')
		.option('--threshold <score>', 'Confidence score threshold [0..1]', '0.82')
		.option('--dpi <dpi>', 'Render DPI for OCR', '220')
		.option('--engine <engine>', 'Force OCR engine (macos-vision | tesseract)')
		.option('--profile <profile>', 'Layout profile (auto | narrative | technical)', 'auto')
		.option('--title <title>', 'Book title override')
		.option('--author <author>', 'Book author override')
		.option('--auto-llm', 'Automatically repair flagged blocks via LLM after scanning', false)
		.option('--max-llm-blocks <count>', 'Maximum flagged blocks to repair with --auto-llm', '20')
		.action(async (pdfArg: string, opts: Record<string, string | boolean | undefined>) => {
			const pdfPath = resolve(pdfArg);
			const outDir = resolve(String(opts.out));
			const pages = parsePagesOption(typeof opts.pages === 'string' ? opts.pages : 'all');
			const threshold = Number.parseFloat(String(opts.threshold ?? '0.82'));
			const dpi = Number.parseInt(String(opts.dpi ?? '220'), 10);
			const lang = String(opts.lang ?? 'en');
			const layoutProfile = LayoutProfileSchema.parse(opts.profile ?? 'auto');
			const preferredEngine =
				opts.engine === 'tesseract' || opts.engine === 'macos-vision'
					? (opts.engine satisfies OcrEngine)
					: undefined;

			const provider = createOcrProvider(preferredEngine);
			console.log(`[pdf2epub] Using OCR engine: ${provider.engine}`);
			console.log(`[pdf2epub] Scanning "${basename(pdfPath)}"...`);

			const startMs = Date.now();
			const ocrBatch = await provider.recognizePages({
				pdfPath,
				pages,
				language: lang,
				dpi,
				onPageProgress: (done, total) => {
					process.stdout.write(`\r[pdf2epub] OCR progress: ${done}/${total} pages`);
				},
			});
			process.stdout.write('\n');

			await writeOcrPagesToWorkspace({
				bookDir: outDir,
				pdfPath,
				pages: ocrBatch.pages,
				provider,
				dpi,
				threshold,
				profile: layoutProfile,
			});

			const allDocs = loadAllPageDocuments(outDir);
			const stats = computeBookStats(allDocs);
			const inferred = inferBookMetadataFromFilename(pdfPath);
			const now = new Date().toISOString();

			const manifest: BookManifest = {
				title: typeof opts.title === 'string' ? opts.title : inferred.title,
				author: typeof opts.author === 'string' ? opts.author : inferred.author,
				language: lang,
				sourcePdf: pdfPath,
				totalPages: ocrBatch.totalPages,
				scannedPages: allDocs.map(d => d.frontmatter.page),
				threshold,
				ocrEngine: provider.engine,
				layoutProfile,
				createdAt: now,
				updatedAt: now,
				stats,
			};
			writeBookManifest(outDir, manifest);

			const elapsedSec = ((Date.now() - startMs) / 1000).toFixed(1);
			const candidates = collectFlaggedCandidates(allDocs);
			const maxLlmBlocks = Number.parseInt(String(opts.maxLlmBlocks ?? '20'), 10);
			const estimate = estimateLlmRepairCost(candidates, maxLlmBlocks);

			console.log(`[pdf2epub] Completed local scan in ${elapsedSec}s -> ${outDir}`);
			console.log(
				`[pdf2epub] Pages: ${allDocs.length} scanned | ${stats.cleanPages} clean | ${stats.flaggedPages} needs-review (${stats.totalFlaggedBlocks} flagged blocks)`,
			);
			if (candidates.length > 0) {
				console.log(
					`[pdf2epub] Estimated LLM fallback cost for ${estimate.selectedBlocks}/${estimate.candidateBlocks} flagged blocks: ~$${estimate.estimatedCostUsd.toFixed(5)} (${estimate.estimatedInputTokens} in / ${estimate.estimatedOutputTokens} out tokens)`,
				);
			}

			if (opts.autoLlm && candidates.length > 0) {
				console.log(`[pdf2epub] Running --auto-llm (capped at ${maxLlmBlocks} blocks)...`);
				const repairResult = await repairFlaggedBlocks({
					bookDir: outDir,
					maxBlocks: maxLlmBlocks,
				});
				console.log(
					`[pdf2epub] Repaired ${repairResult.repairedBlocks} block(s) across pages: ${repairResult.touchedPages.join(', ')}`,
				);
				if (repairResult.reviewRequiredBlocks || repairResult.rejectedBlocks) {
					console.log(
						`[pdf2epub] ${repairResult.reviewRequiredBlocks} suggestion(s) still need review; ${repairResult.rejectedBlocks} invalid response(s) left unchanged.`,
					);
				}
			}
		});

	program
		.command('status')
		.description('Inspect review status and low-confidence blocks in a book workspace')
		.argument('<bookDir>', 'Path to book workspace directory')
		.action((bookDirArg: string) => {
			const bookDir = resolve(bookDirArg);
			const manifest = readBookManifest(bookDir);
			const docs = loadAllPageDocuments(bookDir);
			const stats = computeBookStats(docs);

			console.log(`Book: ${manifest.title} (${manifest.author})`);
			console.log(`OCR Engine: ${manifest.ocrEngine} | Threshold: ${manifest.threshold}`);
			console.log(
				`Scanned Pages: ${docs.length}/${manifest.totalPages} | Clean: ${stats.cleanPages} | Reviewed: ${stats.reviewedPages} | Needs Review: ${stats.flaggedPages} (${stats.totalFlaggedBlocks} blocks)`,
			);

			const flaggedDocs = docs.filter(d => d.frontmatter.status === 'needs-review');
			if (flaggedDocs.length === 0) {
				console.log('\nAll scanned pages are clean or reviewed. Ready to build EPUB!');
				return;
			}

			console.log('\nPages needing review:');
			for (const doc of flaggedDocs) {
				const flaggedBlocks = doc.blocks.filter(b => b.confidence.flagged);
				const reasons = flaggedBlocks.flatMap(b => b.confidence.reasons).slice(0, 4);
				console.log(
					`  - Page ${String(doc.frontmatter.page).padStart(4, ' ')} (pages/page-${String(doc.frontmatter.page).padStart(4, '0')}.md): min-score=${doc.frontmatter.confidence.toFixed(2)}, flaggedBlocks=${flaggedBlocks.length} [${reasons.join('; ')}]`,
				);
			}
		});

	program
		.command('fix-flagged')
		.description('Repair low-confidence blocks using cropped bounding-box images and an LLM')
		.argument('<bookDir>', 'Path to book workspace directory')
		.option('--page <number>', 'Only repair flagged blocks on a specific page')
		.option('--max-blocks <count>', 'Maximum number of blocks to repair', '25')
		.option('--model <model>', 'Override multimodal LLM model name')
		.option('--dry-run', 'Only print token and cost estimate without calling LLM', false)
		.action(async (bookDirArg: string, opts: Record<string, string | boolean | undefined>) => {
			const bookDir = resolve(bookDirArg);
			const maxBlocks = Number.parseInt(String(opts.maxBlocks ?? '25'), 10);
			const dryRun = Boolean(opts.dryRun);

			const repairOpts: {
				bookDir: string;
				maxBlocks: number;
				dryRun: boolean;
				page?: number;
				model?: string;
			} = {
				bookDir,
				maxBlocks,
				dryRun,
			};
			if (typeof opts.page === 'string') {
				repairOpts.page = Number.parseInt(opts.page, 10);
			}
			if (typeof opts.model === 'string') {
				repairOpts.model = opts.model;
			}

			const result = await repairFlaggedBlocks(repairOpts);
			console.log(
				`[pdf2epub] Flagged blocks: ${result.estimate.candidateBlocks} total | Selected (cap ${maxBlocks}): ${result.estimate.selectedBlocks}`,
			);
			console.log(
				`[pdf2epub] Estimated tokens: ${result.estimate.estimatedInputTokens} input / ${result.estimate.estimatedOutputTokens} output (~$${result.estimate.estimatedCostUsd.toFixed(5)})`,
			);

			if (!dryRun) {
				console.log(
					`[pdf2epub] Repaired ${result.repairedBlocks} block(s) across ${result.touchedPages.length} page(s).`,
				);
				if (result.reviewRequiredBlocks || result.rejectedBlocks) {
					console.log(
						`[pdf2epub] ${result.reviewRequiredBlocks} suggestion(s) still need review; ${result.rejectedBlocks} invalid response(s) left unchanged.`,
					);
				}
			}
		});

	program
		.command('review')
		.description('Start local side-by-side PDF & Markdown review web UI')
		.argument('<bookDir>', 'Path to book workspace directory')
		.option('-p, --port <port>', 'HTTP port', '4321')
		.action(async (bookDirArg: string, opts: { port?: string }) => {
			const bookDir = resolve(bookDirArg);
			const port = Number.parseInt(opts.port ?? '4321', 10);
			const handle = await startReviewServer({ bookDir, port });
			console.log(`[pdf2epub] Review UI running at ${handle.url}`);
			console.log('[pdf2epub] Press Ctrl+C to stop');
		});

	program
		.command('build')
		.description('Compile reviewed per-page Markdown files into an EPUB 3 archive')
		.argument('<bookDir>', 'Path to book workspace directory')
		.requiredOption('-o, --out <epubPath>', 'Output .epub file path')
		.option(
			'--allow-unreviewed',
			'Allow building EPUB even if some pages still have status needs-review',
			false,
		)
		.action(async (bookDirArg: string, opts: { out: string; allowUnreviewed?: boolean }) => {
			const result = await buildEpubFromWorkspace({
				bookDir: resolve(bookDirArg),
				outputPath: resolve(opts.out),
				allowUnreviewed: Boolean(opts.allowUnreviewed),
			});
			console.log(`[pdf2epub] Built EPUB 3: ${result.outputPath}`);
			console.log(
				`[pdf2epub] Compiled ${result.pageCount} page(s) into ${result.chapterCount} chapter(s) (${result.stitchedCrossPageParagraphs} cross-page paragraph(s) stitched).`,
			);
		});

	return program;
}

const isMainModule =
	process.argv[1] !== undefined && resolve(process.argv[1]) === resolve(import.meta.filename);

if (isMainModule) {
	const cli = createCliProgram();
	cli.parseAsync(process.argv).catch((err: unknown) => {
		const message = err instanceof Error ? err.message : String(err);
		console.error(`[pdf2epub] Error: ${message}`);
		process.exit(1);
	});
}
