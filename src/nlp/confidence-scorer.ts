import dictionaryEn from 'dictionary-en';
import nspell, { type NSpellInstance } from 'nspell';
import type { ConfidenceBreakdown, OcrLine, OcrPageResult, TextBlock } from '../types/domain.ts';
import type { RawReconstructedBlock } from './layout-reconstructor.ts';

let cachedSpellChecker: NSpellInstance | null = null;

function getSpellChecker(): NSpellInstance {
	if (!cachedSpellChecker) {
		cachedSpellChecker = nspell(dictionaryEn);
	}
	return cachedSpellChecker;
}

const ALLOWED_ALPHANUMERIC_TOKENS = new Set([
	'o1',
	'3d',
	'2d',
	'v1',
	'v2',
	'v3',
	'v4',
	'v8',
	'utf8',
	'utf16',
	'base64',
	'int32',
	'int64',
	'uint8',
	'uint32',
	'float32',
	'float64',
	'arm64',
	'x86',
	'x64',
	'mp3',
	'mp4',
	'h264',
	'sha256',
	'md5',
	'1st',
	'2nd',
	'3rd',
	'4th',
	'5th',
	'6th',
	'7th',
	'8th',
	'9th',
	'10th',
]);

const VALID_SINGLE_LETTER_WORDS = new Set(['a', 'A', 'i', 'I', 'o', 'O', 'Θ', 'θ', 'Ω', 'ω']);

const DIGIT_INSIDE_WORD_REGEX = /\b[A-Za-z]+[0-9]+[A-Za-z0-9]*\b|\b[0-9]+[A-Za-z]+[0-9]+\b/g;
const STRAY_OCR_NOISE_REGEX = /[~^§¤¦¨´¸¬°±]|\|{2,}|(?<=[a-z])\|(?=[a-z])/g;
const IMPOSSIBLE_CONSONANTS_REGEX = /\b[A-Za-z]*[bcdfghjklmnpqrstvwxz]{6,}[A-Za-z]*\b/gi;
const REPEATED_PUNCTUATION_REGEX = /(?:[,;:]{2,}|[!?]{4,}|\.{4,})/g;
// Text OCR confidence does not reliably establish the correctness of symbols or exponents.
const MATHEMATICAL_NOTATION_REGEX =
	/(?:\b[O0]\s*\([^\n)]{1,80}\)|[ΘΩθω]\s*\(|[⁰¹²³⁴⁵⁶⁷⁸⁹¼½¾⅓⅔⅛⅜⅝⅞]|\b[A-Za-z]\s*(?:[=<>≤≥×÷]|\/)\s*[A-Za-z0-9]|\(\s*\d+\s*,\s*\d+\s*\)|\b\d+\s*[+*×÷]\s*\d+|\b\d+[°ºª"])/u;

export function containsMathematicalNotation(text: string): boolean {
	return MATHEMATICAL_NOTATION_REGEX.test(text);
}

function clamp01(val: number): number {
	return Math.min(1, Math.max(0, val));
}

function round4(val: number): number {
	return Math.round(val * 10000) / 10000;
}

export function extractWords(text: string): string[] {
	const rawTokens = text.match(/[\p{L}\p{N}'’_-]+/gu) ?? [];
	return rawTokens
		.map(token => token.replace(/^['’_-]+|['’_-]+$/g, ''))
		.filter(token => token.length > 0);
}

export function buildBookVocabulary(
	pages: readonly OcrPageResult[],
	minFrequency = 3,
): ReadonlySet<string> {
	const counts = new Map<string, number>();
	for (const page of pages) {
		for (const line of page.lines) {
			for (const word of extractWords(line.text)) {
				if (/^\p{L}{3,}$/u.test(word)) {
					const lower = word.toLowerCase();
					counts.set(lower, (counts.get(lower) ?? 0) + 1);
				}
			}
		}
	}

	const frequent = new Set<string>();
	for (const [word, count] of counts) {
		if (count >= minFrequency) {
			frequent.add(word);
		}
	}
	return frequent;
}

function splitIdentifierSubwords(word: string): string[] {
	return word
		.replace(/([a-z])([A-Z])/g, '$1 $2')
		.replace(/([A-Z]+)([A-Z][a-z])/g, '$1 $2')
		.split(/[\s_-]+/)
		.filter(Boolean);
}

export function isWordLexicallyValid(
	rawWord: string,
	spell: NSpellInstance,
	bookVocab: ReadonlySet<string>,
): boolean {
	if (/^\d+(?:[.,]\d+)*$/.test(rawWord)) {
		return true;
	}
	if (/^(?:[IVXLCDM]+|[ivxlcdm]+)$/.test(rawWord) && rawWord.length <= 6) {
		return true;
	}
	if (rawWord.length === 1) {
		return VALID_SINGLE_LETTER_WORDS.has(rawWord) || /^[A-Z]$/.test(rawWord);
	}

	const normalizedApostrophe = rawWord.replace(/’/g, "'");
	const strippedPossessive = normalizedApostrophe.replace(/'s$/i, '');
	const lower = strippedPossessive.toLowerCase();

	if (ALLOWED_ALPHANUMERIC_TOKENS.has(lower)) {
		return true;
	}
	if (bookVocab.has(lower)) {
		return true;
	}
	if (spell.correct(strippedPossessive) || spell.correct(lower)) {
		return true;
	}
	if (lower.endsWith('s') && lower.length > 4) {
		const singular = lower.slice(0, -1);
		if (spell.correct(singular) || bookVocab.has(singular)) {
			return true;
		}
	}

	const subwords = splitIdentifierSubwords(strippedPossessive);
	if (subwords.length > 1) {
		return subwords.every(
			sub =>
				sub.length <= 1 ||
				spell.correct(sub) ||
				spell.correct(sub.toLowerCase()) ||
				bookVocab.has(sub.toLowerCase()),
		);
	}

	return false;
}

export function computeOcrLineScore(lines: readonly OcrLine[]): number {
	if (lines.length === 0) {
		return 1;
	}
	let weightedSum = 0;
	let totalWeight = 0;
	let minLineConf = 1;

	for (const line of lines) {
		const weight = Math.max(1, line.text.length);
		weightedSum += line.confidence * weight;
		totalWeight += weight;
		minLineConf = Math.min(minLineConf, line.confidence);
	}

	const weightedMean = totalWeight > 0 ? weightedSum / totalWeight : 1;
	return clamp01(weightedMean * 0.7 + minLineConf * 0.3);
}

export interface GarbleAnalysis {
	score: number;
	anomalyCount: number;
}

function hasPossibleSectionHeading(block: RawReconstructedBlock): boolean {
	if (block.kind !== 'paragraph') return false;
	const first = block.lines[0];
	if (!first) return false;
	const titlePattern = /^[\p{Lu}][\p{L}\p{N}\s,&:'’–-]{2,75}$/u;
	const bullet = first.text.trim().match(/^[•◦▪▸]\s+(.+)$/u);
	if (bullet?.[1] && titlePattern.test(bullet[1].trim())) return true;
	const next = block.lines[1];
	if (!next || !titlePattern.test(first.text.trim())) return false;
	const gap = next.bbox.y - (first.bbox.y + first.bbox.height);
	return next.bbox.width > first.bbox.width * 1.35 && gap >= first.bbox.height * 0.35;
}

export function analyzeGarbleHeuristics(text: string): GarbleAnalysis {
	if (!text.trim()) {
		return { score: 1, anomalyCount: 0 };
	}

	const normalizedText = text.replace(/\s+\.{3,}\s+(?:\d{1,4}|[ivxlcdm]{1,8})$/i, '');
	let anomalyCount = 0;

	const digitInsideMatches = normalizedText.match(DIGIT_INSIDE_WORD_REGEX) ?? [];
	for (const match of digitInsideMatches) {
		if (!ALLOWED_ALPHANUMERIC_TOKENS.has(match.toLowerCase())) {
			anomalyCount += 1;
		}
	}

	const straySymbols = normalizedText.match(STRAY_OCR_NOISE_REGEX) ?? [];
	anomalyCount += straySymbols.length;

	const consonantRuns = normalizedText.match(IMPOSSIBLE_CONSONANTS_REGEX) ?? [];
	anomalyCount += consonantRuns.length;

	const repeatedPunct = normalizedText.match(REPEATED_PUNCTUATION_REGEX) ?? [];
	anomalyCount += repeatedPunct.length;

	const words = text.split(/\s+/);
	let consecutiveSingleLetters = 0;
	for (const w of words) {
		const clean = w.replace(/[^\p{L}]/gu, '');
		if (clean.length === 1 && !VALID_SINGLE_LETTER_WORDS.has(clean)) {
			consecutiveSingleLetters += 1;
			if (consecutiveSingleLetters >= 2) {
				anomalyCount += 1;
			}
		} else {
			consecutiveSingleLetters = 0;
		}
	}

	const wordCount = Math.max(4, words.length);
	const penalty = (anomalyCount * 2.2) / Math.max(6, Math.sqrt(wordCount) * 3);
	return {
		score: clamp01(1 - penalty),
		anomalyCount,
	};
}

export function scoreBlockConfidence(
	block: RawReconstructedBlock,
	options: {
		threshold?: number;
		bookVocab?: ReadonlySet<string>;
		verifyTechnicalContent?: boolean;
	} = {},
): ConfidenceBreakdown {
	const threshold = options.threshold ?? 0.82;
	const bookVocab = options.bookVocab ?? new Set<string>();
	const spell = getSpellChecker();
	const isStructuredBlock = block.kind === 'code' || block.kind === 'table';

	const ocrScore = computeOcrLineScore(block.lines);

	const words = isStructuredBlock ? [] : extractWords(block.text);
	const oovWords: string[] = [];
	for (const word of words) {
		if (!isWordLexicallyValid(word, spell, bookVocab)) {
			oovWords.push(word);
		}
	}

	const effectiveWordCount = Math.max(3, words.length);
	const oovRatio = oovWords.length / effectiveWordCount;
	const dictScore = clamp01(1 - oovRatio * 1.6);

	const garble = isStructuredBlock
		? { score: 1, anomalyCount: 0 }
		: analyzeGarbleHeuristics(block.text);
	const baseComposite = clamp01(0.45 * ocrScore + 0.3 * dictScore + 0.25 * garble.score);
	const verificationReason = options.verifyTechnicalContent
		? block.kind === 'code'
			? 'code:verify'
			: block.kind === 'table'
				? 'table:verify'
				: containsMathematicalNotation(block.text)
					? 'math:verify'
					: undefined
		: undefined;
	const uncertainLayout = Boolean(
		options.verifyTechnicalContent && hasPossibleSectionHeading(block),
	);
	// Unverified technical content must not appear more certain than the review threshold.
	const composite =
		verificationReason || uncertainLayout
			? Math.min(baseComposite, Math.max(0, threshold - 0.01))
			: baseComposite;

	const reasons: string[] = [];
	if (verificationReason) {
		reasons.push(verificationReason);
	}
	if (uncertainLayout) {
		reasons.push('layout:verify');
	}
	if (
		(block.kind === 'code' || block.kind === 'table') &&
		block.lines.some(line => line.source === 'text-layer')
	) {
		reasons.push(`${block.kind}:text-layer`);
	}
	if (ocrScore < threshold) {
		reasons.push(`ocr:${ocrScore.toFixed(2)}`);
	}
	if (oovWords.length > 0) {
		const sample = oovWords.slice(0, 3).join(',');
		reasons.push(`oov:${oovWords.length}(${sample})`);
	}
	if (garble.anomalyCount > 0) {
		reasons.push(`garble:${garble.anomalyCount}`);
	}

	const flagged =
		verificationReason !== undefined ||
		uncertainLayout ||
		composite < threshold ||
		ocrScore < threshold - 0.1 ||
		garble.anomalyCount >= 2 ||
		(oovWords.length >= 2 && dictScore < 0.72);

	return {
		composite: round4(composite),
		ocr: round4(ocrScore),
		dictionary: round4(dictScore),
		garble: round4(garble.score),
		flagged,
		reasons,
	};
}

export function scorePageBlocks(
	rawBlocks: readonly RawReconstructedBlock[],
	options: {
		threshold?: number;
		bookVocab?: ReadonlySet<string>;
		verifyTechnicalContent?: boolean;
	} = {},
): TextBlock[] {
	return rawBlocks.map(raw => ({
		id: raw.id,
		kind: raw.kind,
		text: raw.text,
		bbox: raw.bbox,
		confidence: scoreBlockConfidence(raw, options),
	}));
}
