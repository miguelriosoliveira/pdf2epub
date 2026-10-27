const OCR_GREEK_SYMBOL = '(?:©|Q|2|22|Θ|Ω|<:>|<>)';

export function normalizeAsymptoticNotation(text: string): string {
	let normalized = text
		.replace(/\bbig\s+[\p{L}\p{N}©ΘΩ<>:]+\s*\(\s*theta\s*\)/giu, 'big Θ (theta)')
		.replace(/\bbig\s+[\p{L}\p{N}©ΘΩ<>:]+\s*\(\s*omega\s*\)/giu, 'big Ω (omega)')
		.replace(/(^|[•*-]\s*)[\p{L}\p{N}©ΘΩ<>:]+\s*(\(\s*big\s+theta\s*\))/gimu, '$1Θ $2')
		.replace(/(^|[•*-]\s*)[\p{L}\p{N}©ΘΩ<>:]+\s*(\(\s*big\s+omega\s*\))/gimu, '$1Ω $2')
		.replace(
			new RegExp(
				`${OCR_GREEK_SYMBOL}\\s+means both\\s+(?:0|O)\\s+and\\s+(?:S|Q|2|22|©|<:>|<>)`,
				'giu',
			),
			'Θ means both O and Ω',
		)
		.replace(
			new RegExp(`In academia,\\s*${OCR_GREEK_SYMBOL}\\s+is the equivalent concept`, 'giu'),
			'In academia, Ω is the equivalent concept',
		)
		.replace(/\b[0o](?=\s*\()/g, 'O');

	if (/\(\s*big\s+omega\s*\)/iu.test(normalized)) {
		normalized = normalized
			.replace(/\b(?:2|22|Q)\s*(?=\()/giu, 'Ω')
			.replace(/(Ω\s*\(\s*)1og(?=\s*N\b)/giu, '$1log');
	}

	if (/\(\s*big\s+theta\s*\)/iu.test(normalized)) {
		normalized = normalized
			.replace(/©(?=\s+gives\b)/gu, 'Θ')
			.replace(/\bQ\s*(?=\(\s*N\b)/giu, 'Ω ');
	}

	return normalized;
}
