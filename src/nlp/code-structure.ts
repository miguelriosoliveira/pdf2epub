/** Count unclosed braces without interpreting braces in strings or comments as code. */
export function unclosedBraceDepth(text: string): number {
	let depth = 0;
	let quote = '';
	let escaped = false;
	let lineComment = false;
	let blockComment = false;
	for (let index = 0; index < text.length; index++) {
		const char = text[index];
		const next = text[index + 1];
		if (lineComment) {
			if (char === '\n') lineComment = false;
			continue;
		}
		if (blockComment) {
			if (char === '*' && next === '/') {
				blockComment = false;
				index += 1;
			}
			continue;
		}
		if (quote) {
			if (escaped) escaped = false;
			else if (char === '\\') escaped = true;
			else if (char === quote) quote = '';
			continue;
		}
		if (char === '/' && next === '/') {
			lineComment = true;
			index += 1;
		} else if (char === '/' && next === '*') {
			blockComment = true;
			index += 1;
		} else if (char === '"' || char === "'" || char === '`') {
			quote = char;
		} else if (char === '{') {
			depth += 1;
		} else if (char === '}') {
			depth = Math.max(0, depth - 1);
		}
	}
	return depth;
}

/** Preserve local indentation while offsetting a continuation into its preceding braces. */
export function indentCodeContinuation(text: string, initialDepth: number): string {
	let outerDepth = initialDepth;
	let localDepth = 0;
	return text
		.split('\n')
		.map(line => {
			const closing = line.trimStart().match(/^}+/)?.[0].length ?? 0;
			outerDepth = Math.max(0, outerDepth - Math.max(0, closing - localDepth));
			const result = `${'  '.repeat(outerDepth)}${line}`;
			localDepth = unclosedBraceDepth(`${'{'.repeat(localDepth)}${line}`);
			return result;
		})
		.join('\n');
}
