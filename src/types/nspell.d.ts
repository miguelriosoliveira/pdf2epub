declare module 'nspell' {
	export interface NSpellInstance {
		correct(word: string): boolean;
		suggest(word: string): string[];
		add(word: string, model?: string): this;
	}

	export interface NSpellDictionaryInput {
		aff: Uint8Array | Buffer | string;
		dic: Uint8Array | Buffer | string;
	}

	export default function nspell(
		affOrDict: NSpellDictionaryInput | Uint8Array | Buffer | string,
		dic?: Uint8Array | Buffer | string,
	): NSpellInstance;
}
