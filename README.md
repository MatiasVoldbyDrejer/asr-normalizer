# asr-normalizer

The text normalizer behind the [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)
and the [Open TTS Leaderboard](https://huggingface.co/spaces/hf-audio/open_tts_leaderboard), as a standalone package.

Before word error rate (WER) is computed, both the reference transcript and the model's output are passed through
this normalizer, so that casing, punctuation, filler words, number formatting, and British/American spelling don't
count as recognition errors. It started as OpenAI Whisper's text normalizer, and the leaderboard has patched it
since (fillers, times, compounds, acronyms, name variants, multilingual numbers, Armenian). This package is that
code, extracted verbatim so that anyone can reproduce leaderboard numbers, or prepare data for the leaderboard,
without cloning the leaderboard repo.

## Install

```bash
pip install asr-normalizer
# before the first PyPI release:
pip install "asr-normalizer @ git+https://github.com/MatiasVoldbyDrejer/asr-normalizer@v0.1.0"
```

Python 3.9 or newer. The only runtime dependencies are `regex` and `num2words`.

## Usage

```python
from asr_normalizer import normalize, normalize_compound_pairs

normalize("I've got 2 cats, uh, at 3:00 PM")                    # 'i have got 2 cats at 3 pm'
normalize("Mr. Smith paid twenty dollars and seven cents")      # 'mister smith paid $20.07'
normalize("The B B C said fifty per cent, um, in nineteen ninety nine.")  # 'the bbc said 50% in 1999'
normalize("Ich habe 2 Katzen und 10 000 Hunde.", lang="de")     # 'ich habe zwei katzen und zehntausend hunde'
normalize("Ես ունեմ 2 կատու։", lang="hy")                        # 'ես ունեմ 2 կատու'

# The leaderboard also aligns compound-word boundaries between reference and prediction:
normalize_compound_pairs(["a white paper"], ["a whitepaper"])   # (['a whitepaper'], ['a whitepaper'])
```

`normalize(text, lang="en")` does exactly what the leaderboard does at scoring time: `lang="en"` uses
`EnglishTextNormalizer()`, and any other value uses `MultilingualNormalizer(remove_diacritics=False)(text, lang=lang)`.
The classes are exported too, with the same names, constructor arguments, and call signatures as in the
leaderboard: `EnglishTextNormalizer`, `BasicTextNormalizer`, `BasicMultilingualTextNormalizer`,
`MultilingualNormalizer`, `ArmenianTextNormalizer`, plus `FILLER_WORDS`.

From the command line (one transcript per line in, one normalized line out, UTF-8, stdin if no file is given):

```bash
echo "I've got 2 cats, uh, at 3:00 PM" | asr-normalize      # i have got 2 cats at 3 pm
asr-normalize --lang de references.txt > references.norm.txt
```

## What the English normalizer does

`EnglishTextNormalizer` applies these steps in order. If you transcribe data for the leaderboard, this is also a
guide to what does and doesn't matter: anything a step below erases or rewrites is a free choice for you, and
everything else is scored.

1. Lowercases everything.
2. Deletes anything in square or angle brackets (`[laughter]`, `<unk>`) and in parentheses (`(inaudible)`).
   Spoken words must therefore never be put inside brackets or parentheses.
3. Deletes filler words and hesitations, including stretched spellings: uh, um, er, erm, ah, eh, em, hm, hmm, mm,
   mhm, huh, heh, hah, ugh, tsk, aha, ah-ha, ahem, hum, hunh, mmkay (`uhhh`, `ummm` and `hmmmm` count too).
   Transcribing them or leaving them out makes no difference.
4. Removes a space before an apostrophe (`don 't` becomes `don't`).
5. Rewrites:
   - on-the-hour times (`3:00` and `11:00 PM` become `3` and `11 pm`);
   - colloquial forms: won't, can't, let's, ain't, y'all, wanna, gotta, gonna, i'ma/imma, woulda/coulda/shoulda,
     ma'am (for example `won't` becomes `will not` and `gonna` becomes `going to`);
   - titles: mr, mrs, dr, st, prof, capt, gov, ald, gen, sen, rep, pres, rev, hon, asst, assoc, lt, col, jr, sr,
     esq (for example `dr` becomes `doctor` and `st` becomes `saint`);
   - `'d been`, `'s been`, `'d gone`, `'s gone`, `'d done` and `'s got` become `had …` or `has …`;
   - other contractions: `n't` becomes `not`, `'re` becomes `are`, `'d` becomes `would`, `'ll` becomes `will`,
     `'ve` becomes `have` and `'m` becomes `am`. `'s` becomes `is` only after it, he, she, what, that, who, here,
     there, how, when, where, why and this. Every other `'s` (possessives like `the dog's`) is split off at
     step 8 and becomes `dog s`.
6. Removes commas between digits (`1,000` becomes `1000`).
7. Removes periods that aren't followed by a digit (`3.14` is kept).
8. Strips accents and diacritics (`résumé` becomes `resume`, `straße` becomes `strasse`, and the same goes for
   æ, ø, œ, ł, đ, ð and þ) and turns every other symbol and punctuation mark (hyphens, quotes, apostrophes, slashes
   and so on) into a space. Only `. % $ ¢ € £` are kept, for numbers.
9. Joins known compounds and fixes spellings that vary freely: et cetera, al right/all right, how ever, wi fi,
   hi fi, lo fi, sci fi, e mail, e book, e commerce, x ray, t shirt, a m, p m, o k, and stretched `ohhh`/`oooo`
   (so `wi-fi` becomes `wifi`, `all right` becomes `alright`, `p.m.` becomes `pm` and `o k` becomes `okay`).
10. Turns spoken numbers into digits:
    - `twenty five` becomes `25`, `nineteen ninety nine` becomes `1999`, `one oh one` becomes `101`,
      `three point one four` becomes `3.14`, `double five` becomes `55` and `five and a half` becomes `5.5`;
    - ordinals and plurals keep their suffix (`twenty first` becomes `21st` and `sixties` becomes `60s`);
    - `minus`, `negative`, `plus` and `positive` become a sign (`minus seven` becomes `-7`);
    - currency words after a number become a symbol in front of it (`twenty dollars and seven cents` becomes
      `$20.07`, and `$2 and 7 cents` also becomes `$2.07`);
    - `percent` and `per cent` become `%`;
    - "oh" is read as zero only inside a run of digits (`oh I see` is left alone);
    - a lone `1` is written as the word `one`, and `1s` as `ones`.

    So digits and words are equivalent: write whichever is natural.
11. Converts British spellings to American (about 1,740 entries: `colour` becomes `color`, `organise` becomes
    `organize`), and also `ok`/`kay` to `okay` and `etcetera` to `etc`.
12. Converts name spelling variants to one canonical form (152 entries: `allan` becomes `allen` and `kris` becomes
    `chris`).
13. Joins runs of single letters or digits into one token (`b b c` becomes `bbc` and `5 g` becomes `5g`). A run
    containing the words "a" or "i" needs at least 3 tokens, so `a i` stays as it is.
14. Removes `. $ ¢ € £` where no digit follows and `%` where no digit comes before it.
15. Collapses all whitespace to single spaces.

Scoring also passes each normalized reference/prediction pair through `normalize_compound_pairs`, which joins a
compound written as one word on one side and as several words on the other (`white paper` against `whitepaper`).

## Language support

| `lang` | Normalizer | What it does |
|---|---|---|
| `en` | `EnglishTextNormalizer` | The full pipeline above. |
| `hy` | `ArmenianTextNormalizer` | A dedicated Armenian normalizer. It handles Armenian punctuation and quotes, and canonicalizes written and spoken numbers, ordinals, percentages, currency and clock times (`H։MM`). |
| any other code, or `None` | `MultilingualNormalizer(remove_diacritics=False)` | Lowercases. Removes `[...]`, `<...>` and `(...)`. Applies Unicode NFKC and replaces symbols and punctuation with spaces, but keeps combining marks (Devanagari vowel signs, for example) and diacritics. Drops other non-word characters and collapses whitespace. **If a `lang` is given**, it also removes that language's filler words from `FILLER_WORDS` (empty for every language in 0.1), joins space-separated thousands groups (`10 000` becomes `10000`) and spells out digits with `num2words(…, lang=lang)`. Digits stay as they are if `num2words` doesn't support the language. |

`MultilingualNormalizer()` uses `remove_diacritics=True` by default, which strips diacritics instead (NFKD). The
leaderboard doesn't use that setting. Hindi is not a supported scoring language in 0.1, because the leaderboard
scores `hi` with voi_oiwer, which normalizes internally.

## Versioning policy

The version number identifies the normalization behaviour. Scores computed with the same version are comparable.

- **Patch** (`0.1.x`): no output changes for any input (packaging, docs, CLI or speed).
- **Minor** (`0.x.0`): any change to any golden output. The CHANGELOG entry must list the affected datasets.
- **Major**: incompatible API changes.

## Provenance

Extracted verbatim from
[huggingface/open_asr_leaderboard](https://github.com/huggingface/open_asr_leaderboard) at commit `60e2e3b`:

| Original (`normalizer/…`) | Here |
|---|---|
| `normalizer.py`: `ADDITIONAL_DIACRITICS`, `remove_symbols*`, `BasicTextNormalizer`, `BasicMultilingualTextNormalizer` | `asr_normalizer/basic.py` |
| `normalizer.py`: `English*Normalizer`, `EnglishTextNormalizer` | `asr_normalizer/english/normalizer.py` |
| `english_abbreviations.py` (3 dicts) | `asr_normalizer/english/{spelling,names,compounds}.json`, loaded by `asr_normalizer/english/data.py` |
| `armenian_normalizer.py` | `asr_normalizer/languages/hy.py` |
| `data_utils.py`: `FILLER_WORDS`, `MultilingualNormalizer` | `asr_normalizer/multilingual.py` |
| `eval_utils.py`: `normalize_compound_pairs` | `asr_normalizer/pairs.py` |

Function and class bodies are byte-identical to the originals; only imports and module placement differ. The
dictionaries were converted to JSON with key order preserved. The English normalizer is derived from OpenAI
Whisper's text normalizer, and the original copyright headers are kept.

## Contributing

```bash
uv venv && uv pip install -e ".[dev]"
pytest -q                                   # golden fixtures: normalize(input, lang) == expected
ASR_NORMALIZER_FULL=1 pytest -q -m full     # re-fetch every leaderboard reference and check the manifest hashes
```

The golden fixtures in `tests/golden/` were generated from the original leaderboard code, and are what proves
that this package behaves exactly like it. A pull request that changes behaviour must regenerate the fixtures
(`scripts/make_golden.py`), bump the minor version, and add a CHANGELOG entry that lists the affected datasets.
The fixture diff is what reviewers check.

## License

Apache-2.0 (see [LICENSE](LICENSE)). Copyright the OpenAI team, the Hugging Face team and contributors.
