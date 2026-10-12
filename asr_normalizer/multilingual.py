# Extracted from huggingface/open_asr_leaderboard@60e2e3b (normalizer/data_utils.py: FILLER_WORDS, MultilingualNormalizer).
# Function and class bodies are verbatim; only imports and module placement differ.

import re

import num2words

from .basic import BasicMultilingualTextNormalizer
from .languages.hy import ArmenianTextNormalizer


# Language-specific filler words / hesitations, removed when the matching
# lang= is passed to the multilingual normalizer. Multi-word entries are
# supported (matched on whitespace boundaries). Currently empty
FILLER_WORDS = {}


class MultilingualNormalizer(BasicMultilingualTextNormalizer):
    """BasicMultilingualTextNormalizer with optional number normalization.

    Call with just text for standard normalization (backward-compatible).
    Pass lang= to also convert digits to words via num2words and remove
    language-specific filler words (see FILLER_WORDS).
    """

    def __init__(self, remove_diacritics: bool = True):
        super().__init__(remove_diacritics)
        self._language_normalizers = {"hy": ArmenianTextNormalizer()}
        # Pre-compile filler patterns. Each filler word is passed through the
        # base normalization itself, so the pattern matches the normalized
        # text exactly (base normalization may strip punctuation such as "…"
        # or combining marks). Longest-first so that multi-word and longer
        # variants match before their prefixes. Matched on whitespace
        # boundaries ((?<!\S) / (?!\S)) rather than \b, which is unreliable
        # next to combining marks.
        self._filler_patterns = {}
        base_normalize = super().__call__
        for lang, words in FILLER_WORDS.items():
            normalized_words = {base_normalize(w) for w in words}
            normalized_words.discard("")
            self._filler_patterns[lang] = re.compile(
                r"(?<!\S)(?:"
                + "|".join(re.escape(w) for w in sorted(normalized_words, key=len, reverse=True))
                + r")(?!\S)"
            )

    def _remove_fillers(self, text, lang):
        pattern = self._filler_patterns.get(lang)
        if pattern is None:
            return text
        text = pattern.sub("", text)
        return re.sub(r"\s+", " ", text).strip()

    def _normalize_numbers(self, text, lang):
        # Join space-separated thousand groups (e.g. "10 000" -> "10000")
        text = re.sub(r"(\d)\s+(\d{3})\b", r"\1\2", text)

        # Convert remaining digit sequences to words
        def _replace(m):
            try:
                return num2words.num2words(int(m.group()), lang=lang)
            except Exception:
                return m.group()

        return re.sub(r"\d+", _replace, text)

    def __call__(self, s, lang=None):
        language_normalizer = self._language_normalizers.get(lang)
        if language_normalizer is not None:
            return language_normalizer(s)

        s = super().__call__(s)
        if lang is not None:
            s = self._remove_fillers(s, lang)
            s = self._normalize_numbers(s, lang)
        return s
