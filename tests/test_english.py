"""EnglishTextNormalizer, one focused test per pipeline stage.

Inputs and expected values live in tests/_unit_cases.py, generated from the ORIGINAL code by
scripts/make_unit_cases.py (never hand-typed).
"""

import pytest
from _unit_cases import CASES, META

from asr_normalizer import EnglishTextNormalizer
from asr_normalizer.basic import remove_symbols, remove_symbols_and_diacritics, ADDITIONAL_DIACRITICS
from asr_normalizer.english import normalizer as en_mod

EN = EnglishTextNormalizer()


def _cases(group):
    return pytest.mark.parametrize("text,expected", CASES[group], ids=[repr(c[0])[:40] for c in CASES[group]])


# --- EnglishTextNormalizer.__call__, in pipeline order -----------------------------------
@_cases("en_lowercase")
def test_lowercase(text, expected):
    assert EN(text) == expected


@_cases("en_brackets")
def test_remove_bracketed(text, expected):
    assert EN(text) == expected


@_cases("en_parentheses")
def test_remove_parenthesized(text, expected):
    assert EN(text) == expected


@_cases("en_fillers")
def test_fillers(text, expected):
    assert EN(text) == expected


@_cases("en_apostrophe_space")
def test_space_before_apostrophe(text, expected):
    assert EN(text) == expected


@_cases("en_oclock_times")
def test_oclock_times(text, expected):
    assert EN(text) == expected


@_cases("en_contractions")
def test_contractions(text, expected):
    assert EN(text) == expected


@_cases("en_titles")
def test_titles(text, expected):
    assert EN(text) == expected


@_cases("en_perfect_tenses")
def test_perfect_tenses(text, expected):
    assert EN(text) == expected


@_cases("en_digit_commas")
def test_commas_between_digits(text, expected):
    assert EN(text) == expected


@_cases("en_periods")
def test_periods(text, expected):
    assert EN(text) == expected


@_cases("en_symbols_diacritics")
def test_symbols_and_diacritics(text, expected):
    assert EN(text) == expected


@_cases("en_compounds")
def test_compounds(text, expected):
    assert EN(text) == expected


@_cases("en_numbers")
def test_numbers(text, expected):
    assert EN(text) == expected


@_cases("en_spelling")
def test_british_spellings(text, expected):
    assert EN(text) == expected


@_cases("en_names")
def test_names(text, expected):
    assert EN(text) == expected


@_cases("en_acronyms")
def test_acronyms(text, expected):
    assert EN(text) == expected


@_cases("en_symbol_cleanup")
def test_currency_and_percent_cleanup(text, expected):
    assert EN(text) == expected


@_cases("en_whitespace")
def test_whitespace(text, expected):
    assert EN(text) == expected


@_cases("en_realistic")
def test_realistic_sentences(text, expected):
    assert EN(text) == expected


@_cases("custom_spelling_mapping")
def test_custom_spelling_mapping(text, expected):
    assert EnglishTextNormalizer({})(text) == expected


def test_instances_are_reusable_and_stateless():
    texts = [c[0] for c in CASES["en_realistic"] + CASES["en_numbers"]]
    first = [EN(t) for t in texts]
    assert [EN(t) for t in reversed(texts)] == list(reversed(first))
    assert [EnglishTextNormalizer()(t) for t in texts] == first


# --- Components --------------------------------------------------------------------------
@_cases("num_component")
def test_number_normalizer(text, expected):
    assert en_mod.EnglishNumberNormalizer()(text) == expected


@_cases("spell_component")
def test_spelling_normalizer(text, expected):
    from asr_normalizer.english.data import english_spelling_normalizer

    assert en_mod.EnglishSpellingNormalizer(english_spelling_normalizer)(text) == expected


@_cases("name_component")
def test_name_normalizer(text, expected):
    assert en_mod.EnglishNameNormalizer()(text) == expected


@_cases("acr_component")
def test_acronym_normalizer(text, expected):
    assert en_mod.EnglishAcronymNormalizer()(text) == expected


@pytest.mark.parametrize("args,expected", CASES["rsd_component"], ids=[repr(c[0])[:40] for c in CASES["rsd_component"]])
def test_remove_symbols_and_diacritics(args, expected):
    text, keep = args
    assert remove_symbols_and_diacritics(text, keep=keep) == expected


@_cases("rs_component")
def test_remove_symbols(text, expected):
    assert remove_symbols(text) == expected


# --- Verbatim tables ---------------------------------------------------------------------
def test_replacers_verbatim():
    assert [list(kv) for kv in EN.replacers.items()] == META["replacers"]


def test_filler_pattern_verbatim():
    assert EN.ignore_patterns == META["ignore_patterns"]


def test_additional_diacritics_verbatim():
    assert ADDITIONAL_DIACRITICS == META["additional_diacritics"]
    assert list(ADDITIONAL_DIACRITICS) == list(META["additional_diacritics"])


@pytest.mark.parametrize("name", sorted(META["data"]))
def test_english_data_tables_verbatim(name):
    """Loaded JSON tables must equal the original dicts, including key order."""
    import hashlib
    import json

    from asr_normalizer.english import data

    table = getattr(data, name)
    n, digest = META["data"][name]
    assert len(table) == n
    got = hashlib.sha256(json.dumps(list(table.items()), ensure_ascii=False).encode("utf-8")).hexdigest()
    assert got == digest
