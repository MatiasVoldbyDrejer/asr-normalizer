"""MultilingualNormalizer / BasicMultilingualTextNormalizer / BasicTextNormalizer.

Expected values come from tests/_unit_cases.py, generated from the ORIGINAL code by
scripts/make_unit_cases.py.
"""

import pytest
from _unit_cases import CASES, META

import asr_normalizer
from asr_normalizer import (
    FILLER_WORDS,
    ArmenianTextNormalizer,
    BasicMultilingualTextNormalizer,
    BasicTextNormalizer,
    MultilingualNormalizer,
)
from asr_normalizer import multilingual as ml_mod
from asr_normalizer.basic import remove_symbols_keep_marks

ML = MultilingualNormalizer(remove_diacritics=False)
ML_RD = MultilingualNormalizer(remove_diacritics=True)


def _ml_cases(group):
    return pytest.mark.parametrize(
        "text,lang,expected", [(a[0], a[1], e) for a, e in CASES[group]], ids=[repr(c[0])[:40] for c in CASES[group]]
    )


def _cases(group):
    return pytest.mark.parametrize("text,expected", CASES[group], ids=[repr(c[0])[:40] for c in CASES[group]])


# --- Fillers -----------------------------------------------------------------------------
def test_filler_words_table():
    assert FILLER_WORDS == META["filler_words"]
    assert ml_mod.FILLER_WORDS is FILLER_WORDS


@_ml_cases("ml_fillers_noop")
def test_fillers_are_noop_while_table_is_empty(text, lang, expected):
    assert FILLER_WORDS == {}
    assert ML(text, lang=lang) == expected


@_ml_cases("ml_fillers_configured")
def test_filler_removal_mechanism(monkeypatch, text, lang, expected):
    # The original reads FILLER_WORDS when the normalizer is constructed.
    monkeypatch.setattr(ml_mod, "FILLER_WORDS", META["patched_fillers"])
    assert MultilingualNormalizer(remove_diacritics=False)(text, lang=lang) == expected


# --- Numbers -----------------------------------------------------------------------------
@_ml_cases("ml_thousand_groups")
def test_thousand_groups_joined(text, lang, expected):
    assert ML(text, lang=lang) == expected


@_ml_cases("ml_num2words")
def test_num2words(text, lang, expected):
    assert ML(text, lang=lang) == expected


@_ml_cases("ml_unknown_lang")
def test_unknown_lang_leaves_digits(text, lang, expected):
    assert ML(text, lang=lang) == expected


@_ml_cases("ml_no_lang")
def test_no_lang_is_basic_path(text, lang, expected):
    assert lang is None
    assert ML(text) == expected
    assert ML(text, lang=None) == expected
    assert BasicMultilingualTextNormalizer(remove_diacritics=False)(text) == expected


@_ml_cases("ml_large_numbers")
def test_large_numbers(text, lang, expected):
    assert ML(text, lang=lang) == expected


# --- Scripts with combining marks --------------------------------------------------------
@_ml_cases("ml_devanagari")
def test_devanagari_marks_kept(text, lang, expected):
    assert ML(text, lang=lang) == expected


@_ml_cases("ml_devanagari_remove_diacritics")
def test_devanagari_marks_stripped_with_remove_diacritics(text, lang, expected):
    assert ML_RD(text, lang=lang) == expected


@_ml_cases("ml_remove_diacritics_default")
def test_remove_diacritics_default(text, lang, expected):
    assert MultilingualNormalizer()(text, lang=lang) == expected


@_cases("rskm_component")
def test_remove_symbols_keep_marks(text, expected):
    assert remove_symbols_keep_marks(text) == expected


# --- Armenian routing --------------------------------------------------------------------
@_ml_cases("ml_hy_route")
def test_hy_routes_to_armenian(text, lang, expected):
    assert ML(text, lang="hy") == expected
    assert ArmenianTextNormalizer()(text) == expected


# --- Basic normalizers -------------------------------------------------------------------
@_cases("bml_default")
def test_basic_multilingual_default(text, expected):
    assert BasicMultilingualTextNormalizer()(text) == expected


@_cases("bml_keep_diacritics")
def test_basic_multilingual_keep_diacritics(text, expected):
    assert BasicMultilingualTextNormalizer(remove_diacritics=False)(text) == expected


@_cases("basic_default")
def test_basic_default(text, expected):
    assert BasicTextNormalizer()(text) == expected


@_cases("basic_remove_diacritics")
def test_basic_remove_diacritics(text, expected):
    assert BasicTextNormalizer(remove_diacritics=True)(text) == expected


@_cases("basic_split_letters")
def test_basic_split_letters(text, expected):
    assert BasicTextNormalizer(split_letters=True)(text) == expected


def test_multilingual_is_basic_subclass():
    assert issubclass(MultilingualNormalizer, BasicMultilingualTextNormalizer)
    assert asr_normalizer.MultilingualNormalizer is ml_mod.MultilingualNormalizer
