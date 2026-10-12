"""ArmenianTextNormalizer (asr_normalizer.languages.hy).

Expected values come from tests/_unit_cases.py, generated from the ORIGINAL code by
scripts/make_unit_cases.py.
"""

import pytest
from _unit_cases import CASES, META

from asr_normalizer import ArmenianTextNormalizer
from asr_normalizer.languages import hy

HY = ArmenianTextNormalizer()


def _cases(group):
    return pytest.mark.parametrize("text,expected", CASES[group], ids=[repr(c[0])[:40] for c in CASES[group]])


@_cases("hy_numbers")
def test_numbers(text, expected):
    assert HY(text) == expected


@_cases("hy_ordinals")
def test_ordinals(text, expected):
    assert HY(text) == expected


@_cases("hy_percent")
def test_percent(text, expected):
    assert HY(text) == expected


@_cases("hy_time")
def test_clock_times(text, expected):
    assert HY(text) == expected


@_cases("hy_years")
def test_year_markers_and_era(text, expected):
    assert HY(text) == expected


@_cases("hy_fractions_decimals")
def test_fractions_and_decimals(text, expected):
    assert HY(text) == expected


@_cases("hy_money_units")
def test_money_and_units(text, expected):
    assert HY(text) == expected


@_cases("hy_latin_folding")
def test_latin_diacritic_folding(text, expected):
    assert HY(text) == expected


@_cases("hy_punctuation")
def test_punctuation_case_and_whitespace(text, expected):
    assert HY(text) == expected


@_cases("hy_preprocessing")
def test_preprocessing_function(text, expected):
    assert hy.preprocessing(text) == expected


@_cases("hy_normalization")
def test_normalization_function(text, expected):
    assert hy.normalization(text) == expected


@_cases("hy_numbernormalization")
def test_numbernormalization_function(text, expected):
    assert hy.numbernormalization(text) == expected


def test_call_is_normalization_numbernormalization():
    for text, _ in CASES["hy_numbers"] + CASES["hy_years"]:
        assert HY(text) == hy.normalization_numbernormalization(text)


def test_module_exports_verbatim():
    assert list(hy.__all__) == META["hy_all"]
    assert hy.NORMALIZATION_METADATA == META["hy_metadata"]
