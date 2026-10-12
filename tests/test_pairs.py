"""normalize_compound_pairs (asr_normalizer.pairs).

Expected values come from tests/_unit_cases.py (PAIRS), generated from the ORIGINAL code by
scripts/make_unit_cases.py. Real leaderboard pairs are covered by tests/golden/pairs.jsonl.
"""

import copy

import pytest
from _unit_cases import PAIRS

from asr_normalizer import normalize_compound_pairs
from asr_normalizer import pairs as pairs_mod


@pytest.mark.parametrize("args,expected", PAIRS, ids=[repr(a)[:50] for a, _ in PAIRS])
def test_normalize_compound_pairs(args, expected):
    refs, preds = args
    got_refs, got_preds = normalize_compound_pairs(refs, preds)
    assert (got_refs, got_preds) == expected


def test_returns_two_lists_and_does_not_mutate_inputs():
    refs, preds = ["white paper", "a b"], ["whitepaper", "a b"]
    before = copy.deepcopy((refs, preds))
    result = normalize_compound_pairs(refs, preds)
    assert isinstance(result, tuple) and len(result) == 2
    assert all(isinstance(x, list) for x in result)
    assert (refs, preds) == before


def test_submodule_export():
    assert pairs_mod.normalize_compound_pairs is normalize_compound_pairs
