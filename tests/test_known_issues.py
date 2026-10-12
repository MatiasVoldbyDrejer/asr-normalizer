"""Known upstream bugs, kept as-is in 0.1 (zero behaviour change).

The xfail tests are strict: if a future version fixes the bug they XPASS and fail, so the fix is
noticed and the golden fixtures are regenerated deliberately. Values come from
tests/_unit_cases.py, generated from the ORIGINAL code.
"""

import pytest
from _unit_cases import CASES

from asr_normalizer import normalize

ISSUE66 = dict(CASES["issue66"])


def test_issue66_current_behaviour_is_pinned():
    # Original output today: "c o 2 emissions" -> "c02 emissions" (the lone "o" is read as a zero
    # by EnglishNumberNormalizer and the acronym collapser then joins "c 0 2").
    assert normalize("c o 2 emissions", "en") == ISSUE66["c o 2 emissions"]
    assert normalize("CO 2 emissions", "en") == ISSUE66["CO 2 emissions"]


@pytest.mark.xfail(strict=True, reason="open_asr_leaderboard#66")
def test_issue66_spelled_co2_matches_written_co2():
    # Spelled-out "c o 2" should normalize like the written forms "CO2" / "CO 2", which the
    # original maps to "co 2 emissions".
    assert normalize("c o 2 emissions", "en") == ISSUE66["CO2 emissions"]
