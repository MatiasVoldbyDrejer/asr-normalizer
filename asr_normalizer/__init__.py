"""asr-normalizer: the text normalizer behind the Open ASR Leaderboard and the Open TTS Leaderboard.

Originally OpenAI Whisper's text normalizer, plus the leaderboard's patches. Extracted verbatim
from huggingface/open_asr_leaderboard@60e2e3b.

    >>> from asr_normalizer import normalize
    >>> normalize("I've got 2 cats, uh, at 3:00 PM")
    'i have got 2 cats at 3 pm'
"""

__version__ = "0.1.0"

from .basic import BasicMultilingualTextNormalizer, BasicTextNormalizer
from .english import EnglishTextNormalizer
from .languages.hy import ArmenianTextNormalizer
from .multilingual import FILLER_WORDS, MultilingualNormalizer
from .pairs import normalize_compound_pairs

__all__ = [
    "__version__",
    "normalize",
    "EnglishTextNormalizer",
    "BasicTextNormalizer",
    "BasicMultilingualTextNormalizer",
    "MultilingualNormalizer",
    "ArmenianTextNormalizer",
    "normalize_compound_pairs",
    "FILLER_WORDS",
]

# The same two instances the leaderboard builds at import time in normalizer/data_utils.py.
_EN = EnglishTextNormalizer()
_ML = MultilingualNormalizer(remove_diacritics=False)


def normalize(text: str, lang: str = "en") -> str:
    """Normalize ``text`` exactly as the Open ASR Leaderboard does at scoring time.

    ``lang == "en"`` uses ``EnglishTextNormalizer()``; any other value (including ``None``)
    uses ``MultilingualNormalizer(remove_diacritics=False)(text, lang=lang)``, which routes
    ``"hy"`` to ``ArmenianTextNormalizer``. Mirrors ``eval_utils._normalize`` in the leaderboard.
    """
    return _EN(text) if lang == "en" else _ML(text, lang=lang)
