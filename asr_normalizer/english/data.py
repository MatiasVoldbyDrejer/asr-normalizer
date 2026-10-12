"""English normalization data, loaded from the JSON files next to this module.

These are the three mappings that lived in ``normalizer/english_abbreviations.py`` of
huggingface/open_asr_leaderboard@60e2e3b, converted to JSON with key order preserved
(``json`` keeps insertion order, and the order of ``english_compound_normalizer`` matters
because its patterns are applied one after another):

- ``spelling.json``  -> ``english_spelling_normalizer``: British -> American spellings
  (https://www.tysto.com/uk-us-spelling-list.html), plus "ok"/"kay" -> "okay" and
  "etcetera" -> "etc". Applied per whitespace-separated token.
- ``names.json``     -> ``english_name_normalizer``: name spelling variants -> one canonical
  form. Applied per token. The JSON keeps the section order of the original file, whose
  section comments cannot be carried over to JSON: double-letter variants (from "alan"),
  Chr/Kr ("kris"), C/K ("karl"), silent letters / alternate spellings ("ann"), Mac/Mc
  ("macarthur"), international ("mohamad"), miscellaneous homophones ("alphonso").
- ``compounds.json`` -> ``english_compound_normalizer``: regex-based multi-word ->
  single-token mappings, plus spellings that vary too freely to enumerate in the spelling
  list (e.g. elongations). Applied after symbol removal, so hyphens/punctuation are
  already stripped. Keys are used with ``re.sub``; values are the replacement strings.
"""

import json
from importlib.resources import files


def _load(name):
    return json.loads(files(__package__).joinpath(name).read_text(encoding="utf-8"))


english_spelling_normalizer = _load("spelling.json")
english_name_normalizer = _load("names.json")
english_compound_normalizer = _load("compounds.json")

__all__ = ["english_spelling_normalizer", "english_name_normalizer", "english_compound_normalizer"]
