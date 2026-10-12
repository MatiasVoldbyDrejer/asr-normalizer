#!/usr/bin/env python3
"""Generate tests/_unit_cases.py: expected values for the unit tests, computed by the ORIGINAL code.

The unit tests (tests/test_english.py, test_multilingual.py, test_armenian.py, test_pairs.py,
test_known_issues.py) keep their inputs here and never hand-type an expected string: every
expected value is produced by running huggingface/open_asr_leaderboard's ``normalizer/``
(commit 60e2e3b) in a subprocess with ``PYTHONPATH=<source repo>``. This process never
imports the original package or ``asr_normalizer``.

Requirements: the interpreter given by ``--python`` (default: this one) must be able to import the
ORIGINAL ``normalizer`` package, i.e. have ``datasets``, ``huggingface_hub``, ``kaldialign``,
``num2words`` and ``regex`` installed (``normalizer.data_utils`` / ``eval_utils`` import them at
module level).

Usage (``--source-repo PATH`` is required unless ``ASR_NORMALIZER_SOURCE_REPO`` is set; PATH is a clone
of huggingface/open_asr_leaderboard at commit 60e2e3b):
    python scripts/make_unit_cases.py --source-repo PATH            # rewrite tests/_unit_cases.py
    python scripts/make_unit_cases.py --source-repo PATH --check    # exit 1 if the file is out of date
"""

from __future__ import annotations

import argparse
import json
import os
import pprint
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parent
# Path to a checkout of huggingface/open_asr_leaderboard (the ORIGINAL normalizer). No built-in
# default: pass --source-repo or set ASR_NORMALIZER_SOURCE_REPO.
DEFAULT_SOURCE_REPO = os.environ.get("ASR_NORMALIZER_SOURCE_REPO")

# Each group: (callable spec, [args...]) where args are the call arguments.
# Callable specs are resolved by the worker against the ORIGINAL code.
EN = "en"  # EnglishTextNormalizer()(s)

GROUPS = {
    # ---------------- English, one group per pipeline stage ----------------
    "en_lowercase": (EN, ["HELLO World", "ÀÉÎ ÕÜ", "ǅungla", "İstanbul", "ΣΊΣΥΦΟΣ"]),
    "en_brackets": (EN, ["hello [noise] world", "<unk> yes", "[a] b <c> d", "unbalanced [x", "a [b] [c] d",
                         "nested [a [b] c] d", "<b>bold</b>", "[]", "x <y z> w"]),
    "en_parentheses": (EN, ["keep (drop this) keep", "(laughs) ok", "a (b) c (d)", "x (y (z)) w", "empty ()",
                            "unbalanced (x", "f(x)=y"]),
    "en_fillers": (EN, ["uh i think", "uhhhm so", "um-hmm right", "ah-ha got it", "mm hmm", "hmmm",
                        "erm well", "ugh fine", "tsk", "mmkay", "ahem", "hum", "umbrella", "hummus", "error",
                        "emma", "aha", "ahaaa moment", "huh?", "eh", "mhm", "uhuh", "uhum", "ehehe", "hmmph"]),
    "en_apostrophe_space": (EN, ["it 's fine", "they ' re here", "she  'll go", "rock 'n' roll"]),
    "en_oclock_times": (EN, ["2:00 am", "at 11:00", "10:30", "9:00:00", "123:00", "3:000", "3 o'clock",
                             "at 2:00 p.m."]),
    "en_contractions": (EN, ["won't", "can't", "let's", "ain't", "y'all", "wanna", "gotta", "gonna",
                             "i'ma", "imma", "woulda", "coulda", "shoulda", "ma'am", "isn't", "you're",
                             "it's", "that's", "she'd", "we'll", "don't", "we've", "i'm", "it’s"]),
    "en_titles": (EN, ["mr smith", "Mrs. Jones", "St. Louis", "Dr. Who", "prof x", "capt hook", "gov brown",
                       "ald x", "gen lee", "sen y", "rep z", "pres q", "rev w", "hon v", "asst u",
                       "assoc t", "lt dan", "col s", "jr", "sr", "esq", "mister"]),
    "en_perfect_tenses": (EN, ["she'd been", "it's been", "they'd gone", "he's gone", "we'd done",
                               "she's got", "he's done"]),
    "en_digit_commas": (EN, ["1,000", "1,000,000", "10,5", "a,b", "1, 000"]),
    "en_periods": (EN, ["end.", "e.g. this", "3.14", "a.m.", "U.S.A.", "1.", "...", "v1.2"]),
    "en_symbols_diacritics": (EN, ["café", "naïve", "Œuvre", "straße", "smørrebrød", "þorn", "Łódź",
                                   "a—b", "«q»", "x²", "½", "ﬁne", "C++", "AT&T", "#tag", "@me",
                                   "日本語", "Привет", "🙂 ok"]),
    "en_compounds": (EN, ["wi fi", "wi-fi", "et cetera", "all right", "al right", "how ever", "e mail",
                          "e-book", "x ray", "t-shirt", "sci fi", "hi fi", "lo fi", "a m", "p m",
                          "o k", "ooh", "ohhh", "oooo", "e commerce"]),
    "en_numbers": (EN, ["one", "ones", "twenty one", "one hundred and five", "two thousand twenty four",
                        "nineteen ninety nine", "nineteen oh five", "four oh one", "oh seven nine eight",
                        "oh no", "oh oh", "zero zero seven", "double seven", "triple five", "first",
                        "twenty first", "the 1960s", "sixties", "two and a half", "a half",
                        "five point two", "point five", "$20 million", "twenty dollars",
                        "two dollars and seven cents", "$0.07", "50 percent", "fifty per cent",
                        "minus five", "plus two", "1/2", "192.168.0.1", "a million dollars", "11",
                        "1s", "one oh one", "sixes", "millionth", "hundreds"]),
    "en_spelling": (EN, ["colour", "theatre", "centre", "grey", "organisation", "travelling", "aluminium"]),
    "en_names": (EN, ["alan", "kris", "katherine", "micheal", "stacey", "allison", "carroll"]),
    "en_acronyms": (EN, ["b b c", "the b b c news", "u s a", "5 g", "a big cat", "a i", "i b m",
                         "a b c", "i o u", "x y z", "j f k", "c o 2"]),
    "en_symbol_cleanup": (EN, ["$", "price $", "50%", "a %", "€ euro", "5 £", ". dot", "100 %"]),
    "en_whitespace": (EN, ["", " ", "   ", "a   b", "\ttab", "line\nbreak", " lead", "trail "]),
    "en_realistic": (EN, [
        "Uh, Mr. Smith said he'd been to St. Louis at 3:00 PM, and it cost $20.50 — roughly 50%.",
        "[noise] So, um, the C.E.O. — y'know — gonna announce Q3 earnings of $1.2 billion (approx).",
        "The colour of the theatre's centre was grey, Dr. Kris said.",
    ]),
    # ---------------- English components ----------------
    "num_component": ("num", ["one two three", "twenty one", "oh seven nine eight", "1,000", "$20 million",
                              "two and a half", "1st", "the 1960s", "", "x"]),
    "spell_component": ("spell", ["colour centre", "favourite", "unchanged words"]),
    "name_component": ("name", ["alan kris", "nobody"]),
    "acr_component": ("acr", ["b b c", "a b", "a b c", "5 g", "a big cat", "", "c 0 2"]),
    "custom_spelling_mapping": ("en_nospell", ["colour", "theatre centre"]),
    "rsd_component": ("rsd", [("café, naïve!", ""), ("$5.00 50%", ".%$"), ("Œ æ ß ø", ""), ("x²", "")]),
    "rs_component": ("rs", ["café, naïve!", "a—b", "ﬁ x²"]),
    "rskm_component": ("rskm", ["हिन्दी!", "café, naïve!"]),
    "basic_default": ("basic", ["Hello, World! [x] (y)", "café", "a—b"]),
    "basic_remove_diacritics": ("basic_rd", ["café naïve", "Œuvre"]),
    "basic_split_letters": ("basic_split", ["abc", "हिन्दी", "café"]),
    "bml_default": ("bml", ["Café, naïve!", "हिन्दी", "[x] a (b) c", "a_b"]),
    "bml_keep_diacritics": ("bml_keep", ["Café, naïve!", "हिन्दी", "a_b c-d"]),
    # ---------------- Multilingual ----------------
    "ml_fillers_noop": ("ml", [("äh ähm also", "de"), ("euh ben", "fr"), ("eh bueno", "es"),
                               ("ehm allora", "it"), ("é pá", "pt"), ("eh nou", "nl")]),
    "ml_fillers_configured": ("ml_patched_fillers", [("äh das ist ähm gut", "de"), ("Äh… ja", "de"),
                                                     ("euh oui", "fr"), ("äh nicht hier", "fr"),
                                                     ("äh hmm", None)]),
    "ml_thousand_groups": ("ml", [("10 000", "de"), ("10 000 Einwohner", "fr"), ("1 000 000", "fr"),
                                  ("2 500", "es"), ("12 34", "de")]),
    "ml_num2words": ("ml", [("21", "de"), ("21", "fr"), ("21", "es"), ("1999", "de"), ("80", "fr"),
                            ("100", "es"), ("7", "it"), ("7", "pt"), ("7", "nl"), ("Es kostet 3,5 Euro", "de")]),
    "ml_unknown_lang": ("ml", [("21 apples", "xx"), ("21", "tlh"), ("21", "")]),
    "ml_no_lang": ("ml", [("21 Birnen", None), ("Café 3!", None)]),
    "ml_devanagari": ("ml", [("हिन्दी में", "hi"), ("क़िला", "hi"), ("हिन्दी 5", None)]),
    "ml_devanagari_remove_diacritics": ("ml_rd", [("हिन्दी में", None)]),
    "ml_remove_diacritics_default": ("ml_rd", [("Café crème", None), ("Café 21", "fr")]),
    "ml_hy_route": ("ml", [("Ես ունեմ 25 խնձոր։", "hy"), ("1-ին դասարան", "hy")]),
    "ml_large_numbers": ("ml", [("1" + "0" * 30, "de"), ("9" * 40, "fr")]),
    # ---------------- Armenian ----------------
    "hy_numbers": ("hy", ["Ես ունեմ 25 խնձոր։", "100 մարդ", "1 000 000", "1,250", "0", "007",
                          "1000000000000", "քսանհինգ", "երկու հազար քսան"]),
    "hy_ordinals": ("hy", ["1-ին դասարան", "2-րդ հարկ", "21-րդ դար", "3րդ անգամ", "առաջին", "հարյուրերորդ"]),
    "hy_percent": ("hy", ["50%", "50 %-ով", "12,5%", "3 տոկոսից", "հարյուր տոկոս"]),
    "hy_time": ("hy", ["10:30", "ժամը 10:30-ին", "ժամը 0930", "9.15-ին", "7։45"]),
    "hy_years": ("hy", ["2023 թ.", "2023թ.-ին", "2023 թվականին", "1990-ական", "Ք.հ. 301 թ.",
                        "քրիստոսից հետո 451"]),
    "hy_fractions_decimals": ("hy", ["3/4", "3,5", "3.14", "մեկուկես ժամ", "կես ժամ"]),
    "hy_money_units": ("hy", ["$100", "100$", "100 դոլար", "5 մլն դրամ", "2 մլրդ", "60 կմ/ժ"]),
    "hy_latin_folding": ("hy", ["café résumé", "Ñandú", "Zoë", "Ångström"]),
    "hy_punctuation": ("hy", ["Ի՞նչ ես անում", "Այո՛", "«Ամենը» ասաց", "<<չակերտներ>>", "Վերջ . . .",
                              "- ցուցակ", "Եվ նա", "ԱՄՆ-ի", "Covid-19-ի", "", "   "]),
    "hy_preprocessing": ("hy_pre", ["բառ ,բառ", "<<բառ>>", "Վերջ . . .", "- ցուցակ", "100 մարդ"]),
    "hy_normalization": ("hy_norm", ["Բարև, ձեզ։", "10:30-ին", "café"]),
    "hy_numbernormalization": ("hy_nn", ["25 խնձոր", "1-ին", "50%"]),
    # ---------------- Known issues ----------------
    "issue66": (EN, ["c o 2 emissions", "CO2 emissions", "CO 2 emissions", "co2 emissions", "c o 2",
                     "CO 2", "3 o'clock"]),
}

PAIR_CASES = [
    (["white paper"], ["whitepaper"]),
    (["whitepaper"], ["white paper"]),
    (["haus tür offen"], ["haustür offen"]),
    (["the cat sat"], ["the dog sat"]),
    (["a b c"], ["abc"]),
    (["ab c d"], ["a bc d"]),
    (["x  y   z"], ["x y z"]),
    (["a b", "c d"], ["ab"]),
    ([], []),
    ([""], [""]),
    (["same words here"], ["same words here"]),
    (["new york city"], ["newyork city now"]),
]

WORKER = r'''
import json, sys, hashlib
req_path, resp_path = sys.argv[1], sys.argv[2]
req = json.load(open(req_path, encoding="utf-8"))

from normalizer import EnglishTextNormalizer, BasicMultilingualTextNormalizer, ArmenianTextNormalizer
from normalizer import normalizer as nm
from normalizer import armenian_normalizer as hy
from normalizer import english_abbreviations as ea
from normalizer import data_utils
from normalizer.data_utils import MultilingualNormalizer, FILLER_WORDS
from normalizer.eval_utils import normalize_compound_pairs

PATCHED_FILLERS = {"de": ["äh", "ähm", "Äh…"], "fr": ["euh"]}


def ml_patched(s, lang):
    saved = dict(data_utils.FILLER_WORDS)
    data_utils.FILLER_WORDS.clear()
    data_utils.FILLER_WORDS.update(PATCHED_FILLERS)
    try:
        n = MultilingualNormalizer(remove_diacritics=False)
    finally:
        data_utils.FILLER_WORDS.clear()
        data_utils.FILLER_WORDS.update(saved)
    return n(s, lang=lang)


EN = EnglishTextNormalizer()
ML = MultilingualNormalizer(remove_diacritics=False)
ML_RD = MultilingualNormalizer(remove_diacritics=True)
FN = {
    "en": lambda s: EN(s),
    "en_nospell": lambda s: EnglishTextNormalizer({})(s),
    "num": lambda s: nm.EnglishNumberNormalizer()(s),
    "spell": lambda s: nm.EnglishSpellingNormalizer(ea.english_spelling_normalizer)(s),
    "name": lambda s: nm.EnglishNameNormalizer()(s),
    "acr": lambda s: nm.EnglishAcronymNormalizer()(s),
    "rsd": lambda a: nm.remove_symbols_and_diacritics(a[0], keep=a[1]),
    "rs": lambda s: nm.remove_symbols(s),
    "rskm": lambda s: nm.remove_symbols_keep_marks(s),
    "basic": lambda s: nm.BasicTextNormalizer()(s),
    "basic_rd": lambda s: nm.BasicTextNormalizer(remove_diacritics=True)(s),
    "basic_split": lambda s: nm.BasicTextNormalizer(split_letters=True)(s),
    "bml": lambda s: BasicMultilingualTextNormalizer()(s),
    "bml_keep": lambda s: BasicMultilingualTextNormalizer(remove_diacritics=False)(s),
    "ml": lambda a: ML(a[0], lang=a[1]),
    "ml_rd": lambda a: ML_RD(a[0], lang=a[1]),
    "ml_patched_fillers": lambda a: ml_patched(a[0], a[1]),
    "hy": lambda s: ArmenianTextNormalizer()(s),
    "hy_pre": lambda s: hy.preprocessing(s),
    "hy_norm": lambda s: hy.normalization(s),
    "hy_nn": lambda s: hy.numbernormalization(s),
}

out = {}
for name, (spec, args) in req["groups"].items():
    rows = []
    for a in args:
        try:
            rows.append([a, FN[spec](a), None])
        except Exception as e:
            rows.append([a, None, type(e).__name__])
    out[name] = rows

pairs = []
for refs, preds in req["pairs"]:
    r, p = normalize_compound_pairs(refs, preds)
    pairs.append([refs, preds, r, p])


def digest(d):
    return hashlib.sha256(json.dumps(list(d.items()), ensure_ascii=False).encode("utf-8")).hexdigest()


import inspect


def sig(obj):
    out = []
    for p in inspect.signature(obj).parameters.values():
        d = p.default
        if d is inspect.Parameter.empty:
            d = None
        elif isinstance(d, (bool, int, str, type(None))):
            d = repr(d)
        else:
            d = "<" + type(d).__name__ + ">"
        out.append([p.name, p.kind.name, d])
    return out


meta = {
    "signatures": {
        "EnglishTextNormalizer.__init__": sig(EnglishTextNormalizer.__init__),
        "EnglishTextNormalizer.__call__": sig(EnglishTextNormalizer.__call__),
        "BasicTextNormalizer.__init__": sig(nm.BasicTextNormalizer.__init__),
        "BasicTextNormalizer.__call__": sig(nm.BasicTextNormalizer.__call__),
        "BasicMultilingualTextNormalizer.__init__": sig(BasicMultilingualTextNormalizer.__init__),
        "BasicMultilingualTextNormalizer.__call__": sig(BasicMultilingualTextNormalizer.__call__),
        "MultilingualNormalizer.__init__": sig(MultilingualNormalizer.__init__),
        "MultilingualNormalizer.__call__": sig(MultilingualNormalizer.__call__),
        "ArmenianTextNormalizer.__call__": sig(ArmenianTextNormalizer.__call__),
        "normalize_compound_pairs": sig(normalize_compound_pairs),
    },
    "filler_words": FILLER_WORDS,
    "patched_fillers": PATCHED_FILLERS,
    "data": {
        "english_spelling_normalizer": [len(ea.english_spelling_normalizer), digest(ea.english_spelling_normalizer)],
        "english_name_normalizer": [len(ea.english_name_normalizer), digest(ea.english_name_normalizer)],
        "english_compound_normalizer": [len(ea.english_compound_normalizer), digest(ea.english_compound_normalizer)],
    },
    "additional_diacritics": nm.ADDITIONAL_DIACRITICS,
    "replacers": list(EN.replacers.items()),
    "ignore_patterns": EN.ignore_patterns,
    "hy_all": list(hy.__all__),
    "hy_metadata": hy.NORMALIZATION_METADATA,
}
json.dump({"groups": out, "pairs": pairs, "meta": meta}, open(resp_path, "w", encoding="utf-8"), ensure_ascii=False)
'''


def run_original(source_repo: Path, python: str) -> dict:
    groups = {k: [spec, [list(a) if isinstance(a, tuple) else a for a in args]] for k, (spec, args) in GROUPS.items()}
    with tempfile.TemporaryDirectory() as td:
        req, resp = Path(td) / "req.json", Path(td) / "resp.json"
        req.write_text(json.dumps({"groups": groups, "pairs": PAIR_CASES}, ensure_ascii=False), encoding="utf-8")
        env = dict(os.environ, PYTHONPATH=str(source_repo), PYTHONDONTWRITEBYTECODE="1")
        try:
            subprocess.run([python, "-c", WORKER, str(req), str(resp)], check=True, env=env, cwd=str(source_repo),
                           stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
        except subprocess.CalledProcessError as e:
            # Print only the worker's stderr: the CalledProcessError message would embed the whole worker source.
            sys.stderr.write(e.stderr or "")
            sys.exit(e.returncode if e.returncode > 0 else 1)
        return json.loads(resp.read_text(encoding="utf-8"))


def render(result: dict, head: str) -> str:
    lines = [
        "# GENERATED by scripts/make_unit_cases.py -- do not edit by hand.",
        f"# Expected values computed by the ORIGINAL normalizer: huggingface/open_asr_leaderboard @ {head}.",
        "# Each case is (args, expected) where args is the input string or a tuple (input, lang | keep),",
        "# and expected is the original output (or {'raises': ExcName} if the original raised).",
        "# fmt: off",
        "",
    ]
    cases = {}
    for name, rows in result["groups"].items():
        conv = []
        for a, o, err in rows:
            a = tuple(a) if isinstance(a, list) else a
            conv.append((a, {"raises": err} if err else o))
        cases[name] = conv
    lines.append("CASES = " + pprint.pformat(cases, width=110, sort_dicts=False))
    lines.append("")
    pairs = [(tuple([r, p]), (er, ep)) for r, p, er, ep in result["pairs"]]
    lines.append("PAIRS = " + pprint.pformat(pairs, width=110, sort_dicts=False))
    lines.append("")
    lines.append("META = " + pprint.pformat(result["meta"], width=110, sort_dicts=False))
    lines.append("")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source-repo", default=DEFAULT_SOURCE_REPO, required=DEFAULT_SOURCE_REPO is None,)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--out", default=str(PACKAGE_ROOT / "tests" / "_unit_cases.py"))
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    source_repo = Path(args.source_repo).resolve()
    head = subprocess.run(["git", "-C", str(source_repo), "rev-parse", "--short=7", "HEAD"],
                          check=True, capture_output=True, text=True).stdout.strip()
    text = render(run_original(source_repo, args.python), head)
    out = Path(args.out)
    if args.check:
        if not out.exists() or out.read_text(encoding="utf-8") != text:
            sys.exit(f"{out} is out of date; run scripts/make_unit_cases.py")
        print(f"{out} is up to date")
        return
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
