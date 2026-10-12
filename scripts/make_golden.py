#!/usr/bin/env python3
"""Regenerate the golden-output fixtures under tests/golden/ from the ORIGINAL normalizer.

The expected values are produced exclusively by the original code in a clone of
huggingface/open_asr_leaderboard (``normalizer/`` at commit 60e2e3b). That code runs
in a *subprocess* with ``PYTHONPATH=<source repo>``; this driver process never imports
it (nor ``asr_normalizer``), so the two packages are never loaded in one interpreter.

What it writes (all under ``--out``, default ``tests/golden``):

* ``<dataset id with / -> __>__<config>__<split>.jsonl``: a deterministic sample of the
  reference transcripts of every public leaderboard eval set, one
  ``{"input", "lang", "expected"}`` row per line.
* ``synthetic__en.jsonl``, ``synthetic__ml.jsonl``, ``synthetic__hy.jsonl``: crafted
  strings covering every stage of the English, multilingual and Armenian normalizers.
* ``pairs.jsonl``: ``normalize_compound_pairs`` cases built from real reference /
  prediction pairs in the public results bucket, plus edge cases.
* ``manifest.json``: provenance (dataset revisions, files, row counts, sha256 of the
  newline-joined outputs of the FULL sweep over every reference of each source).

Only the transcript column is ever read: parquet files are range-read through
``HfFileSystem`` + ``pyarrow.parquet.ParquetFile(...).read(columns=[...])``, so no
audio is downloaded.

Routing mirrors the leaderboard's scoring (``normalizer.eval_utils._normalize``):
``lang == "en"`` -> ``EnglishTextNormalizer``; any other lang ->
``MultilingualNormalizer(remove_diacritics=False)(text, lang=lang)`` ("hy" is routed to
``ArmenianTextNormalizer`` inside it).

Filtering: the multilingual run scripts (``*/run_eval_ml.py``) drop references failing
``data_utils.is_target_text_in_range`` *before* normalization, so multilingual and
Armenian sources are pre-filtered with the original function. The English pipeline
(``data_utils.prepare_data``) applies that filter to the *normalized* text, i.e. after
normalization, so English sources are not pre-filtered (rows normalizing to "" are kept).

Requirements: this driver needs ``huggingface_hub`` and ``pyarrow``; the interpreter given by
``--python`` (default: this one) must be able to import the ORIGINAL ``normalizer`` package, i.e.
have ``regex``, ``num2words``, ``datasets``, ``huggingface_hub`` and ``kaldialign`` installed
(``normalizer.data_utils`` / ``eval_utils`` import them at module level).

Usage (``--source-repo PATH`` is required unless ``ASR_NORMALIZER_SOURCE_REPO`` is set; PATH is a clone
of huggingface/open_asr_leaderboard at commit 60e2e3b):
    python scripts/make_golden.py --source-repo PATH                      # full regeneration
    python scripts/make_golden.py --source-repo PATH --no-full-sweep      # keep manifest sha256s, resample only
    python scripts/make_golden.py --source-repo PATH --only fleurs_de     # regenerate a subset (same sample; manifest kept)
    python scripts/make_golden.py --source-repo PATH --refresh-revisions  # move every dataset to its latest Hub revision

Dataset revisions are pinned: by default the revisions recorded in an existing manifest.json are
reused, so a rerun reproduces the committed fixtures byte for byte (except the "generated" date).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parent
# Path to a checkout of huggingface/open_asr_leaderboard (the ORIGINAL normalizer). No built-in
# default: pass --source-repo or set ASR_NORMALIZER_SOURCE_REPO.
DEFAULT_SOURCE_REPO = os.environ.get("ASR_NORMALIZER_SOURCE_REPO")
EXPECTED_SOURCE_COMMIT = "60e2e3b"
SOURCE_REPO_NAME = "huggingface/open_asr_leaderboard"

# --------------------------------------------------------------------------------------
# Sources: every public (dataset, config, split) the leaderboard scores, plus one legacy
# config still present in the main repo. `files` is a glob inside the dataset repo.
# --------------------------------------------------------------------------------------
OAL = "hf-audio/open-asr-leaderboard"  # hf-audio/esb-datasets-test-only-sorted redirects here
ML = "hf-audio/open-asr-leaderboard-multilingual-datasets"
HY = "Metric-AI/open-asr-leaderboard-multilingual-datasets"


def _src(dataset, config, split, files, lang, family, text_column="text", fmt="parquet"):
    return {
        "dataset": dataset,
        "config": config,
        "split": split,
        "files": files,
        "text_column": text_column,
        "format": fmt,
        "lang": lang,
        "family": family,
        # Multilingual run scripts filter raw refs before normalization; English does not.
        "prefilter": lang != "en",
    }


SOURCES = [
    # English short-form, "public" family (main leaderboard columns)
    _src(OAL, "ami_cleaned", "test", "ami_cleaned/test-*.parquet", "en", "public"),
    _src(OAL, "gigaspeech_cleaned", "test", "gigaspeech_cleaned/test-*.parquet", "en", "public"),
    _src(OAL, "librispeech", "test.clean", "librispeech/test.clean-*.parquet", "en", "public"),
    _src(OAL, "librispeech", "test.other", "librispeech/test.other-*.parquet", "en", "public"),
    _src(OAL, "spgispeech", "test", "spgispeech/test-*.parquet", "en", "public"),
    _src(OAL, "urgent2024", "test", "urgent2024/test-*.parquet", "en", "public"),
    _src(OAL, "voxpopuli_cleaned_aa", "test", "voxpopuli_cleaned_aa/test-*.parquet", "en", "public"),
    # Earnings22-Cleaned-AA-chunked is scored against its parent's session-level references.
    _src("ArtificialAnalysis/Earnings22-Cleaned-AA", "default", "test",
         "earnings22_cleaned_aa_v1.jsonl", "en", "public", text_column="transcript", fmt="jsonl"),
    _src("VoiceArena/Monsoon_en_IN_test", "default", "test", "data/test-*.parquet", "en", "public"),
    # English short-form, "extra" family
    _src(OAL, "ami", "test", "ami/test-*.parquet", "en", "extra"),
    _src(OAL, "earnings22", "test", "earnings22/test-*.parquet", "en", "extra"),
    _src(OAL, "gigaspeech", "test", "gigaspeech/test-*.parquet", "en", "extra"),
    _src(OAL, "voxpopuli", "test", "voxpopuli/test-*.parquet", "en", "extra"),
    _src(OAL, "urgent2024_clean", "test", "urgent2024_clean/test-*.parquet", "en", "extra"),
    # Legacy ESB config still in the repo (not a current leaderboard column)
    _src(OAL, "common_voice", "test", "common_voice/test-*.parquet", "en", "legacy"),
    # English long-form
    _src("hf-audio/asr-leaderboard-longform", "earnings21", "test", "earnings21/test-*.parquet", "en", "longform"),
    _src("hf-audio/asr-leaderboard-longform", "earnings22", "test", "earnings22/test-*.parquet", "en", "longform"),
    *[
        _src("bezzam/coraal", sub, "test", f"{sub}/test-*.parquet", "en", "longform")
        for sub in ["ATL", "DCA", "DCB", "DTA", "LES", "PRV", "ROC", "VLD"]
    ],
    # Multilingual (ML_LANG_DATASETS in eval_utils.score_results)
    *[
        _src(ML, f"{ds}_{lang}", "test", f"data/{ds}/{lang}_test.parquet", lang, f"ml_{lang}")
        for ds, langs in [
            ("fleurs", ["de", "fr", "it", "es", "pt", "nl"]),
            ("mcv", ["de", "fr", "it", "es", "nl"]),
            ("mls", ["fr", "it", "es", "pt", "nl"]),
        ]
        for lang in langs
    ],
    # Armenian
    _src(HY, "fleurs_hy", "test", "fleurs_hy/hy_test.parquet", "hy", "ml_hy"),
    _src(HY, "mcv_hy", "test", "mcv_hy/hy_test.parquet", "hy", "ml_hy"),
]

# Public results bucket used for real (reference, prediction) pairs.
PAIRS_BUCKET = "hf-audio/asr_leaderboard_multilingual"
PAIRS_EN_BUCKET = "hf-audio/asr_leaderboard_h200"
PAIRS_MODEL = "openai-whisper-large-v3"


def source_id(src: dict) -> str:
    return f"{src['dataset'].replace('/', '__')}__{src['config']}__{src['split']}"


# --------------------------------------------------------------------------------------
# Worker: runs inside the subprocess with PYTHONPATH=<source repo>. Imports ONLY the
# original `normalizer` package.
# --------------------------------------------------------------------------------------
WORKER = r'''
import json, sys, platform, unicodedata, importlib.metadata as md

req_path, resp_path = sys.argv[1], sys.argv[2]
with open(req_path, encoding="utf-8") as f:
    req = json.load(f)

from normalizer import data_utils, eval_utils
from normalizer import EnglishTextNormalizer
from normalizer import english_abbreviations as ea
from normalizer.eval_utils import normalize_compound_pairs


def _ver(name):
    try:
        return md.version(name)
    except Exception:
        return None


def norm_one(text, lang):
    try:
        return eval_utils._normalize(text, lang), None
    except Exception as e:  # recorded, never swallowed silently
        return None, f"{type(e).__name__}: {e}"[:300]


op = req["op"]
if op == "introspect":
    en = EnglishTextNormalizer()
    resp = {
        "python": platform.python_version(),
        "unidata_version": unicodedata.unidata_version,
        "num2words": _ver("num2words"),
        "regex": _ver("regex"),
        "replacers": list(en.replacers.keys()),
        "ignore_patterns": en.ignore_patterns,
        "names": list(ea.english_name_normalizer.items()),
        "compounds": list(ea.english_compound_normalizer.items()),
        "spelling": list(ea.english_spelling_normalizer.items()),
        "filler_words": data_utils.FILLER_WORDS,
    }
elif op == "normalize":
    out = []
    for text, lang, prefilter in req["items"]:
        if prefilter and not data_utils.is_target_text_in_range(text):
            out.append([False, None, None])
            continue
        o, err = norm_one(text, lang)
        out.append([True, o, err])
    resp = {"results": out}
elif op == "filter":
    resp = {"results": [bool(data_utils.is_target_text_in_range(t)) for t in req["items"]]}
elif op == "pairs":
    out = []
    for refs, preds in req["items"]:
        try:
            r, p = normalize_compound_pairs(refs, preds)
            out.append([r, p, None])
        except Exception as e:
            out.append([None, None, f"{type(e).__name__}: {e}"[:300]])
    resp = {"results": out}
else:
    raise SystemExit(f"unknown op {op}")

with open(resp_path, "w", encoding="utf-8") as f:
    json.dump(resp, f, ensure_ascii=False)
'''


class Original:
    """Runs the original normalizer in a subprocess (never in this interpreter)."""

    def __init__(self, source_repo: Path, python: str):
        self.source_repo = source_repo
        self.python = python

    def call(self, request: dict) -> dict:
        with tempfile.TemporaryDirectory() as td:
            req = Path(td) / "req.json"
            resp = Path(td) / "resp.json"
            req.write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")
            env = dict(os.environ)
            env["PYTHONPATH"] = str(self.source_repo)
            env["PYTHONDONTWRITEBYTECODE"] = "1"  # never write into the source clone
            env.pop("PYTHONHASHSEED", None)
            subprocess.run(
                [self.python, "-c", WORKER, str(req), str(resp)],
                check=True,
                env=env,
                cwd=str(self.source_repo),
            )
            return json.loads(resp.read_text(encoding="utf-8"))

    def normalize(self, items):
        """items: [(text, lang, prefilter)] -> [(kept, output, error)]"""
        results = []
        chunk = 50_000
        for i in range(0, len(items), chunk):
            results += self.call({"op": "normalize", "items": [list(x) for x in items[i:i + chunk]]})["results"]
        return results


# --------------------------------------------------------------------------------------
# Fetching (text column only)
# --------------------------------------------------------------------------------------
def resolve_revision(dataset: str, cache: dict) -> str:
    if dataset not in cache:
        from huggingface_hub import HfApi

        cache[dataset] = HfApi().dataset_info(dataset).sha
    return cache[dataset]


def fetch_texts(src: dict, revision: str, cache_dir: Path) -> tuple[list, list]:
    """Return (texts, files) for a source at a pinned revision. Cached on disk."""
    cache_file = cache_dir / f"{source_id(src)}@{revision}.json"
    if cache_file.exists():
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        return data["texts"], data["files"]

    from concurrent.futures import ThreadPoolExecutor

    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    base = f"datasets/{src['dataset']}@{revision}/"
    files = sorted(fs.glob(base + src["files"]))
    if not files:
        raise RuntimeError(f"no files for {src['dataset']} {src['files']} @ {revision}")
    rel_files = [path[len(base):] if path.startswith(base) else path.split("/", 3)[-1] for path in files]

    def read_one(path):
        if src["format"] == "parquet":
            import pyarrow.parquet as pq

            with fs.open(path, "rb", block_size=1 << 20) as f:
                table = pq.ParquetFile(f).read(columns=[src["text_column"]])
            return table.column(src["text_column"]).to_pylist()
        if src["format"] == "jsonl":
            with fs.open(path, "rb") as f:
                return [json.loads(line)[src["text_column"]]
                        for line in f.read().decode("utf-8").splitlines() if line.strip()]
        raise ValueError(src["format"])

    # Files are independent range reads; fetch them concurrently, keep sorted-file order.
    with ThreadPoolExecutor(max_workers=8) as pool:
        texts = [t for chunk in pool.map(read_one, files) for t in chunk]
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps({"texts": texts, "files": rel_files}, ensure_ascii=False), encoding="utf-8")
    return texts, rel_files


def sha256_lines(lines) -> str:
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------------------
def row_bytes(inp, lang, expected) -> int:
    return len(json.dumps({"input": inp, "lang": lang, "expected": expected}, ensure_ascii=False).encode("utf-8")) + 1


def stratified_sample(rows, lang, quota_lines, quota_bytes, rng):
    """rows: [(input, expected)] (unique inputs). Deterministic, stratified by length.

    Four length buckets (quartiles); buckets are visited round-robin, each in its own
    shuffled order. A row that would exceed the byte quota is skipped; at least one
    row (the shortest) is always taken. Row sizes are estimated from the input alone
    (as if expected == input), so the sample does not depend on the normalizer's output
    and --no-full-sweep picks exactly the same rows.
    """
    if not rows:
        return []
    order = sorted(range(len(rows)), key=lambda i: (len(rows[i][0]), rows[i][0]))
    n_buckets = 4
    buckets = [order[k * len(order) // n_buckets:(k + 1) * len(order) // n_buckets] for k in range(n_buckets)]
    buckets = [b for b in buckets if b]
    for b in buckets:
        rng.shuffle(b)
    picked, used = [], 0
    cursors = [0] * len(buckets)
    while len(picked) < quota_lines and any(c < len(b) for c, b in zip(cursors, buckets)):
        for k, b in enumerate(buckets):
            if len(picked) >= quota_lines:
                break
            while cursors[k] < len(b):
                i = b[cursors[k]]
                cursors[k] += 1
                size = row_bytes(rows[i][0], lang, rows[i][0])
                if used + size <= quota_bytes:
                    picked.append(i)
                    used += size
                    break
    if not picked:
        picked = [order[0]]
    picked.sort()  # keep dataset order in the committed file
    return [rows[i] for i in picked]


def allocate(sizes: dict, total: int) -> dict:
    """Water-filling: split `total` evenly over the sources, capped by each source's size."""
    alloc = {k: 0 for k in sizes}
    active = [k for k in sizes if sizes[k] > 0]
    budget = total
    while active and budget > 0:
        share = budget // len(active)
        if share == 0:
            for k in active[:budget]:
                alloc[k] += 1
            break
        nxt = []
        for k in active:
            give = min(share, sizes[k] - alloc[k])
            alloc[k] += give
            budget -= give
            if alloc[k] < sizes[k]:
                nxt.append(k)
        active = nxt
    return alloc


# --------------------------------------------------------------------------------------
# Synthetic inputs (inputs only; expected values come from the original code)
# --------------------------------------------------------------------------------------
def _expand_filler(pattern: str) -> list:
    """Minimal and elongated spellings of a filler regex like ``u+h+m*``."""
    if not re.fullmatch(r"(?:[a-z](?:\+|\*|\{2,\})?|\[hm\]\*)+", pattern):
        return [pattern]  # literal alternative ("ahem", "ah-ha", ...)
    tokens = re.findall(r"([a-z])(\+|\*|\{2,\}|)|\[hm\]\*", pattern)
    minimal, long_ = "", ""
    for ch, q in tokens:
        if not ch:  # [hm]*
            long_ += "hm"
            continue
        if q == "+":
            minimal += ch
            long_ += ch * 3
        elif q == "*":
            long_ += ch * 2
        elif q == "{2,}":
            minimal += ch * 2
            long_ += ch * 4
        else:
            minimal += ch
            long_ += ch
    out = [minimal, long_]
    return [w for i, w in enumerate(out) if w and w not in out[:i]]


def _replacer_examples(pattern: str) -> list:
    if pattern == r"\b(\d{1,2}):00\b":
        return ["2:00", "11:00", "9:00", "10:00:00", "123:00", "3:000"]
    m = re.fullmatch(r"\\b\(([a-z|]+)\)'s\\b", pattern)
    if m:
        return [f"{w}'s" for w in m.group(1).split("|")]
    body = pattern.replace(r"\b", "")
    if re.search(r"[\\()\[\]{}+*?|^$]", body):
        raise ValueError(f"cannot build an example for replacer {pattern!r}")
    if body.startswith("'"):
        # suffix contractions: attach to a plausible stem
        stems = {"'d been": "she", "'s been": "it", "'d gone": "they", "'s gone": "he",
                 "'d done": "we", "'s got": "she", "n't": "is", "'re": "you", "'d": "i",
                 "'ll": "we", "'t": "don", "'ve": "they", "'m": "i"}
        if body not in stems:
            raise ValueError(f"no stem for {body!r}")
        return [stems[body] + body]
    if body == "n't":
        return ["isn't", "doesn't"]
    return [body]


def synthetic_en(intro: dict) -> list:
    S = []
    add = S.append

    # Empty / whitespace
    for s in ["", " ", "   ", "\t", "\n", " \t\n ", "\u00a0", "\u3000", ".", "...", "?!", "-", "'"]:
        add(s)

    # Brackets and parentheses
    for s in [
        "hello [noise] world", "hello <unk> world", "<laugh> yes", "[inaudible]", "(laughs) okay",
        "keep (this part) out", "nested [a [b] c] d", "unbalanced [bracket here", "unbalanced (paren here",
        "a (b) c (d) e", "a <b> [c] (d) e", "empty () parens", "brackets [] empty", "x (y (z)) w",
        "<b>bold</b> text", "{curly} braces", "(full sentence in parens)", "foo[bar]baz",
    ]:
        add(s)

    # Fillers, incl. elongations and look-alikes
    filler_alts = re.match(r"\\b\((.*)\)\\b$", intro["ignore_patterns"]).group(1).split("|")
    for alt in filler_alts:
        for w in _expand_filler(alt):
            add(f"well {w} i think")
            add(f"{w.capitalize()}, yes.")
    for s in [
        "uhhhm", "Uhhhm, so", "UMMM what", "uh-huh", "um-hmm right", "mm-hmm", "ah-ha got it", "aha moment",
        "hmmm interesting", "err no", "ehm well", "ugh fine", "tsk tsk", "hmmph", "mmkay", "hunh",
        "umbrella", "hummus", "ahead", "error", "emma", "hum", "huh?", "ha ha", "hah", "heh", "eheh", "ehehe",
        "erm, uh, um", "uh uh uh", "so...uh...yeah", "uh,um,er", "Um. Uh. Er.", "mhm", "mmhm", "mh",
    ]:
        add(s)

    # Every replacer (contractions, titles, perfect tenses, o'clock times)
    for pattern in intro["replacers"]:
        for ex in _replacer_examples(pattern):
            add(f"well {ex} today")
            add(f"{ex.upper()}.")
    for s in [
        "Mr. Smith and Mrs. Jones", "Dr. Who", "St. Louis", "Prof. X", "Lt. Col. Smith", "Jr. and Sr.",
        "it's 3 o'clock", "at 2:00 AM", "at 2:00 p.m.", "10:30 am", "11:00", "at 3:00 o'clock", "5 o'clock shadow",
        "won't can't don't", "I'd've", "y'all'll", "ma'am", "I'ma go", "gonna wanna gotta", "she 's here",
        "rock 'n' roll", "'tis the season", "o'neill's", "the '90s", "it 's", "they ' re",
        "it’s curly", "don’t", "let's go", "ain't", "woulda coulda shoulda",
    ]:
        add(s)

    # Numbers, ordinals, fractions, currency, percent, decimals, years, phone-like runs
    for s in [
        "one", "ones", "one two three", "twenty one", "twenty-one", "a hundred", "one hundred and five",
        "two thousand and twenty four", "nineteen ninety nine", "nineteen oh five", "four oh one",
        "oh seven nine eight", "oh oh", "oh no", "oh", "zero", "zero zero seven", "double seven",
        "triple five", "double o seven", "double you", "first", "second", "twenty first", "hundredth",
        "the 1st", "the 2nd", "3rd", "4th", "21st", "the 1960s", "1960's", "the sixties", "eighties",
        "two and a half", "three and a half million", "and a half", "a half", "one half", "1/2", "3/4",
        "five point two", "point five", "3.14", "0.5", "1.0", "1.", ".5", "1.2.3", "192.168.0.1",
        "$20", "$20 million", "20 dollars", "twenty dollars", "$2.50", "2 dollars and 7 cents",
        "two dollars and seven cents", "$0.07", "€5", "£3.50", "5 euros", "ten pounds", "50 cents",
        "¢5", "50%", "50 percent", "fifty per cent", "12.5%", "100 percent sure", "per cent",
        "minus five", "negative 3", "plus two", "positive", "-5", "+5", "1,000", "1,000,000",
        "10,5", "1 000", "2024", "the year 2024", "in 1999", "1999-2000", "555-1234", "(555) 123-4567",
        "+1 800 555 0199", "800 555 0199", "0049 30 1234", "000", "007", "0", "01", "1", "1s", "11",
        "one million", "a million dollars", "two billion", "3 trillion", "a quadrillion", "decillions",
        "123456789012345678901234567890", "1e5", "10^3", "2x", "x2", "mp3", "covid-19", "b2b", "4k",
        "5g", "5 g", "v8", "a4 paper", "1980s", "80s", "'80s", "the 80's", "1st-class",
        "twenty twenty", "twenty twenty-four", "number nine", "nine eleven", "seven eleven",
        "one oh one", "one o one", "zero point five", "point", "and", "double", "triple",
        "two hundred thousand", "thousand", "hundreds of people", "thousands", "millions of dollars",
        "a dozen", "twelve dozen", "fifteenth", "nineteenth", "ninetieth", "millionth",
        "sixes and sevens", "twos", "1 and a half", "3 and a half", "seventies", "fortieth",
        # number-normalizer corner cases (dropped words, odd joins)
        "point point", "and a half and a half", "and a half", "an hour and a half", "1.2.3.", "$.5",
        "1/0", "0/0 x", "point hundred", "double dollars", "minus", "dollars", "per cent", "percent",
        # digit runs at/over Python's 4300-digit int() limit: the original raises ValueError above it
        "9" * 4300, "1" * 4301,
    ]:
        add(s)

    # Spelled letters / acronyms
    for s in [
        "c o 2", "c o 2 emissions", "co2 emissions", "CO 2", "CO2", "h 2 o", "b b c", "the b b c news",
        "u s a", "U.S.A.", "u. s. a.", "the U.S.", "a b c", "a big cat", "i am a", "i b m", "a i",
        "x y z", "f b i agent", "n a s a", "5 g network", "j f k", "t v", "a m", "p m", "o k",
        "a.m.", "p.m.", "9 a m", "9 am", "i o u", "a 1", "e e", "o o", "m i 6",
    ]:
        add(s)

    # Names (all keys of the original name map, several per line) and their canonicals
    names = [k for k, _ in intro["names"]]
    for i in range(0, len(names), 4):
        add(" and ".join(n.capitalize() for n in names[i:i + 4]) + " came by")
    add(" ".join(sorted({v for _, v in intro["names"]})[:30]))

    # Compounds (every pattern of the original compound map)
    compound_examples = {
        r"\bo+h+\b": ["ooh", "ohh", "oooohhh", "oh"],
        r"\booo+\b": ["ooo", "oooooo"],
    }
    for pattern, _repl in intro["compounds"]:
        if pattern in compound_examples:
            exs = compound_examples[pattern]
        else:
            m = re.fullmatch(r"\\b([a-z]+)\\s\+([a-z]+)\\b", pattern)
            if not m:
                raise ValueError(f"cannot build an example for compound {pattern!r}")
            a, b = m.groups()
            exs = [f"{a} {b}", f"{a}-{b}", f"{a.upper()}‑{b}", f"{a}  {b}", f"{a}{b}"]
        for ex in exs:
            add(f"the {ex} works")

    # British spellings (deterministic slice of the original mapping)
    spelling = intro["spelling"]
    step = max(1, len(spelling) // 40)
    for i in range(0, len(spelling), step):
        add(f"the {spelling[i][0]} was fine")

    # Accented and non-ASCII text, symbols, punctuation
    for s in [
        "café résumé naïve", "Crème Brûlée", "Ångström", "smørrebrød", "Œuvre", "straße", "ẞ", "þorn",
        "Łódź", "Đorđe", "æsthetic", "façade", "jalapeño", "São Paulo", "Zürich", "Dvořák", "İstanbul",
        "ﬁne ligature", "Ⅻ roman", "x²", "½ cup", "①", "ｆｕｌｌｗｉｄｔｈ", "日本語のテキスト", "中文 3 个",
        "Привет мир", "مرحبا", "שלום", "नमस्ते दुनिया", "🙂 emoji", "rock ♫ music", "a—b–c", "«quoted»",
        "“smart quotes”", "‘single’", "C++ and C#", "AT&T", "R&D", "50/50", "and/or", "e.g.", "i.e.",
        "etc.", "et cetera", "vs.", "#hashtag", "@mention", "user@example.com", "http://example.com/a?b=c",
        "a_b", "snake_case_word", "tab\tseparated", "line\nbreak", "multiple   spaces", "trailing space ",
        " leading space", "ALL CAPS SENTENCE", "MiXeD CaSe", "hello... world", "wait—what?", "so-called",
        "well-known", "e-mail", "Wi-Fi", "T-shirt", "x-ray", "sci-fi", "all right", "al right", "how ever",
        "e book", "e commerce", "hi fi", "lo fi", "o k then", "it's ok", "okay", "OK", "o.k.",
    ]:
        add(s)

    # Realistic sentences mixing stages
    for s in [
        "Uh, Mr. Smith said he'd been to St. Louis at 3:00 PM, and it cost $20.50 — uh, roughly 50%.",
        "[noise] So, um, the C.E.O. — y'know — gonna announce Q3 earnings of $1.2 billion (approx).",
        "I can't believe it's 2024; we've got twenty-one items, two and a half hours, and the b b c.",
        "The colour of the theatre's centre was grey, Dr. Kris said.",
        "We'll meet at 10:00, uh, or maybe eleven o'clock, ok?",
        "Revenue grew 12.5% year-over-year to €3.4 million in the second quarter.",
        "Call 1-800-555-0199 or nine one one.",
        "He won't go; she'll stay — that's what ma'am said.",
        "So the, uh, the c o 2 levels went from three hundred to four hundred ppm.",
        "In nineteen eighty four, twenty percent of the ones were sixes.",
    ]:
        add(s)

    # Unique, order-preserving
    seen, out = set(), []
    for s in S:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return [(s, "en") for s in out]


ML_LANGS = ["de", "fr", "es", "it", "pt", "nl", "hi", "ja", "zh", None]

ML_STRINGS = [
    "", "   ", "Hallo Welt", "Ich habe 3 Äpfel und 21 Birnen.", "10 000 Einwohner", "1 000 000", "10 000 000 000",
    "Es kostet 1.234,56 €.", "Am 3. Oktober 2023 um 14:30 Uhr", "12,5 %", "Nummer 007", "1234567890123456789012345",
    "äh ähm also", "euh ben voilà", "eh bueno pues", "ehm allora", "é pá então", "eh nou ja", "uh um",
    "C'est l'été à 20 h", "Señor Núñez tiene 2 años", "Il y a 1 000 personnes", "São 300 mil pessoas",
    "Het is 25 graden", "[ruido] hola (risas) mundo", "<unk> test", "«Bonjour»", "„Guten Tag“",
    "Straße Œuvre Ærø", "naïve café", "Ελληνικά 5", "Русский текст 42", "हिन्दी में 5 शब्द", "किताब 12 रुपये",
    "क़िला ज़रूर", "日本語 3 つ", "東京は2024年", "中文 100 个", "我有3个苹果", "١٢٣ arabic digits", "２０２４ fullwidth",
    "x² ½ ①", "e-mail und so", "o'clock 3", "50/50", "3.14", "-5 Grad", "+33 1 23 45 67 89",
    "tab\tsep", "line\nbreak", "😀 emoji", "UPPER lower MiXeD",
]


HY_STRINGS = [
    "", "   ", "Բարեւ", "Բարև ձեզ։", "Ես ունեմ 25 խնձոր։", "Նա 3 տարեկան է", "100 մարդ եկավ",
    "1-ին դասարան", "2-րդ հարկ", "21-րդ դար", "3րդ անգամ", "11-րդ", "1-ինը", "5-ից ավելի", "10-ում",
    "20-ական թվականներ", "1990-ական", "1990 ական", "2-ի", "4-ով", "3-երորդ",
    "50%", "50 %", "50%-ով", "50 %-ից", "12,5%", "12.5 տոկոս", "3 տոկոսից", "հարյուր տոկոս",
    "10:30", "ժամը 10:30-ին", "ժամը 0930", "9.15-ին", "0930", "23:59", "24:00", "7։45",
    "2023 թ.", "2023թ.", "2023 թվականին", "2023թ.-ին", "2023 թ. -ից", "1918 թվականի մայիսի 28-ին",
    "Ք.հ. 301 թ.", "քրիստոսից հետո 451", "ք. հ. 5-րդ դար",
    "3/4", "1/2 մաս", "3,5", "3.14", "0,5 կգ", "1 000 000", "1,250", "1.250", "1 250 000 դրամ",
    "5 մլն դրամ", "2 մլրդ", "60 կմ/ժ", "$100", "100$", "100 դոլար", "100 դոլլար", "$ 5",
    "քսանհինգ", "երկու հազար քսան", "մեկուկես ժամ", "կես ժամ", "քառորդ", "առաջին", "հարյուրերորդ",
    "երկուսը", "մեկին", "տասներկուսն", "զրո", "իրեք", "իննսունինը",
    "café résumé", "Ñandú", "Ångström", "Zoë", "naïve", "iPhone 15", "Covid-19-ի", "ԱՄՆ-ի", "ՆԱՏՕ-ն",
    "Москва 2020", "Ռուսաստան-Հայաստան", "«Ամենը» ասաց", "<<չակերտներ>>", "‹‹բառ››", "Եվ նա", "ԵՎ", "եւ",
    "Ի՞նչ ես անում", "Ո՜վ", "Այո՛", "բառ՝ բառ", "Վերջ . . .", "Վերջ...", "բառ ,բառ", "- ցուցակ",
    "– գիծ", "a—b", "1.2.3a", "v1.2.3", "10.5.1b", "13:00-ից", "մաս / մաս",
    "123456789012345", "1000000000000", "999999999999", "0", "007", "01",
    "Տող\nնոր տող", "Tab\tհետո", "բառ (փակագծեր) բառ", "[աղմուկ] խոսք",
]


# Extra (string, lang) rows not crossed with every language
ML_EXTRA = [
    ("1" * 4301, "de"),  # over the int() digit limit: num2words fails, digits are kept
    ("1 000 000 000", "de"),
    ("1 000 000", "fr"),
    ("20 000 000 Menschen", "de"),
    ("80 ans", "fr"),
    ("71 años", "es"),
    ("1e10", "de"),
]


def synthetic_ml():
    rows = []
    for lang in ML_LANGS:
        for s in ML_STRINGS:
            rows.append((s, lang))
    return rows + ML_EXTRA


def synthetic_hy():
    return [(s, "hy") for s in HY_STRINGS] + [("1" * 4301, "hy")]


# --------------------------------------------------------------------------------------
# Pairs for normalize_compound_pairs
# --------------------------------------------------------------------------------------
PAIRS_SPECS = [
    # (bucket, filename suffix, lang)
    (PAIRS_BUCKET, "fleurs_de_test.jsonl", "de"),
    (PAIRS_BUCKET, "fleurs_nl_test.jsonl", "nl"),
    (PAIRS_BUCKET, "fleurs_it_test.jsonl", "it"),
    (PAIRS_BUCKET, "fleurs_fr_test.jsonl", "fr"),
    (PAIRS_BUCKET, "fleurs_es_test.jsonl", "es"),
    (PAIRS_BUCKET, "fleurs_pt_test.jsonl", "pt"),
    (PAIRS_BUCKET, "fleurs_hy_test.jsonl", "hy"),
    (PAIRS_EN_BUCKET, "voxpopuli_cleaned_aa_test.jsonl", "en"),
]


def fetch_pair_manifests(cache_dir: Path):
    """Real (ref, pred) pairs of one model from the public results buckets."""
    from huggingface_hub import HfApi, HfFileSystem

    api = HfApi()
    fs = HfFileSystem()
    out = []
    listings = {}
    for bucket, suffix, lang in PAIRS_SPECS:
        if bucket not in listings:
            listings[bucket] = sorted(
                (f.path, f.size)
                for f in api.list_bucket_tree(bucket, prefix=PAIRS_MODEL, recursive=True)
                if getattr(f, "type", None) == "file"
            )
        cands = [(p, s) for p, s in listings[bucket] if p.startswith(PAIRS_MODEL + "/") and p.endswith(suffix)]
        if not cands:
            print(f"  pairs: no {suffix} for {PAIRS_MODEL} in {bucket}", file=sys.stderr)
            continue
        path, size = cands[0]
        cache_file = cache_dir / ("bucket__" + bucket.replace("/", "__") + "__" + path.replace("/", "__"))
        if cache_file.exists():
            raw = cache_file.read_bytes()
        else:
            with fs.open(f"buckets/{bucket}/{path}", "rb") as f:
                raw = f.read()
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file.write_bytes(raw)
        rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
        out.append({
            "bucket": bucket, "path": path, "size": size, "lang": lang,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "pairs": [(r["text"], r["pred_text"] or "") for r in rows],
        })
    return out


def build_pairs(original: Original, manifests, rng, n_changed=12, n_random=6):
    """Normalize refs/preds per language with the original code, then pick pairs."""
    cases = []  # (refs, preds) as passed to normalize_compound_pairs (already normalized)
    for m in manifests:
        items = [(t, m["lang"], False) for pair in m["pairs"] for t in pair]
        res = original.normalize(items)
        normed = []
        for i in range(0, len(res), 2):
            (_, r, er), (_, p, ep) = res[i], res[i + 1]
            if er or ep:
                continue
            normed.append((r, p))
        out = original.call({"op": "pairs", "items": [[[r], [p]] for r, p in normed]})["results"]
        changed = [i for i, (o, (r, p)) in enumerate(zip(out, normed)) if o[2] is None and (o[0][0] != r or o[1][0] != p)]
        changed_set = set(changed)
        unchanged = [i for i in range(len(normed)) if i not in changed_set]
        rng.shuffle(changed)
        rng.shuffle(unchanged)
        pick = sorted(changed[:n_changed]) + sorted(unchanged[:n_random])
        for i in pick:
            cases.append(([normed[i][0]], [normed[i][1]]))
        # one multi-pair call per language (lists of 5)
        multi = sorted((changed[n_changed:] + unchanged[n_random:])[:5])
        if multi:
            cases.append(([normed[i][0] for i in multi], [normed[i][1] for i in multi]))
    # Edge cases
    cases += [
        ([], []),
        ([""], [""]),
        ([""], ["a b"]),
        (["a b"], [""]),
        (["white paper"], ["whitepaper"]),
        (["whitepaper"], ["white paper"]),
        (["haus tür offen"], ["haustür offen"]),
        (["ab c d"], ["a bc d"]),
        (["a b c"], ["abc"]),
        (["new york city"], ["newyork city"]),
        (["the cat sat"], ["the dog sat"]),
        (["x  y   z"], ["x y z"]),
        (["a b", "c d"], ["ab"]),  # zip truncates to the shorter list
        (["ab"], ["a b", "c"]),
        (["eins zwei drei"], ["einszwei drei", ]),
        (["fußball spiel"], ["fußballspiel"]),
        (["über all"], ["überall"]),
        (["a a a"], ["aa a"]),
        (["a b a b"], ["ab ab"]),
        (["re do it"], ["redo it", ]),
        (["1 000"], ["1000"]),
        (["\u0570\u0561\u0575 \u0565\u0580\u0565\u0576"], ["\u0570\u0561\u0575\u0565\u0580\u0565\u0576"]),
        (["tab\tsep"], ["tabsep"]),
        (["line\nbreak"], ["linebreak"]),
    ]
    out = original.call({"op": "pairs", "items": [[r, p] for r, p in cases]})["results"]
    rows = []
    for (refs, preds), (er, ep, err) in zip(cases, out):
        if err:
            raise RuntimeError(f"normalize_compound_pairs raised on {refs!r}/{preds!r}: {err}")
        rows.append({"refs": refs, "preds": preds, "expected_refs": er, "expected_preds": ep})
    return rows


# --------------------------------------------------------------------------------------
def git_head(repo: Path) -> str:
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()


def write_jsonl(path: Path, rows) -> tuple[int, int]:
    data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    path.write_text(data, encoding="utf-8")
    return len(rows), len(data.encode("utf-8"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source-repo", default=DEFAULT_SOURCE_REPO, required=DEFAULT_SOURCE_REPO is None,
                    help="clone of huggingface/open_asr_leaderboard at commit 60e2e3b")
    ap.add_argument("--out", default=str(PACKAGE_ROOT / "tests" / "golden"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-total", type=int, default=10_000, help="max fixture lines over all files")
    ap.add_argument("--max-bytes", type=int, default=2_900_000, help="max fixture bytes over all files")
    ap.add_argument("--full-sweep", action=argparse.BooleanOptionalAction, default=True,
                    help="normalize every reference and record sha256 (default); with --no-full-sweep "
                         "the sha256/total_rows of the existing manifest are kept")
    ap.add_argument("--python", default=sys.executable, help="interpreter that runs the original code")
    ap.add_argument("--cache-dir", default=os.path.expanduser("~/.cache/asr-normalizer-golden"))
    ap.add_argument("--only", nargs="*", default=None,
                    help="regenerate only sources whose id contains one of these (needs an existing manifest; "
                         "the sample is identical to a full run's; manifest.json is not rewritten)")
    ap.add_argument("--refresh-revisions", action="store_true",
                    help="resolve the latest Hub revision of every dataset instead of reusing the manifest's")
    ap.add_argument("--allow-other-commit", action="store_true")
    args = ap.parse_args(argv)

    source_repo = Path(args.source_repo).resolve()
    out_dir = Path(args.out).resolve()
    cache_dir = Path(args.cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    head = git_head(source_repo)
    if not head.startswith(EXPECTED_SOURCE_COMMIT) and not args.allow_other_commit:
        sys.exit(f"source repo is at {head}, expected {EXPECTED_SOURCE_COMMIT} (use --allow-other-commit)")

    original = Original(source_repo, args.python)
    intro = original.call({"op": "introspect"})
    print(f"original normalizer: {source_repo} @ {head[:12]}; python {intro['python']}, "
          f"unidata {intro['unidata_version']}, num2words {intro['num2words']}, regex {intro['regex']}")

    manifest_path = out_dir / "manifest.json"
    old_manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    old_sources = {s["file"]: s for s in old_manifest.get("sources", [])}

    sources = SOURCES
    if args.only:
        sources = [s for s in SOURCES if any(o in source_id(s) for o in args.only)]

    # ---- synthetic files first (they reserve part of the budget) ----
    synth_specs = [
        ("synthetic__en.jsonl", synthetic_en(intro)),
        ("synthetic__ml.jsonl", synthetic_ml()),
        ("synthetic__hy.jsonl", synthetic_hy()),
    ]
    synth_meta = []
    used_lines = used_bytes = 0
    for fname, items in synth_specs:
        res = original.normalize([(t, lang, False) for t, lang in items])
        rows = []
        for (t, lang), (_, o, err) in zip(items, res):
            row = {"input": t, "lang": lang, "expected": o}
            if err:
                row["error"] = err
            rows.append(row)
        n, b = write_jsonl(out_dir / fname, rows)
        used_lines += n
        used_bytes += b
        synth_meta.append({"file": fname, "rows": n, "bytes": b,
                           "errors": sum(1 for r in rows if "error" in r)})
        print(f"  {fname}: {n} rows, {b} bytes")

    # ---- pairs ----
    rng_pairs = random.Random(f"{args.seed}|pairs")
    pair_manifests = fetch_pair_manifests(cache_dir)
    pair_rows = build_pairs(original, pair_manifests, rng_pairs)
    n, b = write_jsonl(out_dir / "pairs.jsonl", pair_rows)
    used_lines += n
    used_bytes += b
    pairs_meta = {
        "file": "pairs.jsonl", "rows": n, "bytes": b, "model": PAIRS_MODEL,
        "sources": [{k: m[k] for k in ("bucket", "path", "size", "sha256", "lang")} for m in pair_manifests],
        "note": "refs/preds are the leaderboard-normalized (original eval_utils._normalize) texts of real "
                "reference/prediction pairs, as passed to normalize_compound_pairs at scoring time; "
                "plus synthetic edge cases at the end",
    }
    print(f"  pairs.jsonl: {n} rows, {b} bytes")

    # ---- datasets ----
    revisions = {}
    if not args.refresh_revisions:
        for old in old_sources.values():
            revisions.setdefault(old["dataset"], old["revision"])
    fetched = {}
    for src in sources:
        rev = resolve_revision(src["dataset"], revisions)
        texts, files = fetch_texts(src, rev, cache_dir)
        fetched[source_id(src)] = (src, rev, texts, files)
        print(f"  fetched {source_id(src)} @ {rev[:12]}: {len(texts)} rows from {len(files)} file(s)")

    # Normalize: everything (full sweep) or nothing yet (sample-only mode normalizes after sampling)
    results = {}
    for sid, (src, rev, texts, files) in fetched.items():
        nulls = sum(1 for t in texts if t is None)
        if nulls:
            print(f"  WARNING {sid}: {nulls} null transcripts skipped", file=sys.stderr)
        valid = [t for t in texts if t is not None]
        if args.full_sweep:
            res = original.normalize([(t, src["lang"], src["prefilter"]) for t in valid])
        else:
            res = None
            if src["prefilter"]:  # same pre-filter as the full sweep, with the original function
                keep = original.call({"op": "filter", "items": valid})["results"]
                valid = [t for t, k in zip(valid, keep) if k]
        results[sid] = (valid, res, nulls)

    # Pre-filter + unique inputs per source, for sampling
    candidates = {}
    for sid, (valid, res, _) in results.items():
        src = fetched[sid][0]
        if res is not None:
            rows = [(t, o) for t, (kept, o, err) in zip(valid, res) if kept and err is None]
        else:
            rows = [(t, None) for t in valid]
        seen, uniq = set(), []
        for t, o in rows:
            if t not in seen:
                seen.add(t)
                uniq.append((t, o))
        candidates[sid] = uniq

    line_budget = max(0, args.max_total - used_lines)
    byte_budget = max(0, args.max_bytes - used_bytes)
    # Budgets are always split over ALL sources (sizes of unselected ones come from the manifest),
    # so --only reproduces exactly the sample a full run would write.
    sizes = {}
    for src in SOURCES:
        sid = source_id(src)
        if sid in candidates:
            sizes[sid] = len(candidates[sid])
        else:
            old = old_sources.get(f"{sid}.jsonl")
            if not old or old.get("unique_inputs") is None:
                sys.exit(f"--only needs an existing manifest entry for {sid}; run without --only first")
            sizes[sid] = old["unique_inputs"]
    line_alloc = allocate(sizes, line_budget)
    per_source_bytes = byte_budget // max(1, len(SOURCES))

    manifest_sources = []
    for sid, uniq in candidates.items():
        src, rev, texts, files = fetched[sid]
        valid, res, nulls = results[sid]
        rng = random.Random(f"{args.seed}|{sid}")
        if res is None:
            # sample on inputs only, normalize the sample afterwards
            approx = [(t, t) for t, _ in uniq]
            sample = stratified_sample(approx, src["lang"], line_alloc[sid], per_source_bytes, rng)
            sres = original.normalize([(t, src["lang"], False) for t, _ in sample])
            sample = [(t, o) for (t, _), (_, o, err) in zip(sample, sres) if err is None]
        else:
            sample = stratified_sample(uniq, src["lang"], line_alloc[sid], per_source_bytes, rng)
        fname = f"{sid}.jsonl"
        n, b = write_jsonl(out_dir / fname, [{"input": t, "lang": src["lang"], "expected": o} for t, o in sample])

        if res is not None:
            kept = [(t, o, err) for t, (k, o, err) in zip(valid, res) if k]
            errors = [(t, err) for t, o, err in kept if err]
            outputs = [o if err is None else "" for t, o, err in kept]
            total_rows = len(kept)
            sha = sha256_lines(outputs)
            filtered_out = len(valid) - len(kept)
            empty_outputs = sum(1 for o in outputs if o == "")
            err_count = len(errors)
            if errors:
                print(f"  WARNING {sid}: {len(errors)} inputs raised, e.g. {errors[0][1]}", file=sys.stderr)
        else:
            old = old_sources.get(fname, {})
            if old.get("revision") != rev:
                print(f"  WARNING {sid}: no full sweep and no matching manifest entry; sha256 left null",
                      file=sys.stderr)
                old = {}
            total_rows = old.get("total_rows")
            sha = old.get("sha256_full")
            filtered_out = old.get("prefiltered_out")
            empty_outputs = old.get("empty_outputs")
            err_count = old.get("errors")

        manifest_sources.append({
            "dataset": src["dataset"],
            "revision": rev,
            "config": src["config"],
            "split": src["split"],
            "text_column": src["text_column"],
            "lang": src["lang"],
            "family": src["family"],
            "format": src["format"],
            "data_files": files,
            "prefilter": "is_target_text_in_range" if src["prefilter"] else None,
            "raw_rows": len(texts),
            "null_rows": nulls,
            "prefiltered_out": filtered_out,
            "total_rows": total_rows,
            "unique_inputs": len(uniq),
            "empty_outputs": empty_outputs,
            "errors": err_count,
            "sampled_rows": n,
            "sampled_bytes": b,
            "sha256_full": sha,
            "file": fname,
        })
        used_lines += n
        used_bytes += b
        print(f"  {fname}: total {total_rows}, sampled {n} ({b} bytes), sha {str(sha)[:12]}")

    if args.only:
        print("--only given: manifest not rewritten")
        return

    manifest = {
        "source_repo": SOURCE_REPO_NAME,
        "source_commit": EXPECTED_SOURCE_COMMIT,
        "source_commit_full": head,
        "generated": _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat(),
        "seed": args.seed,
        "python": intro["python"],
        "unidata_version": intro["unidata_version"],
        "num2words": intro["num2words"],
        "regex": intro["regex"],
        "routing": "lang 'en' -> EnglishTextNormalizer(); else MultilingualNormalizer(remove_diacritics=False)"
                   "(text, lang=lang) ('hy' -> ArmenianTextNormalizer); i.e. eval_utils._normalize",
        "sha256_full_definition": "sha256 of '\\n'.join(normalized outputs of every kept row, in dataset "
                                  "order: data_files sorted, row order within file), utf-8",
        "prefilter_note": "multilingual/Armenian rows failing data_utils.is_target_text_in_range(raw) are "
                          "dropped before normalization, as in */run_eval_ml.py; English rows are not "
                          "(prepare_data filters on the normalized text instead)",
        "totals": {"lines": used_lines, "bytes": used_bytes},
        "synthetic": synth_meta,
        "pairs": pairs_meta,
        "sources": manifest_sources,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {manifest_path}: {used_lines} lines, {used_bytes} bytes in fixtures")


if __name__ == "__main__":
    main()
