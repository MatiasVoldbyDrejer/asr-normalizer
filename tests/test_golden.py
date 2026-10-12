"""Behaviour-preservation gate: outputs of asr_normalizer == outputs of the original code.

Every expected value under tests/golden/ was produced by the ORIGINAL normalizer of
huggingface/open_asr_leaderboard (commit 60e2e3b) via scripts/make_golden.py; see
tests/golden/manifest.json for provenance. Regenerate with that script, never by hand.
"""

import hashlib
import json
from pathlib import Path

import pytest

from asr_normalizer import MultilingualNormalizer, normalize, normalize_compound_pairs

GOLDEN = Path(__file__).parent / "golden"
MANIFEST = json.loads((GOLDEN / "manifest.json").read_text(encoding="utf-8"))
FILES = sorted(p.name for p in GOLDEN.glob("*.jsonl") if p.name != "pairs.jsonl")
MAX_REPORTED = 20

_ML_NO_LANG = MultilingualNormalizer(remove_diacritics=False)


def _normalize(text, lang):
    # lang None means "no lang": the plain BasicMultilingualTextNormalizer path
    if lang is None:
        return _ML_NO_LANG(text)
    return normalize(text, lang)


def _load(name):
    with open(GOLDEN / name, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _report(name, total, mismatches):
    lines = [f"{len(mismatches)}/{total} mismatching rows in {name} (showing first {MAX_REPORTED}):"]
    for lineno, row, got in mismatches[:MAX_REPORTED]:
        lines.append(f"  line {lineno} lang={row.get('lang')!r}")
        for key in ("input", "refs", "preds"):
            if key in row:
                lines.append(f"    {key + ':':10}{row[key]!r}")
        lines.append(f"    {'expected:':10}{row.get('expected', (row.get('expected_refs'), row.get('expected_preds')))!r}")
        lines.append(f"    {'got:':10}{got!r}")
    return "\n".join(lines)


def test_fixture_files_present():
    listed = {s["file"] for s in MANIFEST["sources"]} | {s["file"] for s in MANIFEST["synthetic"]}
    assert listed == set(FILES)
    assert (GOLDEN / MANIFEST["pairs"]["file"]).exists()


@pytest.mark.parametrize("name", FILES)
def test_golden_file(name):
    rows = _load(name)
    assert rows, f"{name} is empty"
    mismatches = []
    for lineno, row in enumerate(rows, 1):
        try:
            got = _normalize(row["input"], row["lang"])
        except Exception as e:  # noqa: BLE001 -- reported below, compared against the original
            got = {"raised": f"{type(e).__name__}: {e}"}
            if "error" in row and row["error"].split(":")[0] == type(e).__name__:
                continue  # the original raised the same exception type
        if "error" in row or got != row["expected"]:
            mismatches.append((lineno, row, got))
    if mismatches:
        pytest.fail(_report(name, len(rows), mismatches), pytrace=False)


def test_golden_pairs():
    rows = _load(MANIFEST["pairs"]["file"])
    assert rows
    mismatches = []
    for lineno, row in enumerate(rows, 1):
        got_refs, got_preds = normalize_compound_pairs(row["refs"], row["preds"])
        if (list(got_refs), list(got_preds)) != (row["expected_refs"], row["expected_preds"]):
            mismatches.append((lineno, row, (got_refs, got_preds)))
    if mismatches:
        pytest.fail(_report(MANIFEST["pairs"]["file"], len(rows), mismatches), pytrace=False)


# ----------------------------------------------------------------------------------------
# Full sweep: re-fetch every reference of every source at the recorded revision and compare
# the sha256 of the newline-joined outputs. Network + several minutes; ASR_NORMALIZER_FULL=1.
# ----------------------------------------------------------------------------------------
def _is_target_text_in_range(ref):
    # Same as the leaderboard's data_utils.is_target_text_in_range (not part of asr_normalizer).
    if ref.strip() == "ignore time segment in scoring":
        return False
    return ref.strip() != ""


def _fetch_texts(source):
    """Transcript column only (parquet range reads), files in manifest order."""
    from concurrent.futures import ThreadPoolExecutor

    hf = pytest.importorskip("huggingface_hub")
    pq = pytest.importorskip("pyarrow.parquet") if source["format"] == "parquet" else None
    fs = hf.HfFileSystem()
    column = source["text_column"]

    def read_one(rel):
        path = f"datasets/{source['dataset']}@{source['revision']}/{rel}"
        if source["format"] == "parquet":
            with fs.open(path, "rb", block_size=1 << 20) as f:
                return pq.ParquetFile(f).read(columns=[column]).column(column).to_pylist()
        if source["format"] == "jsonl":
            with fs.open(path, "rb") as f:
                return [json.loads(line)[column] for line in f.read().decode("utf-8").splitlines() if line.strip()]
        raise ValueError(source["format"])

    with ThreadPoolExecutor(max_workers=8) as pool:
        return [t for chunk in pool.map(read_one, source["data_files"]) for t in chunk]


@pytest.mark.full
@pytest.mark.parametrize("source", MANIFEST["sources"], ids=[s["file"] for s in MANIFEST["sources"]])
def test_full_sweep_sha256(source):
    texts = _fetch_texts(source)
    assert len(texts) == source["raw_rows"]
    texts = [t for t in texts if t is not None]
    if source["prefilter"]:
        texts = [t for t in texts if _is_target_text_in_range(t)]
    outputs, errors = [], 0
    for t in texts:
        try:
            outputs.append(normalize(t, source["lang"]))
        except Exception:  # noqa: BLE001 -- counted; the generator records "" for such rows
            errors += 1
            outputs.append("")
    assert len(outputs) == source["total_rows"]
    assert errors == source["errors"]
    sha = hashlib.sha256("\n".join(outputs).encode("utf-8")).hexdigest()
    assert sha == source["sha256_full"], (
        f"{source['file']}: full-sweep sha256 differs (fixtures generated with Python "
        f"{MANIFEST['python']}, unicodedata {MANIFEST['unidata_version']}, num2words {MANIFEST['num2words']})"
    )
