"""Public API of asr_normalizer (see SPEC): exports, version, routing of normalize(), CLI."""

import importlib
import inspect
import os
import subprocess
import sys

import pytest
from _unit_cases import CASES, META

import asr_normalizer
from asr_normalizer import (
    ArmenianTextNormalizer,
    BasicMultilingualTextNormalizer,
    BasicTextNormalizer,
    EnglishTextNormalizer,
    MultilingualNormalizer,
    normalize,
    normalize_compound_pairs,
)

EXPORTS = [
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
SUBMODULES = [
    "asr_normalizer.basic",
    "asr_normalizer.english",
    "asr_normalizer.english.data",
    "asr_normalizer.multilingual",
    "asr_normalizer.languages.hy",
    "asr_normalizer.pairs",
]
EN_SAMPLES = [c[0] for c in CASES["en_realistic"] + CASES["en_numbers"] + CASES["en_fillers"] + CASES["issue66"]]
ML_SAMPLES = ["Ich habe 3 Äpfel und 10 000 Einwohner.", "C'est l'été à 20 h", "Ես ունեմ 25 խնձոր։", "हिन्दी 5", ""]


@pytest.mark.parametrize("name", EXPORTS)
def test_export_exists(name):
    assert hasattr(asr_normalizer, name)


def test_version():
    assert asr_normalizer.__version__ == "0.1.0"


@pytest.mark.parametrize("module", SUBMODULES)
def test_submodule_importable(module):
    importlib.import_module(module)


def _params(obj):
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


_SIG_OBJECTS = {
    "EnglishTextNormalizer.__init__": EnglishTextNormalizer.__init__,
    "EnglishTextNormalizer.__call__": EnglishTextNormalizer.__call__,
    "BasicTextNormalizer.__init__": BasicTextNormalizer.__init__,
    "BasicTextNormalizer.__call__": BasicTextNormalizer.__call__,
    "BasicMultilingualTextNormalizer.__init__": BasicMultilingualTextNormalizer.__init__,
    "BasicMultilingualTextNormalizer.__call__": BasicMultilingualTextNormalizer.__call__,
    "MultilingualNormalizer.__init__": MultilingualNormalizer.__init__,
    "MultilingualNormalizer.__call__": MultilingualNormalizer.__call__,
    "ArmenianTextNormalizer.__call__": ArmenianTextNormalizer.__call__,
    "normalize_compound_pairs": normalize_compound_pairs,
}


@pytest.mark.parametrize("name", sorted(_SIG_OBJECTS))
def test_signatures_match_original(name):
    """Parameter names, kinds and defaults are identical to the original (annotations ignored)."""
    assert _params(_SIG_OBJECTS[name]) == META["signatures"][name]


def test_normalize_signature():
    params = inspect.signature(normalize).parameters
    assert list(params)[:2] == ["text", "lang"]
    assert params["lang"].default == "en"


@pytest.mark.parametrize("text", EN_SAMPLES)
def test_normalize_en_is_english_normalizer(text):
    assert normalize(text, "en") == EnglishTextNormalizer()(text)
    assert normalize(text) == normalize(text, "en")


@pytest.mark.parametrize("lang", ["de", "fr", "es", "it", "pt", "nl", "hy", "xx"])
@pytest.mark.parametrize("text", ML_SAMPLES)
def test_normalize_other_lang_is_multilingual(text, lang):
    assert normalize(text, lang) == MultilingualNormalizer(remove_diacritics=False)(text, lang=lang)


@pytest.mark.parametrize("text", ML_SAMPLES)
def test_normalize_hy_is_armenian(text):
    assert normalize(text, "hy") == ArmenianTextNormalizer()(text)


# --- CLI ---------------------------------------------------------------------------------
def _cli_command():
    """Command prefix to run the CLI, or None (skip) if asr_normalizer.cli has no main()."""
    try:
        cli = importlib.import_module("asr_normalizer.cli")
    except ImportError:
        return None
    if not callable(getattr(cli, "main", None)):
        return None
    try:
        source = inspect.getsource(cli)
    except (OSError, TypeError):
        source = ""
    if "__main__" in source:
        return [sys.executable, "-m", "asr_normalizer.cli"]
    return [sys.executable, "-c", "import sys; from asr_normalizer.cli import main; sys.exit(main())"]


CLI_LINES = ["Mr. Smith's colour is grey.", "uh twenty one dollars", "", "c o 2 emissions", "[noise] (laughs) ok"]


def _run_cli(args, stdin=None):
    cmd = _cli_command()
    if cmd is None:
        pytest.skip("asr_normalizer.cli has no main()")
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    return subprocess.run(cmd + args, input=stdin, capture_output=True, text=True, encoding="utf-8",
                          env=env, check=True, timeout=120)


def test_cli_stdin_default_lang_en():
    r = _run_cli([], stdin="\n".join(CLI_LINES) + "\n")
    assert r.stdout.splitlines() == [normalize(line, "en") for line in CLI_LINES]


def test_cli_files_with_lang(tmp_path):
    lines = ["Ich habe 3 Äpfel.", "10 000 Einwohner", "Ünïcödé, bitte!"]
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("\n".join(lines[:2]) + "\n", encoding="utf-8")
    b.write_text(lines[2] + "\n", encoding="utf-8")
    r = _run_cli(["--lang", "de", str(a), str(b)])
    assert r.stdout.splitlines() == [normalize(line, "de") for line in lines]
