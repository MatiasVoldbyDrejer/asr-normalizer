"""Command-line interface: ``asr-normalize [--lang LANG] [FILE ...]``.

Reads UTF-8 text, one transcript per line, from the given files (or stdin when no file is
given, or for ``-``) and writes one normalized line per input line to stdout.
"""

import argparse
import os
import sys

from . import __version__, normalize


def _use_utf8(stream):
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is not None:
        try:
            reconfigure(encoding="utf-8")
        except (ValueError, OSError):  # e.g. already read from / not a real text stream
            pass


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="asr-normalize",
        description=(
            "Normalize transcripts exactly as the Open ASR Leaderboard does before scoring. "
            "One transcript per input line in, one normalized line out."
        ),
    )
    parser.add_argument(
        "--lang",
        default="en",
        help=(
            'language code (default: "en"). "en" uses EnglishTextNormalizer; any other code uses '
            'MultilingualNormalizer(remove_diacritics=False) with lang=LANG ("hy" routes to '
            "ArmenianTextNormalizer)"
        ),
    )
    parser.add_argument(
        "files",
        nargs="*",
        metavar="FILE",
        help="UTF-8 input file(s), one transcript per line; '-' or no FILE reads stdin",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = parser.parse_args(argv)

    out = sys.stdout
    _use_utf8(out)

    def emit(stream):
        for line in stream:
            if line.endswith("\n"):
                line = line[:-1]
            out.write(normalize(line, args.lang) + "\n")

    try:
        for path in args.files or ["-"]:
            if path == "-":
                _use_utf8(sys.stdin)
                emit(sys.stdin)
            else:
                with open(path, encoding="utf-8") as f:
                    emit(f)
        out.flush()
    except BrokenPipeError:
        # Downstream closed the pipe (e.g. `| head`): exit quietly.
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        return 1
    except (OSError, UnicodeDecodeError) as e:
        parser.exit(1, f"{parser.prog}: error: {e}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
