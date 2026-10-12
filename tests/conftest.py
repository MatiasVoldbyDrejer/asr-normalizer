import os
import sys
from pathlib import Path

import pytest

# tests/_unit_cases.py (generated expectations) must be importable whatever --import-mode is used.
sys.path.insert(0, str(Path(__file__).resolve().parent))

FULL_ENV = "ASR_NORMALIZER_FULL"


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "full: re-fetches every golden source from the Hugging Face Hub and verifies the sha256 of the "
        f"full sweep (network, slow); skipped unless {FULL_ENV}=1",
    )


def pytest_collection_modifyitems(config, items):
    if os.environ.get(FULL_ENV) == "1":
        return
    skip_full = pytest.mark.skip(reason=f"full sweep: set {FULL_ENV}=1 to run")
    for item in items:
        if "full" in item.keywords:
            item.add_marker(skip_full)
