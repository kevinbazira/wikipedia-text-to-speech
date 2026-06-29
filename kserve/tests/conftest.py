import importlib
import sys
from pathlib import Path

import pytest

KSERVE_ROOT = Path(__file__).resolve().parents[1]
REAL_NUMPY = importlib.import_module("numpy")

if str(KSERVE_ROOT) not in sys.path:
    sys.path.insert(0, str(KSERVE_ROOT))


@pytest.fixture(autouse=True)
def restore_numpy_module():
    previous_numpy = sys.modules.get("numpy")
    sys.modules["numpy"] = REAL_NUMPY
    try:
        yield
    finally:
        if previous_numpy is None:
            sys.modules.pop("numpy", None)
        else:
            sys.modules["numpy"] = previous_numpy
