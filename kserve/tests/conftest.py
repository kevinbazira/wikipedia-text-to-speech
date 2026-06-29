import logging
import sys
import types
from pathlib import Path

KSERVE_ROOT = Path(__file__).resolve().parents[1]

if str(KSERVE_ROOT) not in sys.path:
    sys.path.insert(0, str(KSERVE_ROOT))


errors_module = types.ModuleType("kserve.errors")
errors_module.InferenceError = type("InferenceError", (Exception,), {})
errors_module.InvalidInput = type("InvalidInputError", (Exception,), {})
errors_module.ModelMissingError = type("ModelMissingError", (Exception,), {})


kserve_module = types.ModuleType("kserve")
kserve_module.constants = types.SimpleNamespace(KSERVE_LOGLEVEL=logging.INFO)


class Model:
    def __init__(self, name: str) -> None:
        self.name = name
        self.ready = False


class ModelServer:
    def start(self, models) -> None:
        return None


kserve_module.Model = Model
kserve_module.ModelServer = ModelServer
kserve_module.errors = errors_module

sys.modules["kserve"] = kserve_module
sys.modules["kserve.errors"] = errors_module
