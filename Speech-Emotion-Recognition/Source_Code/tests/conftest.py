"""Skip (instead of error) the TensorFlow-dependent tests when TensorFlow is not installed."""
import importlib.util
import inspect

import pytest

HAS_TF = importlib.util.find_spec("tensorflow") is not None
_TF_HINTS = ("keras", "predict", "build_model", "_models()")


def pytest_collection_modifyitems(config, items):
    if HAS_TF:
        return
    skip = pytest.mark.skip(reason="TensorFlow not installed (supported Python: 3.10-3.13); run in the project .venv")
    for item in items:
        try:
            src = inspect.getsource(item.function)
        except (OSError, TypeError, AttributeError):
            continue
        if any(h in src for h in _TF_HINTS):
            item.add_marker(skip)
