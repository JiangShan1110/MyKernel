import logging
from typing import Any

LOG = logging.getLogger(__name__)


def make_json_friendly(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, (list, tuple)):
        return [make_json_friendly(item) for item in obj]
    if isinstance(obj, dict):
        return {str(k): make_json_friendly(v) for k, v in obj.items()}
    return str(obj)
