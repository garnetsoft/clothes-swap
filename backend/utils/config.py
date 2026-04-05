import yaml
from pathlib import Path
from types import SimpleNamespace


def _to_ns(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in d.items()})
    return d


def load_config(path: str = None) -> SimpleNamespace:
    if path is None:
        path = Path(__file__).parents[2] / "config.yaml"
    with open(path) as f:
        return _to_ns(yaml.safe_load(f))


cfg = load_config()
