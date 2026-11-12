"""
Run configuration: YAML with `inherit:` (deep merge) and ${ENV:VAR} substitution
"""
import copy
import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into a copy of base"""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _substitute_env(obj):
    if isinstance(obj, dict):
        return {k: _substitute_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_substitute_env(v) for v in obj]
    if isinstance(obj, str) and obj.startswith('${ENV:') and obj.endswith('}'):
        return os.getenv(obj[6:-1])
    return obj


def load_run_config(path: str) -> dict:
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if 'inherit' in cfg:
        base_path = Path(cfg.pop('inherit'))
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        cfg = deep_merge(load_run_config(base_path), cfg)
    return _substitute_env(cfg)


def load_secrets(path: str = None):
    """Export KEY=VALUE lines of secrets/keys.env into os.environ (never printed)"""
    path = Path(path) if path else PROJECT_ROOT / 'secrets' / 'keys.env'
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        if '=' not in line:
            continue
        k, v = line.split('=', 1)
        k, v = k.strip(), v.strip()
        if k and k not in os.environ:
            os.environ[k] = v
