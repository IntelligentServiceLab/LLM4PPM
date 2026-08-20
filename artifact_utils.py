from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Container, Dict, Iterable, List, Optional


def add_src_to_path() -> None:
    src = Path(__file__).resolve().parent / "src"
    if src.exists() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


def config_to_argv(config: Dict[str, Any], negatable_keys: Optional[Container[str]] = None) -> List[str]:
    argv: List[str] = []
    for key, value in config.items():
        flag = "--" + key
        if isinstance(value, bool):
            if value:
                argv.append(flag)
            elif negatable_keys is not None and key in negatable_keys:
                argv.append("--no-" + key)
        elif isinstance(value, (list, tuple)):
            argv.append(flag)
            argv.extend(str(v) for v in value)
        elif value is not None:
            argv.extend([flag, str(value)])
    return argv


def load_yaml_args(args: Iterable[str], negatable_keys: Optional[Container[str]] = None) -> List[str]:
    args = list(args)
    if "--config" not in args:
        return args
    idx = args.index("--config")
    if idx + 1 >= len(args):
        raise SystemExit("--config requires a YAML path.")
    config_path = Path(args[idx + 1])
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("PyYAML is required for --config. Install requirements.txt first.") from exc
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise SystemExit(f"Config must be a mapping: {config_path}")
    return config_to_argv(config, negatable_keys=negatable_keys) + args[:idx] + args[idx + 2 :]
