from __future__ import annotations
from pathlib import Path
import json


def save_strategy(folder: str | Path, name: str, config: dict) -> Path:
    folder = Path(folder).expanduser()
    folder.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name).strip("_") or "strategy"
    path = folder / f"{safe}.json"
    path.write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    return path


def list_strategies(folder: str | Path):
    folder = Path(folder).expanduser()
    return sorted(folder.glob("*.json")) if folder.exists() else []


def load_strategy(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
