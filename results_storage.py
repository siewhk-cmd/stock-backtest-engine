from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re


def safe_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip())
    return cleaned.strip("_") or "backtest"


def create_run_folder(base_dir: str | Path, strategy_name: str, mode: str) -> Path:
    base = Path(base_dir).expanduser()
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    folder = base / f"{stamp}_{safe_name(strategy_name)}_{safe_name(mode)}"
    counter = 1
    original = folder
    while folder.exists():
        folder = Path(str(original) + f"_{counter}")
        counter += 1
    folder.mkdir(parents=True, exist_ok=False)
    return folder
