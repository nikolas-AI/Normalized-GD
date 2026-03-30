from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def make_run_dir(base: str | Path = "runs", name: str | None = None) -> Path:
    base_path = Path(base)
    base_path.mkdir(parents=True, exist_ok=True)

    if name is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        name = ts

    run_dir = base_path / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def save_csv(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rows_list = list(rows)
    if not rows_list:
        p.write_text("", encoding="utf-8")
        return

    fieldnames: list[str] = sorted({k for r in rows_list for k in r.keys()})
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows_list:
            writer.writerow(r)

