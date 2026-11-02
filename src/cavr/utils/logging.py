"""Structured logging helpers: JSONL (optionally gzipped), CSV, markdown."""
from __future__ import annotations

import csv
import gzip
import json
import os
from typing import Any, Iterable, Sequence


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


class JsonlWriter:
    """Append-only JSONL writer; gzips on close when gz=True."""

    def __init__(self, path: str, gz: bool = False):
        ensure_dir(os.path.dirname(path))
        self.path = path
        self.gz = gz
        self._fh = gzip.open(path, "at", encoding="utf-8") if gz else open(path, "a", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        self._fh.write(json.dumps(record, separators=(",", ":"), default=float) + "\n")

    def flush(self) -> None:
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def append_csv(path: str, header: Sequence[str], row: Sequence[Any]) -> None:
    """Append one row; write header if the file does not exist yet."""
    ensure_dir(os.path.dirname(path))
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(list(header))
        w.writerow([r if not isinstance(r, float) else round(r, 6) for r in row])


def read_csv_rows(path: str) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def markdown_table(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(str(h) for h in header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(c) for c in row) + " |")
    return "\n".join(lines)


def _fmt(x: Any) -> str:
    if isinstance(x, float):
        return f"{x:.4g}"
    return str(x)
