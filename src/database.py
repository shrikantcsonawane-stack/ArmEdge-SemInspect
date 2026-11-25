"""SQLite inspection log."""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

COLS = ["timestamp", "image_name", "mode", "pred_class", "confidence", "severity", "decision",
        "area_pct", "area_source", "latency_ms", "advisor_source", "recommendation", "model_trained_on"]


class InspectionDB:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self._conn() as c:
            c.execute(f"CREATE TABLE IF NOT EXISTS inspections (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                      f"{', '.join(c_ + ' TEXT' for c_ in COLS)})")

    def _conn(self):
        return sqlite3.connect(self.path)

    def log(self, rec: dict) -> None:
        rec = {**rec, "timestamp": rec.get("timestamp") or datetime.now().isoformat(timespec="seconds")}
        with self._conn() as c:
            c.execute(f"INSERT INTO inspections ({', '.join(COLS)}) VALUES ({', '.join('?' * len(COLS))})",
                      [None if rec.get(k) is None else str(rec.get(k)) for k in COLS])

    def history(self, limit: int = 1000) -> pd.DataFrame:
        with self._conn() as c:
            return pd.read_sql_query(f"SELECT * FROM inspections ORDER BY id DESC LIMIT {int(limit)}", c)

    def clear(self) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM inspections")
