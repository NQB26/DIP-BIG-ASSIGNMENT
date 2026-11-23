"""Persistent, process-safe daily event ID sequence using the stdlib SQLite."""
from __future__ import annotations

import os
import sqlite3
import threading
from pathlib import Path


class EventIdStore:
    """Generate EVT_YYYYMMDD_NNNN IDs without resetting on process restart.

    SQLite's BEGIN IMMEDIATE serializes writers, so separate camera processes
    sharing one database do not allocate the same sequence number.
    """

    def __init__(self, db_path: str = "runs/event_sequence.sqlite3") -> None:
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            self.db_path, timeout=30.0, check_same_thread=False
        )
        self._conn.execute("PRAGMA busy_timeout = 30000")
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS event_sequences (
                   event_date TEXT PRIMARY KEY,
                   last_seq INTEGER NOT NULL CHECK(last_seq >= 0)
               )"""
        )
        self._conn.commit()

    def next_id(self, day: str) -> str:
        if len(day) != 8 or not day.isdigit():
            raise ValueError("day must be YYYYMMDD")
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                self._conn.execute(
                    "INSERT OR IGNORE INTO event_sequences(event_date, last_seq) VALUES (?, 0)",
                    (day,),
                )
                self._conn.execute(
                    "UPDATE event_sequences SET last_seq = last_seq + 1 WHERE event_date = ?",
                    (day,),
                )
                row = self._conn.execute(
                    "SELECT last_seq FROM event_sequences WHERE event_date = ?",
                    (day,),
                ).fetchone()
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return f"EVT_{day}_{int(row[0]):04d}"

    def close(self) -> None:
        with self._lock:
            self._conn.close()
