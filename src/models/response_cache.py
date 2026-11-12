"""
Disk cache for LLM responses (SQLite, thread-safe)
Same request (model, messages, schema, temperature, max_tokens) -> same response,
so ablations that reuse a prompt never pay for it twice.
"""
import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Dict, Optional, Tuple


def request_key(payload: Dict) -> str:
    """Stable hash of a request payload"""
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()


class ResponseCache:
    """key -> response text, stored in one SQLite file"""

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False, timeout=60)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS responses ("
            "key TEXT PRIMARY KEY, model TEXT, response TEXT, "
            "input_tokens INTEGER, output_tokens INTEGER)"
        )
        self._conn.commit()

    def get(self, key: str) -> Optional[str]:
        entry = self.get_entry(key)
        return entry[0] if entry else None

    def get_entry(self, key: str) -> Optional[Tuple[str, int, int]]:
        """(response, input_tokens, output_tokens) as recorded by the physical call"""
        try:
            with self._lock:
                row = self._conn.execute(
                    "SELECT response, input_tokens, output_tokens FROM responses WHERE key = ?", (key,)
                ).fetchone()
        except sqlite3.Error as e:  # a cache failure must never fail the LLM call
            print(f"cache get failed: {e}")
            return None
        return (row[0], row[1] or 0, row[2] or 0) if row else None

    def put(self, key: str, model: str, response: str, input_tokens: int = 0, output_tokens: int = 0):
        try:
            with self._lock:
                self._conn.execute(
                    "INSERT OR REPLACE INTO responses VALUES (?, ?, ?, ?, ?)",
                    (key, model, response, input_tokens, output_tokens),
                )
                self._conn.commit()
        except sqlite3.Error as e:
            print(f"cache put failed: {e}")

    def delete(self, key: str):
        with self._lock:
            self._conn.execute("DELETE FROM responses WHERE key = ?", (key,))
            self._conn.commit()

    def __len__(self):
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0]
