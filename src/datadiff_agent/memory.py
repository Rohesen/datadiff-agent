"""Long-term memory: past runs and human feedback, stored in SQLite.

Agents need memory to stop repeating the same mistakes. Two things are
remembered per dataset:

* runs: every decision the agent made, with its findings
* feedback: a human's verdict on a run ("that was a false positive",
  "you missed something")

Before deciding, the agent reads this history through a tool. If a
person marked an issue as a false positive before, the agent can treat
the same issue as expected noise next time. That is a simple, honest
form of learning from feedback, with no model training involved.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

FEEDBACK_VERDICTS = ("correct", "false_positive", "missed_issue")

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset     TEXT NOT NULL,
    batch       TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    decision    TEXT NOT NULL,
    severity    TEXT NOT NULL,
    summary     TEXT NOT NULL,
    findings    TEXT NOT NULL,   -- JSON list of findings
    mode        TEXT NOT NULL    -- which model made the decision
);

CREATE TABLE IF NOT EXISTS feedback (
    feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER NOT NULL REFERENCES runs(run_id),
    verdict     TEXT NOT NULL,
    note        TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL
);
"""


@dataclass
class RunRecord:
    run_id: int
    dataset: str
    batch: str
    created_at: str
    decision: str
    severity: str
    summary: str
    findings: list[dict]
    mode: str
    feedback: list[dict]

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class Memory:
    """A small SQLite-backed store. Use ':memory:' for tests."""

    def __init__(self, path: str | Path = "datadiff_agent.db"):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)

    def close(self) -> None:
        self._conn.close()

    # --- writing ---------------------------------------------------------

    def record_run(
        self,
        dataset: str,
        batch: str,
        decision: str,
        severity: str,
        summary: str,
        findings: list[dict],
        mode: str,
    ) -> int:
        with self._conn:
            cursor = self._conn.execute(
                "INSERT INTO runs (dataset, batch, created_at, decision, "
                "severity, summary, findings, mode) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    dataset,
                    batch,
                    _now(),
                    decision,
                    severity,
                    summary,
                    json.dumps(findings),
                    mode,
                ),
            )
        return int(cursor.lastrowid)

    def add_feedback(self, run_id: int, verdict: str, note: str = "") -> None:
        if verdict not in FEEDBACK_VERDICTS:
            raise ValueError(
                f"verdict must be one of {FEEDBACK_VERDICTS}, got {verdict!r}"
            )
        if self.get_run(run_id) is None:
            raise ValueError(f"No run with id {run_id}")
        with self._conn:
            self._conn.execute(
                "INSERT INTO feedback (run_id, verdict, note, created_at) "
                "VALUES (?, ?, ?, ?)",
                (run_id, verdict, note, _now()),
            )

    # --- reading ---------------------------------------------------------

    def get_run(self, run_id: int) -> RunRecord | None:
        row = self._conn.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        return self._to_record(row) if row else None

    def recent_runs(self, dataset: str, limit: int = 5) -> list[RunRecord]:
        rows = self._conn.execute(
            "SELECT * FROM runs WHERE dataset = ? "
            "ORDER BY run_id DESC LIMIT ?",
            (dataset, limit),
        ).fetchall()
        return [self._to_record(row) for row in rows]

    def known_false_positives(self, dataset: str) -> list[dict]:
        """Issues a person has marked as false positives for this dataset.

        Returns a de-duplicated list of {"kind", "column", "note"}.
        """
        rows = self._conn.execute(
            "SELECT r.findings, f.note FROM feedback f "
            "JOIN runs r ON r.run_id = f.run_id "
            "WHERE r.dataset = ? AND f.verdict = 'false_positive' "
            "ORDER BY f.feedback_id",
            (dataset,),
        ).fetchall()

        seen: dict[tuple, dict] = {}
        for row in rows:
            for finding in json.loads(row["findings"]):
                key = (finding.get("kind"), finding.get("column"))
                if key[0] and key not in seen:
                    seen[key] = {
                        "kind": key[0],
                        "column": key[1],
                        "note": row["note"],
                    }
        return list(seen.values())

    def _to_record(self, row: sqlite3.Row) -> RunRecord:
        feedback = [
            dict(f)
            for f in self._conn.execute(
                "SELECT verdict, note, created_at FROM feedback "
                "WHERE run_id = ? ORDER BY feedback_id",
                (row["run_id"],),
            ).fetchall()
        ]
        return RunRecord(
            run_id=row["run_id"],
            dataset=row["dataset"],
            batch=row["batch"],
            created_at=row["created_at"],
            decision=row["decision"],
            severity=row["severity"],
            summary=row["summary"],
            findings=json.loads(row["findings"]),
            mode=row["mode"],
            feedback=feedback,
        )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def open_memory(path: str | Path) -> closing:
    """Context manager: ``with open_memory(p) as mem: ...``."""
    return closing(Memory(path))
