"""A small local pipeline with the agent as a quality gate.

    landing/<file>.csv
          │
          ▼
    agent compares it with the last good batch (baseline)
          │
    ┌─────┼──────────────┐
  PASS   WARN          BLOCK
    │     │ + alert       │ + alert, ask a human (optional)
    ▼     ▼               ▼
  warehouse (SQLite)    quarantine/
  (only PASS batches become the new baseline)

The same shape works on Databricks or Airflow: put ``check_batch`` in front
of the load step, and only load when the decision is PASS or WARN.
"""

from __future__ import annotations

import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .actions import ApprovalGate, Notifier, write_report
from .agent import AgentResult, run_agent
from .contracts import DatasetContract
from .llm import ChatModel
from .memory import Memory
from .tools import RunContext


@dataclass
class BatchOutcome:
    batch: str
    loaded: bool
    result: AgentResult | None
    report: Path | None
    note: str = ""


class LocalPipeline:
    def __init__(
        self,
        root: str | Path,
        contract: DatasetContract,
        model: ChatModel,
        notifier: Notifier | None = None,
        approval: ApprovalGate | None = None,
    ):
        self.root = Path(root)
        self.contract = contract
        self.model = model
        self.notifier = notifier or Notifier()
        self.approval = approval or ApprovalGate("deny")
        for name in ("landing", "quarantine", "reports", "warehouse"):
            (self.root / name).mkdir(parents=True, exist_ok=True)
        self.memory = Memory(self.root / "warehouse" / "agent_memory.db")

    @property
    def dataset(self) -> str:
        return self.contract.name

    @property
    def baseline_path(self) -> Path:
        return self.root / "warehouse" / f"{self.dataset}__baseline.csv"

    @property
    def warehouse_db(self) -> Path:
        return self.root / "warehouse" / "warehouse.db"

    def process(self, batch_file: str | Path, on_step=None) -> BatchOutcome:
        batch_file = Path(batch_file)
        new = pd.read_csv(batch_file)

        if not self.baseline_path.exists():
            # First batch ever: nothing to compare with yet.
            self._load(new, batch_file.name)
            return BatchOutcome(
                batch=batch_file.name,
                loaded=True,
                result=None,
                report=None,
                note="First batch: loaded as the baseline (nothing to compare with).",
            )

        old = pd.read_csv(self.baseline_path)
        ctx = RunContext(
            dataset=self.dataset,
            batch=batch_file.name,
            old=old,
            new=new,
            contract=self.contract,
            memory=self.memory,
        )
        result = run_agent(ctx, self.model, on_step=on_step)
        report = write_report(self.root / "reports", self.dataset, batch_file.name, result)
        self.notifier.notify(self.dataset, batch_file.name, result)

        if result.decision == "PASS":
            self._load(new, batch_file.name, update_baseline=True)
            return BatchOutcome(batch_file.name, True, result, report)

        if result.decision == "WARN":
            # Loaded, but not used as the reference for future batches:
            # otherwise a problem would quietly become the new "normal".
            self._load(new, batch_file.name, update_baseline=False)
            return BatchOutcome(
                batch_file.name, True, result, report,
                note="Loaded with a warning; baseline kept on the last clean batch.",
            )

        if self.approval.allow_load_anyway(self.dataset, batch_file.name, result):
            self._load(new, batch_file.name, update_baseline=False)
            return BatchOutcome(
                batch_file.name, True, result, report,
                note="Blocked by the agent, loaded after human approval.",
            )

        shutil.copy(batch_file, self.root / "quarantine" / batch_file.name)
        return BatchOutcome(
            batch_file.name, False, result, report, note="Quarantined."
        )

    def _load(self, df: pd.DataFrame, batch: str, update_baseline: bool = True) -> None:
        """Append to the warehouse table; optionally make it the new baseline."""
        with sqlite3.connect(self.warehouse_db) as conn:
            # Schema evolution: add columns the table doesn't have yet,
            # so an accepted batch with a new column can still be appended.
            existing = {
                row[1]
                for row in conn.execute(f'PRAGMA table_info("{self.dataset}")')
            }
            if existing:
                for column in df.columns:
                    if column not in existing:
                        conn.execute(
                            f'ALTER TABLE "{self.dataset}" ADD COLUMN "{column}"'
                        )
            df.assign(_source_batch=batch).to_sql(
                self.dataset, conn, if_exists="append", index=False
            )
        if update_baseline:
            df.to_csv(self.baseline_path, index=False)

    def warehouse_row_count(self) -> int:
        if not self.warehouse_db.exists():
            return 0
        with sqlite3.connect(self.warehouse_db) as conn:
            try:
                return conn.execute(f'SELECT COUNT(*) FROM "{self.dataset}"').fetchone()[0]
            except sqlite3.OperationalError:
                return 0

    def close(self) -> None:
        self.memory.close()
