"""Guardrails: deterministic rules the model cannot override.

Language models are good at judgement and explanation but can be wrong,
inconsistent, or manipulated. So the agent works inside hard limits:

* A **floor**: some findings always require at least WARN or BLOCK,
  no matter what the model says. A missing critical column must never be
  waved through because the model talked itself into it.
* The model may always be *stricter* than the floor (e.g. BLOCK a batch
  the rules would only WARN about). It can never be *more lenient*.

This "LLM proposes, rules constrain" pattern is how agents are usually
made safe enough for production systems.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from datadiff_engine import DatasetDiff

from .contracts import DatasetContract

RANK = {"PASS": 0, "WARN": 1, "BLOCK": 2}


@dataclass
class Floor:
    decision: str = "PASS"
    reasons: list[str] = field(default_factory=list)

    def raise_to(self, decision: str, reason: str) -> None:
        if RANK[decision] > RANK[self.decision]:
            self.decision = decision
        self.reasons.append(f"{decision}: {reason}")


def compute_floor(diff: DatasetDiff, contract: DatasetContract) -> Floor:
    """The minimum decision the rules require for this diff."""
    floor = Floor()
    critical = contract.critical_columns

    for issue in diff.issues():
        on_critical = issue.column in critical

        if issue.kind == "empty_dataset":
            floor.raise_to("BLOCK", issue.message)

        elif issue.kind == "row_count_change" and issue.severity == "CRITICAL":
            floor.raise_to("BLOCK", issue.message)

        elif issue.kind == "column_removed" and on_critical:
            floor.raise_to("BLOCK", f"critical {issue.message[0].lower()}{issue.message[1:]}")

        elif (
            issue.kind == "type_changed"
            and on_critical
            and issue.severity != "INFO"
        ):
            floor.raise_to("BLOCK", f"critical column: {issue.message}")

        elif (
            issue.kind == "null_rate_increase"
            and on_critical
            and issue.severity == "CRITICAL"
        ):
            floor.raise_to("BLOCK", f"critical column: {issue.message}")

        elif on_critical and issue.severity in ("WARNING", "CRITICAL"):
            floor.raise_to("WARN", f"critical column: {issue.message}")

    return floor


@dataclass
class Enforcement:
    decision: str
    overridden: bool
    note: str = ""


def enforce(proposed: str, floor: Floor) -> Enforcement:
    """Apply the floor to the model's proposed decision."""
    if RANK[proposed] >= RANK[floor.decision]:
        return Enforcement(decision=proposed, overridden=False)
    return Enforcement(
        decision=floor.decision,
        overridden=True,
        note=(
            f"Guardrail raised the decision from {proposed} to "
            f"{floor.decision}. Rule(s): " + "; ".join(floor.reasons)
        ),
    )
