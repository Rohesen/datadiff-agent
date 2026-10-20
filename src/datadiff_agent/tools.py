"""The tools the agent can call.

Each tool is a plain Python function plus a JSON schema that describes it
to the language model. The model never sees file paths, SQL or code: it
can only call these narrow, read-only tools about the batch under review.
Keeping tools small and safe is one of the most important guardrails in
agent design.

The one special tool is ``submit_decision``: calling it ends the agent
loop. Making the final answer a tool call (instead of free text) forces
the model to return a structured, validated decision.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd
from datadiff_engine import DatasetDiff, compare

from .contracts import DatasetContract
from .memory import Memory

DECISIONS = ("PASS", "WARN", "BLOCK")
SEVERITIES = ("INFO", "WARNING", "CRITICAL")
MAX_TOOL_OUTPUT_CHARS = 6000


@dataclass
class RunContext:
    """Everything the tools need about the batch under review."""

    dataset: str
    batch: str
    old: pd.DataFrame
    new: pd.DataFrame
    contract: DatasetContract
    memory: Memory
    _diff: DatasetDiff | None = field(default=None, repr=False)

    @property
    def diff(self) -> DatasetDiff:
        if self._diff is None:
            self._diff = compare(
                self.old, self.new, config=self.contract.diff_config()
            )
        return self._diff


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    func: Callable[..., Any]

    def schema(self) -> dict:
        """OpenAI-style function schema (also used by Groq, Ollama, etc.)."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolError(Exception):
    """An error the model can see and recover from."""


# --- tool implementations -------------------------------------------------


def get_dataset_contract(ctx: RunContext) -> dict:
    return ctx.contract.to_dict()


def compare_batches(ctx: RunContext) -> dict:
    report = ctx.diff.to_dict()
    critical = ctx.contract.critical_columns
    issues = report["summary"]["issues"]
    for issue in issues:
        issue["column_is_critical"] = issue["column"] in critical
    return {
        "batch": ctx.batch,
        "overall_severity": report["summary"]["overall_severity"],
        "rows": report["rows"],
        "schema": report["schema"],
        "issues": issues,
    }


def get_column_details(ctx: RunContext, column: str) -> dict:
    diff = ctx.diff
    if column in diff.column_diffs:
        details = diff.column(column).to_dict()
        details.pop("config", None)
    elif column in diff.removed_columns:
        details = {"name": column, "status": "removed in new batch"}
    elif column in diff.added_columns:
        details = {"name": column, "status": "added in new batch"}
    else:
        raise ToolError(
            f"Unknown column '{column}'. Columns in the new batch: "
            f"{sorted(ctx.new.columns)}"
        )
    contract = ctx.contract.columns.get(column)
    details["contract"] = (
        {"description": contract.description, "critical": contract.critical}
        if contract
        else None
    )
    return details


def sample_values(ctx: RunContext, column: str, n: int = 5) -> dict:
    n = max(1, min(int(n), 10))
    result: dict[str, Any] = {"column": column}
    for label, df in (("old", ctx.old), ("new", ctx.new)):
        if column in df.columns:
            values = df[column].dropna().head(n).tolist()
            result[label] = [_jsonable(v) for v in values]
        else:
            result[label] = "column not present"
    return result


def get_run_history(ctx: RunContext, limit: int = 5) -> dict:
    limit = max(1, min(int(limit), 20))
    runs = ctx.memory.recent_runs(ctx.dataset, limit=limit)
    return {
        "recent_runs": [
            {
                "run_id": r.run_id,
                "batch": r.batch,
                "decision": r.decision,
                "severity": r.severity,
                "summary": r.summary,
                "feedback": r.feedback,
            }
            for r in runs
        ],
        "known_false_positives": ctx.memory.known_false_positives(ctx.dataset),
    }


def submit_decision(
    ctx: RunContext,
    decision: str,
    severity: str,
    summary: str,
    findings: list[dict],
    recommended_actions: list[str] | None = None,
) -> dict:
    """Validated in agent.py; returning the payload ends the loop."""
    return {
        "decision": decision,
        "severity": severity,
        "summary": summary,
        "findings": findings,
        "recommended_actions": recommended_actions or [],
    }


# --- registry -------------------------------------------------------------

TOOLS: dict[str, Tool] = {
    tool.name: tool
    for tool in [
        Tool(
            name="get_dataset_contract",
            description=(
                "Get the dataset contract: what the dataset is, which "
                "columns are critical, what each column means, and which "
                "reports depend on it. Call this first."
            ),
            parameters={"type": "object", "properties": {}},
            func=get_dataset_contract,
        ),
        Tool(
            name="compare_batches",
            description=(
                "Compare the new batch with the last good batch using "
                "datadiff-engine. Returns row counts, schema changes and a "
                "list of issues with severities. Each issue says whether "
                "its column is critical."
            ),
            parameters={"type": "object", "properties": {}},
            func=compare_batches,
        ),
        Tool(
            name="get_column_details",
            description=(
                "Full comparison for one column: types, null rates, unique "
                "counts, numeric statistics and drift, top categories, and "
                "the column's contract."
            ),
            parameters={
                "type": "object",
                "properties": {"column": {"type": "string"}},
                "required": ["column"],
            },
            func=get_column_details,
        ),
        Tool(
            name="sample_values",
            description=(
                "A few example non-null values of a column from the old and "
                "new batch. Useful to see *how* values changed, e.g. a new "
                "currency code or a different date format."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "column": {"type": "string"},
                    "n": {"type": "integer", "minimum": 1, "maximum": 10},
                },
                "required": ["column"],
            },
            func=sample_values,
        ),
        Tool(
            name="get_run_history",
            description=(
                "Recent decisions for this dataset with any human feedback, "
                "plus issues people have marked as false positives before. "
                "Use it to tell recurring, accepted noise from new problems."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "minimum": 1, "maximum": 20}
                },
            },
            func=get_run_history,
        ),
        Tool(
            name="submit_decision",
            description=(
                "Submit your final decision. Call exactly once, at the end. "
                "PASS = load the batch. WARN = load it and alert the team. "
                "BLOCK = do not load; quarantine it for a human."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "decision": {"type": "string", "enum": list(DECISIONS)},
                    "severity": {"type": "string", "enum": list(SEVERITIES)},
                    "summary": {
                        "type": "string",
                        "description": (
                            "2-4 plain-English sentences for the data team: "
                            "what changed, why it matters, what you decided."
                        ),
                    },
                    "findings": {
                        "type": "array",
                        "description": "The issues that drove the decision.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "kind": {"type": "string"},
                                "column": {"type": ["string", "null"]},
                                "severity": {
                                    "type": "string",
                                    "enum": list(SEVERITIES),
                                },
                                "explanation": {"type": "string"},
                            },
                            "required": ["kind", "severity", "explanation"],
                        },
                    },
                    "recommended_actions": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["decision", "severity", "summary", "findings"],
            },
            func=submit_decision,
        ),
    ]
}

FINAL_TOOL = "submit_decision"


def run_tool(ctx: RunContext, name: str, arguments: dict) -> str:
    """Execute a tool and return its result as a JSON string for the model.

    Errors are returned to the model as text instead of crashing the run,
    so the model can correct itself (e.g. after a misspelled column name).
    """
    tool = TOOLS.get(name)
    if tool is None:
        return json.dumps({"error": f"Unknown tool '{name}'."})
    try:
        result = tool.func(ctx, **arguments)
    except ToolError as error:
        return json.dumps({"error": str(error)})
    except TypeError as error:
        return json.dumps({"error": f"Bad arguments for {name}: {error}"})

    text = json.dumps(result, default=_jsonable, allow_nan=False)
    if len(text) > MAX_TOOL_OUTPUT_CHARS:
        text = text[:MAX_TOOL_OUTPUT_CHARS] + '..." [truncated]'
    return text


def _jsonable(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value
