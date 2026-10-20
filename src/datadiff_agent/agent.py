"""The agent loop: observe -> think -> act, until a decision is made.

    ┌──────────────────────────────────────────────────────────┐
    │ messages = [system prompt, task]                         │
    │ loop (max_steps):                                        │
    │     turn = model.complete(messages, tools)               │
    │     if turn has tool calls:                              │
    │         run each tool, append results to messages        │
    │         if the tool was submit_decision: validate, stop  │
    │     else: remind the model to call submit_decision       │
    │ apply guardrails (policy.py)   ← rules beat the model    │
    │ save the run to memory         ← so it can learn later   │
    └──────────────────────────────────────────────────────────┘

Every step is recorded in a trace, so you can always see *why* the agent
decided what it did. Being able to inspect an agent's reasoning
(observability) matters as much as the decision itself.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable

from .llm import AssistantTurn, ChatModel
from .policy import Enforcement, compute_floor, enforce
from .tools import DECISIONS, FINAL_TOOL, SEVERITIES, TOOLS, RunContext, run_tool

SYSTEM_PROMPT = """\
You are a data quality agent guarding a production data pipeline.

A new batch of the dataset has arrived. Your job: decide whether it is safe
to load into the data warehouse that feeds business reports.

How to work:
1. Call get_dataset_contract to learn what the dataset is and which columns
   are critical.
2. Call compare_batches to see what changed versus the last good batch.
3. Call get_run_history to check past decisions and known false positives.
4. Investigate anything suspicious with get_column_details and
   sample_values. Look at actual values: a changed currency code or
   date format explains a lot.
5. Call submit_decision exactly once.

How to decide:
- PASS: nothing notable, or only changes that are expected or harmless.
- WARN: load it, but a person should look today (e.g. problems in
  non-critical columns, moderate drift).
- BLOCK: loading it would put wrong numbers in front of the business
  (e.g. a critical column missing, mostly null, wrong unit or currency,
  or the file is clearly incomplete).
- Weigh issues by the contract: the same change matters more in a
  critical column than in a non-critical one.
- If people marked an issue as a false positive before, treat the same
  issue as expected unless it got clearly worse.
- Base every finding on tool results. Never invent numbers.

Write the summary for a busy data engineer: what changed, why it matters,
what you decided. Hard safety rules are applied after your decision, so
be accurate rather than cautious.
"""


@dataclass
class Step:
    number: int
    tool: str
    arguments: dict
    result_preview: str


@dataclass
class AgentResult:
    decision: str
    severity: str
    summary: str
    findings: list[dict]
    recommended_actions: list[str]
    proposed_decision: str
    guardrail: Enforcement
    model: str
    steps: list[Step] = field(default_factory=list)
    run_id: int | None = None
    completed: bool = True

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "decision": self.decision,
            "severity": self.severity,
            "summary": self.summary,
            "findings": self.findings,
            "recommended_actions": self.recommended_actions,
            "proposed_decision": self.proposed_decision,
            "guardrail_overrode": self.guardrail.overridden,
            "guardrail_note": self.guardrail.note,
            "model": self.model,
            "completed": self.completed,
            "steps": [step.__dict__ for step in self.steps],
        }


class DecisionError(ValueError):
    pass


def validate_decision(arguments: dict) -> dict:
    """Check the model's final answer before trusting it."""
    decision = str(arguments.get("decision", "")).upper()
    severity = str(arguments.get("severity", "")).upper()
    summary = arguments.get("summary")
    findings = arguments.get("findings")
    actions = arguments.get("recommended_actions") or []

    problems = []
    if decision not in DECISIONS:
        problems.append(f"decision must be one of {DECISIONS}")
    if severity not in SEVERITIES:
        problems.append(f"severity must be one of {SEVERITIES}")
    if not isinstance(summary, str) or not summary.strip():
        problems.append("summary must be a non-empty string")
    if not isinstance(findings, list):
        problems.append("findings must be a list")
    if not isinstance(actions, list):
        problems.append("recommended_actions must be a list of strings")
    if problems:
        raise DecisionError("; ".join(problems))

    clean_findings = [
        {
            "kind": str(f.get("kind", "")),
            "column": f.get("column"),
            "severity": str(f.get("severity", "")).upper(),
            "explanation": str(f.get("explanation", "")),
        }
        for f in findings
        if isinstance(f, dict)
    ]
    return {
        "decision": decision,
        "severity": severity,
        "summary": summary.strip(),
        "findings": clean_findings,
        "recommended_actions": [str(a) for a in actions],
    }


def run_agent(
    ctx: RunContext,
    model: ChatModel,
    max_steps: int = 12,
    on_step: Callable[[Step], None] | None = None,
    record: bool = True,
) -> AgentResult:
    """Run the agent on one batch and return its (guarded) decision."""
    tools = [tool.schema() for tool in TOOLS.values()]
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"New batch '{ctx.batch}' of dataset '{ctx.dataset}' has "
                "arrived. Review it and submit your decision."
            ),
        },
    ]
    steps: list[Step] = []
    final: dict | None = None

    for _ in range(max_steps):
        turn: AssistantTurn = model.complete(messages, tools)
        messages.append(turn.to_message())

        if not turn.tool_calls:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Use the tools to investigate, and finish by calling "
                        f"{FINAL_TOOL}. Do not answer in plain text."
                    ),
                }
            )
            continue

        for call in turn.tool_calls:
            if call.name == FINAL_TOOL:
                try:
                    final = validate_decision(call.arguments)
                    output = json.dumps({"status": "decision recorded"})
                except DecisionError as error:
                    output = json.dumps(
                        {"error": f"Invalid decision: {error}. Fix and resubmit."}
                    )
            else:
                output = run_tool(ctx, call.name, call.arguments)

            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": output}
            )
            step = Step(
                number=len(steps) + 1,
                tool=call.name,
                arguments=call.arguments,
                result_preview=output[:200],
            )
            steps.append(step)
            if on_step:
                on_step(step)

        if final is not None:
            break

    completed = final is not None
    if final is None:
        # Fail closed: an agent that cannot decide must not let data through.
        final = {
            "decision": "BLOCK",
            "severity": "CRITICAL",
            "summary": (
                f"The agent did not reach a decision within {max_steps} "
                "steps, so the batch was blocked for manual review."
            ),
            "findings": [],
            "recommended_actions": ["Review the batch manually."],
        }

    floor = compute_floor(ctx.diff, ctx.contract)
    guardrail = enforce(final["decision"], floor)
    severity = final["severity"]
    if guardrail.overridden and guardrail.decision == "BLOCK":
        severity = "CRITICAL"
    elif guardrail.overridden and severity == "INFO":
        severity = "WARNING"

    result = AgentResult(
        decision=guardrail.decision,
        severity=severity,
        summary=final["summary"],
        findings=final["findings"],
        recommended_actions=final["recommended_actions"],
        proposed_decision=final["decision"],
        guardrail=guardrail,
        model=model.name,
        steps=steps,
        completed=completed,
    )

    if record:
        result.run_id = ctx.memory.record_run(
            dataset=ctx.dataset,
            batch=ctx.batch,
            decision=result.decision,
            severity=result.severity,
            summary=result.summary,
            findings=result.findings,
            mode=model.name,
        )
    return result
