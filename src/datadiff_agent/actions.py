"""What happens after a decision: reports, alerts and human approval.

An agent that only *says* things is a chatbot. An agent *acts*. The actions
here are deliberately small and auditable:

* ``write_report``  a Markdown incident report per batch
* ``Notifier``      console output, plus Slack if SLACK_WEBHOOK_URL is set
* ``ApprovalGate``  human-in-the-loop: a person can override a BLOCK

Anything risky (loading data the agent blocked) needs a human.
"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path
from typing import Callable

from .agent import AgentResult

ICON = {"PASS": "✅", "WARN": "⚠️", "BLOCK": "⛔"}


def format_step(step) -> str:
    """One-line description of a tool call, for traces."""
    if step.tool == "submit_decision":
        args = step.arguments
        return (
            f"submit_decision(decision={args.get('decision')!r}, "
            f"severity={args.get('severity')!r}, "
            f"findings={len(args.get('findings') or [])})"
        )
    args = json.dumps(step.arguments, ensure_ascii=False) if step.arguments else ""
    return f"{step.tool}({args})"


def render_report(dataset: str, batch: str, result: AgentResult) -> str:
    lines = [
        f"# {ICON[result.decision]} {result.decision}: {dataset} / {batch}",
        "",
        f"- **Severity:** {result.severity}",
        f"- **Decided by:** {result.model}",
        f"- **Run id:** {result.run_id}",
    ]
    if result.guardrail.overridden:
        lines.append(
            f"- **Guardrail:** model proposed {result.proposed_decision}; "
            f"{result.guardrail.note}"
        )
    if not result.completed:
        lines.append("- **Note:** the agent did not finish; failed closed.")

    lines += ["", "## Summary", "", result.summary, ""]

    if result.findings:
        lines += ["## Findings", "", "| Severity | Issue | Column | Explanation |", "|---|---|---|---|"]
        for f in result.findings:
            lines.append(
                f"| {f['severity']} | {f['kind']} | {f.get('column') or '—'} "
                f"| {f['explanation'].replace('|', '/')} |"
            )
        lines.append("")

    if result.recommended_actions:
        lines += ["## Recommended actions", ""]
        lines += [f"- {action}" for action in result.recommended_actions]
        lines.append("")

    lines += ["## Agent trace", ""]
    for step in result.steps:
        lines.append(f"{step.number}. `{format_step(step)}`")
    lines.append("")
    return "\n".join(lines)


def write_report(
    folder: str | Path, dataset: str, batch: str, result: AgentResult
) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{dataset}__{Path(batch).stem}__run{result.run_id}.md"
    path.write_text(render_report(dataset, batch, result), encoding="utf-8")
    return path


class Notifier:
    """Prints alerts; also posts to Slack when a webhook URL is configured."""

    def __init__(
        self,
        slack_webhook_url: str | None = None,
        printer: Callable[[str], None] = print,
    ):
        self.slack_webhook_url = slack_webhook_url or os.getenv("SLACK_WEBHOOK_URL")
        self.printer = printer

    def notify(self, dataset: str, batch: str, result: AgentResult) -> None:
        if result.decision == "PASS":
            return
        text = (
            f"{ICON[result.decision]} *{result.decision}* {dataset}/{batch} "
            f"({result.severity})\n{result.summary}"
        )
        self.printer(f"[alert] {text}")
        if self.slack_webhook_url:
            self._post_slack(text)

    def _post_slack(self, text: str) -> None:
        request = urllib.request.Request(
            self.slack_webhook_url,
            data=json.dumps({"text": text}).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            urllib.request.urlopen(request, timeout=10)
        except OSError as error:  # never let alerting break the pipeline
            self.printer(f"[alert] Slack post failed: {error}")


class ApprovalGate:
    """Human-in-the-loop for blocked batches.

    mode="ask"     prompt a person in the terminal
    mode="deny"    never override (safe default for scheduled runs)
    mode="approve" always override (for tests only)
    """

    def __init__(self, mode: str = "deny", ask: Callable[[str], str] = input):
        if mode not in ("ask", "deny", "approve"):
            raise ValueError("mode must be ask, deny or approve")
        self.mode = mode
        self.ask = ask

    def allow_load_anyway(self, dataset: str, batch: str, result: AgentResult) -> bool:
        if self.mode == "deny":
            return False
        if self.mode == "approve":
            return True
        answer = self.ask(
            f"\nThe agent BLOCKED {dataset}/{batch}: {result.summary}\n"
            "Load it anyway? Type 'load' to override, anything else to keep "
            "it quarantined: "
        )
        return answer.strip().lower() == "load"
