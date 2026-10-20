"""The "brain" of the agent, behind one small interface.

``ChatModel.complete(messages, tools)`` returns either some text or a list
of tool calls. That is the whole contract between the agent loop and a
model, which is why the same loop can run on:

* ``OpenAICompatibleModel``: any server that speaks the OpenAI chat
  completions API with tool calling: Groq, OpenAI, Ollama, vLLM,
  LM Studio, ...
* ``RuleBasedModel``: an offline planner that follows a fixed script and
  decides with plain rules. No API key, fully deterministic. It is used
  for tests and demos, and as a baseline to compare a real LLM against.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Protocol

RANK = {"PASS": 0, "WARN": 1, "BLOCK": 2}
SEV_RANK = {"INFO": 0, "WARNING": 1, "CRITICAL": 2}


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class AssistantTurn:
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)

    def to_message(self) -> dict:
        """The assistant message to append to the conversation history."""
        message: dict = {"role": "assistant", "content": self.content or ""}
        if self.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments),
                    },
                }
                for call in self.tool_calls
            ]
        return message


class ChatModel(Protocol):
    name: str

    def complete(self, messages: list[dict], tools: list[dict]) -> AssistantTurn:
        ...


# --- real LLMs over the OpenAI-compatible API -----------------------------

PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "default_model": "llama-3.3-70b-versatile",
    },
    "openai": {
        "base_url": None,  # library default
        "api_key_env": "OPENAI_API_KEY",
        "default_model": None,  # set DATADIFF_AGENT_MODEL
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "api_key_env": None,  # local, no key
        "default_model": "llama3.1",
    },
}


class OpenAICompatibleModel:
    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        client=None,
    ):
        if client is None:
            from openai import OpenAI

            client = OpenAI(base_url=base_url, api_key=api_key or "not-needed")
        self.client = client
        self.model = model
        self.temperature = temperature
        self.name = f"llm:{model}"

    def complete(self, messages: list[dict], tools: list[dict]) -> AssistantTurn:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=self.temperature,
        )
        message = response.choices[0].message
        calls = []
        for call in message.tool_calls or []:
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {"_invalid_json": call.function.arguments}
            calls.append(
                ToolCall(id=call.id, name=call.function.name, arguments=arguments)
            )
        return AssistantTurn(content=message.content, tool_calls=calls)


def model_from_env(
    provider: str | None = None,
    model: str | None = None,
) -> ChatModel:
    """Pick a model from arguments or environment variables.

    DATADIFF_AGENT_PROVIDER  groq | openai | ollama | offline  (default: groq
                             if GROQ_API_KEY is set, otherwise offline)
    DATADIFF_AGENT_MODEL     model name override
    DATADIFF_AGENT_BASE_URL  custom OpenAI-compatible endpoint
    """
    provider = provider or os.getenv("DATADIFF_AGENT_PROVIDER")
    if provider is None:
        provider = "groq" if os.getenv("GROQ_API_KEY") else "offline"

    if provider == "offline":
        return RuleBasedModel()

    if provider not in PROVIDERS:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Use one of: offline, {', '.join(PROVIDERS)}."
        )

    spec = PROVIDERS[provider]
    model = model or os.getenv("DATADIFF_AGENT_MODEL") or spec["default_model"]
    if not model:
        raise ValueError(
            f"Set DATADIFF_AGENT_MODEL (or --model) for provider '{provider}'."
        )

    api_key = None
    if spec["api_key_env"]:
        api_key = os.getenv(spec["api_key_env"])
        if not api_key:
            raise ValueError(
                f"Provider '{provider}' needs {spec['api_key_env']} to be set."
            )

    return OpenAICompatibleModel(
        model=model,
        base_url=os.getenv("DATADIFF_AGENT_BASE_URL") or spec["base_url"],
        api_key=api_key,
    )


# --- offline rule-based planner -------------------------------------------


class RuleBasedModel:
    """A scripted stand-in for an LLM.

    It walks through the same tools a good LLM would use, in a fixed order:

        contract -> compare -> history -> column details -> decision

    and decides with transparent rules that use the dataset contract:

    * CRITICAL issue on a critical column, or dataset-wide -> BLOCK
    * CRITICAL issue on a non-critical column              -> WARN
      (e.g. a CRM-only email column: alert, but don't stop finance reports)
    * any WARNING                                          -> WARN
    * problems in 2+ different critical columns at once    -> BLOCK
    * issues people previously marked as false positives are downgraded
      to INFO (memory in action)

    It reads tool results back out of the conversation, exactly like an
    LLM does, so the agent loop can't tell the difference.
    """

    name = "offline-rules"
    MAX_DETAIL_LOOKUPS = 3

    def complete(self, messages: list[dict], tools: list[dict]) -> AssistantTurn:
        results = _tool_results(messages)
        called = [name for name, _ in results]

        for step in ("get_dataset_contract", "compare_batches", "get_run_history"):
            if step not in called:
                return _call(step, {}, len(called))

        comparison = _last_result(results, "compare_batches")
        looked_up = {
            args.get("column")
            for name, args in _tool_calls(messages)
            if name == "get_column_details"
        }
        to_inspect = [
            issue["column"]
            for issue in comparison.get("issues", [])
            if issue["severity"] != "INFO"
            and issue["column"]
            and issue["kind"] != "column_removed"
            and issue["column"] not in looked_up
        ]
        if to_inspect and len(looked_up) < self.MAX_DETAIL_LOOKUPS:
            return _call(
                "get_column_details", {"column": to_inspect[0]}, len(called)
            )

        history = _last_result(results, "get_run_history")
        return _call(
            "submit_decision", self._decide(comparison, history), len(called)
        )

    @staticmethod
    def _decide(comparison: dict, history: dict) -> dict:
        accepted = {
            (fp["kind"], fp["column"])
            for fp in history.get("known_false_positives", [])
        }
        findings = []
        decision = "PASS"
        for issue in comparison.get("issues", []):
            severity = issue["severity"]
            explanation = issue["message"]
            critical_column = issue["column_is_critical"]
            dataset_wide = issue["column"] is None

            if (issue["kind"], issue["column"]) in accepted and severity != "INFO":
                severity = "INFO"
                explanation += " (previously marked as a false positive; treated as expected)"
            elif severity != "INFO" and critical_column:
                explanation += " This is a critical column."
            elif severity != "INFO" and not dataset_wide:
                explanation += " This column is not critical for downstream reports."

            if severity == "CRITICAL" and (critical_column or dataset_wide):
                issue_decision = "BLOCK"
            elif severity in ("WARNING", "CRITICAL"):
                issue_decision = "WARN"
            else:
                issue_decision = "PASS"
            if RANK[issue_decision] > RANK[decision]:
                decision = issue_decision

            findings.append(
                {
                    "kind": issue["kind"],
                    "column": issue["column"],
                    "severity": severity,
                    "explanation": explanation,
                }
            )

        # Correlated problems: when two or more *different* critical columns
        # are affected at once (e.g. a new currency code and a jump in
        # fares), something systematic broke upstream.
        affected_critical = {
            issue["column"]
            for issue, finding in zip(comparison.get("issues", []), findings)
            if issue["column_is_critical"] and finding["severity"] != "INFO"
        }
        if len(affected_critical) >= 2:
            decision = "BLOCK"
            findings.append(
                {
                    "kind": "correlated_issues",
                    "column": None,
                    "severity": "CRITICAL",
                    "explanation": (
                        "Several critical columns changed at the same time ("
                        + ", ".join(sorted(affected_critical))
                        + "), which points to a systematic upstream change."
                    ),
                }
            )

        worst = max((SEV_RANK[f["severity"]] for f in findings), default=0)
        severity = {0: "INFO", 1: "WARNING", 2: "CRITICAL"}[worst]

        notable = [f for f in findings if f["severity"] != "INFO"]
        known = [f for f in findings if "previously marked as a false positive" in f["explanation"]]
        if not notable and known:
            columns = ", ".join(sorted({str(f["column"]) for f in known}))
            summary = (
                f"Only previously accepted issue(s) seen again ({columns}); "
                "a person marked them as expected before, so the batch passes."
            )
        elif not notable:
            summary = "No notable changes compared with the last good batch."
        else:
            top = "; ".join(f["explanation"] for f in notable[:3])
            summary = f"{len(notable)} notable issue(s): {top}"

        actions = []
        if decision == "BLOCK":
            actions.append("Quarantine the batch and ask the source team to resend it.")
        if decision == "WARN":
            actions.append("Load the batch and review the flagged columns today.")

        return {
            "decision": decision,
            "severity": severity,
            "summary": summary,
            "findings": [f for f in findings if f["severity"] != "INFO"] or findings[:3],
            "recommended_actions": actions,
        }


def _call(name: str, arguments: dict, n: int) -> AssistantTurn:
    return AssistantTurn(tool_calls=[ToolCall(id=f"call_{n}", name=name, arguments=arguments)])


def _tool_calls(messages: list[dict]) -> list[tuple[str, dict]]:
    calls = []
    for message in messages:
        for call in message.get("tool_calls", []) or []:
            calls.append(
                (call["function"]["name"], json.loads(call["function"]["arguments"]))
            )
    return calls


def _tool_results(messages: list[dict]) -> list[tuple[str, dict]]:
    """Pair each tool message with the name of the tool that produced it."""
    names = {}
    for message in messages:
        for call in message.get("tool_calls", []) or []:
            names[call["id"]] = call["function"]["name"]
    results = []
    for message in messages:
        if message.get("role") == "tool":
            try:
                payload = json.loads(message["content"])
            except json.JSONDecodeError:
                payload = {"raw": message["content"]}
            results.append((names.get(message["tool_call_id"], "?"), payload))
    return results


def _last_result(results: list[tuple[str, dict]], name: str) -> dict:
    for tool_name, payload in reversed(results):
        if tool_name == name:
            return payload
    return {}
