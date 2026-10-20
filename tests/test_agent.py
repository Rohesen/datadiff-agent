from types import SimpleNamespace

import json
import pytest

from datadiff_agent.agent import run_agent, validate_decision, DecisionError
from datadiff_agent.llm import (
    AssistantTurn,
    OpenAICompatibleModel,
    RuleBasedModel,
    ToolCall,
    model_from_env,
)


# --- offline rule-based model on every demo scenario -----------------------

@pytest.mark.parametrize(
    ("new_day", "old_day", "expected"),
    [
        (2, 1, "PASS"),   # normal variation
        (3, 2, "WARN"),   # non-critical email column mostly null
        (4, 2, "BLOCK"),  # USD fares: correlated critical changes
        (5, 2, "BLOCK"),  # critical column renamed / removed
        (6, 2, "BLOCK"),  # partial file
    ],
)
def test_offline_agent_decisions(make_ctx, new_day, old_day, expected):
    result = run_agent(make_ctx(new_day, old_day), RuleBasedModel())
    assert result.decision == expected
    assert result.completed
    assert result.run_id is not None
    assert result.steps[0].tool == "get_dataset_contract"
    assert result.steps[-1].tool == "submit_decision"


def test_agent_learns_from_false_positive_feedback(make_ctx, memory):
    first = run_agent(make_ctx(3), RuleBasedModel())
    assert first.decision == "WARN"

    memory.add_feedback(first.run_id, "false_positive", "known CRM gap")

    second = run_agent(make_ctx(7), RuleBasedModel())
    assert second.decision == "PASS"
    assert "previously accepted" in second.summary


# --- scripted models to test the loop itself --------------------------------

class ScriptedModel:
    """Returns pre-written turns, one per call."""

    name = "scripted"

    def __init__(self, turns):
        self.turns = list(turns)
        self.seen_messages = []

    def complete(self, messages, tools):
        self.seen_messages.append(list(messages))
        return self.turns.pop(0) if self.turns else AssistantTurn(content="...")


def decide(decision, severity="INFO", call_id="d"):
    return AssistantTurn(tool_calls=[ToolCall(
        id=call_id, name="submit_decision",
        arguments={"decision": decision, "severity": severity,
                   "summary": "Looks fine to me.", "findings": []},
    )])


def test_guardrail_overrides_a_reckless_model(make_ctx):
    # The model ignores everything and says PASS on a batch missing fare_amount.
    result = run_agent(make_ctx(5), ScriptedModel([decide("PASS")]))

    assert result.proposed_decision == "PASS"
    assert result.decision == "BLOCK"
    assert result.guardrail.overridden
    assert result.severity == "CRITICAL"


def test_model_may_be_stricter_than_rules(make_ctx):
    result = run_agent(make_ctx(2, old_day=1), ScriptedModel([decide("WARN", "WARNING")]))
    assert result.decision == "WARN"
    assert not result.guardrail.overridden


def test_invalid_decision_is_sent_back_and_corrected(make_ctx):
    bad = AssistantTurn(tool_calls=[ToolCall(
        id="bad", name="submit_decision",
        arguments={"decision": "MAYBE", "severity": "INFO", "summary": "", "findings": []},
    )])
    model = ScriptedModel([bad, decide("PASS", call_id="good")])

    result = run_agent(make_ctx(2, old_day=1), model)

    assert result.decision == "PASS"
    error_message = model.seen_messages[1][-1]
    assert error_message["role"] == "tool"
    assert "Invalid decision" in error_message["content"]


def test_plain_text_answer_gets_a_reminder(make_ctx):
    model = ScriptedModel([AssistantTurn(content="It is fine."), decide("PASS")])

    result = run_agent(make_ctx(2, old_day=1), model)

    assert result.decision == "PASS"
    assert "submit_decision" in model.seen_messages[1][-1]["content"]


def test_agent_that_never_decides_fails_closed(make_ctx):
    looping = [
        AssistantTurn(tool_calls=[ToolCall(id=f"c{i}", name="compare_batches", arguments={})])
        for i in range(10)
    ]
    result = run_agent(make_ctx(2, old_day=1), ScriptedModel(looping), max_steps=4)

    assert not result.completed
    assert result.decision == "BLOCK"


def test_validate_decision_normalises_case():
    clean = validate_decision({
        "decision": "warn", "severity": "warning", "summary": " ok ",
        "findings": [{"kind": "x", "severity": "info", "explanation": "e"}],
    })
    assert clean["decision"] == "WARN"
    assert clean["findings"][0]["severity"] == "INFO"
    with pytest.raises(DecisionError):
        validate_decision({"decision": "PASS"})


# --- the real-LLM code path, with a fake OpenAI client ---------------------

class FakeOpenAIClient:
    """Mimics openai.OpenAI().chat.completions.create responses."""

    def __init__(self, script):
        self.script = list(script)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        # Snapshot the messages: the agent keeps appending to the same list.
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        name, args = self.script.pop(0)
        call = SimpleNamespace(
            id=f"call_{len(self.requests)}",
            function=SimpleNamespace(name=name, arguments=json.dumps(args)),
        )
        message = SimpleNamespace(content=None, tool_calls=[call])
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_openai_compatible_model_runs_the_full_loop(make_ctx):
    client = FakeOpenAIClient([
        ("get_dataset_contract", {}),
        ("compare_batches", {}),
        ("sample_values", {"column": "currency"}),
        ("submit_decision", {
            "decision": "BLOCK", "severity": "CRITICAL",
            "summary": "Fares arrived in USD instead of OMR.",
            "findings": [{"kind": "new_category_values", "column": "currency",
                          "severity": "CRITICAL", "explanation": "USD"}],
        }),
    ])
    model = OpenAICompatibleModel(model="test-model", client=client)

    result = run_agent(make_ctx(4), model)

    assert result.decision == "BLOCK"
    assert result.model == "llm:test-model"
    assert [s.tool for s in result.steps] == [
        "get_dataset_contract", "compare_batches", "sample_values", "submit_decision",
    ]
    # Every request carries the tool schemas, and the conversation grows
    # with assistant tool calls followed by tool results.
    last = client.requests[-1]
    assert {t["function"]["name"] for t in last["tools"]} >= {"compare_batches", "submit_decision"}
    roles = [m["role"] for m in last["messages"]]
    assert roles[:2] == ["system", "user"]
    assert roles.count("tool") == 3


def test_model_from_env_defaults_to_offline(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("DATADIFF_AGENT_PROVIDER", raising=False)
    assert model_from_env().name == "offline-rules"


def test_model_from_env_requires_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        model_from_env("groq")
