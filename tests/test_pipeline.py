from datadiff_agent.actions import ApprovalGate, Notifier
from datadiff_agent.demo import SCENARIOS, write_week
from datadiff_agent.llm import RuleBasedModel
from datadiff_agent.pipeline import LocalPipeline


def build(tmp_path, contract, approval="deny"):
    alerts = []
    pipeline = LocalPipeline(
        tmp_path / "run",
        contract=contract,
        model=RuleBasedModel(),
        notifier=Notifier(printer=alerts.append, slack_webhook_url=""),
        approval=ApprovalGate(approval),
    )
    return pipeline, alerts


def test_full_week(tmp_path, contract):
    pipeline, alerts = build(tmp_path, contract)
    files = write_week(tmp_path / "run" / "landing")

    outcomes = []
    for scenario, path in zip(SCENARIOS, files):
        outcome = pipeline.process(path)
        outcomes.append(outcome)
        if scenario.day == 3:
            pipeline.memory.add_feedback(outcome.result.run_id, "false_positive", "known")

    decisions = ["LOAD"] + [o.result.decision for o in outcomes[1:]]
    assert decisions == [s.expected for s in SCENARIOS]

    quarantined = sorted(p.name for p in (tmp_path / "run" / "quarantine").iterdir())
    assert quarantined == [
        "bookings_2026-10-04.csv", "bookings_2026-10-05.csv", "bookings_2026-10-06.csv",
    ]
    assert len(list((tmp_path / "run" / "reports").glob("*.md"))) == 6
    assert len(alerts) == 4  # 1 WARN + 3 BLOCK
    loaded_rows = sum(len(__import__("pandas").read_csv(f)) for f, o in zip(files, outcomes) if o.loaded)
    assert pipeline.warehouse_row_count() == loaded_rows
    pipeline.close()


def test_warn_batch_does_not_become_baseline(tmp_path, contract):
    pipeline, _ = build(tmp_path, contract)
    files = write_week(tmp_path / "run" / "landing")
    pipeline.process(files[1])  # day 2 -> baseline
    baseline_before = pipeline.baseline_path.read_text()

    pipeline.process(files[2])  # day 3 -> WARN, loaded

    assert pipeline.baseline_path.read_text() == baseline_before
    pipeline.close()


def test_human_can_override_a_block(tmp_path, contract):
    pipeline, _ = build(tmp_path, contract, approval="approve")
    files = write_week(tmp_path / "run" / "landing")
    pipeline.process(files[0])

    outcome = pipeline.process(files[5])  # partial file -> BLOCK

    assert outcome.result.decision == "BLOCK"
    assert outcome.loaded
    assert "human approval" in outcome.note
    pipeline.close()


def test_interactive_gate_only_loads_on_explicit_yes(tmp_path, contract):
    answers = iter(["yes please", "load"])
    gate = ApprovalGate("ask", ask=lambda prompt: next(answers))
    result = type("R", (), {"summary": "partial file"})()

    assert gate.allow_load_anyway("bookings", "b", result) is False
    assert gate.allow_load_anyway("bookings", "b", result) is True
