import pytest


def test_record_and_read_runs(memory):
    run_id = memory.record_run(
        "bookings", "day3.csv", "WARN", "CRITICAL", "email gap",
        [{"kind": "null_rate_increase", "column": "passenger_email"}], "test",
    )
    run = memory.get_run(run_id)
    assert run.decision == "WARN"
    assert run.findings[0]["column"] == "passenger_email"
    assert memory.recent_runs("bookings")[0].run_id == run_id
    assert memory.recent_runs("other") == []


def test_false_positive_feedback_is_remembered(memory):
    run_id = memory.record_run(
        "bookings", "day3.csv", "WARN", "CRITICAL", "email gap",
        [{"kind": "null_rate_increase", "column": "passenger_email"}], "test",
    )
    memory.add_feedback(run_id, "false_positive", "known CRM issue")

    assert memory.known_false_positives("bookings") == [
        {"kind": "null_rate_increase", "column": "passenger_email", "note": "known CRM issue"}
    ]
    assert memory.get_run(run_id).feedback[0]["verdict"] == "false_positive"


def test_correct_feedback_is_not_a_false_positive(memory):
    run_id = memory.record_run(
        "bookings", "b", "BLOCK", "CRITICAL", "s",
        [{"kind": "column_removed", "column": "fare_amount"}], "test",
    )
    memory.add_feedback(run_id, "correct")
    assert memory.known_false_positives("bookings") == []


def test_invalid_feedback_is_rejected(memory):
    with pytest.raises(ValueError):
        memory.add_feedback(999, "correct")
    run_id = memory.record_run("bookings", "b", "PASS", "INFO", "s", [], "test")
    with pytest.raises(ValueError):
        memory.add_feedback(run_id, "maybe")


def test_memory_persists_on_disk(tmp_path):
    from datadiff_agent.memory import Memory

    path = tmp_path / "mem.db"
    first = Memory(path)
    first.record_run("bookings", "b", "PASS", "INFO", "s", [], "test")
    first.close()

    second = Memory(path)
    assert len(second.recent_runs("bookings")) == 1
    second.close()
