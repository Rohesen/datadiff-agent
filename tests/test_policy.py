from datadiff_agent.policy import Floor, compute_floor, enforce


def test_clean_batch_has_pass_floor(make_ctx, contract):
    assert compute_floor(make_ctx(2, old_day=1).diff, contract).decision == "PASS"


def test_removed_critical_column_forces_block(make_ctx, contract):
    floor = compute_floor(make_ctx(5).diff, contract)
    assert floor.decision == "BLOCK"
    assert any("fare_amount" in reason for reason in floor.reasons)


def test_partial_file_forces_block(make_ctx, contract):
    assert compute_floor(make_ctx(6).diff, contract).decision == "BLOCK"


def test_warning_on_critical_column_forces_at_least_warn(make_ctx, contract):
    # Day 4: new currency value + fare drift, both on critical columns.
    assert compute_floor(make_ctx(4).diff, contract).decision == "WARN"


def test_problem_in_non_critical_column_has_no_floor(make_ctx, contract):
    # Day 3: passenger_email (non-critical) mostly null.
    assert compute_floor(make_ctx(3).diff, contract).decision == "PASS"


def test_enforce_raises_lenient_decision():
    floor = Floor()
    floor.raise_to("BLOCK", "critical column removed")

    result = enforce("PASS", floor)

    assert result.decision == "BLOCK"
    assert result.overridden
    assert "critical column removed" in result.note


def test_enforce_allows_stricter_decision():
    floor = Floor()
    floor.raise_to("WARN", "drift")

    result = enforce("BLOCK", floor)

    assert result.decision == "BLOCK"
    assert not result.overridden
