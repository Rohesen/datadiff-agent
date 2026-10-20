import json

from datadiff_agent.tools import MAX_TOOL_OUTPUT_CHARS, TOOLS, run_tool


def test_every_tool_has_a_valid_schema():
    for tool in TOOLS.values():
        schema = tool.schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == tool.name
        assert schema["function"]["parameters"]["type"] == "object"


def test_compare_batches_marks_critical_columns(make_ctx):
    result = json.loads(run_tool(make_ctx(4), "compare_batches", {}))

    by_column = {i["column"]: i for i in result["issues"]}
    assert by_column["currency"]["kind"] == "new_category_values"
    assert by_column["currency"]["column_is_critical"] is True


def test_compare_batches_on_clean_day_has_no_issues(make_ctx):
    result = json.loads(run_tool(make_ctx(2, old_day=1), "compare_batches", {}))
    assert result["overall_severity"] == "INFO"
    assert result["issues"] == []


def test_unknown_column_returns_error_the_model_can_read(make_ctx):
    result = json.loads(run_tool(make_ctx(3), "get_column_details", {"column": "nope"}))
    assert "Unknown column" in result["error"]
    assert "passenger_email" in result["error"]  # tells the model what exists


def test_removed_column_details(make_ctx):
    result = json.loads(run_tool(make_ctx(5), "get_column_details", {"column": "fare_amount"}))
    assert result["status"] == "removed in new batch"
    assert result["contract"]["critical"] is True


def test_sample_values_shows_new_currency(make_ctx):
    result = json.loads(run_tool(make_ctx(4), "sample_values", {"column": "currency", "n": 2}))
    assert result["old"] == ["OMR", "OMR"]
    assert result["new"] == ["USD", "USD"]


def test_unknown_tool_and_bad_arguments_do_not_crash(make_ctx):
    ctx = make_ctx(2)
    assert "Unknown tool" in run_tool(ctx, "drop_table", {})
    assert "Bad arguments" in run_tool(ctx, "get_column_details", {"colum": "x"})


def test_tool_output_is_size_limited(make_ctx):
    ctx = make_ctx(2)
    for i in range(40):
        ctx.memory.record_run("bookings", f"b{i}", "PASS", "INFO", "x" * 400, [], "test")
    output = run_tool(ctx, "get_run_history", {"limit": 20})
    assert len(output) <= MAX_TOOL_OUTPUT_CHARS + 20
