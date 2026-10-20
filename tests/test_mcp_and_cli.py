import asyncio
import json

import pytest

from datadiff_agent.cli import main
from datadiff_agent.demo import make_day

mcp = pytest.importorskip("mcp")


@pytest.fixture
def data_root(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    make_day(2).to_csv(root / "day2.csv", index=False)
    make_day(5).to_csv(root / "day5.csv", index=False)
    (tmp_path / "secret.csv").write_text("password\nhunter2\n")
    return root


def call(server, name, args):
    """Call a tool in-process and return its structured result."""
    result = asyncio.run(server.call_tool(name, args))
    assert not result.is_error, result.content
    structured = result.structured_content
    if structured is not None:
        # Non-dict return values are wrapped as {"result": value}.
        return structured.get("result", structured)
    return json.loads(result.content[0].text)


def test_mcp_server_lists_tools(data_root, tmp_path):
    from datadiff_agent.mcp_server import build_server

    server = build_server(data_root, tmp_path / "mem.db")
    tools = {t.name for t in asyncio.run(server.list_tools())}
    assert {"list_data_files", "compare_datasets", "column_report", "agent_history"} <= tools


def test_mcp_compare_datasets(data_root, tmp_path):
    from datadiff_agent.mcp_server import build_server

    server = build_server(data_root, tmp_path / "mem.db")
    result = call(server, "compare_datasets", {
        "old_file": "day2.csv", "new_file": "day5.csv", "contract": "bookings",
    })
    assert result["summary"]["overall_severity"] == "CRITICAL"
    assert "fare_amount" in result["schema"]["removed"]


def test_mcp_path_guard_blocks_files_outside_root(data_root):
    from datadiff_agent.mcp_server import PathGuard

    guard = PathGuard(data_root)
    with pytest.raises(ValueError, match="outside the data root"):
        guard.resolve("../secret.csv")
    assert guard.resolve("day2.csv").name == "day2.csv"


def test_cli_check_exit_codes(data_root, tmp_path, capsys):
    db = str(tmp_path / "mem.db")
    old, new = str(data_root / "day2.csv"), str(data_root / "day5.csv")

    code = main(["check", old, new, "--contract", "bookings", "--db", db,
                 "--provider", "offline", "--json"])
    report = json.loads(capsys.readouterr().out)

    assert code == 2  # BLOCK
    assert report["decision"] == "BLOCK"
    assert main(["check", old, old, "--contract", "bookings", "--db", db,
                 "--provider", "offline", "--json"]) == 0


def test_cli_feedback_and_history(tmp_path, data_root, capsys):
    db = str(tmp_path / "mem.db")
    main(["check", str(data_root / "day2.csv"), str(data_root / "day5.csv"),
          "--contract", "bookings", "--db", db, "--provider", "offline", "--json"])
    capsys.readouterr()

    assert main(["feedback", "1", "correct", "--note", "good catch", "--db", db]) == 0
    assert main(["history", "--dataset", "bookings", "--db", db]) == 0
    out = capsys.readouterr().out
    assert "BLOCK" in out and "correct" in out
    assert main(["feedback", "99", "correct", "--db", db]) == 3
