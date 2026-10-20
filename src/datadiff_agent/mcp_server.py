"""An MCP server that exposes datadiff-engine to any MCP-capable AI app.

MCP (Model Context Protocol) is a standard way to give AI applications
tools. With this server running, an assistant like Claude Desktop or
Claude Code can compare datasets on your machine and read the agent's
history, without any custom integration code.

Run it:      datadiff-agent mcp --data-root ./data
Claude Desktop config (claude_desktop_config.json):

    {
      "mcpServers": {
        "datadiff": {
          "command": "datadiff-agent",
          "args": ["mcp", "--data-root", "/absolute/path/to/data"]
        }
      }
    }

Security: tools only read files inside --data-root. Paths outside it are
rejected. Exposing a whole disk to an AI tool is a common, avoidable risk.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from datadiff_engine import compare

from .contracts import default_contract, load_contract
from .memory import Memory

SUPPORTED = {".csv", ".parquet", ".pq"}


class PathGuard:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()

    def resolve(self, relative: str, allowed: set[str] = SUPPORTED) -> Path:
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"Path '{relative}' is outside the data root.")
        if not path.is_file():
            raise ValueError(f"File '{relative}' not found under the data root.")
        if path.suffix.lower() not in allowed:
            raise ValueError(f"Unsupported file type '{path.suffix}'.")
        return path


def _read(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def build_server(data_root: str | Path, memory_path: str | Path):
    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError

    path_guard = PathGuard(data_root)

    class guard:  # noqa: N801 - small adapter around PathGuard
        """Turn path problems into ToolErrors, whose message reaches the model
        (other exceptions are treated as crashes and their text is hidden)."""

        root = path_guard.root

        @staticmethod
        def resolve(relative: str, allowed: set[str] = SUPPORTED) -> Path:
            try:
                return path_guard.resolve(relative, allowed)
            except ValueError as error:
                raise ToolError(str(error)) from None
    server = MCPServer(
        name="datadiff",
        instructions=(
            "Tools to compare two versions of a dataset and inspect data "
            "quality decisions. File paths are relative to the data root; "
            "call list_data_files first to see what is available."
        ),
    )

    @server.tool()
    def list_data_files() -> list[str]:
        """List CSV/Parquet files under the data root (relative paths)."""
        return sorted(
            str(p.relative_to(guard.root))
            for p in guard.root.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED
        )[:500]

    @server.tool()
    def compare_datasets(old_file: str, new_file: str, contract: str | None = None) -> dict:
        """Compare two dataset files and return issues, row and schema changes.

        Args:
            old_file: baseline file, relative to the data root.
            new_file: new file, relative to the data root.
            contract: optional bundled contract name (e.g. "bookings") or a
                contract JSON path relative to the data root, to apply that
                dataset's thresholds.
        """
        if contract and contract.endswith(".json"):
            spec = load_contract(guard.resolve(contract, allowed={".json"}))
        elif contract:
            spec = load_contract(contract)
        else:
            spec = default_contract(Path(new_file).stem)
        diff = compare(
            _read(guard.resolve(old_file)),
            _read(guard.resolve(new_file)),
            config=spec.diff_config(),
        )
        report = diff.to_dict()
        return {
            "summary": report["summary"],
            "rows": report["rows"],
            "schema": report["schema"],
        }

    @server.tool()
    def column_report(old_file: str, new_file: str, column: str) -> dict:
        """Detailed comparison of one column between two dataset files."""
        diff = compare(_read(guard.resolve(old_file)), _read(guard.resolve(new_file)))
        if column not in diff.column_diffs:
            return {"error": f"'{column}' is not present in both files."}
        return diff.column(column).to_dict()

    @server.tool()
    def agent_history(dataset: str, limit: int = 10) -> dict:
        """Recent decisions the data quality agent made for a dataset."""
        memory = Memory(memory_path)
        try:
            runs = memory.recent_runs(dataset, limit=max(1, min(limit, 50)))
            return {
                "runs": [
                    {k: v for k, v in r.to_dict().items() if k != "findings"}
                    for r in runs
                ],
                "known_false_positives": memory.known_false_positives(dataset),
            }
        finally:
            memory.close()

    return server


def serve(data_root: str | Path, memory_path: str | Path) -> None:
    build_server(data_root, memory_path).run("stdio")
