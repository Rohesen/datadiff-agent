"""Command line interface.

    datadiff-agent demo                      run the one-week airline demo
    datadiff-agent check OLD NEW             review one batch
    datadiff-agent history                   past decisions and feedback
    datadiff-agent feedback RUN_ID VERDICT   tell the agent if it was right
    datadiff-agent mcp --data-root DIR       run the MCP server

Model selection (all commands that run the agent):
    --provider offline|groq|openai|ollama   (default: groq if GROQ_API_KEY
                                             is set, otherwise offline)
    --model NAME
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from . import __version__
from .actions import ApprovalGate, Notifier, format_step, render_report
from .agent import run_agent
from .contracts import default_contract, load_contract
from .demo import SCENARIOS, write_week
from .llm import model_from_env
from .memory import FEEDBACK_VERDICTS, Memory
from .pipeline import LocalPipeline
from .tools import RunContext

EXIT_CODES = {"PASS": 0, "WARN": 1, "BLOCK": 2}
EXIT_ERROR = 3


def _print_step(step) -> None:
    print(f"   {step.number:>2}. {format_step(step)}")


def _read_table(path: str) -> pd.DataFrame:
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in (".parquet", ".pq"):
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {suffix}")


# --- commands --------------------------------------------------------------


def cmd_demo(args) -> int:
    root = Path(args.dir)
    if root.exists() and any(root.iterdir()) and not args.force:
        print(f"{root} is not empty. Use --force to reuse it, or pick another --dir.")
        return EXIT_ERROR

    model = model_from_env(args.provider, args.model)
    pipeline = LocalPipeline(
        root,
        contract=load_contract("bookings"),
        model=model,
        notifier=Notifier(printer=lambda text: print("   " + text.split("\n")[0])),
        approval=ApprovalGate("ask" if args.interactive else "deny"),
    )
    files = write_week(root / "landing")

    print(f"datadiff-agent demo · model: {model.name}")
    print(f"Working folder: {root.resolve()}\n")

    rows = []
    try:
        for scenario, path in zip(SCENARIOS, files):
            print(f"── Day {scenario.day}: {scenario.title} ({path.name})")
            outcome = pipeline.process(path, on_step=_print_step if args.verbose else None)

            if outcome.result is None:
                print(f"   {outcome.note}\n")
                rows.append((scenario, "LOAD", True))
                continue

            result = outcome.result
            guard = " (raised by guardrail)" if result.guardrail.overridden else ""
            print(f"   Decision: {result.decision}{guard} · severity {result.severity}")
            print(f"   {result.summary}")
            status = "loaded" if outcome.loaded else "quarantined"
            extra = f" {outcome.note}" if outcome.note and outcome.note != "Quarantined." else ""
            print(f"   → {status}.{extra} Report: {outcome.report.relative_to(root)}\n")
            rows.append((scenario, result.decision, outcome.loaded))

            if scenario.day == 3 and not args.no_feedback:
                pipeline.memory.add_feedback(
                    result.run_id,
                    "false_positive",
                    "Known issue: the booking engine stopped sending emails "
                    "for agent bookings. CRM team is aware; not a data error.",
                )
                print(
                    "   💬 Simulated human feedback: a data engineer marks this "
                    f"run (#{result.run_id}) as a false positive. The agent "
                    "will remember.\n"
                )
    finally:
        loaded_rows = pipeline.warehouse_row_count()
        pipeline.close()

    print("Day  Scenario                                Expected  Agent   Loaded")
    for scenario, decision, loaded in rows:
        mark = "✓" if decision == scenario.expected else "✗"
        print(
            f"{scenario.day:>3}  {scenario.title:<38}  {scenario.expected:<8}  "
            f"{decision:<6}  {'yes' if loaded else 'no '} {mark}"
        )
    print(f"\nWarehouse rows: {loaded_rows:,}. Reports: {root / 'reports'}")
    return 0


def cmd_check(args) -> int:
    try:
        old = _read_table(args.old)
        new = _read_table(args.new)
        contract = (
            load_contract(args.contract) if args.contract
            else default_contract(args.dataset or Path(args.new).stem)
        )
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_ERROR

    model = model_from_env(args.provider, args.model)
    memory = Memory(args.db)
    try:
        ctx = RunContext(
            dataset=args.dataset or contract.name,
            batch=Path(args.new).name,
            old=old,
            new=new,
            contract=contract,
            memory=memory,
        )
        result = run_agent(
            ctx, model, max_steps=args.max_steps,
            on_step=None if args.json else _print_step,
        )
    finally:
        memory.close()

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print()
        print(render_report(ctx.dataset, ctx.batch, result))
    return EXIT_CODES[result.decision]


def cmd_history(args) -> int:
    memory = Memory(args.db)
    try:
        runs = memory.recent_runs(args.dataset, limit=args.limit)
        if not runs:
            print(f"No runs recorded for '{args.dataset}' in {args.db}.")
        for r in runs:
            fb = ", ".join(f["verdict"] for f in r.feedback) or "no feedback"
            print(f"#{r.run_id:<4} {r.created_at}  {r.decision:<5} {r.severity:<8} {r.batch}  [{fb}]")
            print(f"       {r.summary[:160]}")
        fps = memory.known_false_positives(args.dataset)
        if fps:
            print("\nKnown false positives:")
            for fp in fps:
                print(f"  - {fp['kind']} on {fp['column']}: {fp['note']}")
    finally:
        memory.close()
    return 0


def cmd_feedback(args) -> int:
    memory = Memory(args.db)
    try:
        memory.add_feedback(args.run_id, args.verdict, args.note or "")
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_ERROR
    finally:
        memory.close()
    print(f"Recorded '{args.verdict}' for run #{args.run_id}.")
    return 0


def cmd_mcp(args) -> int:
    try:
        from .mcp_server import serve
    except ImportError:
        print("Install the MCP extra: pip install 'datadiff-agent[mcp]'", file=sys.stderr)
        return EXIT_ERROR
    serve(args.data_root, args.db)
    return 0


# --- parser ----------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="datadiff-agent",
        description="An AI agent that guards data pipelines with datadiff-engine.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    def model_args(p):
        p.add_argument("--provider", choices=["offline", "groq", "openai", "ollama"])
        p.add_argument("--model")

    demo = sub.add_parser("demo", help="Run the one-week airline bookings demo.")
    demo.add_argument("--dir", default="demo_run")
    demo.add_argument("--force", action="store_true", help="Reuse a non-empty --dir.")
    demo.add_argument("--verbose", "-v", action="store_true", help="Show every tool call.")
    demo.add_argument("--interactive", action="store_true",
                      help="Ask before keeping blocked batches out of the warehouse.")
    demo.add_argument("--no-feedback", action="store_true",
                      help="Skip the simulated human feedback on day 3.")
    model_args(demo)
    demo.set_defaults(func=cmd_demo)

    check = sub.add_parser(
        "check",
        help="Review one batch. Exit code: 0 PASS, 1 WARN, 2 BLOCK, 3 error.",
    )
    check.add_argument("old", help="Last good batch (CSV or Parquet).")
    check.add_argument("new", help="New batch to review.")
    check.add_argument("--contract", help="Contract JSON path or bundled name (e.g. bookings).")
    check.add_argument("--dataset", help="Dataset name (defaults to the contract name).")
    check.add_argument("--db", default="datadiff_agent.db", help="Memory database path.")
    check.add_argument("--max-steps", type=int, default=12)
    check.add_argument("--json", action="store_true")
    model_args(check)
    check.set_defaults(func=cmd_check)

    history = sub.add_parser("history", help="Show past decisions and feedback.")
    history.add_argument("--dataset", default="bookings")
    history.add_argument("--db", default="datadiff_agent.db")
    history.add_argument("--limit", type=int, default=10)
    history.set_defaults(func=cmd_history)

    feedback = sub.add_parser("feedback", help="Tell the agent whether a decision was right.")
    feedback.add_argument("run_id", type=int)
    feedback.add_argument("verdict", choices=FEEDBACK_VERDICTS)
    feedback.add_argument("--note")
    feedback.add_argument("--db", default="datadiff_agent.db")
    feedback.set_defaults(func=cmd_feedback)

    mcp = sub.add_parser("mcp", help="Run the MCP server (stdio).")
    mcp.add_argument("--data-root", required=True)
    mcp.add_argument("--db", default="datadiff_agent.db")
    mcp.set_defaults(func=cmd_mcp)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ValueError as error:  # e.g. missing API key
        print(f"error: {error}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
