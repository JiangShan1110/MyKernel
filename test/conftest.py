from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

import pytest
from _pytest.reports import TestReport
from rich import box
from rich.console import Console
from rich.logging import RichHandler
from rich.markup import escape
from rich.table import Table
from rich.text import Text
from test_framework.utils import make_json_friendly

# ---------- global storage ----------
CASES: List[Dict[str, Any]] = []
console = Console()
_first_of_case = True


# ---------- hook: configure rich logging ----------
class CaseRichHandler(RichHandler):
    def emit(self, record: logging.LogRecord) -> None:
        global _first_of_case
        if _first_of_case:
            console.print()
            _first_of_case = False
        super().emit(record)


def pytest_configure(config):
    config.option.logging_disable = True
    handler = CaseRichHandler(
        level=logging.DEBUG,
        show_time=True,
        show_path=True,
        rich_tracebacks=True,
        console=console,
        omit_repeated_times=False,
    )
    logging.getLogger().setLevel(logging.DEBUG)
    logging.getLogger().addHandler(handler)


def pytest_runtest_logfinish(nodeid, location):
    global _first_of_case
    _first_of_case = True


# ---------- hook: collect test report ----------
# ---------- 1. worker report ----------
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call) -> None:
    outcome = yield
    report: TestReport = outcome.get_result()
    if report.when == "call":
        case_name = report.nodeid.replace("::()::", "::")
        params = {
            k: v
            for k, v in getattr(item, "funcargs", {}).items()
            if not k.startswith("_")
        }
        report.user_properties.append(("case_name", case_name))
        report.user_properties.append(
            ("params", json.dumps(make_json_friendly(params), ensure_ascii=False))
        )


# ---------- 2. master collect ----------
def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if hasattr(config, "workerinput"):
        return

    for report in terminalreporter.stats.get("passed", []):
        _collect_report(report)
    for report in terminalreporter.stats.get("failed", []):
        _collect_report(report)

    _print_rich_table()


def _collect_report(report: TestReport) -> None:
    props = dict(report.user_properties)
    CASES.append(
        {
            "name": props.get("case_name", report.nodeid),
            "status": "PASS" if report.outcome == "passed" else "FAIL",
            "param": props.get("params", {}),
        }
    )


def _print_rich_table() -> None:
    if not CASES:
        Console().print("[yellow]No test cases collected![/]")
        return

    passed = sum(1 for c in CASES if c["status"] == "PASS")
    failed = len(CASES) - passed

    width = console.size.width
    summary = Text(
        f" Total={len(CASES)}  Pass={passed}  Fail={failed} ".center(width, "="),
        style="italic bold",
    )

    table = Table(
        title=summary,
        header_style="italic cyan",
        expand=True,
        box=box.ASCII,
    )
    table.add_column("Name")
    table.add_column("Status")
    table.add_column("Parameters")

    for c in CASES:
        if c["status"] == "PASS":
            table.add_row(escape(c["name"]), "[green]PASS[/]", c["param"])
        else:
            table.add_row(escape(c["name"]), "[bold red]FAIL[/]", c["param"])

    Console().print(table)
