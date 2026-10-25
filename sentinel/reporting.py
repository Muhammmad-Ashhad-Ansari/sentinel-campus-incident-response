"""Small reporting helpers for the CLI and tests."""

from __future__ import annotations

from typing import Dict, List


def format_path(path: List[str]) -> str:
    if not path:
        return "(no path)"
    try:
        return " -> ".join(path)
    except Exception:
        return " -> ".join(str(p) for p in path)


def print_section(title: str) -> None:
    line = "=" * 60
    print(f"\n{line}\n{title}\n{line}")


def print_table(rows: List[List[object]], headers: List[str] | None = None) -> None:
    if headers:
        rows = [list(headers)] + [list(row) for row in rows]
    if not rows:
        return
    widths = [0] * len(rows[0])
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(str(cell)))
    for r, row in enumerate(rows):
        parts = []
        for idx, cell in enumerate(row):
            parts.append(str(cell).ljust(widths[idx]))
        print(" | ".join(parts))
        if headers and r == 0:
            print("-+-".join("-" * width for width in widths))


def summarise_search(result: Dict[str, object]) -> str:
    if not result.get("success"):
        return f"{result.get('algorithm')}: FAILED - {result.get('error')}"
    path = format_path(result.get("path", []))
    base = (
        f"{result.get('algorithm'):12} {path:25} cost={result.get('travel_cost'):>2} "
        f"risk={result.get('risk'):>2} edges={result.get('edges')}"
    )
    if result.get("objective") is not None:
        base += f"  J={result.get('objective')} (lambda={result.get('objective_lambda')})"
    return base


def summarise_agent(decision: Dict[str, object]) -> str:
    if decision.get("action") is None:
        return f"{decision.get('incident_id')}: NO ACTION - {decision.get('explanation')}"
    return (
        f"{decision.get('incident_id'):4} -> {decision.get('action'):12} "
        f"[{decision.get('triggered_rule')}]  {decision.get('explanation')}"
    )


def summarise_integration(result: Dict[str, object]) -> str:
    if result.get("action") is None:
        return f"{result.get('incident_id')}: INTEGRATION ERROR - {result.get('explanation')}"
    route = result.get("route") or {}
    route_str = "no route"
    if route.get("success"):
        route_str = f"route {format_path(route.get('path',[]))} (cost={route.get('travel_cost')}, risk={route.get('risk')})"
    elif route:
        route_str = f"route error: {route.get('error')}"
    return (
        f"{result.get('incident_id'):4} predicted={result.get('predicted_class') or 'N/A':9} "
        f"action={result.get('action'):12} rule={result.get('triggered_rule')}  {route_str}"
    )
