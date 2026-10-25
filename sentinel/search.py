"""Graph search algorithms used by SENTINEL.

Every public search function returns the same result dictionary:

    {
        "algorithm": "BFS",
        "start": "A",
        "goal": "F",
        "path": ["A", "B", "D", "F"],
        "edges": 3,
        "travel_cost": 7,
        "risk": 23,
        "objective": None,          # only risk-aware search fills this
        "objective_lambda": None,   # only risk-aware search fills this
        "success": True,
        "error": None
    }

Metrics are calculated from the graph after the path is found.  Nothing is
copied from the assignment's expected-results table.
"""

from __future__ import annotations

import heapq
import itertools
from collections import deque
from typing import Dict, List

from .graph import CampusGraph, GraphError, UnknownNodeError

Path = List[str]
SearchResult = Dict[str, object]


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _failure(algorithm: str, start: str, goal: str, error: str) -> SearchResult:
    return {
        "algorithm": algorithm,
        "start": start,
        "goal": goal,
        "path": [],
        "edges": 0,
        "travel_cost": 0,
        "risk": 0,
        "objective": None,
        "objective_lambda": None,
        "success": False,
        "error": error,
    }


def _success(
    algorithm: str,
    start: str,
    goal: str,
    path: Path,
    graph: CampusGraph,
    objective_lambda: float | None = None,
) -> SearchResult:
    metrics = graph.path_metrics(path, objective_lambda=objective_lambda)
    return {
        "algorithm": algorithm,
        "start": start,
        "goal": goal,
        "path": metrics["path"],
        "edges": metrics["edges"],
        "travel_cost": metrics["travel_cost"],
        "risk": metrics["risk"],
        "objective": metrics.get("objective"),
        "objective_lambda": metrics.get("objective_lambda"),
        "success": True,
        "error": None,
    }


def _validate_endpoints(graph: CampusGraph, start: str, goal: str, algorithm: str) -> SearchResult | None:
    """Return a failure result for bad endpoints, otherwise ``None``."""
    if not graph.has_node(start):
        return _failure(algorithm, start, goal, f"Unknown start node: {start!r}")
    if not graph.has_node(goal):
        return _failure(algorithm, start, goal, f"Unknown goal node: {goal!r}")
    return None


# ----------------------------------------------------------------------
# A. Breadth-First Search - fewest edges
# ----------------------------------------------------------------------
def bfs(graph: CampusGraph, start: str = "A", goal: str = "F") -> SearchResult:
    """Find a path with the fewest edges using a FIFO queue.

    Neighbours are enqueued in alphabetical order, which makes the result
    deterministic.  A visited set prevents cycles from causing infinite work.
    """
    algorithm = "BFS"
    invalid = _validate_endpoints(graph, start, goal, algorithm)
    if invalid:
        return invalid

    queue = deque([[start]])
    visited = {start}

    while queue:
        path = queue.popleft()
        node = path[-1]
        if node == goal:
            return _success(algorithm, start, goal, path, graph)

        for neighbour in graph.neighbors(node):  # already alphabetical
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append(path + [neighbour])

    return _failure(algorithm, start, goal, f"No path from {start!r} to {goal!r}.")


# ----------------------------------------------------------------------
# B. Depth-First Search - alphabetical neighbour order
# ----------------------------------------------------------------------
def dfs(graph: CampusGraph, start: str = "A", goal: str = "F") -> SearchResult:
    """Find a path by exploring the first alphabetical neighbour first.

    The traversal rule is recursive DFS with neighbours sorted A..Z.  On the
    assignment graph this produces A -> B -> D -> C -> E -> F (cost 13,
    risk 19), matching the DOCX correction note.  A visited set prevents
    cycles, and the current path is copied before returning.
    """
    algorithm = "DFS"
    invalid = _validate_endpoints(graph, start, goal, algorithm)
    if invalid:
        return invalid

    visited = set()
    found_path: List[str] | None = None

    def visit(node: str, path: Path) -> bool:
        nonlocal found_path
        visited.add(node)
        if node == goal:
            found_path = list(path)
            return True

        for neighbour in graph.neighbors(node):  # alphabetical
            if neighbour not in visited:
                if visit(neighbour, path + [neighbour]):
                    return True
        return False

    visit(start, [start])
    if found_path is None:
        return _failure(algorithm, start, goal, f"No path from {start!r} to {goal!r}.")

    return _success(algorithm, start, goal, found_path, graph)


# ----------------------------------------------------------------------
# C. Uniform-Cost Search - minimum travel cost
# ----------------------------------------------------------------------
def uniform_cost_search(graph: CampusGraph, start: str = "A", goal: str = "F") -> SearchResult:
    """Dijkstra's algorithm using a min-heap keyed only by travel cost.

    The heap can contain stale entries after a cheaper route is discovered.
    Those entries are skipped by comparing the popped cost with ``best``.
    """
    algorithm = "UCS"
    invalid = _validate_endpoints(graph, start, goal, algorithm)
    if invalid:
        return invalid

    counter = itertools.count()
    frontier: List[tuple[float, int, str, Path]] = [(0.0, next(counter), start, [start])]
    best: Dict[str, float] = {start: 0.0}

    while frontier:
        cost, _, node, path = heapq.heappop(frontier)
        if cost > best.get(node, float("inf")):
            continue  # stale entry
        if node == goal:
            return _success(algorithm, start, goal, path, graph)

        for neighbour in graph.neighbors(node):
            edge = graph.get_edge(node, neighbour)
            new_cost = cost + edge["cost"]
            if new_cost < best.get(neighbour, float("inf")):
                best[neighbour] = new_cost
                heapq.heappush(
                    frontier,
                    (new_cost, next(counter), neighbour, path + [neighbour]),
                )

    return _failure(algorithm, start, goal, f"No path from {start!r} to {goal!r}.")


# ----------------------------------------------------------------------
# D. Risk-aware search - minimum J(P) = sum(cost) + lambda * sum(risk)
# ----------------------------------------------------------------------
def risk_aware_search(
    graph: CampusGraph,
    start: str = "A",
    goal: str = "F",
    lam: float = 1.0,
) -> SearchResult:
    """Dijkstra search over the additive risk-aware objective.

    ``lam`` (lambda) controls how strongly risk is penalised.  The assignment
    uses ``lam = 1``.  Because both cost and risk are non-negative, the edge
    weight ``cost + lam * risk`` is non-negative, so Dijkstra is valid.
    """
    algorithm = "Risk-Aware"
    if not _is_number(lam) or lam < 0:
        return _failure(algorithm, start, goal, "lambda must be a non-negative number.")

    invalid = _validate_endpoints(graph, start, goal, algorithm)
    if invalid:
        return invalid

    counter = itertools.count()
    # heap entry: (objective, travel_cost, counter, node, path)
    frontier: List[tuple[float, float, int, str, Path]] = [
        (0.0, 0.0, next(counter), start, [start])
    ]
    best: Dict[str, float] = {start: 0.0}

    while frontier:
        objective, travel_cost, _, node, path = heapq.heappop(frontier)
        if objective > best.get(node, float("inf")):
            continue  # stale entry
        if node == goal:
            return _success(algorithm, start, goal, path, graph, objective_lambda=lam)

        for neighbour in graph.neighbors(node):
            edge = graph.get_edge(node, neighbour)
            new_cost = travel_cost + edge["cost"]
            new_objective = objective + edge["cost"] + lam * edge["risk"]
            if new_objective < best.get(neighbour, float("inf")):
                best[neighbour] = new_objective
                heapq.heappush(
                    frontier,
                    (
                        new_objective,
                        new_cost,
                        next(counter),
                        neighbour,
                        path + [neighbour],
                    ),
                )

    return _failure(algorithm, start, goal, f"No path from {start!r} to {goal!r}.")


def run_all_searches(graph: CampusGraph, start: str = "A", goal: str = "F", lam: float = 1.0) -> List[SearchResult]:
    """Run the four required algorithms in the assignment's order."""
    return [
        bfs(graph, start, goal),
        dfs(graph, start, goal),
        uniform_cost_search(graph, start, goal),
        risk_aware_search(graph, start, goal, lam=lam),
    ]


def search_by_name(name: str, graph: CampusGraph, start: str = "A", goal: str = "F", lam: float = 1.0) -> SearchResult:
    """Dispatch helper used by the integration layer for route requests."""
    key = (name or "").strip().lower()
    if key in {"bfs", "breadth-first", "breadth_first"}:
        return bfs(graph, start, goal)
    if key in {"dfs", "depth-first", "depth_first"}:
        return dfs(graph, start, goal)
    if key in {"ucs", "uniform-cost", "uniform_cost", "dijkstra"}:
        return uniform_cost_search(graph, start, goal)
    if key in {"risk-aware", "risk_aware", "riskaware", "risk"}:
        return risk_aware_search(graph, start, goal, lam=lam)
    return _failure(name, start, goal, f"Unknown route algorithm: {name!r}")


def route_result_to_dict(result: SearchResult) -> Dict[str, object]:
    """Explicit conversion keeps the integration output JSON-compatible."""
    return {
        "algorithm": result["algorithm"],
        "start": result["start"],
        "goal": result["goal"],
        "path": list(result["path"]),
        "edges": result["edges"],
        "travel_cost": result["travel_cost"],
        "risk": result["risk"],
        "objective": result["objective"],
        "objective_lambda": result["objective_lambda"],
        "success": result["success"],
        "error": result["error"],
    }
