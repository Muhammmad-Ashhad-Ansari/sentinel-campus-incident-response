"""Campus graph for SENTINEL.

The graph is undirected (the assignment says edges are bidirectional).  Every
edge stores two independent numbers:

* ``cost`` - travel cost, a non-negative number
* ``risk`` - security risk on a 0..10 scale

Keeping the graph in its own module means the search algorithms never contain
campus-specific data or hardcoded answers.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple


class GraphError(ValueError):
    """Base class for graph validation errors."""


class UnknownNodeError(GraphError):
    """Raised when an operation references a node that is not in the graph."""


class InvalidEdgeError(GraphError):
    """Raised when an edge has an invalid cost or risk value."""


def _is_number(value: object) -> bool:
    """Return True for real numbers but not for booleans or strings."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_node_name(node: object) -> str:
    if not isinstance(node, str) or not node.strip():
        raise UnknownNodeError("Node names must be non-empty strings.")
    return node


class CampusGraph:
    """A small, validated, undirected campus graph.

    Example
    -------
    >>> g = CampusGraph()
    >>> g.add_edge("A", "B", 2, 8)
    >>> g.get_edge("B", "A")
    {'cost': 2, 'risk': 8}
    >>> g.neighbors("A")
    ['B']
    """

    def __init__(self) -> None:
        # adjacency: node -> {neighbor: {"cost": number, "risk": number}}
        self._adj: Dict[str, Dict[str, Dict[str, float]]] = {}

    # ------------------------------------------------------------------
    # Building and changing the graph
    # ------------------------------------------------------------------
    def add_node(self, node: str) -> None:
        """Add an isolated node. Adding an existing node is harmless."""
        node = _validate_node_name(node)
        self._adj.setdefault(node, {})

    def add_edge(self, start: str, goal: str, cost: float, risk: float) -> None:
        """Add or replace an undirected edge after validating both values.

        Travel cost must be >= 0.  Risk must be in the inclusive range 0..10.
        Adding the same edge twice updates it instead of creating a duplicate.
        """
        start = _validate_node_name(start)
        goal = _validate_node_name(goal)
        if start == goal:
            raise InvalidEdgeError("Self-loops are not supported by this campus graph.")
        if not _is_number(cost):
            raise InvalidEdgeError("Travel cost must be a number.")
        if not _is_number(risk):
            raise InvalidEdgeError("Risk must be a number.")
        if cost < 0:
            raise InvalidEdgeError("Travel cost cannot be negative.")
        if risk < 0 or risk > 10:
            raise InvalidEdgeError("Risk must be between 0 and 10 (inclusive).")

        self.add_node(start)
        self.add_node(goal)
        payload = {"cost": float(cost), "risk": float(risk)}
        self._adj[start][goal] = dict(payload)
        self._adj[goal][start] = dict(payload)

    def set_edge(self, start: str, goal: str, cost: float, risk: float) -> None:
        """Convenience alias used for live demonstrations."""
        self.add_edge(start, goal, cost, risk)

    def remove_edge(self, start: str, goal: str) -> None:
        """Remove an edge in both directions. Missing edges are ignored."""
        if start in self._adj:
            self._adj[start].pop(goal, None)
        if goal in self._adj:
            self._adj[goal].pop(start, None)

    # ------------------------------------------------------------------
    # Queries used by the search algorithms
    # ------------------------------------------------------------------
    def has_node(self, node: str) -> bool:
        return isinstance(node, str) and node in self._adj

    def nodes(self) -> List[str]:
        return sorted(self._adj)

    def neighbors(self, node: str) -> List[str]:
        """Return neighbours in alphabetical order.

        Alphabetical ordering is the documented traversal rule for BFS and DFS.
        """
        if not self.has_node(node):
            raise UnknownNodeError(f"Unknown campus node: {node!r}")
        return sorted(self._adj[node])

    def get_edge(self, start: str, goal: str) -> Dict[str, float]:
        if not self.has_node(start):
            raise UnknownNodeError(f"Unknown campus node: {start!r}")
        if not self.has_node(goal):
            raise UnknownNodeError(f"Unknown campus node: {goal!r}")
        try:
            return dict(self._adj[start][goal])
        except KeyError as exc:
            raise UnknownNodeError(f"No edge between {start!r} and {goal!r}.") from exc

    def edges(self) -> List[Tuple[str, str, float, float]]:
        """Return each undirected edge once, sorted for deterministic output."""
        result: List[Tuple[str, str, float, float]] = []
        for start in self.nodes():
            for goal in self.neighbors(start):
                if start < goal:
                    edge = self._adj[start][goal]
                    result.append((start, goal, edge["cost"], edge["risk"]))
        return result

    def path_metrics(self, path: Iterable[str], objective_lambda: float | None = None) -> Dict[str, object]:
        """Calculate metrics for a path from the current graph data.

        This function never trusts metrics supplied by a caller.  It raises
        ``GraphError`` if the path is empty or contains a missing edge.
        """
        path_list = list(path)
        if not path_list:
            raise GraphError("Cannot calculate metrics for an empty path.")
        for node in path_list:
            if not self.has_node(node):
                raise UnknownNodeError(f"Unknown campus node in path: {node!r}")

        total_cost = 0.0
        total_risk = 0.0
        for start, goal in zip(path_list, path_list[1:]):
            try:
                edge = self._adj[start][goal]
            except KeyError as exc:
                raise GraphError(f"No edge between {start!r} and {goal!r} in path.") from exc
            total_cost += edge["cost"]
            total_risk += edge["risk"]

        result: Dict[str, object] = {
            "path": path_list,
            "edges": max(0, len(path_list) - 1),
            "travel_cost": _clean_number(total_cost),
            "risk": _clean_number(total_risk),
        }
        if objective_lambda is not None:
            result["objective"] = _clean_number(total_cost + objective_lambda * total_risk)
            result["objective_lambda"] = objective_lambda
        return result

    # ------------------------------------------------------------------
    # Serialisation / display helpers
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, object]:
        return {
            "nodes": self.nodes(),
            "edges": [
                {"from": a, "to": b, "cost": cost, "risk": risk}
                for a, b, cost, risk in self.edges()
            ],
        }

    def copy(self) -> "CampusGraph":
        """Return an independent copy so callers can experiment safely."""
        clone = CampusGraph()
        for start, goal, cost, risk in self.edges():
            clone.add_edge(start, goal, cost, risk)
        for node in self.nodes():
            clone.add_node(node)
        return clone


def _clean_number(value: float) -> float | int:
    """Turn 7.0 into 7 so JSON and console output stay readable."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def build_campus_graph() -> CampusGraph:
    """Create the exact graph from the SENTINEL assignment."""
    graph = CampusGraph()
    edges = [
        ("A", "B", 2, 8),
        ("A", "C", 4, 1),
        ("B", "D", 2, 7),
        ("B", "E", 5, 1),
        ("C", "D", 2, 1),
        ("C", "E", 5, 2),
        ("D", "F", 3, 8),
        ("E", "F", 2, 1),
    ]
    for start, goal, cost, risk in edges:
        graph.add_edge(start, goal, cost, risk)
    return graph
