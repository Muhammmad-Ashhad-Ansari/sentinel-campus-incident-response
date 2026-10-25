from __future__ import annotations

import unittest

from sentinel.graph import CampusGraph, GraphError, UnknownNodeError, build_campus_graph
from sentinel.search import bfs, dfs, risk_aware_search, uniform_cost_search


class TestCampusGraph(unittest.TestCase):
    def test_exact_assignment_graph_is_bidirectional(self):
        graph = build_campus_graph()
        self.assertTrue(graph.has_node("A"))
        self.assertTrue(graph.has_node("F"))
        edge_ab = graph.get_edge("A", "B")
        edge_ba = graph.get_edge("B", "A")
        self.assertEqual(edge_ab, {"cost": 2, "risk": 8})
        self.assertEqual(edge_ba, {"cost": 2, "risk": 8})

    def test_reject_negative_travel_cost(self):
        graph = CampusGraph()
        with self.assertRaises(GraphError):
            graph.add_edge("A", "B", -1, 1)

    def test_reject_invalid_risk(self):
        graph = CampusGraph()
        with self.assertRaises(GraphError):
            graph.add_edge("A", "B", 1, -1)
        with self.assertRaises(GraphError):
            graph.add_edge("A", "B", 1, 11)

    def test_unknown_node(self):
        graph = build_campus_graph()
        with self.assertRaises(UnknownNodeError):
            graph.neighbors("Z")


class TestSearch(unittest.TestCase):
    def setUp(self):
        self.graph = build_campus_graph()

    def test_bfs_expected_path_and_metrics(self):
        result = bfs(self.graph, "A", "F")
        self.assertTrue(result["success"])
        self.assertEqual(result["path"], ["A", "B", "D", "F"])
        self.assertEqual(result["edges"], 3)
        self.assertEqual(result["travel_cost"], 7)
        self.assertEqual(result["risk"], 23)

    def test_dfs_expected_path_and_metrics(self):
        result = dfs(self.graph, "A", "F")
        self.assertTrue(result["success"])
        self.assertEqual(result["path"], ["A", "B", "D", "C", "E", "F"])
        self.assertEqual(result["edges"], 5)
        self.assertEqual(result["travel_cost"], 13)
        self.assertEqual(result["risk"], 19)

    def test_ucs_expected_path_and_metrics(self):
        result = uniform_cost_search(self.graph, "A", "F")
        self.assertTrue(result["success"])
        # Computed from graph: minimum travel cost is 7 via A -> B -> D -> F
        self.assertEqual(result["path"], ["A", "B", "D", "F"])
        self.assertEqual(result["edges"], 3)
        self.assertEqual(result["travel_cost"], 7)
        self.assertEqual(result["risk"], 23)

    def test_risk_aware_expected_path_and_metrics(self):
        result = risk_aware_search(self.graph, "A", "F", lam=1.0)
        self.assertTrue(result["success"])
        self.assertEqual(result["path"], ["A", "C", "E", "F"])
        self.assertEqual(result["edges"], 3)
        self.assertEqual(result["travel_cost"], 11)
        self.assertEqual(result["risk"], 4)
        self.assertEqual(result["objective"], 15)
        self.assertEqual(result["objective_lambda"], 1.0)

    def test_changing_edge_weights_changes_result(self):
        graph = self.graph.copy()
        # Make the cheapest route A-B-D-F expensive
        graph.set_edge("B", "D", 100, 1)
        graph.set_edge("D", "F", 100, 1)
        result = uniform_cost_search(graph, "A", "F")
        self.assertTrue(result["success"])
        # UCS should now avoid the expensive route; result must change
        self.assertNotEqual(result["travel_cost"], 7)

    def test_unreachable_destination(self):
        graph = CampusGraph()
        graph.add_edge("A", "B", 1, 1)
        result = bfs(graph, "A", "C")
        self.assertFalse(result["success"])
        self.assertEqual(result["path"], [])

    def test_unknown_nodes(self):
        result = bfs(self.graph, "A", "Z")
        self.assertFalse(result["success"])

    def test_cyclic_graph_does_not_loop(self):
        graph = CampusGraph()
        graph.add_edge("A", "B", 1, 1)
        graph.add_edge("B", "C", 1, 1)
        graph.add_edge("C", "A", 1, 1)  # cycle
        graph.add_edge("C", "F", 1, 1)
        result = bfs(graph, "A", "F")
        self.assertTrue(result["success"])
        self.assertEqual(result["path"][0], "A")
        self.assertEqual(result["path"][-1], "F")

    def test_path_metrics_calculated_from_graph(self):
        result = dfs(self.graph, "A", "F")
        self.assertTrue(result["success"])
        # metrics computed by graph.path_metrics
        metrics = self.graph.path_metrics(result["path"])
        self.assertEqual(metrics["travel_cost"], result["travel_cost"])
        self.assertEqual(metrics["risk"], result["risk"])


if __name__ == "__main__":
    unittest.main()
