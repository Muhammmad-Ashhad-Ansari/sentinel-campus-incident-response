from __future__ import annotations

import unittest

from sentinel.classifier import IncidentClassifier
from sentinel.dataset import generate_records
from sentinel.graph import build_campus_graph
from sentinel.integration import process_incident
from sentinel.validation import IncidentRegistry


class TestIntegration(unittest.TestCase):
    def setUp(self):
        self.graph = build_campus_graph()
        self.rows = generate_records(400, seed=42)
        self.model = IncidentClassifier(random_state=42)
        self.model.train(self.rows)

    def test_i05_integration(self):
        event = {
            "id": "I05",
            "failed_logins": 12,
            "requests_per_min": 120,
            "unusual_hour": 1,
            "known_device": 0,
            "duplicate_count": 4,
            "risk": 9,
            "authorized": 0,
            "confirmed_compromise": 0,
            "critical_alert": 1,
            "duplicate_conflict": 0,
            "route_request": {"start": "A", "goal": "F"},
        }
        result = process_incident(event, self.graph, self.model, registry=IncidentRegistry())
        self.assertIsNotNone(result["predicted_class"])
        self.assertEqual(result["action"], "ESCALATE")
        self.assertIn("RULE_2", result["triggered_rule"])
        self.assertIsNotNone(result["route"])
        self.assertTrue(result["route"]["success"])
        self.assertIn("path", result["route"])
        self.assertIn("travel_cost", result["route"])
        self.assertIn("risk", result["route"])

    def test_route_less_incident_works(self):
        event = {
            "id": "I01",
            "failed_logins": 0,
            "requests_per_min": 5,
            "unusual_hour": 0,
            "known_device": 1,
            "duplicate_count": 0,
            "risk": 1,
            "authorized": 1,
            "confirmed_compromise": 0,
            "critical_alert": 0,
            "duplicate_conflict": 0,
        }
        result = process_incident(event, self.graph, self.model, registry=IncidentRegistry())
        self.assertEqual(result["action"], "ALLOW")
        self.assertIsNone(result["route"])

    def test_invalid_event_safe_error(self):
        result = process_incident({"id": "BAD"}, self.graph, self.model, registry=IncidentRegistry())
        self.assertIsNone(result["action"])
        self.assertIn("validation_status", result)

    def test_output_json_serializable(self):
        import json

        event = {
            "id": "I02",
            "failed_logins": 3,
            "requests_per_min": 40,
            "unusual_hour": 1,
            "known_device": 0,
            "duplicate_count": 1,
            "risk": 8,
            "authorized": 0,
            "confirmed_compromise": 0,
            "critical_alert": 0,
            "duplicate_conflict": 0,
        }
        result = process_incident(event, self.graph, self.model, registry=IncidentRegistry())
        json.dumps(result)  # should not raise


if __name__ == "__main__":
    unittest.main()
