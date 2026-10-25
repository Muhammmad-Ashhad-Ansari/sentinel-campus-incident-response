from __future__ import annotations

import unittest

from sentinel.agent import decide_action
from sentinel.validation import IncidentRegistry


class TestAgent(unittest.TestCase):
    def test_i01_allow(self):
        event = {"id": "I01", "risk": 1, "authorized": 1, "confirmed_compromise": 0, "critical_alert": 0, "duplicate_conflict": 0}
        result = decide_action(event)
        self.assertEqual(result["action"], "ALLOW")
        self.assertEqual(result["triggered_rule"], "RULE_3")

    def test_i02_reject(self):
        event = {"id": "I02", "risk": 8, "authorized": 0, "confirmed_compromise": 0, "critical_alert": 0, "duplicate_conflict": 0}
        result = decide_action(event)
        self.assertEqual(result["action"], "REJECT")
        self.assertEqual(result["triggered_rule"], "RULE_5")

    def test_i03_escalate_compromise(self):
        event = {"id": "I03", "risk": 2, "authorized": 1, "confirmed_compromise": 1, "critical_alert": 0, "duplicate_conflict": 0}
        result = decide_action(event)
        self.assertEqual(result["action"], "ESCALATE")
        self.assertEqual(result["triggered_rule"], "RULE_1")

    def test_i04_investigate_duplicate_conflict(self):
        event = {"id": "I04", "risk": 4, "authorized": 1, "confirmed_compromise": 0, "critical_alert": 0, "duplicate_conflict": 1}
        result = decide_action(event)
        self.assertEqual(result["action"], "INVESTIGATE")
        self.assertEqual(result["triggered_rule"], "RULE_4")

    def test_confirmed_compromise_overrides_authorized_low_risk(self):
        event = {"id": "X1", "risk": 0, "authorized": 1, "confirmed_compromise": 1, "critical_alert": 0, "duplicate_conflict": 0}
        result = decide_action(event)
        self.assertEqual(result["action"], "ESCALATE")
        self.assertEqual(result["triggered_rule"], "RULE_1")

    def test_critical_alert_overrides_authorized_low_risk(self):
        event = {"id": "X2", "risk": 1, "authorized": 1, "confirmed_compromise": 0, "critical_alert": 1, "duplicate_conflict": 0}
        result = decide_action(event)
        self.assertEqual(result["action"], "ESCALATE")
        self.assertEqual(result["triggered_rule"], "RULE_2")

    def test_low_risk_authorized_allow(self):
        event = {"id": "X3", "risk": 2, "authorized": 1, "confirmed_compromise": 0, "critical_alert": 0, "duplicate_conflict": 0}
        result = decide_action(event)
        self.assertEqual(result["action"], "ALLOW")
        self.assertEqual(result["triggered_rule"], "RULE_3")

    def test_duplicate_conflict_investigate(self):
        event = {"id": "X4", "risk": 5, "authorized": 1, "confirmed_compromise": 0, "critical_alert": 0, "duplicate_conflict": 1}
        result = decide_action(event)
        self.assertEqual(result["action"], "INVESTIGATE")
        self.assertEqual(result["triggered_rule"], "RULE_4")

    def test_high_risk_reject(self):
        event = {"id": "X5", "risk": 7, "authorized": 0, "confirmed_compromise": 0, "critical_alert": 0, "duplicate_conflict": 0}
        result = decide_action(event)
        self.assertEqual(result["action"], "REJECT")
        self.assertEqual(result["triggered_rule"], "RULE_5")

    def test_default_investigate(self):
        event = {"id": "X6", "risk": 5, "authorized": 0, "confirmed_compromise": 0, "critical_alert": 0, "duplicate_conflict": 0}
        result = decide_action(event)
        self.assertEqual(result["action"], "INVESTIGATE")
        self.assertEqual(result["triggered_rule"], "RULE_6")

    def test_missing_fields_invalid(self):
        result = decide_action({"id": "X7", "risk": 1})
        self.assertIsNone(result["action"])
        self.assertEqual(result["validation_status"], "INVALID")

    def test_invalid_flag(self):
        event = {"id": "X8", "risk": 1, "authorized": 2, "confirmed_compromise": 0, "critical_alert": 0, "duplicate_conflict": 0}
        result = decide_action(event)
        self.assertIsNone(result["action"])
        self.assertEqual(result["validation_status"], "INVALID")


if __name__ == "__main__":
    unittest.main()
