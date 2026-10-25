"""Rule-based incident decision agent.

The rules are applied in exactly the order given in the assignment.  Later
rules are never allowed to override an earlier one, and the ML classifier is
never consulted here.  The policy remains authoritative for the final action.
"""

from __future__ import annotations

from typing import Dict

from .validation import (
    normalise_flags,
    scan_description_for_prompt_injection,
    validate_agent_event,
    check_evidence_warnings,
)

ACTIONS = ("ALLOW", "INVESTIGATE", "REJECT", "ESCALATE")

RULE_1 = "RULE_1"
RULE_2 = "RULE_2"
RULE_3 = "RULE_3"
RULE_4 = "RULE_4"
RULE_5 = "RULE_5"
RULE_6 = "RULE_6"

# PEAS description, kept next to the agent so it is easy to point to in a viva.
PEAS = {
    "performance_measure": [
        "Decision accuracy against the policy",
        "Response time",
        "Unnecessary rejections",
    ],
    "environment": [
        "Campus graph",
        "Users and transactions",
        "Incident records and security alerts",
    ],
    "actuators": ["ALLOW", "REJECT", "INVESTIGATE", "ESCALATE"],
    "sensors": [
        "Event logs",
        "Authorization status",
        "Risk score",
        "Compromise and critical alerts",
        "Duplicate-conflict evidence",
    ],
}


def _invalid_result(event: object, validation: Dict[str, object]) -> Dict[str, object]:
    incident_id = event.get("id") if isinstance(event, dict) else None
    risk = event.get("risk") if isinstance(event, dict) else None
    return {
        "incident_id": incident_id,
        "action": None,
        "triggered_rule": None,
        "explanation": "Incident rejected before decision: " + "; ".join(validation["errors"]),  # type: ignore[arg-type]
        "evidence": {
            "risk_score": risk,
            "flags": normalise_flags(event) if isinstance(event, dict) else {},
        },
        "risk_score": risk,
        "validation_status": "INVALID",
        "errors": list(validation["errors"]),  # type: ignore[arg-type]
    }


def decide_action(event: object) -> Dict[str, object]:
    """Select exactly one action for a validated incident.

    Priority order (mandatory):

    1. confirmed_compromise == 1 -> ESCALATE
    2. critical_alert == 1       -> ESCALATE
    3. authorized == 1 and risk <= 2 -> ALLOW
    4. duplicate_conflict == 1   -> INVESTIGATE
    5. risk >= 7                 -> REJECT
    6. otherwise                 -> INVESTIGATE
    """
    validation = validate_agent_event(event)
    if not validation["valid"]:
        return _invalid_result(event, validation)

    assert isinstance(event, dict)  # guaranteed by validate_agent_event
    incident_id = event["id"]
    risk = event["risk"]
    authorized = int(bool(event["authorized"]))
    confirmed_compromise = int(bool(event["confirmed_compromise"]))
    critical_alert = int(bool(event["critical_alert"]))
    duplicate_conflict = int(bool(event["duplicate_conflict"]))
    flags = normalise_flags(event)

    if confirmed_compromise == 1:
        action, rule = "ESCALATE", RULE_1
        explanation = (
            "Confirmed compromise is present. RULE_1 takes priority over "
            "authorization and the low risk score; the incident is escalated."
        )
    elif critical_alert == 1:
        action, rule = "ESCALATE", RULE_2
        explanation = (
            "A critical alert is present. RULE_2 escalates the incident even "
            "when other fields look permissive."
        )
    elif authorized == 1 and risk <= 2:
        action, rule = "ALLOW", RULE_3
        explanation = (
            f"The event is authorized and risk {risk} is at or below the "
            "low-risk threshold of 2. RULE_3 allows it."
        )
    elif duplicate_conflict == 1:
        action, rule = "INVESTIGATE", RULE_4
        explanation = (
            "The event is a duplicate transaction with conflicting evidence. "
            "RULE_4 sends it for investigation."
        )
    elif risk >= 7:
        action, rule = "REJECT", RULE_5
        explanation = (
            f"Risk {risk} is at least 7 without a higher-priority escalation "
            "or authorization condition. RULE_5 rejects it."
        )
    else:
        action, rule = "INVESTIGATE", RULE_6
        explanation = (
            "No higher-priority rule matched. RULE_6 is the safe default: "
            "investigate rather than allow or reject."
        )

    injection = scan_description_for_prompt_injection(event.get("description"))
    warnings = check_evidence_warnings(event)

    return {
        "incident_id": incident_id,
        "action": action,
        "triggered_rule": rule,
        "explanation": explanation,
        "evidence": {
            "risk_score": risk,
            "flags": flags,
            "low_risk_threshold": 2,
            "high_risk_threshold": 7,
            "prompt_injection_check": injection,
            "evidence_warnings": warnings,
        },
        "risk_score": risk,
        "validation_status": "VALID",
        "errors": [],
    }


def apply_agent(event: object) -> Dict[str, object]:
    """Alias kept for readability in tests and the integration layer."""
    return decide_action(event)


# A short validation helper for the CLI's PEAS section.
def describe_peas() -> Dict[str, list[str]]:
    return PEAS
