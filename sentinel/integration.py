"""Integration of search, agent and ML classifier for SENTINEL.

The main entry point is ``process_incident(event, graph, model)``.  It
validates, predicts, applies policy (policy is authoritative), computes any
requested route, and returns a JSON-compatible dictionary.
"""

from __future__ import annotations

import copy
from typing import Dict

from .agent import decide_action
from .graph import CampusGraph, GraphError, UnknownNodeError
from .search import bfs, risk_aware_search, route_result_to_dict, uniform_cost_search
from .validation import IncidentRegistry, validate_event, normalise_flags


def _error_result(
    incident_id: object,
    message: str,
    validation_status: str = "ERROR",
    errors: list[str] | None = None,
) -> Dict[str, object]:
    return {
        "incident_id": incident_id,
        "predicted_class": None,
        "action": None,
        "triggered_rule": None,
        "explanation": message,
        "evidence": {},
        "risk_score": None,
        "route": None,
        "validation_status": validation_status,
        "errors": errors or [message],
    }


def _get_ml_prediction(event: Dict[str, object], model) -> tuple[str | None, str | None]:
    """Generate an ML prediction if possible.  Failure never blocks the agent."""
    if model is None:
        return None, "No model supplied; ML prediction skipped."
    try:
        # Some wrappers expose predict_event; fall back to predict_features.
        if hasattr(model, "predict_event"):
            return str(model.predict_event(event)), None
        if hasattr(model, "predict_features"):
            features = [
                event["failed_logins"],
                event["requests_per_min"],
                event["unusual_hour"],
                event["known_device"],
                event["duplicate_count"],
            ]
            return str(model.predict_features(features)), None
        if hasattr(model, "predict"):
            features = [
                [event["failed_logins"], event["requests_per_min"], event["unusual_hour"], event["known_device"], event["duplicate_count"]]
            ]
            prediction = model.predict(features)[0]
            return str(prediction), None
    except Exception as exc:
        return None, f"ML prediction failed safely: {exc}"
    return None, "Model could not produce a prediction."


def _compute_route(
    route_request: Dict[str, object],
    graph: CampusGraph,
    default_lam: float = 1.0,
) -> Dict[str, object]:
    if not isinstance(route_request, dict):
        return {"success": False, "error": "route_request must be a dictionary."}

    start = route_request.get("start")
    goal = route_request.get("goal")
    if not isinstance(start, str) or not isinstance(goal, str):
        return {"success": False, "error": "route_request.start and route_request.goal must be strings."}

    algorithm = route_request.get("algorithm") or "risk-aware"
    lam = route_request.get("lambda", default_lam)
    try:
        if lam is not None and (not isinstance(lam, (int, float)) or isinstance(lam, bool) or lam < 0):
            return {"success": False, "error": "route_request.lambda must be a non-negative number."}
        if algorithm == "bfs":
            result = bfs(graph, start, goal)
        elif algorithm == "ucs":
            result = uniform_cost_search(graph, start, goal)
        elif algorithm == "risk-aware":
            result = risk_aware_search(graph, start, goal, lam=float(lam) if lam is not None else default_lam)
        else:
            # default to risk-aware to match the assignment's "risk-aware" framing
            result = risk_aware_search(graph, start, goal, lam=float(lam) if lam is not None else default_lam)
    except (UnknownNodeError, GraphError) as exc:
        return {"success": False, "error": str(exc)}
    except Exception as exc:
        return {"success": False, "error": f"Route computation failed safely: {exc}"}

    route = route_result_to_dict(result)
    if not route["success"]:
        return {"success": False, "error": route.get("error") or "Route computation failed."}
    return {"success": True, "route": route}


# A module-level registry is useful for CLI tests; callers can use their own.
DEFAULT_REGISTRY = IncidentRegistry()


def process_incident(
    event: object,
    graph: CampusGraph,
    model=None,
    registry: IncidentRegistry | None = None,
) -> Dict[str, object]:
    """Process a single incident end to end.

    The ML prediction does not override the rule-based agent.  If the event is
    invalid or requests a route to an unknown node, the function returns safe
    errors instead of crashing.
    """
    incident_id = event.get("id") if isinstance(event, dict) else None
    if registry is None:
        registry = DEFAULT_REGISTRY

    # Duplicate ID check (robustness case 5)
    if isinstance(event, dict) and isinstance(event.get("id"), str):
        if not registry.register(event["id"]):
            return _error_result(
                event["id"],
                f"Duplicate incident ID detected: {event['id']!r}. "
                "Each incident ID may be processed only once.",
                validation_status="DUPLICATE",
                errors=[f"Duplicate incident ID: {event['id']!r}"],
            )

    validation = validate_event(event)
    if not validation["valid"]:
        return _error_result(
            incident_id,
            "Incident failed validation: " + "; ".join(validation["errors"]),  # type: ignore[arg-type]
            validation_status="INVALID",
            errors=list(validation["errors"]),  # type: ignore[arg-type]
        )

    assert isinstance(event, dict)
    agent_decision = decide_action(event)
    predicted_class, ml_error = _get_ml_prediction(event, model)

    evidence = dict(agent_decision.get("evidence") or {})
    evidence["flags"] = normalise_flags(event)
    if ml_error:
        evidence["ml_warning"] = ml_error
    if validation.get("warnings"):
        evidence["validation_warnings"] = validation["warnings"]
    if validation.get("prompt_injection"):
        evidence["prompt_injection"] = validation["prompt_injection"]

    route_payload = None
    if "route_request" in event and event["route_request"] is not None:
        route_request = copy.deepcopy(event["route_request"])
        computed = _compute_route(route_request, graph)
        if not computed.get("success"):
            return _error_result(
                incident_id,
                f"Route request failed: {computed.get('error')}",
                validation_status="INVALID_ROUTE",
                errors=[computed.get("error") or "Route request failed"],
            )
        route_payload = computed["route"]

    explanation = agent_decision["explanation"]
    if predicted_class:
        explanation = (
            f"{explanation} ML classifier predicted {predicted_class} from "
            "the five incident features; the rule-based agent remains "
            "authoritative for the action."
        )

    return {
        "incident_id": incident_id,
        "predicted_class": predicted_class,
        "action": agent_decision["action"],
        "triggered_rule": agent_decision["triggered_rule"],
        "explanation": explanation,
        "evidence": evidence,
        "risk_score": agent_decision.get("risk_score"),
        "route": route_payload,
        "validation_status": "VALID",
        "errors": [],
    }
