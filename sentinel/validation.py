"""Input validation and the six robustness cases for SENTINEL.

Design rule: validation reports problems; it never silently repairs an
incident.  Invalid incidents receive ``action = None`` (no approval) and a
clear list of errors.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List

BINARY_FIELDS = (
    "authorized",
    "confirmed_compromise",
    "critical_alert",
    "duplicate_conflict",
)

ML_FEATURES = (
    "failed_logins",
    "requests_per_min",
    "unusual_hour",
    "known_device",
    "duplicate_count",
)

# Text patterns are used only for a warning in the evidence.  The decision
# policy never reads the description, so an injected instruction cannot change
# the selected action.
_INJECTION_PATTERNS = (
    r"ignore\s+(all\s+|the\s+|your\s+)?(previous\s+|prior\s+)?(security\s+)?rules",
    r"ignore\s+(all\s+|the\s+|your\s+)?instructions",
    r"disregard\s+.{0,30}(rules|instructions|policy)",
    r"override\s+.{0,30}(rules|policy|decision)",
    r"allow\s+this\s+(transaction|incident|event)",
    r"you\s+are\s+now",
    r"system\s+prompt",
    r"jailbreak",
    r"bypass\s+.{0,20}(security|policy|rules)",
    r"do\s+not\s+(investigate|reject|escalate)",
)
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_flag(value: object) -> bool:
    """Accept JSON-style 0/1 and also Python booleans (normalised to 0/1)."""
    return isinstance(value, bool) or (_is_int(value) and value in (0, 1))


def _flag_to_int(value: object) -> int:
    return 1 if value is True or value == 1 else 0


def new_validation_result() -> Dict[str, object]:
    return {
        "valid": True,
        "errors": [],
        "warnings": [],
        "contradictions": [],
        "prompt_injection": None,
    }


def _finish(result: Dict[str, object]) -> Dict[str, object]:
    result["valid"] = not bool(result["errors"])
    return result


def validate_agent_event(event: object) -> Dict[str, object]:
    """Validate the fields needed by the six decision rules (I01-I05)."""
    result = new_validation_result()
    errors: List[str] = result["errors"]  # type: ignore[assignment]

    if not isinstance(event, dict):
        errors.append("Incident must be a JSON object / Python dictionary.")
        return _finish(result)

    incident_id = event.get("id")
    if not isinstance(incident_id, str) or not incident_id.strip():
        errors.append("Missing or invalid required field: id (non-empty string).")

    if "risk" not in event:
        errors.append("Missing required field: risk.")
    elif not _is_number(event["risk"]):
        errors.append("Field risk must be a number.")
    elif event["risk"] < 0 or event["risk"] > 10:
        errors.append("Field risk must be between 0 and 10 (inclusive).")

    for field in BINARY_FIELDS:
        if field not in event:
            errors.append(f"Missing required field: {field}.")
        elif not _is_flag(event[field]):
            errors.append(f"Field {field} must be 0 or 1.")

    return _finish(result)


def validate_ml_features(event: object) -> Dict[str, object]:
    """Validate the five input features used by the ML classifier."""
    result = new_validation_result()
    errors: List[str] = result["errors"]  # type: ignore[assignment]

    if not isinstance(event, dict):
        errors.append("Incident must be a JSON object / Python dictionary.")
        return _finish(result)

    if "failed_logins" not in event:
        errors.append("Missing required field: failed_logins.")
    elif not _is_int(event["failed_logins"]) or event["failed_logins"] < 0:
        errors.append("Field failed_logins must be a non-negative integer.")

    if "requests_per_min" not in event:
        errors.append("Missing required field: requests_per_min.")
    elif not _is_number(event["requests_per_min"]) or event["requests_per_min"] < 0:
        errors.append("Field requests_per_min must be a non-negative number.")

    for field in ("unusual_hour", "known_device"):
        if field not in event:
            errors.append(f"Missing required field: {field}.")
        elif not _is_flag(event[field]):
            errors.append(f"Field {field} must be 0 or 1.")

    if "duplicate_count" not in event:
        errors.append("Missing required field: duplicate_count.")
    elif not _is_int(event["duplicate_count"]) or event["duplicate_count"] < 0:
        errors.append("Field duplicate_count must be a non-negative integer.")

    return _finish(result)


def scan_description_for_prompt_injection(description: object) -> Dict[str, object]:
    """Detect suspicious instructions but never execute them.

    The returned dictionary is evidence only.  The agent ignores it when
    selecting an action.
    """
    if description is None:
        return {"detected": False, "matches": [], "note": "No description supplied."}
    if not isinstance(description, str):
        return {
            "detected": True,
            "matches": [],
            "note": "Description was not text; treated as untrusted data and ignored.",
        }

    matches = sorted({match.group(0) for match in _INJECTION_RE.finditer(description)})
    return {
        "detected": bool(matches),
        "matches": matches,
        "note": (
            "Description treated as untrusted data; instructions inside it are not executed."
            if matches
            else "No known injection pattern found; description still not used for policy."
        ),
    }


def validate_route_request(route_request: object) -> Dict[str, object]:
    result = new_validation_result()
    errors: List[str] = result["errors"]  # type: ignore[assignment]

    if not isinstance(route_request, dict):
        errors.append("route_request must be an object with start and goal.")
        return _finish(result)

    for field in ("start", "goal"):
        if field not in route_request:
            errors.append(f"route_request is missing {field}.")
        elif not isinstance(route_request[field], str) or not route_request[field].strip():
            errors.append(f"route_request.{field} must be a non-empty string.")

    if "algorithm" in route_request and not isinstance(route_request["algorithm"], str):
        errors.append("route_request.algorithm must be a string when provided.")
    if "lambda" in route_request and (
        not _is_number(route_request["lambda"]) or route_request["lambda"] < 0
    ):
        errors.append("route_request.lambda must be a non-negative number when provided.")

    return _finish(result)


def check_contradictions(event: Dict[str, object]) -> List[str]:
    """Return structural contradictions that no policy rule can sensibly fix.

    Deliberately narrow:

    * ``duplicate_conflict = 1`` with an explicitly supplied
      ``duplicate_count = 0`` is impossible: a conflict needs at least one
      repeated record.
    * ``duplicate_conflict = 1`` with a negative count is already a type error.

    Apparent conflicts such as an authorized session with a confirmed
    compromise (incident I03) are *not* invalid.  They are exactly what the
    priority policy exists to resolve, so they become warnings instead.
    """
    contradictions: List[str] = []
    if event.get("duplicate_conflict") == 1 and "duplicate_count" in event:
        count = event.get("duplicate_count")
        if _is_int(count) and count == 0:
            contradictions.append(
                "duplicate_conflict=1 requires duplicate_count >= 1; "
                "a conflict cannot exist with zero duplicate records."
            )
    return contradictions


def check_evidence_warnings(event: Dict[str, object]) -> List[str]:
    """Flag conflicting evidence that the priority rules must resolve."""
    warnings: List[str] = []
    if event.get("authorized") == 1 and event.get("confirmed_compromise") == 1:
        warnings.append(
            "Authorized session conflicts with confirmed compromise; "
            "RULE_1 intentionally takes priority."
        )
    if event.get("critical_alert") == 1:
        risk = event.get("risk")
        if _is_number(risk) and risk <= 2:
            warnings.append(
                "Critical alert conflicts with a low risk score; "
                "RULE_2 intentionally takes priority."
            )
    return warnings


def validate_event(event: object) -> Dict[str, object]:
    """Comprehensive validation used by ``process_incident``."""
    result = new_validation_result()
    errors: List[str] = result["errors"]  # type: ignore[assignment]

    if not isinstance(event, dict):
        errors.append("Incident must be a JSON object / Python dictionary.")
        return _finish(result)

    # Reuse the two focused validators and merge their messages.
    for validator in (validate_agent_event, validate_ml_features):
        part = validator(event)
        errors.extend(message for message in part["errors"] if message not in errors)  # type: ignore[arg-type]

    # Optional route request.
    if "route_request" in event and event["route_request"] is not None:
        route_part = validate_route_request(event["route_request"])
        errors.extend(message for message in route_part["errors"] if message not in errors)  # type: ignore[arg-type]

    # Optional untrusted description.
    injection = None
    if "description" in event and event["description"] is not None:
        if not isinstance(event["description"], str):
            errors.append("Field description must be a string when provided.")
        injection = scan_description_for_prompt_injection(event["description"])

    contradictions = check_contradictions(event)
    if contradictions:
        result["contradictions"] = contradictions  # type: ignore[assignment]
        errors.extend(contradictions)

    warnings = check_evidence_warnings(event)
    result["warnings"] = warnings  # type: ignore[assignment]
    result["prompt_injection"] = injection

    return _finish(result)


def normalise_flags(event: Dict[str, object]) -> Dict[str, int]:
    """Expose boolean flags as 0/1 for evidence without mutating the input."""
    return {
        field: _flag_to_int(event.get(field))
        for field in BINARY_FIELDS
        if field in event
    }


class IncidentRegistry:
    """Remember incident IDs so duplicate records are never silently merged."""

    def __init__(self) -> None:
        self._ids: set[str] = set()

    def is_duplicate(self, incident_id: str) -> bool:
        return incident_id in self._ids

    def register(self, incident_id: str) -> bool:
        """Register an ID. Return False when it was already present."""
        if incident_id in self._ids:
            return False
        self._ids.add(incident_id)
        return True

    def clear(self) -> None:
        self._ids.clear()

    def __len__(self) -> int:
        return len(self._ids)


def find_duplicate_ids(records: Iterable[object]) -> List[str]:
    """Return every incident ID that occurs more than once, sorted."""
    seen: set[str] = set()
    duplicates: set[str] = set()
    for record in records:
        if isinstance(record, dict):
            incident_id = record.get("id")
            if isinstance(incident_id, str):
                if incident_id in seen:
                    duplicates.add(incident_id)
                seen.add(incident_id)
    return sorted(duplicates)
