from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json(
    path: str | Path,
) -> dict[str, Any]:
    """Load a JSON file and validate its root structure."""
    input_path = Path(path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {input_path.resolve()}"
        )

    with input_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError(
            f"Le fichier {input_path} doit contenir un objet JSON."
        )

    return payload


def action_matches_condition(
    action: dict[str, Any],
    condition: dict[str, Any],
) -> bool:
    """
    Check whether a semantic action satisfies a behavioral condition.

    A condition can specify:
    - verb;
    - object;
    - state.

    Missing condition fields are ignored.
    """
    for field in ("verb", "object", "state"):
        expected_value = condition.get(field)

        if expected_value is None:
            continue

        if action.get(field) != expected_value:
            return False

    return True


def find_matching_action(
    semantic_actions: list[dict[str, Any]],
    condition: dict[str, Any],
) -> dict[str, Any] | None:
    """
    Return the strongest semantic action matching a condition.
    """
    matching_actions = [
        action
        for action in semantic_actions
        if action_matches_condition(action, condition)
    ]

    if not matching_actions:
        return None

    return max(
        matching_actions,
        key=lambda action: float(action.get("weight", 0.0)),
    )


def build_evidence(
    condition: dict[str, Any],
    action: dict[str, Any],
    evidence_type: str,
) -> dict[str, Any]:
    """Build a traceable evidence item for an activated pattern."""
    return {
        "evidence_type": evidence_type,
        "condition": {
            "verb": condition.get("verb"),
            "object": condition.get("object"),
            "state": condition.get("state"),
        },
        "matched_action": {
            "verb": action.get("verb"),
            "object": action.get("object"),
            "state": action.get("state"),
            "weight": float(action.get("weight", 0.0)),
            "concept": action.get("concept"),
            "source_feature": action.get("source_feature"),
            "source_value": action.get("source_value"),
            "signed_contribution": float(
                action.get("signed_contribution", 0.0)
            ),
            "direction": action.get("direction"),
        },
    }


def evaluate_behavior_pattern(
    pattern: dict[str, Any],
    semantic_actions: list[dict[str, Any]],
    prediction_label: str | None = None,
) -> dict[str, Any]:
    """
    Evaluate one behavioral pattern against semantic actions.

    Activation requires:
    1. all required conditions;
    2. the configured minimum number of supporting conditions.
    """
    required_conditions = pattern.get(
        "required_conditions",
        [],
    )

    supporting_conditions = pattern.get(
        "supporting_conditions",
        [],
    )

    required_evidence: list[dict[str, Any]] = []
    supporting_evidence: list[dict[str, Any]] = []
    missing_required: list[dict[str, Any]] = []

    for condition in required_conditions:
        action = find_matching_action(
            semantic_actions,
            condition,
        )

        if action is None:
            missing_required.append(condition)
            continue

        required_evidence.append(
            build_evidence(
                condition=condition,
                action=action,
                evidence_type="required",
            )
        )

    for condition in supporting_conditions:
        action = find_matching_action(
            semantic_actions,
            condition,
        )

        if action is None:
            continue

        supporting_evidence.append(
            build_evidence(
                condition=condition,
                action=action,
                evidence_type="supporting",
            )
        )

    minimum_supporting = int(
        pattern.get(
            "minimum_supporting_conditions",
            0,
        )
    )

    all_required_matched = (
        len(required_evidence)
        == len(required_conditions)
    )

    enough_supporting = (
        len(supporting_evidence)
        >= minimum_supporting
    )

    activated = (
        all_required_matched
        and enough_supporting
    )

    required_weight = sum(
        item["matched_action"]["weight"]
        for item in required_evidence
    )

    supporting_weight = sum(
        item["matched_action"]["weight"]
        for item in supporting_evidence
    )

    total_evidence_weight = round(
        required_weight + supporting_weight,
        4,
    )

    required_coverage = (
        len(required_evidence)
        / len(required_conditions)
        if required_conditions
        else 1.0
    )

    supporting_coverage = (
        len(supporting_evidence)
        / len(supporting_conditions)
        if supporting_conditions
        else 1.0
    )

    associated_labels = pattern.get(
        "associated_labels",
        [],
    )

    label_consistent = (
        prediction_label in associated_labels
        if prediction_label is not None
        else None
    )

    return {
        "pattern_id": pattern.get("pattern_id"),
        "name": pattern.get("name"),
        "description": pattern.get("description"),
        "category": pattern.get("category"),
        "severity": pattern.get("severity"),
        "activated": activated,
        "associated_labels": associated_labels,
        "prediction_label": prediction_label,
        "label_consistent": label_consistent,
        "required_condition_count": len(
            required_conditions
        ),
        "matched_required_count": len(
            required_evidence
        ),
        "supporting_condition_count": len(
            supporting_conditions
        ),
        "matched_supporting_count": len(
            supporting_evidence
        ),
        "minimum_supporting_conditions": minimum_supporting,
        "required_coverage": round(
            required_coverage,
            4,
        ),
        "supporting_coverage": round(
            supporting_coverage,
            4,
        ),
        "required_weight": round(
            required_weight,
            4,
        ),
        "supporting_weight": round(
            supporting_weight,
            4,
        ),
        "total_evidence_weight": total_evidence_weight,
        "required_evidence": required_evidence,
        "supporting_evidence": supporting_evidence,
        "missing_required_conditions": missing_required,
    }


def infer_behavior_patterns(
    semantic_actions: list[dict[str, Any]],
    patterns_config: dict[str, Any],
    prediction_label: str | None = None,
) -> list[dict[str, Any]]:
    """
    Evaluate all configured patterns and return their diagnostics.

    Activated patterns are ranked before inactive patterns.
    """
    results = [
        evaluate_behavior_pattern(
            pattern=pattern,
            semantic_actions=semantic_actions,
            prediction_label=prediction_label,
        )
        for pattern in patterns_config.get("patterns", [])
    ]

    return sorted(
        results,
        key=lambda result: (
            not result["activated"],
            -result["total_evidence_weight"],
            result["pattern_id"] or "",
        ),
    )


def get_activated_patterns(
    pattern_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return only activated behavioral patterns."""
    return [
        result
        for result in pattern_results
        if result.get("activated")
    ]