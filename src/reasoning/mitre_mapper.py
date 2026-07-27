from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json(
    path: str | Path,
) -> dict[str, Any]:
    """Load and validate a JSON object."""
    input_path = Path(path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {input_path.resolve()}"
        )

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError(
            f"Le fichier {input_path} doit contenir un objet JSON."
        )

    return payload


def find_pattern_mappings(
    pattern_id: str,
    mitre_config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return all MITRE mappings associated with a pattern."""
    return [
        mapping
        for mapping in mitre_config.get("mappings", [])
        if mapping.get("pattern_id") == pattern_id
    ]


def extract_pattern_evidence(
    pattern: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Extract traceable evidence from an activated behavioral pattern.

    The trace is preserved from:
    MITRE candidate
        → behavioral pattern
        → semantic action
        → semantic concept
        → source feature
        → SHAP contribution.
    """
    evidence_items: list[dict[str, Any]] = []

    evidence_groups = (
        (
            "required",
            pattern.get("required_evidence", []),
        ),
        (
            "supporting",
            pattern.get("supporting_evidence", []),
        ),
    )

    for evidence_type, evidence_group in evidence_groups:
        for evidence in evidence_group:
            action = evidence.get(
                "matched_action",
                {},
            )

            evidence_items.append(
                {
                    "evidence_type": evidence_type,
                    "verb": action.get("verb"),
                    "object": action.get("object"),
                    "state": action.get("state"),
                    "action_weight": float(
                        action.get("weight", 0.0)
                    ),
                    "concept": action.get("concept"),
                    "source_feature": action.get(
                        "source_feature"
                    ),
                    "source_value": action.get(
                        "source_value"
                    ),
                    "signed_contribution": float(
                        action.get(
                            "signed_contribution",
                            0.0,
                        )
                    ),
                    "direction": action.get("direction"),
                }
            )

    return evidence_items


def build_mitre_candidate(
    pattern: dict[str, Any],
    mapping: dict[str, Any],
) -> dict[str, Any]:
    """
    Build one MITRE candidate while preserving its justification.

    The expert-defined base confidence is not modified by the
    prediction label. Label consistency remains diagnostic only.
    """
    return {
        "mapping_id": mapping.get("mapping_id"),
        "pattern_id": pattern.get("pattern_id"),
        "pattern_name": pattern.get("name"),
        "pattern_category": pattern.get("category"),
        "pattern_severity": pattern.get("severity"),
        "technique_id": mapping.get("technique_id"),
        "technique_name": mapping.get("technique_name"),
        "tactic_id": mapping.get("tactic_id"),
        "tactic_name": mapping.get("tactic_name"),
        "granularity": mapping.get(
            "granularity",
            "technique",
        ),
        "base_confidence": float(
            mapping.get("base_confidence", 0.0)
        ),
        "mapping_status": mapping.get(
            "mapping_status",
            "candidate",
        ),
        "justification": mapping.get("justification"),
        "supporting_behavior": mapping.get(
            "supporting_behavior",
            [],
        ),
        "excluded_subtechniques": mapping.get(
            "excluded_subtechniques",
            [],
        ),
        "required_additional_context": mapping.get(
            "required_additional_context",
            [],
        ),
        "pattern_evidence_weight": float(
            pattern.get(
                "total_evidence_weight",
                0.0,
            )
        ),
        "required_coverage": float(
            pattern.get("required_coverage", 0.0)
        ),
        "supporting_coverage": float(
            pattern.get("supporting_coverage", 0.0)
        ),
        "prediction_label": pattern.get(
            "prediction_label"
        ),
        "label_consistent": pattern.get(
            "label_consistent"
        ),
        "evidence": extract_pattern_evidence(pattern),
    }


def map_patterns_to_mitre(
    activated_patterns: list[dict[str, Any]],
    mitre_config: dict[str, Any],
) -> dict[str, Any]:
    """
    Map activated behavioral patterns to MITRE candidates.

    Only activated patterns are considered. A prediction label never
    activates a mapping by itself.
    """
    candidates: list[dict[str, Any]] = []
    unmapped_patterns: list[dict[str, Any]] = []

    for pattern in activated_patterns:
        if not pattern.get("activated", False):
            continue

        pattern_id = str(
            pattern.get("pattern_id", "")
        )

        mappings = find_pattern_mappings(
            pattern_id=pattern_id,
            mitre_config=mitre_config,
        )

        if not mappings:
            unmapped_patterns.append(
                {
                    "pattern_id": pattern_id,
                    "pattern_name": pattern.get("name"),
                    "reason": (
                        "No explicit MITRE mapping is configured "
                        "for this activated pattern."
                    ),
                }
            )
            continue

        for mapping in mappings:
            candidates.append(
                build_mitre_candidate(
                    pattern=pattern,
                    mapping=mapping,
                )
            )

    ranked_candidates = sorted(
        candidates,
        key=lambda candidate: (
            -candidate["base_confidence"],
            -candidate["pattern_evidence_weight"],
            candidate["technique_id"] or "",
        ),
    )

    return {
        "mitre_candidates": ranked_candidates,
        "unmapped_patterns": unmapped_patterns,
        "summary": {
            "activated_pattern_count": len(
                activated_patterns
            ),
            "mapped_pattern_count": len(
                {
                    candidate["pattern_id"]
                    for candidate in ranked_candidates
                }
            ),
            "candidate_count": len(
                ranked_candidates
            ),
            "unmapped_pattern_count": len(
                unmapped_patterns
            ),
        },
    }