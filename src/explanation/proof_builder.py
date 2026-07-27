from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json(
    path: str | Path,
) -> dict[str, Any]:
    """Load a JSON file whose root must be an object."""
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


def format_value(
    value: Any,
) -> str:
    """Format an observed value for a human-readable explanation."""
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")

    return str(value)


def build_observation(
    evidence: dict[str, Any],
) -> dict[str, Any]:
    """
    Build one traceable observation from a MITRE evidence item.

    O contains the source feature, observed value and local
    contribution supporting or opposing the prediction.
    """
    signed_contribution = float(
        evidence.get("signed_contribution", 0.0)
    )

    return {
        "feature": evidence.get("source_feature"),
        "value": evidence.get("source_value"),
        "concept": evidence.get("concept"),
        "semantic_action": {
            "verb": evidence.get("verb"),
            "object": evidence.get("object"),
            "state": evidence.get("state"),
        },
        "action_weight": float(
            evidence.get("action_weight", 0.0)
        ),
        "signed_contribution": signed_contribution,
        "direction": evidence.get("direction"),
        "trace": (
            f"{evidence.get('source_feature')}="
            f"{format_value(evidence.get('source_value'))}"
            f" → {evidence.get('concept')}"
            f" → {evidence.get('verb')}"
            f"({evidence.get('object')})"
        ),
    }


def deduplicate_observations(
    observations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Remove duplicate observations while preserving their order."""
    unique_observations: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()

    for observation in observations:
        action = observation.get(
            "semantic_action",
            {},
        )

        key = (
            observation.get("feature"),
            observation.get("value"),
            observation.get("concept"),
            action.get("verb"),
            action.get("object"),
            action.get("state"),
        )

        if key in seen:
            continue

        seen.add(key)
        unique_observations.append(observation)

    return unique_observations


def extract_observations(
    mitre_candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Extract and rank observations from MITRE candidates."""
    observations = [
        build_observation(evidence)
        for candidate in mitre_candidates
        for evidence in candidate.get("evidence", [])
    ]

    unique_observations = deduplicate_observations(
        observations
    )

    return sorted(
        unique_observations,
        key=lambda observation: (
            -abs(
                float(
                    observation.get(
                        "signed_contribution",
                        0.0,
                    )
                )
            ),
            str(observation.get("feature", "")),
        ),
    )


def build_reasoning_rules(
    mitre_candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Build the explicit rule chain R.

    R describes the transformations:
    Feature → Concept → Action → Pattern → MITRE.
    """
    rules: list[dict[str, Any]] = []

    for candidate in mitre_candidates:
        pattern_id = candidate.get("pattern_id")
        pattern_name = candidate.get("pattern_name")
        technique_id = candidate.get("technique_id")
        technique_name = candidate.get("technique_name")

        for evidence in candidate.get("evidence", []):
            rules.append(
                {
                    "rule_type": "semantic_trace",
                    "source_feature": evidence.get(
                        "source_feature"
                    ),
                    "semantic_concept": evidence.get(
                        "concept"
                    ),
                    "semantic_action": {
                        "verb": evidence.get("verb"),
                        "object": evidence.get("object"),
                        "state": evidence.get("state"),
                    },
                    "behavior_pattern": {
                        "pattern_id": pattern_id,
                        "name": pattern_name,
                    },
                    "mitre_candidate": {
                        "technique_id": technique_id,
                        "technique_name": technique_name,
                    },
                }
            )

        rules.append(
            {
                "rule_type": "pattern_to_mitre",
                "pattern_id": pattern_id,
                "pattern_name": pattern_name,
                "technique_id": technique_id,
                "technique_name": technique_name,
                "tactic_id": candidate.get("tactic_id"),
                "tactic_name": candidate.get("tactic_name"),
                "mapping_status": candidate.get(
                    "mapping_status"
                ),
                "base_confidence": float(
                    candidate.get(
                        "base_confidence",
                        0.0,
                    )
                ),
                "justification": candidate.get(
                    "justification"
                ),
            }
        )

    return rules


def build_justification(
    prediction: dict[str, Any],
    mitre_candidates: list[dict[str, Any]],
    observations: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Build J: numerical, semantic and contextual justification.
    """
    principal_observations = observations[:5]

    return {
        "prediction_probability": float(
            prediction.get("probability", 0.0)
        ),
        "principal_observations": [
            {
                "feature": observation.get("feature"),
                "value": observation.get("value"),
                "signed_contribution": observation.get(
                    "signed_contribution"
                ),
                "direction": observation.get("direction"),
            }
            for observation in principal_observations
        ],
        "pattern_diagnostics": [
            {
                "pattern_id": candidate.get("pattern_id"),
                "pattern_name": candidate.get(
                    "pattern_name"
                ),
                "evidence_weight": float(
                    candidate.get(
                        "pattern_evidence_weight",
                        0.0,
                    )
                ),
                "required_coverage": float(
                    candidate.get(
                        "required_coverage",
                        0.0,
                    )
                ),
                "supporting_coverage": float(
                    candidate.get(
                        "supporting_coverage",
                        0.0,
                    )
                ),
                "label_consistent": candidate.get(
                    "label_consistent"
                ),
            }
            for candidate in mitre_candidates
        ],
        "mitre_diagnostics": [
            {
                "technique_id": candidate.get(
                    "technique_id"
                ),
                "technique_name": candidate.get(
                    "technique_name"
                ),
                "base_confidence": float(
                    candidate.get(
                        "base_confidence",
                        0.0,
                    )
                ),
                "mapping_status": candidate.get(
                    "mapping_status"
                ),
                "granularity": candidate.get(
                    "granularity"
                ),
            }
            for candidate in mitre_candidates
        ],
    }


def build_limitations(
    mitre_candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build explicit limitations and missing-context statements."""
    limitations: list[dict[str, Any]] = []

    for candidate in mitre_candidates:
        excluded_subtechniques = candidate.get(
            "excluded_subtechniques",
            [],
        )

        additional_context = candidate.get(
            "required_additional_context",
            [],
        )

        if excluded_subtechniques:
            limitations.append(
                {
                    "type": "subtechnique_uncertainty",
                    "technique_id": candidate.get(
                        "technique_id"
                    ),
                    "message": (
                        "Les preuves disponibles soutiennent la "
                        "technique générale, mais pas une "
                        "sous-technique précise."
                    ),
                    "excluded_subtechniques": (
                        excluded_subtechniques
                    ),
                }
            )

        if additional_context:
            limitations.append(
                {
                    "type": "missing_context",
                    "technique_id": candidate.get(
                        "technique_id"
                    ),
                    "message": (
                        "Un contexte supplémentaire est requis "
                        "pour augmenter la précision du mapping."
                    ),
                    "required_context": additional_context,
                }
            )

    return limitations


def generate_natural_language_explanation(
    prediction: dict[str, Any],
    mitre_candidates: list[dict[str, Any]],
    observations: list[dict[str, Any]],
) -> dict[str, str]:
    """Generate a concise SOC-oriented explanation."""
    prediction_label = prediction.get(
        "label",
        "inconnue",
    )

    probability = float(
        prediction.get("probability", 0.0)
    )

    if mitre_candidates:
        candidate = mitre_candidates[0]

        technique_text = (
            f"{candidate.get('technique_id')} — "
            f"{candidate.get('technique_name')}"
        )

        tactic_text = (
            f"{candidate.get('tactic_id')} — "
            f"{candidate.get('tactic_name')}"
        )

        pattern_text = (
            f"{candidate.get('pattern_id')} — "
            f"{candidate.get('pattern_name')}"
        )
    else:
        technique_text = "aucune technique identifiée"
        tactic_text = "aucune tactique identifiée"
        pattern_text = "aucun motif activé"

    principal = observations[:2]

    evidence_fragments = [
        (
            f"{observation.get('feature')}="
            f"{format_value(observation.get('value'))} "
            f"(SHAP "
            f"{float(observation.get('signed_contribution', 0.0)):+.4f})"
        )
        for observation in principal
    ]

    evidence_text = (
        " et ".join(evidence_fragments)
        if evidence_fragments
        else "aucune observation principale disponible"
    )

    return {
        "summary": (
            f"La connexion est classée {prediction_label} "
            f"avec une probabilité de {probability:.4f}."
        ),
        "behavior": (
            f"Les observations activent le motif "
            f"{pattern_text}."
        ),
        "mitre": (
            f"Ce motif est compatible avec "
            f"{technique_text}, dans la tactique "
            f"{tactic_text}."
        ),
        "evidence": (
            f"Les preuves locales principales sont "
            f"{evidence_text}."
        ),
        "caution": (
            "Le résultat MITRE est un candidat contextuel "
            "issu de règles explicites, et non une certitude."
        ),
    }


def build_explainable_proof(
    prediction: dict[str, Any],
    mitre_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Build the final explainable proof E = <O, R, J> and conclusion C.
    """
    observations = extract_observations(
        mitre_candidates
    )

    reasoning_rules = build_reasoning_rules(
        mitre_candidates
    )

    justification = build_justification(
        prediction=prediction,
        mitre_candidates=mitre_candidates,
        observations=observations,
    )

    limitations = build_limitations(
        mitre_candidates
    )

    candidate_conclusions = [
        {
            "pattern_id": candidate.get("pattern_id"),
            "pattern_name": candidate.get("pattern_name"),
            "technique_id": candidate.get("technique_id"),
            "technique_name": candidate.get(
                "technique_name"
            ),
            "tactic_id": candidate.get("tactic_id"),
            "tactic_name": candidate.get("tactic_name"),
            "mapping_status": candidate.get(
                "mapping_status"
            ),
        }
        for candidate in mitre_candidates
    ]

    conclusion = {
        "prediction": {
            "label": prediction.get("label"),
            "probability": float(
                prediction.get("probability", 0.0)
            ),
        },
        "contextual_candidates": candidate_conclusions,
    }

    return {
        "proof_id": "E-CASE-0",
        "formalism": "E=<O,R,J>",
        "O_observations": observations,
        "R_reasoning_rules": reasoning_rules,
        "J_justification": justification,
        "C_conclusion": conclusion,
        "limitations": limitations,
        "natural_language_explanation": (
            generate_natural_language_explanation(
                prediction=prediction,
                mitre_candidates=mitre_candidates,
                observations=observations,
            )
        ),
        "audit": {
            "observation_count": len(observations),
            "rule_count": len(reasoning_rules),
            "mitre_candidate_count": len(
                mitre_candidates
            ),
            "traceable": bool(
                observations
                and reasoning_rules
                and mitre_candidates
            ),
        },
    }