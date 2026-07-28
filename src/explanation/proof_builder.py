from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json(
    path: str | Path,
) -> dict[str, Any]:
    """Charge un fichier JSON dont la racine doit être un objet."""
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


def format_value(value: Any) -> str:
    """Formate une valeur observée pour une explication lisible."""
    if value is None:
        return "inconnue"

    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")

    return str(value)


def safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    """Convertit une valeur en nombre flottant sans interrompre le pipeline."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_observation(
    evidence: dict[str, Any],
) -> dict[str, Any]:
    """
    Construit une observation traçable à partir d'un élément de preuve.

    Une observation relie :
    Feature → Concept → Action sémantique.
    """
    signed_contribution = safe_float(
        evidence.get("signed_contribution")
    )

    semantic_action = {
        "verb": evidence.get("verb"),
        "object": evidence.get("object"),
        "state": evidence.get("state"),
    }

    return {
        "observation_type": "xai_semantic_evidence",
        "feature": evidence.get("source_feature"),
        "value": evidence.get("source_value"),
        "concept": evidence.get("concept"),
        "semantic_action": semantic_action,
        "action_weight": safe_float(
            evidence.get("action_weight")
        ),
        "signed_contribution": signed_contribution,
        "direction": evidence.get("direction"),
        "supports_prediction": evidence.get(
            "supports_prediction"
        ),
        "trace": (
            f"{evidence.get('source_feature')}="
            f"{format_value(evidence.get('source_value'))}"
            f" → {evidence.get('concept')}"
            f" → {evidence.get('verb')}"
            f"({evidence.get('object')}, "
            f"{evidence.get('state')})"
        ),
    }


def build_semantic_action_observation(
    action: dict[str, Any],
) -> dict[str, Any]:
    """Transforme une action sémantique en observation explicable."""
    return {
        "observation_type": "semantic_action",
        "verb": action.get("verb"),
        "object": action.get("object"),
        "state": action.get("state"),
        "concept": action.get("concept"),
        "source_feature": action.get("source_feature"),
        "source_value": action.get("source_value"),
        "signed_contribution": safe_float(
            action.get("signed_contribution")
        ),
        "direction": action.get("direction"),
        "supports_prediction": action.get(
            "supports_prediction"
        ),
        "trace": (
            f"{action.get('source_feature')}="
            f"{format_value(action.get('source_value'))}"
            f" → {action.get('concept')}"
            f" → {action.get('verb')}"
            f"({action.get('object')}, "
            f"{action.get('state')})"
        ),
    }


def deduplicate_observations(
    observations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Supprime les observations en double en conservant leur ordre."""
    unique_observations: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()

    for observation in observations:
        action = observation.get(
            "semantic_action",
            {},
        )

        key = (
            observation.get("observation_type"),
            observation.get("feature")
            or observation.get("source_feature"),
            str(
                observation.get("value")
                if "value" in observation
                else observation.get("source_value")
            ),
            observation.get("concept"),
            action.get("verb")
            or observation.get("verb"),
            action.get("object")
            or observation.get("object"),
            action.get("state")
            or observation.get("state"),
        )

        if key in seen:
            continue

        seen.add(key)
        unique_observations.append(observation)

    return unique_observations


def extract_observations(
    mitre_candidates: list[dict[str, Any]],
    semantic_actions: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """
    Extrait et classe les observations disponibles.

    Les observations peuvent provenir :
    1. des preuves associées aux candidats MITRE ;
    2. des actions sémantiques produites avant le raisonnement.
    """
    observations = [
        build_observation(evidence)
        for candidate in mitre_candidates
        for evidence in candidate.get("evidence", [])
    ]

    for action in semantic_actions or []:
        observations.append(
            build_semantic_action_observation(action)
        )

    unique_observations = deduplicate_observations(
        observations
    )

    return sorted(
        unique_observations,
        key=lambda observation: (
            -abs(
                safe_float(
                    observation.get(
                        "signed_contribution"
                    )
                )
            ),
            str(
                observation.get("feature")
                or observation.get("source_feature")
                or ""
            ),
        ),
    )


def build_reasoning_rules(
    mitre_candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Reconstruit la chaîne explicite de transformation :

    Feature → Concept → Action → Pattern → MITRE.
    """
    rules: list[dict[str, Any]] = []

    for candidate in mitre_candidates:
        pattern_id = candidate.get("pattern_id")
        pattern_name = (
            candidate.get("pattern_name")
            or candidate.get("name")
        )

        technique_id = candidate.get("technique_id")
        technique_name = candidate.get(
            "technique_name"
        )

        for evidence in candidate.get("evidence", []):
            rules.append(
                {
                    "rule_type": "semantic_trace",
                    "source_feature": evidence.get(
                        "source_feature"
                    ),
                    "source_value": evidence.get(
                        "source_value"
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
                "tactic_name": candidate.get(
                    "tactic_name"
                ),
                "mapping_status": candidate.get(
                    "mapping_status"
                ),
                "confidence": safe_float(
                    candidate.get(
                        "confidence",
                        candidate.get(
                            "base_confidence"
                        ),
                    )
                ),
                "justification": candidate.get(
                    "justification"
                ),
            }
        )

    return rules


def build_reasoning_section(
    reasoning_result: dict[str, Any] | None,
    mitre_candidates: list[dict[str, Any]],
    attack_chain: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Construit R à partir de la trace native du moteur et des règles
    explicites de transformation.
    """
    reasoning_result = reasoning_result or {}

    trace = reasoning_result.get("trace", {})

    return {
        "engine": (
            reasoning_result.get("backend")
            or trace.get("engine")
            or "unknown"
        ),
        "status": trace.get(
            "status",
            "unknown",
        ),
        "native_execution": trace.get(
            "native_execution",
            False,
        ),
        "rules_fired": trace.get(
            "rules_fired",
            [],
        ),
        "evaluated_pattern_count": trace.get(
            "evaluated_patterns",
            0,
        ),
        "activated_pattern_count": trace.get(
            "activated_pattern_count",
            len(
                reasoning_result.get(
                    "activated_patterns",
                    [],
                )
            ),
        ),
        "activated_patterns": reasoning_result.get(
            "activated_patterns",
            [],
        ),
        "derived_relations": reasoning_result.get(
            "derived_relations",
            [],
        ),
        "explicit_reasoning_rules": (
            build_reasoning_rules(
                mitre_candidates
            )
        ),
        "mitre_mappings": mitre_candidates,
        "attack_chain": attack_chain,
        "context": reasoning_result.get(
            "context",
            {},
        ),
    }


def build_justification(
    prediction: dict[str, Any],
    mitre_candidates: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    reasoning_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Construit J : justification numérique, sémantique et symbolique.
    """
    reasoning_result = reasoning_result or {}

    principal_observations = observations[:5]

    activated_patterns = reasoning_result.get(
        "activated_patterns",
        [],
    )

    return {
        "prediction_probability": safe_float(
            prediction.get("probability")
        ),
        "principal_observations": [
            {
                "feature": (
                    observation.get("feature")
                    or observation.get(
                        "source_feature"
                    )
                ),
                "value": (
                    observation.get("value")
                    if "value" in observation
                    else observation.get(
                        "source_value"
                    )
                ),
                "concept": observation.get("concept"),
                "signed_contribution": safe_float(
                    observation.get(
                        "signed_contribution"
                    )
                ),
                "direction": observation.get(
                    "direction"
                ),
                "trace": observation.get("trace"),
            }
            for observation in principal_observations
        ],
        "pattern_diagnostics": [
            {
                "pattern_id": pattern.get(
                    "pattern_id"
                ),
                "pattern_name": (
                    pattern.get("name")
                    or pattern.get("pattern_name")
                ),
                "severity": pattern.get("severity"),
                "required_coverage": safe_float(
                    pattern.get(
                        "required_coverage"
                    )
                ),
                "supporting_coverage": safe_float(
                    pattern.get(
                        "supporting_coverage"
                    )
                ),
                "matched_required_conditions": (
                    pattern.get(
                        "matched_required_conditions"
                    )
                ),
                "total_required_conditions": (
                    pattern.get(
                        "total_required_conditions"
                    )
                ),
                "matched_supporting_conditions": (
                    pattern.get(
                        "matched_supporting_conditions"
                    )
                ),
                "minimum_supporting_conditions": (
                    pattern.get(
                        "minimum_supporting_conditions"
                    )
                ),
            }
            for pattern in activated_patterns
        ],
        "mitre_diagnostics": [
            {
                "technique_id": candidate.get(
                    "technique_id"
                ),
                "technique_name": candidate.get(
                    "technique_name"
                ),
                "tactic_id": candidate.get(
                    "tactic_id"
                ),
                "tactic_name": candidate.get(
                    "tactic_name"
                ),
                "confidence": safe_float(
                    candidate.get(
                        "confidence",
                        candidate.get(
                            "base_confidence"
                        ),
                    )
                ),
                "mapping_status": candidate.get(
                    "mapping_status"
                ),
                "justification": candidate.get(
                    "justification"
                ),
                "supporting_behavior": candidate.get(
                    "supporting_behavior"
                ),
            }
            for candidate in mitre_candidates
        ],
        "explanatory_statements": {
            "why_prediction": (
                "La conclusion du modèle est soutenue par les "
                "observations ayant les contributions locales "
                "les plus importantes."
            ),
            "why_pattern": (
                "Un motif comportemental est activé lorsque toutes "
                "ses conditions obligatoires et le nombre minimal "
                "de conditions de soutien sont satisfaits."
            ),
            "why_mitre": (
                "Chaque technique MITRE ATT&CK est associée à un "
                "motif activé au moyen d'une règle explicite de la "
                "base de connaissances."
            ),
            "why_attack_chain": (
                "La chaîne d'attaque ordonne les techniques selon "
                "la progression logique des tactiques MITRE ATT&CK."
            ),
        },
    }


def build_limitations(
    mitre_candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Construit les limites et les besoins de contexte supplémentaires."""
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

        mapping_status = candidate.get(
            "mapping_status"
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

        if mapping_status and mapping_status not in {
            "confirmed",
            "validated",
        }:
            limitations.append(
                {
                    "type": "mapping_uncertainty",
                    "technique_id": candidate.get(
                        "technique_id"
                    ),
                    "mapping_status": mapping_status,
                    "message": (
                        "Le mapping MITRE doit être interprété "
                        "comme une hypothèse contextuelle."
                    ),
                }
            )

    return limitations


def determine_risk_level(
    prediction: dict[str, Any],
    reasoning_result: dict[str, Any] | None,
) -> str:
    """Détermine un niveau de risque simple et explicite."""
    reasoning_result = reasoning_result or {}

    probability = safe_float(
        prediction.get("probability")
    )

    activated_patterns = reasoning_result.get(
        "activated_patterns",
        [],
    )

    severities = {
        str(pattern.get("severity", "")).lower()
        for pattern in activated_patterns
    }

    if "critical" in severities:
        return "critical"

    if "high" in severities:
        return "high"

    label = str(
        prediction.get("label", "")
    ).lower()

    if probability >= 0.90 and label not in {
        "normal",
        "benign",
    }:
        return "high"

    if probability >= 0.70 and label not in {
        "normal",
        "benign",
    }:
        return "medium"

    if label in {"normal", "benign"}:
        return "low"

    return "medium"


def determine_soc_priority(
    risk_level: str,
) -> str:
    """Convertit le niveau de risque en priorité opérationnelle."""
    priorities = {
        "critical": "P1",
        "high": "P1",
        "medium": "P2",
        "low": "P3",
    }

    return priorities.get(
        risk_level,
        "P3",
    )


def determine_recommended_action(
    risk_level: str,
    mitre_candidates: list[dict[str, Any]],
) -> str:
    """Produit une recommandation opérationnelle orientée SOC."""
    if not mitre_candidates:
        return (
            "Conserver l'événement pour corrélation et collecter "
            "davantage de contexte avant escalade."
        )

    if risk_level in {"critical", "high"}:
        return (
            "Examiner immédiatement l'hôte et les flux associés, "
            "valider les indicateurs observés et envisager "
            "l'isolement si l'activité est confirmée."
        )

    if risk_level == "medium":
        return (
            "Corréler l'événement avec les journaux récents, "
            "vérifier l'historique de l'hôte et surveiller toute "
            "progression vers une autre tactique."
        )

    return (
        "Archiver l'observation, maintenir la surveillance et "
        "réévaluer si de nouveaux événements apparaissent."
    )


def generate_natural_language_explanation(
    prediction: dict[str, Any],
    mitre_candidates: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    attack_chain: list[dict[str, Any]] | None = None,
) -> dict[str, str]:
    """Génère une explication concise destinée à un analyste SOC."""
    prediction_label = prediction.get(
        "label",
        "inconnue",
    )

    probability = safe_float(
        prediction.get("probability")
    )

    attack_chain = attack_chain or []

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
            f"{candidate.get('pattern_name') or candidate.get('name')}"
        )
    else:
        technique_text = "aucune technique identifiée"
        tactic_text = "aucune tactique identifiée"
        pattern_text = "aucun motif activé"

    principal = observations[:2]

    evidence_fragments = [
        (
            f"{observation.get('feature') or observation.get('source_feature')}="
            f"{format_value(observation.get('value') if 'value' in observation else observation.get('source_value'))} "
            f"(contribution "
            f"{safe_float(observation.get('signed_contribution')):+.4f})"
        )
        for observation in principal
    ]

    evidence_text = (
        " et ".join(evidence_fragments)
        if evidence_fragments
        else "aucune observation principale disponible"
    )

    if len(attack_chain) > 1:
        chain_text = (
            f"Une progression de {len(attack_chain)} étapes "
            "a été reconstruite."
        )
    elif len(attack_chain) == 1:
        chain_text = (
            "Une étape de la chaîne d'attaque a été identifiée."
        )
    else:
        chain_text = (
            "Aucune chaîne d'attaque multiétape n'a été reconstruite."
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
        "attack_chain": chain_text,
        "evidence": (
            f"Les principales preuves locales sont "
            f"{evidence_text}."
        ),
        "caution": (
            "Le résultat MITRE est un candidat contextuel "
            "issu de règles explicites et doit être validé "
            "avec les journaux et le contexte opérationnel."
        ),
    }


def build_explainable_proof(
    prediction: dict[str, Any],
    mitre_candidates: list[dict[str, Any]],
    *,
    case_id: str | None = None,
    proof_id: str | None = None,
    semantic_actions: list[dict[str, Any]] | None = None,
    reasoning_result: dict[str, Any] | None = None,
    attack_chain: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Construit la preuve explicable finale :

    E = <O, R, J>

    O : observations traçables ;
    R : raisonnement symbolique et chaîne d'inférence ;
    J : justification interprétable et vérifiable ;
    C : conclusion opérationnelle.
    """
    semantic_actions = semantic_actions or []
    reasoning_result = reasoning_result or {}
    attack_chain = attack_chain or []

    resolved_case_id = (
        case_id
        or reasoning_result.get("case_id")
        or "case_0"
    )

    resolved_proof_id = (
        proof_id
        or f"E-{resolved_case_id.upper()}"
    )

    observations = extract_observations(
        mitre_candidates=mitre_candidates,
        semantic_actions=semantic_actions,
    )

    reasoning = build_reasoning_section(
        reasoning_result=reasoning_result,
        mitre_candidates=mitre_candidates,
        attack_chain=attack_chain,
    )

    justification = build_justification(
        prediction=prediction,
        mitre_candidates=mitre_candidates,
        observations=observations,
        reasoning_result=reasoning_result,
    )

    limitations = build_limitations(
        mitre_candidates
    )

    risk_level = determine_risk_level(
        prediction=prediction,
        reasoning_result=reasoning_result,
    )

    soc_priority = determine_soc_priority(
        risk_level
    )

    recommended_action = (
        determine_recommended_action(
            risk_level=risk_level,
            mitre_candidates=mitre_candidates,
        )
    )

    candidate_conclusions = [
        {
            "pattern_id": candidate.get(
                "pattern_id"
            ),
            "pattern_name": (
                candidate.get("pattern_name")
                or candidate.get("name")
            ),
            "technique_id": candidate.get(
                "technique_id"
            ),
            "technique_name": candidate.get(
                "technique_name"
            ),
            "tactic_id": candidate.get(
                "tactic_id"
            ),
            "tactic_name": candidate.get(
                "tactic_name"
            ),
            "mapping_status": candidate.get(
                "mapping_status"
            ),
            "confidence": safe_float(
                candidate.get(
                    "confidence",
                    candidate.get(
                        "base_confidence"
                    ),
                )
            ),
        }
        for candidate in mitre_candidates
    ]

    conclusion = {
        "prediction": {
            "label": prediction.get("label"),
            "probability": safe_float(
                prediction.get("probability")
            ),
        },
        "risk_level": risk_level,
        "soc_priority": soc_priority,
        "recommended_action": recommended_action,
        "contextual_candidates": (
            candidate_conclusions
        ),
    }

    natural_language_explanation = (
        generate_natural_language_explanation(
            prediction=prediction,
            mitre_candidates=mitre_candidates,
            observations=observations,
            attack_chain=attack_chain,
        )
    )

    rule_count = len(
        reasoning.get(
            "explicit_reasoning_rules",
            [],
        )
    )

    derived_relation_count = len(
        reasoning.get(
            "derived_relations",
            [],
        )
    )

    traceable = bool(
        observations
        and (
            rule_count > 0
            or derived_relation_count > 0
        )
    )

    verifiable = bool(
        reasoning.get("engine")
        and reasoning.get(
            "activated_patterns"
        )
    )

    return {
        "proof_id": resolved_proof_id,
        "case_id": resolved_case_id,
        "formalism": "E=<O,R,J>",
        "C_conclusion": conclusion,
        "O_observations": observations,
        "R_reasoning": reasoning,
        "J_justification": justification,
        "limitations": limitations,
        "natural_language_explanation": (
            natural_language_explanation
        ),
        "audit": {
            "observation_count": len(
                observations
            ),
            "semantic_action_count": len(
                semantic_actions
            ),
            "rule_count": rule_count,
            "derived_relation_count": (
                derived_relation_count
            ),
            "activated_pattern_count": len(
                reasoning.get(
                    "activated_patterns",
                    [],
                )
            ),
            "mitre_candidate_count": len(
                mitre_candidates
            ),
            "attack_chain_step_count": len(
                attack_chain
            ),
            "traceable": traceable,
            "verifiable": verifiable,
            "contestable": bool(limitations),
            "reproducible": bool(
                reasoning.get(
                    "native_execution"
                )
                or reasoning.get(
                    "engine"
                )
            ),
        },
    }