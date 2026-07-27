from __future__ import annotations
import re
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_xai_json(path: str | Path) -> dict[str, Any]:
    """Load an XAI explanation stored in COMAT-compatible JSON format."""
    input_path = Path(path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Le fichier XAI est introuvable : {input_path.resolve()}"
        )

    with input_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def xai_to_semantic_concepts(
    xai_payload: dict[str, Any],
    concepts_config: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Transform technical XAI features into interpretable semantic concepts.

    Unknown features are preserved instead of being silently discarded.
    """
    knowledge_base = concepts_config.get("features", {})
    concepts: list[dict[str, Any]] = []

    for feature in xai_payload.get("features", []):
        feature_name = str(feature.get("name", ""))
        feature_value = feature.get("value")
        contribution = float(feature.get("contribution", 0.0))

        signed_contribution = float(
            feature.get("signed_contribution", contribution)
        )

        direction = feature.get("direction")

        if direction is None:
            if signed_contribution > 0:
                direction = "supports_prediction"
            elif signed_contribution < 0:
                direction = "opposes_prediction"
            else:
                direction = "neutral"

        knowledge = knowledge_base.get(feature_name)

        common_fields = {
            "feature": feature_name,
            "value": feature_value,
            "contribution": contribution,
            "signed_contribution": signed_contribution,
            "direction": direction,
        }

        if knowledge is None:
            concepts.append(
                {
                    **common_fields,
                    "mapped": False,
                    "concept": None,
                    "reason": (
                        "No semantic concept is defined for this feature."
                    ),
                }
            )
            continue

        concepts.append(
            {
                **common_fields,
                "mapped": True,
                "concept": knowledge["concept"],
                "category": knowledge.get("category"),
                "value_type": knowledge.get("value_type"),
                "definition": knowledge.get("definition"),
                "soc_interpretation": knowledge.get(
                    "soc_interpretation"
                ),
                "source": knowledge.get("source"),
            }
        )

    return concepts


def evaluate_condition(
    value: float,
    operator: str,
    threshold: float,
) -> bool:
    """Evaluate a numeric semantic condition."""
    if operator == ">":
        return value > threshold
    if operator == ">=":
        return value >= threshold
    if operator == "<":
        return value < threshold
    if operator == "<=":
        return value <= threshold
    if operator == "==":
        return value == threshold
    if operator == "!=":
        return value != threshold

    raise ValueError(
        f"Opérateur sémantique non pris en charge : {operator}"
    )


def semantic_concepts_to_actions(
    semantic_concepts: list[dict[str, Any]],
    actions_config: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Convert semantic concepts into contextual semantic actions.
    """
    rules_by_concept = actions_config.get("actions", {})
    actions: list[dict[str, Any]] = []

    for item in semantic_concepts:
        if not item.get("mapped"):
            continue

        concept_name = item.get("concept")
        concept_rules = rules_by_concept.get(concept_name)

        if not concept_rules:
            continue

        try:
            observed_value = float(item.get("value"))
        except (TypeError, ValueError):
            continue

        contribution = float(item.get("contribution", 0.0))
        signed_contribution = float(
            item.get("signed_contribution", contribution)
        )

        direction = item.get("direction", "neutral")

        for condition in concept_rules.get("conditions", []):
            operator = condition.get("operator")
            threshold = float(condition.get("threshold", 0.0))

            if not evaluate_condition(
                observed_value,
                operator,
                threshold,
            ):
                continue

            base_weight = float(condition.get("weight", 1.0))
            evidence_strength = abs(contribution)

            action_weight = round(
                base_weight * evidence_strength,
                4,
            )

            tag = condition.get("tag")

            actions.append(
                {
                    "verb": condition["verb"],
                    "object": condition["object"],
                    "weight": action_weight,
                    "state": condition.get("state"),
                    "semantic_tags": [tag] if tag else [],
                    "concept": concept_name,
                    "source_feature": item.get("feature"),
                    "source_value": item.get("value"),
                    "contribution": contribution,
                    "signed_contribution": signed_contribution,
                    "direction": direction,
                    "supports_prediction": (
                        direction == "supports_prediction"
                    ),
                    "condition": {
                        "operator": operator,
                        "threshold": threshold,
                    },
                }
            )

            # Une seule condition doit être retenue pour chaque concept.
            break

    return sorted(
        actions,
        key=lambda action: (
            -action["weight"],
            action["verb"],
            action["object"],
        ),
    )


def feature_matches_term(
    feature_name: str,
    term: str,
) -> bool:
    """
    Match a semantic rule against a feature name using complete tokens.

    This prevents false matches such as:
    - "sam" matching "same";
    - "port" matching an unrelated longer word.
    """
    normalized_feature = re.sub(
        r"[^a-z0-9]+",
        " ",
        feature_name.lower(),
    ).strip()

    normalized_term = re.sub(
        r"[^a-z0-9]+",
        " ",
        term.lower(),
    ).strip()

    if not normalized_term:
        return False

    pattern = rf"\b{re.escape(normalized_term)}\b"

    return re.search(
        pattern,
        normalized_feature,
    ) is not None


def xai_to_semantic_actions(
    xai_payload: dict[str, Any],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Convert XAI features into aggregated semantic actions.

    Each matching feature-pattern rule produces an action containing:
    - a verb;
    - an object;
    - a weight;
    - the source feature;
    - the observed value;
    - an optional semantic tag.
    """
    rules = (
        config.get("semantic_mapping", {})
        .get("feature_pattern_rules", [])
    )

    actions: list[dict[str, Any]] = []

    for feature in xai_payload.get("features", []):
        feature_name = str(feature.get("name", ""))
        feature_value = feature.get("value", "")
        feature_text = feature_name

        contribution = float(feature.get("contribution", 0.0))

        for rule in rules:
            terms = rule.get("contains_any", [])

            if any(feature_matches_term(feature_text, term) for term in terms):
                actions.append(
                    {
                        "verb": rule["verb"],
                        "object": rule["object"],
                        "weight": round(
                            float(rule["weight"])
                            * max(contribution, 0.05),
                            4,
                        ),
                        "source_feature": feature_name,
                        "source_value": feature_value,
                        "tag": rule.get("tag"),
                    }
                )

    merged = defaultdict(
        lambda: {
            "weight": 0.0,
            "evidence": [],
            "tags": set(),
        }
    )

    for action in actions:
        key = (action["verb"], action["object"])

        merged[key]["weight"] += action["weight"]

        merged[key]["evidence"].append(
            {
                "feature": action["source_feature"],
                "value": action["source_value"],
                "weight": action["weight"],
            }
        )

        if action.get("tag"):
            merged[key]["tags"].add(action["tag"])

    result: list[dict[str, Any]] = []

    sorted_actions = sorted(
        merged.items(),
        key=lambda item: (
            -item[1]["weight"],
            item[0][0],
            item[0][1],
        ),
    )

    for (verb, obj), payload in sorted_actions:
        result.append(
            {
                "verb": verb,
                "object": obj,
                "weight": round(payload["weight"], 4),
                "semantic_tags": sorted(payload["tags"]),
                "evidence": payload["evidence"],
            }
        )

    return result


def score_technique_against_actions(
    technique_row: dict[str, Any],
    semantic_actions: list[dict[str, Any]],
) -> tuple[float, list[dict[str, Any]]]:
    """Score a MITRE technique against XAI-derived semantic actions."""
    candidate_actions = technique_row.get(
        "threat_actions_generated",
        [],
    )

    score = 0.0
    matches: list[dict[str, Any]] = []

    for semantic_action in semantic_actions:
        for mitre_action in candidate_actions:
            same_verb = (
                semantic_action["verb"]
                == mitre_action["verb"]
            )
            same_object = (
                semantic_action["object"]
                == mitre_action["object"]
            )

            if same_verb and same_object:
                contribution = (
                    semantic_action["weight"]
                    * max(
                        float(mitre_action.get("score", 0.0)),
                        1.0,
                    )
                )

                score += contribution

                matches.append(
                    {
                        "semantic_action": {
                            "verb": semantic_action["verb"],
                            "object": semantic_action["object"],
                            "weight": semantic_action["weight"],
                        },
                        "mitre_action": {
                            "verb": mitre_action["verb"],
                            "object": mitre_action["object"],
                            "score": mitre_action.get("score", 0.0),
                        },
                        "contribution": round(contribution, 4),
                    }
                )

    return round(score, 4), matches


def rank_mitre_candidates(
    xai_payload: dict[str, Any],
    normalized_threat_actions: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Rank MITRE ATT&CK candidates from semantic actions."""
    semantic_actions = xai_to_semantic_actions(
        xai_payload,
        config,
    )

    label = (
        xai_payload.get("prediction", {})
        .get("label", "")
    )

    hinted = set(
        config.get("semantic_mapping", {})
        .get("label_to_mitre_hints", {})
        .get(label, [])
    )

    candidates: list[dict[str, Any]] = []

    for row in normalized_threat_actions:
        score, matches = score_technique_against_actions(
            row,
            semantic_actions,
        )

        if row["technique_id"] in hinted:
            score += 1.5

        if score > 0:
            candidates.append(
                {
                    "technique_id": row["technique_id"],
                    "technique_name": row["technique_name"],
                    "score": round(score, 4),
                    "matches": matches,
                    "hinted_by_label": (
                        row["technique_id"] in hinted
                    ),
                }
            )

    candidates.sort(
        key=lambda candidate: (
            -candidate["score"],
            candidate["technique_id"],
        )
    )

    top_k = int(
        config.get("semantic_mapping", {})
        .get("top_k_candidates", 5)
    )

    return semantic_actions, candidates[:top_k]


def coherence_diagnostic(
    xai_payload: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate coherence between the XAI prediction and MITRE candidates."""
    prediction = xai_payload.get("prediction", {})

    label = prediction.get("label")
    probability = float(
        prediction.get("probability", 0.0)
    )

    if not candidates:
        return {
            "coherent": False,
            "reason": (
                "Aucun candidat MITRE n'a été trouvé à partir "
                "des actions sémantiques dérivées du XAI."
            ),
            "prediction_label": label,
            "prediction_probability": probability,
        }

    top_candidate = candidates[0]
    coherent = top_candidate["score"] >= 1.5

    reason = (
        f"Le meilleur candidat MITRE est "
        f"{top_candidate['technique_id']} "
        f"({top_candidate['technique_name']}) avec un score de "
        f"{top_candidate['score']}. La prédiction XAI "
        f"'{label}' (p={probability}) est "
        f"{'cohérente' if coherent else 'faiblement cohérente'} "
        f"avec les actions sémantiques dérivées."
    )

    return {
        "coherent": coherent,
        "reason": reason,
        "prediction_label": label,
        "prediction_probability": probability,
        "top_candidate": {
            "technique_id": top_candidate["technique_id"],
            "technique_name": top_candidate["technique_name"],
            "score": top_candidate["score"],
        },
    }


def full_semantic_pipeline(
    xai_payload: dict[str, Any],
    normalized_threat_actions: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Run semantic mapping, MITRE ranking, and coherence diagnosis."""
    semantic_actions, candidates = rank_mitre_candidates(
        xai_payload,
        normalized_threat_actions,
        config,
    )

    diagnostic = coherence_diagnostic(
        xai_payload,
        candidates,
    )

    return {
        "prediction": xai_payload.get("prediction", {}),
        "semantic_actions_from_xai": semantic_actions,
        "mitre_candidates": candidates,
        "diagnostic": diagnostic,
    }