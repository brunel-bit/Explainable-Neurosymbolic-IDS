from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from src.explanation.proof_builder import (
    build_explainable_proof,
)
from src.reasoning.mitre_mapper import (
    map_patterns_to_mitre,
)
from src.reasoning.pattern_engine import (
    get_activated_patterns,
    infer_behavior_patterns,
)
from src.semantic.mapper import (
    load_xai_json,
    semantic_concepts_to_actions,
    xai_to_semantic_concepts,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    """Charge un fichier JSON dont la racine doit être un objet."""
    if not path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {path.resolve()}"
        )

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError(
            f"Le fichier {path} doit contenir un objet JSON."
        )

    return payload


def save_json(
    payload: dict[str, Any],
    output_path: Path,
) -> None:
    """Enregistre un objet JSON en créant les dossiers nécessaires."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=4,
            ensure_ascii=False,
        )


def sanitize_case_id(case_id: str) -> str:
    """
    Nettoie l’identifiant du cas pour son utilisation dans les noms
    de fichiers.
    """
    sanitized = re.sub(
        r"[^a-zA-Z0-9_-]+",
        "_",
        case_id.strip(),
    ).strip("_")

    if not sanitized:
        raise ValueError(
            "L’identifiant du cas ne peut pas être vide."
        )

    return sanitized


def infer_case_id(
    input_path: Path,
) -> str:
    """
    Déduit l’identifiant depuis un fichier comme input_case_0.json.
    """
    stem = input_path.stem

    if stem.startswith("input_"):
        stem = stem[len("input_"):]

    return sanitize_case_id(stem)


def build_output_paths(
    case_id: str,
) -> dict[str, Path]:
    """Construit les chemins de sortie du pipeline."""
    return {
        "concepts": (
            PROJECT_ROOT
            / "outputs"
            / "semantic"
            / f"concepts_{case_id}.json"
        ),
        "actions": (
            PROJECT_ROOT
            / "outputs"
            / "semantic"
            / f"actions_{case_id}.json"
        ),
        "patterns": (
            PROJECT_ROOT
            / "outputs"
            / "reasoning"
            / f"patterns_{case_id}.json"
        ),
        "mitre": (
            PROJECT_ROOT
            / "outputs"
            / "mitre"
            / f"mitre_{case_id}.json"
        ),
        "proof": (
            PROJECT_ROOT
            / "outputs"
            / "proofs"
            / f"proof_{case_id}.json"
        ),
    }


def build_concepts_result(
    xai_payload: dict[str, Any],
    concepts_config: dict[str, Any],
) -> dict[str, Any]:
    """Exécute la couche Feature → Concept."""
    concepts = xai_to_semantic_concepts(
        xai_payload=xai_payload,
        concepts_config=concepts_config,
    )

    mapped_count = sum(
        1
        for concept in concepts
        if concept.get("mapped", False)
    )

    total_count = len(concepts)

    return {
        "prediction": xai_payload.get(
            "prediction",
            {},
        ),
        "dataset": concepts_config.get("dataset"),
        "semantic_concepts": concepts,
        "coverage": {
            "mapped_features": mapped_count,
            "total_features": total_count,
            "mapping_rate": (
                round(mapped_count / total_count, 4)
                if total_count
                else 0.0
            ),
        },
    }


def build_actions_result(
    prediction: dict[str, Any],
    concepts: list[dict[str, Any]],
    actions_config: dict[str, Any],
) -> dict[str, Any]:
    """Exécute la couche Concept → Action."""
    actions = semantic_concepts_to_actions(
        semantic_concepts=concepts,
        actions_config=actions_config,
    )

    return {
        "prediction": prediction,
        "semantic_concepts": concepts,
        "semantic_actions": actions,
        "summary": {
            "concept_count": len(concepts),
            "action_count": len(actions),
            "supporting_actions": sum(
                1
                for action in actions
                if action.get(
                    "supports_prediction",
                    False,
                )
            ),
            "opposing_actions": sum(
                1
                for action in actions
                if action.get("direction")
                == "opposes_prediction"
            ),
        },
    }


def build_patterns_result(
    prediction: dict[str, Any],
    actions: list[dict[str, Any]],
    patterns_config: dict[str, Any],
) -> dict[str, Any]:
    """Exécute la couche Action → Motif comportemental."""
    prediction_label = prediction.get("label")

    evaluated_patterns = infer_behavior_patterns(
        semantic_actions=actions,
        patterns_config=patterns_config,
        prediction_label=prediction_label,
    )

    activated_patterns = get_activated_patterns(
        evaluated_patterns
    )

    return {
        "prediction": prediction,
        "semantic_action_count": len(actions),
        "evaluated_patterns": evaluated_patterns,
        "activated_patterns": activated_patterns,
        "summary": {
            "evaluated_pattern_count": len(
                evaluated_patterns
            ),
            "activated_pattern_count": len(
                activated_patterns
            ),
        },
    }


def build_mitre_result(
    prediction: dict[str, Any],
    activated_patterns: list[dict[str, Any]],
    mitre_config: dict[str, Any],
) -> dict[str, Any]:
    """Exécute la couche Motif → MITRE ATT&CK."""
    mapping_result = map_patterns_to_mitre(
        activated_patterns=activated_patterns,
        mitre_config=mitre_config,
    )

    return {
        "prediction": prediction,
        "activated_patterns": [
            {
                "pattern_id": pattern.get(
                    "pattern_id"
                ),
                "name": pattern.get("name"),
                "category": pattern.get("category"),
                "severity": pattern.get("severity"),
                "total_evidence_weight": pattern.get(
                    "total_evidence_weight"
                ),
            }
            for pattern in activated_patterns
        ],
        **mapping_result,
        "metadata": {
            "mapping_version": mitre_config.get(
                "version"
            ),
            "mapping_source": mitre_config.get(
                "source",
                {},
            ),
        },
    }


def run_pipeline(
    input_path: Path,
    case_id: str,
) -> dict[str, Any]:
    """Exécute toutes les couches du pipeline explicable."""
    concepts_config_path = (
        PROJECT_ROOT
        / "config"
        / "semantic"
        / "concepts_nslkdd.json"
    )

    actions_config_path = (
        PROJECT_ROOT
        / "config"
        / "semantic"
        / "actions_nslkdd.json"
    )

    patterns_config_path = (
        PROJECT_ROOT
        / "config"
        / "reasoning"
        / "behavior_patterns_nslkdd.json"
    )

    mitre_config_path = (
        PROJECT_ROOT
        / "config"
        / "mitre"
        / "pattern_to_mitre.json"
    )

    xai_payload = load_xai_json(input_path)
    concepts_config = load_json(
        concepts_config_path
    )
    actions_config = load_json(
        actions_config_path
    )
    patterns_config = load_json(
        patterns_config_path
    )
    mitre_config = load_json(
        mitre_config_path
    )

    output_paths = build_output_paths(case_id)

    concepts_result = build_concepts_result(
        xai_payload=xai_payload,
        concepts_config=concepts_config,
    )

    save_json(
        concepts_result,
        output_paths["concepts"],
    )

    actions_result = build_actions_result(
        prediction=concepts_result["prediction"],
        concepts=concepts_result[
            "semantic_concepts"
        ],
        actions_config=actions_config,
    )

    save_json(
        actions_result,
        output_paths["actions"],
    )

    patterns_result = build_patterns_result(
        prediction=actions_result["prediction"],
        actions=actions_result[
            "semantic_actions"
        ],
        patterns_config=patterns_config,
    )

    save_json(
        patterns_result,
        output_paths["patterns"],
    )

    mitre_result = build_mitre_result(
        prediction=patterns_result["prediction"],
        activated_patterns=patterns_result[
            "activated_patterns"
        ],
        mitre_config=mitre_config,
    )

    save_json(
        mitre_result,
        output_paths["mitre"],
    )

    proof_result = build_explainable_proof(
        prediction=mitre_result["prediction"],
        mitre_candidates=mitre_result[
            "mitre_candidates"
        ],
    )

    proof_result["proof_id"] = (
        f"E-{case_id.upper()}"
    )

    proof_result["case_id"] = case_id
    proof_result["source_xai_file"] = str(
        input_path.resolve()
    )

    save_json(
        proof_result,
        output_paths["proof"],
    )

    return {
        "case_id": case_id,
        "input_path": input_path,
        "output_paths": output_paths,
        "concepts_result": concepts_result,
        "actions_result": actions_result,
        "patterns_result": patterns_result,
        "mitre_result": mitre_result,
        "proof_result": proof_result,
    }


def print_summary(
    pipeline_result: dict[str, Any],
) -> None:
    """Affiche une synthèse compacte du pipeline."""
    case_id = pipeline_result["case_id"]

    concepts_result = pipeline_result[
        "concepts_result"
    ]
    actions_result = pipeline_result[
        "actions_result"
    ]
    patterns_result = pipeline_result[
        "patterns_result"
    ]
    mitre_result = pipeline_result[
        "mitre_result"
    ]
    proof_result = pipeline_result[
        "proof_result"
    ]

    prediction = concepts_result.get(
        "prediction",
        {},
    )

    probability = float(
        prediction.get("probability", 0.0)
    )

    print(
        "=== Pipeline explicable complet ==="
    )
    print(f"Cas         : {case_id}")
    print(
        f"Prédiction  : "
        f"{prediction.get('label')} "
        f"(p={probability:.4f})"
    )

    coverage = concepts_result["coverage"]

    print(
        f"Concepts    : "
        f"{coverage['mapped_features']}/"
        f"{coverage['total_features']} "
        f"({coverage['mapping_rate']:.0%})"
    )

    print(
        f"Actions     : "
        f"{actions_result['summary']['action_count']}"
    )

    print(
        f"Motifs      : "
        f"{patterns_result['summary']['activated_pattern_count']} "
        f"activé(s)"
    )

    print(
        f"MITRE       : "
        f"{mitre_result['summary']['candidate_count']} "
        f"candidat(s)"
    )

    print(
        f"Traçable    : "
        f"{proof_result['audit']['traceable']}"
    )

    print()

    activated_patterns = patterns_result[
        "activated_patterns"
    ]

    for pattern in activated_patterns:
        print(
            f"Motif       : "
            f"{pattern.get('pattern_id')} "
            f"→ {pattern.get('name')}"
        )

    for candidate in mitre_result[
        "mitre_candidates"
    ]:
        print(
            f"MITRE       : "
            f"{candidate.get('technique_id')} "
            f"→ {candidate.get('technique_name')}"
        )

    explanation = proof_result.get(
        "natural_language_explanation",
        {},
    )

    if explanation:
        print()
        print("Explication :")
        print(f"- {explanation.get('summary')}")
        print(f"- {explanation.get('behavior')}")
        print(f"- {explanation.get('mitre')}")
        print(f"- {explanation.get('evidence')}")
        print(f"- {explanation.get('caution')}")

    print()
    print("Fichiers générés :")

    for output_name, output_path in pipeline_result[
        "output_paths"
    ].items():
        print(
            f"- {output_name}: "
            f"{output_path.resolve()}"
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Exécute le pipeline explicable complet "
            "à partir d’un fichier XAI/COMAT."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help=(
            "Chemin du fichier XAI, par exemple "
            "outputs/comat/input_case_0.json."
        ),
    )

    parser.add_argument(
        "--case-id",
        type=str,
        default=None,
        help=(
            "Identifiant utilisé dans les noms de sortie. "
            "S’il est absent, il est déduit du fichier d’entrée."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    input_path = args.input

    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path

    case_id = (
        sanitize_case_id(args.case_id)
        if args.case_id
        else infer_case_id(input_path)
    )

    pipeline_result = run_pipeline(
        input_path=input_path,
        case_id=case_id,
    )

    print_summary(pipeline_result)


if __name__ == "__main__":
    main()