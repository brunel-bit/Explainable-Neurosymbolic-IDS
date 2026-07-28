from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from src.explanation.proof_builder import build_explainable_proof
from src.mitre.mapper import MitreMapper
from src.reasoning.attack_chain_builder import AttackChainBuilder
from src.reasoning.scallop_backend import ScallopReasoningBackend
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
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            indent=4,
            ensure_ascii=False,
        )


def safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    """Convertit une valeur en float sans interrompre le pipeline."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def sanitize_case_id(case_id: str) -> str:
    """Nettoie un identifiant destiné aux noms de fichiers."""
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


def infer_case_id(input_path: Path) -> str:
    """Déduit l’identifiant depuis un fichier comme input_probe_01.json."""
    stem = input_path.stem

    if stem.startswith("input_"):
        stem = stem[len("input_"):]

    return sanitize_case_id(stem)


def build_output_paths(case_id: str) -> dict[str, Path]:
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
        "reasoning": (
            PROJECT_ROOT
            / "outputs"
            / "reasoning"
            / f"reasoning_{case_id}.json"
        ),
        "mitre": (
            PROJECT_ROOT
            / "outputs"
            / "mitre"
            / f"mitre_{case_id}.json"
        ),
        "attack_chain": (
            PROJECT_ROOT
            / "outputs"
            / "reasoning"
            / f"attack_chain_{case_id}.json"
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
    """Exécute la transformation Feature → Concept."""
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
        "prediction": xai_payload.get("prediction", {}),
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
    """Exécute la transformation Concept → Action."""
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
                if action.get("supports_prediction", False)
            ),
            "opposing_actions": sum(
                1
                for action in actions
                if action.get("direction")
                == "opposes_prediction"
            ),
        },
    }


def normalize_reasoning_result(
    reasoning_result: dict[str, Any],
    *,
    case_id: str,
    prediction: dict[str, Any],
    semantic_action_count: int,
) -> dict[str, Any]:
    """Ajoute les métadonnées utiles sans modifier la trace Scallop."""
    activated_patterns = reasoning_result.get(
        "activated_patterns",
        [],
    )
    derived_relations = reasoning_result.get(
        "derived_relations",
        [],
    )
    trace = reasoning_result.get("trace", {})

    return {
        **reasoning_result,
        "case_id": reasoning_result.get("case_id", case_id),
        "prediction": prediction,
        "semantic_action_count": semantic_action_count,
        "summary": {
            "backend": reasoning_result.get(
                "backend",
                trace.get("engine", "Scallop"),
            ),
            "evaluated_pattern_count": trace.get(
                "evaluated_patterns",
                0,
            ),
            "activated_pattern_count": len(activated_patterns),
            "derived_relation_count": len(derived_relations),
        },
    }


def run_scallop_reasoning(
    *,
    case_id: str,
    prediction: dict[str, Any],
    semantic_actions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Exécute Action → Motif avec le backend Scallop."""
    backend = ScallopReasoningBackend()

    reasoning_result = backend.reason(
        case_id=case_id,
        actions=semantic_actions,
        patterns=[],
        context={
            "prediction": prediction,
        },
)

    if not isinstance(reasoning_result, dict):
        raise TypeError(
            "ScallopReasoningBackend.reason() doit retourner un dictionnaire."
        )

    return normalize_reasoning_result(
        reasoning_result,
        case_id=case_id,
        prediction=prediction,
        semantic_action_count=len(semantic_actions),
    )


def build_mitre_result(
    *,
    prediction: dict[str, Any],
    activated_patterns: list[dict[str, Any]],
) -> dict[str, Any]:
    """Exécute Motif → MITRE ATT&CK avec le mapper final."""
    mapper = MitreMapper()
    mitre_candidates = mapper.map_patterns(activated_patterns)

    if not isinstance(mitre_candidates, list):
        raise TypeError(
            "MitreMapper.map_patterns() doit retourner une liste."
        )

    pattern_names = {
        pattern.get("pattern_id"): (
            pattern.get("name")
            or pattern.get("pattern_name")
            or "motif non nommé"
        )
        for pattern in activated_patterns
        if pattern.get("pattern_id")
    }

    for candidate in mitre_candidates:
        pattern_id = candidate.get("pattern_id")

        if not candidate.get("pattern_name"):
            candidate["pattern_name"] = pattern_names.get(
                pattern_id,
                "motif non nommé",
            )

    

    mapped_count = sum(
        1
        for candidate in mitre_candidates
        if candidate.get("mapping_status")
        not in {None, "unmapped"}
    )

    return {
        "prediction": prediction,
        "activated_patterns": activated_patterns,
        "mitre_candidates": mitre_candidates,
        "summary": {
            "activated_pattern_count": len(activated_patterns),
            "candidate_count": len(mitre_candidates),
            "mapped_candidate_count": mapped_count,
        },
    }


def build_attack_chain_result(
    mitre_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Ordonne les candidats selon les tactiques MITRE ATT&CK."""
    builder = AttackChainBuilder()
    attack_chain = builder.build(mitre_candidates)

    if not isinstance(attack_chain, list):
        raise TypeError(
            "AttackChainBuilder.build() doit retourner une liste."
        )

    return {
        "attack_chain": attack_chain,
        "summary": {
            "step_count": len(attack_chain),
            "is_multistage": len(attack_chain) > 1,
            "technique_ids": [
                step.get("technique_id")
                for step in attack_chain
                if step.get("technique_id")
            ],
            "tactics": [
                step.get("tactic")
                for step in attack_chain
                if step.get("tactic")
            ],
        },
    }


def run_pipeline(
    input_path: Path,
    case_id: str,
) -> dict[str, Any]:
    """
    Exécute le pipeline explicable final :

    XAI/COMAT → Concepts → Actions → Scallop → MITRE ATT&CK
    → Chaîne d’attaque → Preuve E=<O,R,J>.
    """
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

    xai_payload = load_xai_json(input_path)
    concepts_config = load_json(concepts_config_path)
    actions_config = load_json(actions_config_path)
    output_paths = build_output_paths(case_id)

    concepts_result = build_concepts_result(
        xai_payload=xai_payload,
        concepts_config=concepts_config,
    )
    save_json(concepts_result, output_paths["concepts"])

    actions_result = build_actions_result(
        prediction=concepts_result["prediction"],
        concepts=concepts_result["semantic_concepts"],
        actions_config=actions_config,
    )
    save_json(actions_result, output_paths["actions"])

    reasoning_result = run_scallop_reasoning(
        case_id=case_id,
        prediction=actions_result["prediction"],
        semantic_actions=actions_result["semantic_actions"],
    )
    save_json(reasoning_result, output_paths["reasoning"])

    mitre_result = build_mitre_result(
        prediction=actions_result["prediction"],
        activated_patterns=reasoning_result["activated_patterns"],
    )
    save_json(mitre_result, output_paths["mitre"])

    attack_chain_result = build_attack_chain_result(
        mitre_result["mitre_candidates"]
    )
    save_json(
        attack_chain_result,
        output_paths["attack_chain"],
    )

    proof_result = build_explainable_proof(
        case_id=case_id,
        prediction=actions_result["prediction"],
        semantic_actions=actions_result["semantic_actions"],
        reasoning_result=reasoning_result,
        mitre_candidates=mitre_result["mitre_candidates"],
        attack_chain=attack_chain_result["attack_chain"],
    )
    proof_result["source_xai_file"] = str(
        input_path.resolve()
    )
    save_json(proof_result, output_paths["proof"])

    return {
        "case_id": case_id,
        "input_path": input_path,
        "output_paths": output_paths,
        "concepts_result": concepts_result,
        "actions_result": actions_result,
        "reasoning_result": reasoning_result,
        "mitre_result": mitre_result,
        "attack_chain_result": attack_chain_result,
        "proof_result": proof_result,
    }


def print_summary(
    pipeline_result: dict[str, Any],
) -> None:
    """Affiche une synthèse SOC compacte du pipeline."""
    case_id = pipeline_result["case_id"]
    concepts_result = pipeline_result["concepts_result"]
    actions_result = pipeline_result["actions_result"]
    reasoning_result = pipeline_result["reasoning_result"]
    mitre_result = pipeline_result["mitre_result"]
    attack_chain_result = pipeline_result[
        "attack_chain_result"
    ]
    proof_result = pipeline_result["proof_result"]

    prediction = concepts_result.get("prediction", {})
    probability = safe_float(
        prediction.get("probability")
    )
    conclusion = proof_result.get("C_conclusion", {})
    audit = proof_result.get("audit", {})

    print("=== Pipeline neurosymbolique explicable ===")
    print(f"Cas         : {case_id}")
    print(
        f"Prédiction  : {prediction.get('label')} "
        f"(p={probability:.4f})"
    )

    coverage = concepts_result["coverage"]
    print(
        f"Concepts    : {coverage['mapped_features']}/"
        f"{coverage['total_features']} "
        f"({coverage['mapping_rate']:.0%})"
    )
    print(
        f"Actions     : "
        f"{actions_result['summary']['action_count']}"
    )
    print(
        f"Scallop     : "
        f"{reasoning_result['summary']['activated_pattern_count']} "
        f"motif(s) activé(s)"
    )
    print(
        f"MITRE       : "
        f"{mitre_result['summary']['candidate_count']} "
        f"candidat(s)"
    )
    print(
        f"Chaîne      : "
        f"{attack_chain_result['summary']['step_count']} "
        f"étape(s)"
    )
    print(f"Risque      : {conclusion.get('risk_level')}")
    print(f"Priorité SOC: {conclusion.get('soc_priority')}")
    print(f"Traçable    : {audit.get('traceable')}")
    print(f"Vérifiable  : {audit.get('verifiable')}")
    print()

    for pattern in reasoning_result.get(
        "activated_patterns",
        [],
    ):
        print(
            f"Motif       : {pattern.get('pattern_id')} → "
            f"{pattern.get('name') or pattern.get('pattern_name')}"
        )

    for candidate in mitre_result.get(
        "mitre_candidates",
        [],
    ):
        print(
            f"MITRE       : {candidate.get('technique_id')} → "
            f"{candidate.get('technique_name')}"
        )

    for index, step in enumerate(
        attack_chain_result.get("attack_chain", []),
        start=1,
    ):
        print(
            f"Étape {index:<3}  : {step.get('tactic')} → "
            f"{step.get('technique_id')} "
            f"({step.get('technique_name')})"
        )

    explanation = proof_result.get(
        "natural_language_explanation",
        {},
    )

    if explanation:
        print()
        print("Explication :")

        for key in (
            "summary",
            "behavior",
            "mitre",
            "attack_chain",
            "evidence",
            "caution",
        ):
            text = explanation.get(key)

            if text:
                print(f"- {text}")

    recommended_action = conclusion.get(
        "recommended_action"
    )

    if recommended_action:
        print()
        print("Action recommandée :")
        print(f"- {recommended_action}")

    print()
    print("Fichiers générés :")

    for output_name, output_path in pipeline_result[
        "output_paths"
    ].items():
        print(f"- {output_name}: {output_path.resolve()}")


def parse_arguments() -> argparse.Namespace:
    """Définit les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(
        description=(
            "Exécute le pipeline neurosymbolique explicable "
            "à partir d’un fichier XAI/COMAT."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help=(
            "Chemin du fichier XAI, par exemple "
            "outputs/comat/input_probe_01.json."
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
    """Point d’entrée du script."""
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
