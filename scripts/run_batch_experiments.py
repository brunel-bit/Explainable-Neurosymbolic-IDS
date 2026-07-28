from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from scripts.run_explainable_pipeline import (
    PROJECT_ROOT,
    infer_case_id,
    run_pipeline,
    safe_float,
)

def infer_expected_label(case_id: str) -> str | None:
    normalized = case_id.lower()

    label_mapping = {
        "normal": "Normal",
        "dos": "DoS",
        "probe": "Probe",
        "r2l": "R2L",
        "u2r": "U2R",
    }

    for token, label in label_mapping.items():
        if token in normalized:
            return label

    return None


def extract_row(pipeline_result: dict[str, Any]) -> dict[str, Any]:
    """Transforme une exécution complète en une ligne synthétique."""
    concepts_result = pipeline_result["concepts_result"]
    actions_result = pipeline_result["actions_result"]
    reasoning_result = pipeline_result["reasoning_result"]
    mitre_result = pipeline_result["mitre_result"]
    attack_chain_result = pipeline_result["attack_chain_result"]
    proof_result = pipeline_result["proof_result"]

    prediction = concepts_result.get("prediction", {})
    conclusion = proof_result.get("C_conclusion", {})
    audit = proof_result.get("audit", {})

    activated_patterns = reasoning_result.get("activated_patterns", [])
    mitre_candidates = mitre_result.get("mitre_candidates", [])
    attack_chain = attack_chain_result.get("attack_chain", [])

    pattern_ids = [
        str(pattern.get("pattern_id"))
        for pattern in activated_patterns
        if pattern.get("pattern_id")
    ]
    pattern_names = [
        str(pattern.get("name") or pattern.get("pattern_name"))
        for pattern in activated_patterns
        if pattern.get("name") or pattern.get("pattern_name")
    ]
    technique_ids = [
        str(candidate.get("technique_id"))
        for candidate in mitre_candidates
        if candidate.get("technique_id")
    ]
    technique_names = [
        str(candidate.get("technique_name"))
        for candidate in mitre_candidates
        if candidate.get("technique_name")
    ]
    tactics = [
        str(step.get("tactic"))
        for step in attack_chain
        if step.get("tactic")
    ]

    coverage = concepts_result.get("coverage", {})
    action_summary = actions_result.get("summary", {})
    reasoning_summary = reasoning_result.get("summary", {})
    mitre_summary = mitre_result.get("summary", {})
    chain_summary = attack_chain_result.get("summary", {})


    case_id = pipeline_result["case_id"]
    expected_label = infer_expected_label(case_id)
    predicted_label = prediction.get("label")

    return {
        "case_id": case_id,
        "expected_label": expected_label,
        "prediction_label": predicted_label,
        "is_correct": (
            expected_label == predicted_label
            if expected_label is not None
            else None
        ),
        "prediction_probability": round(
            safe_float(prediction.get("probability")),
            4,
        ),
        "mapped_features": coverage.get("mapped_features", 0),
        "total_features": coverage.get("total_features", 0),
        "mapping_rate": coverage.get("mapping_rate", 0.0),
        "action_count": action_summary.get("action_count", 0),
        "supporting_actions": action_summary.get("supporting_actions", 0),
        "opposing_actions": action_summary.get("opposing_actions", 0),
        "activated_pattern_count": reasoning_summary.get(
            "activated_pattern_count",
            0,
        ),
        "pattern_ids": "; ".join(pattern_ids),
        "pattern_names": "; ".join(pattern_names),
        "mitre_candidate_count": mitre_summary.get("candidate_count", 0),
        "mitre_technique_ids": "; ".join(technique_ids),
        "mitre_technique_names": "; ".join(technique_names),
        "attack_chain_steps": chain_summary.get("step_count", 0),
        "tactics": "; ".join(tactics),
        "risk_level": conclusion.get("risk_level"),
        "soc_priority": conclusion.get("soc_priority"),
        "traceable": audit.get("traceable"),
        "verifiable": audit.get("verifiable"),
        "proof_file": str(
            pipeline_result["output_paths"]["proof"].resolve()
        ),
    }


def save_json(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=4, ensure_ascii=False)


def save_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    fieldnames = [
        "case_id",
        "expected_label",
        "prediction_label",
        "is_correct",
        "prediction_probability",
        "mapped_features",
        "total_features",
        "mapping_rate",
        "action_count",
        "supporting_actions",
        "opposing_actions",
        "activated_pattern_count",
        "pattern_ids",
        "pattern_names",
        "mitre_candidate_count",
        "mitre_technique_ids",
        "mitre_technique_names",
        "attack_chain_steps",
        "tactics",
        "risk_level",
        "soc_priority",
        "traceable",
        "verifiable",
        "proof_file",
    ]

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def find_inputs(input_dir: Path, pattern: str) -> list[Path]:
    return sorted(path for path in input_dir.glob(pattern) if path.is_file())


def run_batch(
    *,
    input_dir: Path,
    pattern: str,
    continue_on_error: bool,
) -> dict[str, Any]:
    input_files = find_inputs(input_dir, pattern)
    if not input_files:
        raise FileNotFoundError(
            f"Aucun fichier trouvé dans {input_dir.resolve()} "
            f"avec le motif {pattern!r}."
        )

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    print("=== Exécution expérimentale par lots ===")
    print(f"Dossier : {input_dir.resolve()}")
    print(f"Fichiers: {len(input_files)}")
    print()

    for index, input_path in enumerate(input_files, start=1):
        case_id = infer_case_id(input_path)
        print(f"[{index}/{len(input_files)}] {input_path.name} → {case_id}")

        try:
            pipeline_result = run_pipeline(
                input_path=input_path,
                case_id=case_id,
            )
            row = extract_row(pipeline_result)
            rows.append(row)
            print(
                "  OK | "
                f"{row['prediction_label']} | "
                f"MITRE={row['mitre_technique_ids'] or '-'} | "
                f"Risque={row['risk_level']} | "
                f"Priorité={row['soc_priority']}"
            )
        except Exception as exc:  # noqa: BLE001
            failure = {
                "case_id": case_id,
                "input_file": str(input_path.resolve()),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
            failures.append(failure)
            print(
                f"  ÉCHEC | {failure['error_type']}: "
                f"{failure['error_message']}"
            )
            if not continue_on_error:
                raise

    return {
        "input_directory": str(input_dir.resolve()),
        "input_pattern": pattern,
        "total_cases": len(input_files),
        "successful_cases": len(rows),
        "failed_cases": len(failures),
        "results": rows,
        "failures": failures,
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Exécute automatiquement le pipeline explicable "
            "sur tous les fichiers XAI/COMAT d'un dossier."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "comat",
        help="Dossier contenant les fichiers XAI/COMAT.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="input_*.json",
        help="Motif utilisé pour sélectionner les fichiers.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Interrompt l'exécution dès la première erreur.",
    )
    return parser.parse_args()


def main() -> None:
    print(__file__)
    args = parse_arguments()
    input_dir = args.input_dir
    if not input_dir.is_absolute():
        input_dir = PROJECT_ROOT / input_dir

    summary = run_batch(
        input_dir=input_dir,
        pattern=args.pattern,
        continue_on_error=not args.stop_on_error,
    )

    results_dir = PROJECT_ROOT / "outputs" / "experiments"
    json_path = results_dir / "batch_summary.json"
    csv_path = results_dir / "batch_summary.csv"

    print("Colonnes de la première ligne :")
    print(list(summary["results"][0].keys()))
    print(
        "Probabilité première ligne :",
        summary["results"][0].get("prediction_probability"),
    )

    save_json(summary, json_path)
    save_csv(summary["results"], csv_path)

    print()
    print("=== Résumé ===")
    print(f"Cas totaux : {summary['total_cases']}")
    print(f"Réussis    : {summary['successful_cases']}")
    print(f"Échecs     : {summary['failed_cases']}")
    print()
    print("Fichiers générés :")
    print(f"- JSON : {json_path.resolve()}")
    print(f"- CSV  : {csv_path.resolve()}")


if __name__ == "__main__":
    main()
