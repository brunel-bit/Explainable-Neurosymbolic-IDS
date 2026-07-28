from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

from scripts.run_explainable_pipeline import PROJECT_ROOT


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def safe_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value

    if value is None:
        return None

    normalized = str(value).strip().lower()

    if normalized in {"true", "1", "yes", "oui"}:
        return True

    if normalized in {"false", "0", "no", "non"}:
        return False

    return None


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Fichier CSV introuvable : {csv_path.resolve()}"
        )

    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    if not rows:
        raise ValueError("Le fichier CSV ne contient aucune ligne.")

    return rows


def filter_evaluation_rows(
    rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    """
    Exclut les cas sans étiquette réelle connue.

    Exemple :
    case_0 possède expected_label=None et ne doit donc pas être utilisé
    pour calculer l'exactitude de classification.
    """
    return [
        row
        for row in rows
        if row.get("expected_label", "").strip()
    ]


def compute_classification_statistics(
    rows: list[dict[str, str]],
) -> dict[str, Any]:
    evaluation_rows = filter_evaluation_rows(rows)

    total = len(evaluation_rows)

    correct = sum(
        1
        for row in evaluation_rows
        if safe_bool(row.get("is_correct")) is True
    )

    incorrect = total - correct

    accuracy = correct / total if total else 0.0

    class_results: dict[str, dict[str, Any]] = {}

    for row in evaluation_rows:
        expected = row.get("expected_label", "Unknown")
        predicted = row.get("prediction_label", "Unknown")
        is_correct = safe_bool(row.get("is_correct")) is True

        if expected not in class_results:
            class_results[expected] = {
                "total": 0,
                "correct": 0,
                "incorrect": 0,
                "predictions": {},
            }

        class_results[expected]["total"] += 1

        if is_correct:
            class_results[expected]["correct"] += 1
        else:
            class_results[expected]["incorrect"] += 1

        predictions = class_results[expected]["predictions"]
        predictions[predicted] = predictions.get(predicted, 0) + 1

    for class_data in class_results.values():
        class_total = class_data["total"]
        class_correct = class_data["correct"]

        class_data["class_accuracy"] = (
            class_correct / class_total
            if class_total
            else 0.0
        )

    return {
        "evaluated_cases": total,
        "correct_predictions": correct,
        "incorrect_predictions": incorrect,
        "accuracy": accuracy,
        "accuracy_percent": accuracy * 100,
        "class_results": class_results,
    }


def compute_pipeline_statistics(
    rows: list[dict[str, str]],
) -> dict[str, Any]:
    probabilities = [
        safe_float(row.get("prediction_probability"))
        for row in rows
    ]

    mapping_rates = [
        safe_float(row.get("mapping_rate"))
        for row in rows
    ]

    action_counts = [
        safe_int(row.get("action_count"))
        for row in rows
    ]

    pattern_counts = [
        safe_int(row.get("activated_pattern_count"))
        for row in rows
    ]

    mitre_counts = [
        safe_int(row.get("mitre_candidate_count"))
        for row in rows
    ]

    traceable_values = [
        safe_bool(row.get("traceable"))
        for row in rows
    ]

    verifiable_values = [
        safe_bool(row.get("verifiable"))
        for row in rows
    ]

    traceable_count = sum(
        value is True
        for value in traceable_values
    )

    verifiable_count = sum(
        value is True
        for value in verifiable_values
    )

    total = len(rows)

    attack_cases = [
        row
        for row in rows
        if row.get("prediction_label") != "Normal"
    ]

    normal_cases = [
        row
        for row in rows
        if row.get("prediction_label") == "Normal"
    ]

    mitre_mapped_cases = sum(
        1
        for row in rows
        if row.get("mitre_technique_ids", "").strip()
    )

    pattern_activated_cases = sum(
        1
        for row in rows
        if safe_int(row.get("activated_pattern_count")) > 0
    )

    high_risk_cases = sum(
        1
        for row in rows
        if row.get("risk_level", "").lower() == "high"
    )

    p1_cases = sum(
        1
        for row in rows
        if row.get("soc_priority", "").upper() == "P1"
    )

    return {
        "total_pipeline_cases": total,
        "mean_prediction_probability": (
            mean(probabilities)
            if probabilities
            else 0.0
        ),
        "mean_mapping_rate": (
            mean(mapping_rates)
            if mapping_rates
            else 0.0
        ),
        "mean_mapping_rate_percent": (
            mean(mapping_rates) * 100
            if mapping_rates
            else 0.0
        ),
        "mean_action_count": (
            mean(action_counts)
            if action_counts
            else 0.0
        ),
        "mean_activated_pattern_count": (
            mean(pattern_counts)
            if pattern_counts
            else 0.0
        ),
        "mean_mitre_candidate_count": (
            mean(mitre_counts)
            if mitre_counts
            else 0.0
        ),
        "traceable_cases": traceable_count,
        "traceability_rate": (
            traceable_count / total
            if total
            else 0.0
        ),
        "traceability_rate_percent": (
            traceable_count / total * 100
            if total
            else 0.0
        ),
        "verifiable_cases": verifiable_count,
        "verifiability_rate": (
            verifiable_count / total
            if total
            else 0.0
        ),
        "verifiability_rate_percent": (
            verifiable_count / total * 100
            if total
            else 0.0
        ),
        "predicted_attack_cases": len(attack_cases),
        "predicted_normal_cases": len(normal_cases),
        "mitre_mapped_cases": mitre_mapped_cases,
        "mitre_mapping_rate": (
            mitre_mapped_cases / total
            if total
            else 0.0
        ),
        "mitre_mapping_rate_percent": (
            mitre_mapped_cases / total * 100
            if total
            else 0.0
        ),
        "pattern_activated_cases": pattern_activated_cases,
        "pattern_activation_rate": (
            pattern_activated_cases / total
            if total
            else 0.0
        ),
        "pattern_activation_rate_percent": (
            pattern_activated_cases / total * 100
            if total
            else 0.0
        ),
        "high_risk_cases": high_risk_cases,
        "p1_cases": p1_cases,
    }


def generate_latex_statistics(
    statistics: dict[str, Any],
) -> str:
    classification = statistics["classification"]
    pipeline = statistics["pipeline"]

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{Statistiques globales du pipeline expérimental}",
        r"\label{tab:statistiques_globales_pipeline}",
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"\textbf{Indicateur} & \textbf{Valeur} \\",
        r"\midrule",
        (
            "Cas évalués pour la classification"
            f" & {classification['evaluated_cases']} \\\\"
        ),
        (
            "Prédictions correctes"
            f" & {classification['correct_predictions']} \\\\"
        ),
        (
            "Prédictions incorrectes"
            f" & {classification['incorrect_predictions']} \\\\"
        ),
        (
            "Exactitude expérimentale"
            f" & {classification['accuracy_percent']:.1f}\\% \\\\"
        ),
        (
            "Probabilité moyenne de prédiction"
            f" & {pipeline['mean_prediction_probability']:.4f} \\\\"
        ),
        (
            "Couverture sémantique moyenne"
            f" & {pipeline['mean_mapping_rate_percent']:.1f}\\% \\\\"
        ),
        (
            "Nombre moyen d'actions sémantiques"
            f" & {pipeline['mean_action_count']:.2f} \\\\"
        ),
        (
            "Nombre moyen de motifs activés"
            f" & {pipeline['mean_activated_pattern_count']:.2f} \\\\"
        ),
        (
            "Cas associés à une technique MITRE"
            f" & {pipeline['mitre_mapped_cases']} "
            f"({pipeline['mitre_mapping_rate_percent']:.1f}\\%) \\\\"
        ),
        (
            "Cas avec motif activé"
            f" & {pipeline['pattern_activated_cases']} "
            f"({pipeline['pattern_activation_rate_percent']:.1f}\\%) \\\\"
        ),
        (
            "Preuves traçables"
            f" & {pipeline['traceable_cases']} "
            f"({pipeline['traceability_rate_percent']:.1f}\\%) \\\\"
        ),
        (
            "Preuves vérifiables"
            f" & {pipeline['verifiable_cases']} "
            f"({pipeline['verifiability_rate_percent']:.1f}\\%) \\\\"
        ),
        (
            "Cas classés à risque élevé"
            f" & {pipeline['high_risk_cases']} \\\\"
        ),
        (
            "Cas classés en priorité P1"
            f" & {pipeline['p1_cases']} \\\\"
        ),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    return "\n".join(lines) + "\n"


def save_json(
    payload: dict[str, Any],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            indent=4,
            ensure_ascii=False,
        )


def save_text(
    content: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        file.write(content)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calcule les statistiques globales du pipeline "
            "à partir du résumé expérimental CSV."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs"
            / "experiments"
            / "batch_summary.csv"
        ),
        help="Chemin du fichier CSV expérimental.",
    )

    parser.add_argument(
        "--json-output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs"
            / "experiments"
            / "statistics_summary.json"
        ),
        help="Chemin du fichier JSON de statistiques.",
    )

    parser.add_argument(
        "--latex-output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs"
            / "experiments"
            / "statistics_table.tex"
        ),
        help="Chemin du tableau LaTeX.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    input_path = args.input
    json_output = args.json_output
    latex_output = args.latex_output

    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path

    if not json_output.is_absolute():
        json_output = PROJECT_ROOT / json_output

    if not latex_output.is_absolute():
        latex_output = PROJECT_ROOT / latex_output

    rows = load_rows(input_path)

    statistics = {
        "classification": compute_classification_statistics(rows),
        "pipeline": compute_pipeline_statistics(rows),
    }

    latex_content = generate_latex_statistics(statistics)

    save_json(statistics, json_output)
    save_text(latex_content, latex_output)

    print("=== Statistiques expérimentales générées ===")
    print(f"Cas totaux du pipeline : {len(rows)}")
    print(
        "Cas évalués          : "
        f"{statistics['classification']['evaluated_cases']}"
    )
    print(
        "Exactitude           : "
        f"{statistics['classification']['accuracy_percent']:.1f}%"
    )
    print(
        "Couverture moyenne   : "
        f"{statistics['pipeline']['mean_mapping_rate_percent']:.1f}%"
    )
    print(f"JSON                  : {json_output.resolve()}")
    print(f"LaTeX                 : {latex_output.resolve()}")


if __name__ == "__main__":
    main()