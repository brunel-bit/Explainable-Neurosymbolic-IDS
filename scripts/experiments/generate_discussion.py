from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.run_explainable_pipeline import PROJECT_ROOT


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Fichier JSON introuvable : {path.resolve()}"
        )

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def format_percent(value: float) -> str:
    return f"{value:.1f}\\%"


def generate_discussion(statistics: dict[str, Any]) -> str:
    classification = statistics["classification"]
    pipeline = statistics["pipeline"]

    evaluated_cases = classification["evaluated_cases"]
    correct_predictions = classification["correct_predictions"]
    incorrect_predictions = classification["incorrect_predictions"]
    accuracy_percent = classification["accuracy_percent"]

    mean_probability = pipeline["mean_prediction_probability"]
    mapping_percent = pipeline["mean_mapping_rate_percent"]
    traceability_percent = pipeline["traceability_rate_percent"]
    verifiability_percent = pipeline["verifiability_rate_percent"]
    mitre_percent = pipeline["mitre_mapping_rate_percent"]
    pattern_percent = pipeline["pattern_activation_rate_percent"]

    class_results = classification["class_results"]

    correctly_recognized = [
        class_name
        for class_name, values in class_results.items()
        if values["class_accuracy"] == 1.0
    ]

    missed_classes = [
        class_name
        for class_name, values in class_results.items()
        if values["class_accuracy"] == 0.0
    ]

    recognized_text = ", ".join(correctly_recognized)
    missed_text = ", ".join(missed_classes)

    paragraphs = [
        (
            "Le pipeline neurosymbolique explicable a été évalué sur "
            f"{evaluated_cases} cas disposant d'une étiquette réelle. "
            f"Parmi ces cas, {correct_predictions} ont été correctement "
            f"classifiés et {incorrect_predictions} ont été mal classifiés, "
            f"ce qui correspond à une exactitude expérimentale de "
            f"{format_percent(accuracy_percent)}."
        ),
        (
            "Les classes correctement reconnues sont "
            f"{recognized_text}. À l'inverse, les classes "
            f"{missed_text} ont été confondues avec la classe Normal. "
            "Ces résultats indiquent que le modèle distingue correctement "
            "les comportements les plus représentés dans les données "
            "d'entraînement, mais demeure limité pour les classes rares."
        ),
        (
            "La probabilité moyenne associée aux prédictions est de "
            f"{mean_probability:.4f}. Cette valeur relativement élevée "
            "montre que les erreurs observées ne proviennent pas uniquement "
            "d'une faible confiance du modèle. Dans les cas R2L et U2R, "
            "le modèle produit en effet une prédiction Normal avec une "
            "confiance non négligeable."
        ),
        (
            "La couverture sémantique moyenne atteint "
            f"{format_percent(mapping_percent)}. Les cas DoS et Probe "
            "obtiennent une couverture complète, tandis que les cas "
            "prédits comme normaux présentent une couverture partielle. "
            "Cette différence s'explique par l'activation d'un plus grand "
            "nombre de concepts et d'actions sémantiques dans les scénarios "
            "d'attaque correctement identifiés."
        ),
        (
            "Les preuves produites sont traçables dans "
            f"{format_percent(traceability_percent)} des cas. "
            "Chaque conclusion peut ainsi être reliée aux observations "
            "d'origine, aux caractéristiques interprétées et aux règles "
            "ayant contribué au raisonnement."
        ),
        (
            "Le taux de vérifiabilité atteint "
            f"{format_percent(verifiability_percent)}. Les preuves sont "
            "considérées comme vérifiables lorsque le raisonnement active "
            "un motif comportemental et qu'une justification structurée "
            "peut être examinée par un analyste. Les cas classés comme "
            "normaux sans motif activé demeurent traçables, mais ne disposent "
            "pas du même niveau de justification symbolique."
        ),
        (
            "Une technique MITRE ATT\\&CK est associée à "
            f"{format_percent(mitre_percent)} des cas, et un motif "
            "comportemental est activé dans "
            f"{format_percent(pattern_percent)} des cas. "
            "Ces associations concernent principalement les scénarios "
            "DoS et Probe, pour lesquels le pipeline identifie une technique, "
            "une tactique, un niveau de risque élevé et une priorité SOC P1."
        ),
        (
            "Dans l'ensemble, ces résultats montrent que la contribution "
            "principale du pipeline ne réside pas uniquement dans la "
            "performance de classification. Le système enrichit les "
            "prédictions par une représentation sémantique, un raisonnement "
            "symbolique, une correspondance avec MITRE ATT\\&CK et une "
            "preuve explicable structurée. Les erreurs sur les classes R2L "
            "et U2R soulignent toutefois la nécessité d'améliorer le modèle "
            "de classification et d'élargir les règles sémantiques associées "
            "aux attaques rares."
        ),
    ]

    return "\n\n".join(paragraphs) + "\n"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Génère automatiquement une discussion LaTeX "
            "à partir des statistiques expérimentales."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs"
            / "experiments"
            / "statistics_summary.json"
        ),
        help="Fichier JSON contenant les statistiques.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs"
            / "experiments"
            / "results_discussion.tex"
        ),
        help="Fichier LaTeX généré.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    input_path = args.input
    output_path = args.output

    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path

    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    statistics = load_json(input_path)
    discussion = generate_discussion(statistics)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write(discussion)

    print("=== Discussion expérimentale générée ===")
    print(f"Entrée : {input_path.resolve()}")
    print(f"Sortie : {output_path.resolve()}")


if __name__ == "__main__":
    main()