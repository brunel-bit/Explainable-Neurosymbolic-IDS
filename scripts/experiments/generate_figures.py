from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from scripts.run_explainable_pipeline import PROJECT_ROOT


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
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
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError("Le fichier CSV ne contient aucune ligne.")

    return rows


def evaluation_rows(
    rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row.get("expected_label", "").strip()
    ]


def save_figure(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_classification_result_figure(
    rows: list[dict[str, str]],
    output_path: Path,
) -> None:
    rows = evaluation_rows(rows)

    correct = sum(
        safe_bool(row.get("is_correct")) is True
        for row in rows
    )
    incorrect = len(rows) - correct

    labels = ["Correctes", "Incorrectes"]
    values = [correct, incorrect]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, values)

    plt.title("Résultats de classification")
    plt.ylabel("Nombre de cas")
    plt.ylim(0, max(values) + 1)

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.05,
            str(value),
            ha="center",
            va="bottom",
        )

    save_figure(output_path)


def generate_probability_figure(
    rows: list[dict[str, str]],
    output_path: Path,
) -> None:
    rows = evaluation_rows(rows)

    case_ids = [row["case_id"] for row in rows]
    probabilities = [
        safe_float(row.get("prediction_probability"))
        for row in rows
    ]

    plt.figure(figsize=(8, 4.5))
    bars = plt.bar(case_ids, probabilities)

    plt.title("Probabilité de prédiction par cas")
    plt.xlabel("Cas expérimental")
    plt.ylabel("Probabilité")
    plt.ylim(0, 1.1)
    plt.xticks(rotation=25)

    for bar, value in zip(bars, probabilities):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    save_figure(output_path)


def generate_mapping_coverage_figure(
    rows: list[dict[str, str]],
    output_path: Path,
) -> None:
    rows = evaluation_rows(rows)

    case_ids = [row["case_id"] for row in rows]
    mapping_rates = [
        safe_float(row.get("mapping_rate")) * 100
        for row in rows
    ]

    plt.figure(figsize=(8, 4.5))
    bars = plt.bar(case_ids, mapping_rates)

    plt.title("Couverture sémantique par cas")
    plt.xlabel("Cas expérimental")
    plt.ylabel("Couverture sémantique (%)")
    plt.ylim(0, 110)
    plt.xticks(rotation=25)

    for bar, value in zip(bars, mapping_rates):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 2,
            f"{value:.0f} %",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    save_figure(output_path)


def generate_class_comparison_figure(
    rows: list[dict[str, str]],
    output_path: Path,
) -> None:
    rows = evaluation_rows(rows)

    case_ids = [row["case_id"] for row in rows]
    expected_labels = [row["expected_label"] for row in rows]
    predicted_labels = [row["prediction_label"] for row in rows]

    classes = sorted(
        set(expected_labels) | set(predicted_labels)
    )
    class_to_number = {
        class_name: index
        for index, class_name in enumerate(classes)
    }

    expected_values = [
        class_to_number[label]
        for label in expected_labels
    ]
    predicted_values = [
        class_to_number[label]
        for label in predicted_labels
    ]

    positions = list(range(len(case_ids)))
    width = 0.35

    plt.figure(figsize=(9, 4.8))

    plt.bar(
        [position - width / 2 for position in positions],
        expected_values,
        width=width,
        label="Classe réelle",
    )

    plt.bar(
        [position + width / 2 for position in positions],
        predicted_values,
        width=width,
        label="Classe prédite",
    )

    plt.title("Comparaison entre classes réelles et prédites")
    plt.xlabel("Cas expérimental")
    plt.ylabel("Classe")
    plt.xticks(positions, case_ids, rotation=25)
    plt.yticks(
        list(class_to_number.values()),
        list(class_to_number.keys()),
    )
    plt.legend()

    save_figure(output_path)


def generate_confusion_matrix_figure(
    rows: list[dict[str, str]],
    output_path: Path,
) -> None:
    rows = evaluation_rows(rows)

    expected_labels = [row["expected_label"] for row in rows]
    predicted_labels = [row["prediction_label"] for row in rows]

    classes = sorted(
        set(expected_labels) | set(predicted_labels)
    )

    class_to_index = {
        class_name: index
        for index, class_name in enumerate(classes)
    }

    matrix = [
        [0 for _ in classes]
        for _ in classes
    ]

    for expected, predicted in zip(
        expected_labels,
        predicted_labels,
    ):
        row_index = class_to_index[expected]
        column_index = class_to_index[predicted]
        matrix[row_index][column_index] += 1

    plt.figure(figsize=(7, 6))
    image = plt.imshow(matrix)
    plt.colorbar(image)

    plt.title("Matrice de confusion expérimentale")
    plt.xlabel("Classe prédite")
    plt.ylabel("Classe réelle")

    plt.xticks(
        range(len(classes)),
        classes,
        rotation=30,
    )
    plt.yticks(
        range(len(classes)),
        classes,
    )

    for row_index in range(len(classes)):
        for column_index in range(len(classes)):
            plt.text(
                column_index,
                row_index,
                str(matrix[row_index][column_index]),
                ha="center",
                va="center",
            )

    save_figure(output_path)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Génère les figures expérimentales du pipeline "
            "à partir du fichier batch_summary.csv."
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
        help="Chemin du résumé expérimental CSV.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs"
            / "experiments"
            / "figures"
        ),
        help="Dossier de sortie des figures.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    input_path = args.input
    output_dir = args.output_dir

    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path

    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    rows = load_rows(input_path)

    figures = {
        "classification_results": (
            output_dir / "classification_results.png"
        ),
        "prediction_probabilities": (
            output_dir / "prediction_probabilities.png"
        ),
        "semantic_coverage": (
            output_dir / "semantic_coverage.png"
        ),
        "class_comparison": (
            output_dir / "class_comparison.png"
        ),
        "confusion_matrix": (
            output_dir / "confusion_matrix.png"
        ),
    }

    generate_classification_result_figure(
        rows,
        figures["classification_results"],
    )

    generate_probability_figure(
        rows,
        figures["prediction_probabilities"],
    )

    generate_mapping_coverage_figure(
        rows,
        figures["semantic_coverage"],
    )

    generate_class_comparison_figure(
        rows,
        figures["class_comparison"],
    )

    generate_confusion_matrix_figure(
        rows,
        figures["confusion_matrix"],
    )

    print("=== Figures expérimentales générées ===")
    print(f"Entrée : {input_path.resolve()}")

    for name, path in figures.items():
        print(f"- {name}: {path.resolve()}")


if __name__ == "__main__":
    main()