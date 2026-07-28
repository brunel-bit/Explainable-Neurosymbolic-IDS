from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def latex_escape(value: Any) -> str:
    text = "" if value is None else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def normalize_bool(value: str | None) -> bool | None:
    if value is None:
        return None

    normalized = value.strip().lower()

    if normalized in {"true", "1", "yes", "oui"}:
        return True
    if normalized in {"false", "0", "no", "non"}:
        return False

    return None


def display_value(value: str | None, default: str = "--") -> str:
    if value is None:
        return default

    stripped = value.strip()
    return stripped if stripped else default


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    required_columns = {
        "case_id",
        "expected_label",
        "prediction_label",
        "is_correct",
        "prediction_probability",
        "mapping_rate",
        "action_count",
        "activated_pattern_count",
        "mitre_technique_ids",
        "risk_level",
        "soc_priority",
    }

    missing = required_columns.difference(reader.fieldnames or [])
    if missing:
        raise ValueError(
            "Colonnes manquantes dans le CSV : "
            + ", ".join(sorted(missing))
        )

    return rows


def select_rows(
    rows: list[dict[str, str]],
    *,
    include_unlabelled: bool,
) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []

    for row in rows:
        expected = display_value(row.get("expected_label"), default="")

        if not include_unlabelled and not expected:
            continue

        selected.append(row)

    return selected


def format_probability(value: str | None) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "--"


def format_percentage(value: str | None) -> str:
    try:
        return f"{100 * float(value):.1f}\\%"
    except (TypeError, ValueError):
        return "--"


def correctness_symbol(value: str | None) -> str:
    parsed = normalize_bool(value)

    if parsed is True:
        return r"\checkmark"
    if parsed is False:
        return r"$\times$"

    return "--"


def generate_latex(
    rows: list[dict[str, str]],
    *,
    caption: str,
    label: str,
) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        rf"\caption{{{latex_escape(caption)}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{llllllllrrc}",
        r"\toprule",
        (
            r"Cas & Réel & Prédit & Prob. & Couverture & "
            r"MITRE & Risque & Priorité & Actions & Motifs & Correct \\"
        ),
        r"\midrule",
    ]

    for row in rows:
        case_id = latex_escape(display_value(row.get("case_id")))
        expected = latex_escape(
            display_value(row.get("expected_label"))
        )
        predicted = latex_escape(
            display_value(row.get("prediction_label"))
        )
        probability = format_probability(
            row.get("prediction_probability")
        )
        mapping_rate = format_percentage(row.get("mapping_rate"))
        mitre = latex_escape(
            display_value(row.get("mitre_technique_ids"))
        )
        risk = latex_escape(display_value(row.get("risk_level")))
        priority = latex_escape(
            display_value(row.get("soc_priority"))
        )
        actions = latex_escape(
            display_value(row.get("action_count"), default="0")
        )
        patterns = latex_escape(
            display_value(
                row.get("activated_pattern_count"),
                default="0",
            )
        )
        correct = correctness_symbol(row.get("is_correct"))

        lines.append(
            f"{case_id} & {expected} & {predicted} & "
            f"{probability} & {mapping_rate} & {mitre} & "
            f"{risk} & {priority} & {actions} & "
            f"{patterns} & {correct} \\\\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\begin{flushleft}"
                r"\footnotesize "
                r"\textit{Note :} la dernière colonne indique la "
                r"correction de la classification "
                r"(\checkmark : correcte; $\times$ : incorrecte)."
                r"\end{flushleft}"
            ),
            r"\end{table}",
            "",
        ]
    )

    return "\n".join(lines)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Génère un tableau LaTeX à partir du résumé "
            "expérimental CSV."
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
        help="Fichier CSV produit par run_batch_experiments.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs"
            / "experiments"
            / "results_table.tex"
        ),
        help="Fichier LaTeX à générer.",
    )
    parser.add_argument(
        "--include-unlabelled",
        action="store_true",
        help=(
            "Inclut les cas dont la classe attendue est absente."
        ),
    )
    parser.add_argument(
        "--caption",
        default=(
            "Résultats expérimentaux du pipeline "
            "neurosymbolique explicable"
        ),
    )
    parser.add_argument(
        "--label",
        default="tab:resultats_pipeline_explicable",
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

    rows = load_rows(input_path)
    rows = select_rows(
        rows,
        include_unlabelled=args.include_unlabelled,
    )

    latex = generate_latex(
        rows,
        caption=args.caption,
        label=args.label,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(latex, encoding="utf-8")

    print("=== Tableau LaTeX généré ===")
    print(f"Cas inclus : {len(rows)}")
    print(f"Entrée     : {input_path.resolve()}")
    print(f"Sortie     : {output_path.resolve()}")


if __name__ == "__main__":
    main()
