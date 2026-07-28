from __future__ import annotations

import argparse
from pathlib import Path

from scripts.run_explainable_pipeline import PROJECT_ROOT


FIGURE_DEFINITIONS = [
    {
        "filename": "prediction_probabilities.png",
        "caption": (
            "Probabilité de prédiction obtenue pour chacun des cas "
            "expérimentaux."
        ),
        "label": "fig:prediction_probabilities",
        "width": "0.82\\textwidth",
    },
    {
        "filename": "semantic_coverage.png",
        "caption": (
            "Couverture sémantique obtenue après la transformation "
            "des caractéristiques en concepts interprétables."
        ),
        "label": "fig:semantic_coverage",
        "width": "0.82\\textwidth",
    },
    {
        "filename": "confusion_matrix.png",
        "caption": (
            "Matrice de confusion obtenue sur les cinq cas disposant "
            "d'une étiquette réelle."
        ),
        "label": "fig:confusion_matrix",
        "width": "0.72\\textwidth",
    },
    {
        "filename": "classification_results.png",
        "caption": (
            "Répartition des prédictions correctes et incorrectes "
            "sur les cas expérimentaux évalués."
        ),
        "label": "fig:classification_results",
        "width": "0.68\\textwidth",
    },
    {
        "filename": "class_comparison.png",
        "caption": (
            "Comparaison entre les classes réelles et les classes "
            "prédites par le modèle."
        ),
        "label": "fig:class_comparison",
        "width": "0.85\\textwidth",
    },
]


def latex_escape_path(path: str) -> str:
    return path.replace("\\", "/")


def generate_figure_block(
    *,
    image_path: str,
    caption: str,
    label: str,
    width: str,
) -> str:
    return "\n".join(
        [
            r"\begin{figure}[htbp]",
            r"\centering",
            rf"\includegraphics[width={width}]{{{image_path}}}",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\end{figure}",
        ]
    )


def generate_latex_content(
    figures_dir_name: str,
) -> str:
    blocks: list[str] = []

    for figure in FIGURE_DEFINITIONS:
        image_path = latex_escape_path(
            f"{figures_dir_name}/{figure['filename']}"
        )

        blocks.append(
            generate_figure_block(
                image_path=image_path,
                caption=figure["caption"],
                label=figure["label"],
                width=figure["width"],
            )
        )

    return "\n\n".join(blocks) + "\n"


def validate_figures(
    figures_dir: Path,
) -> None:
    missing_files = [
        figure["filename"]
        for figure in FIGURE_DEFINITIONS
        if not (figures_dir / figure["filename"]).exists()
    ]

    if missing_files:
        missing_text = ", ".join(missing_files)
        raise FileNotFoundError(
            "Figures manquantes dans "
            f"{figures_dir.resolve()} : {missing_text}"
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Génère les blocs LaTeX permettant d'insérer "
            "les figures expérimentales dans Overleaf."
        )
    )

    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs"
            / "experiments"
            / "figures"
        ),
        help="Dossier contenant les figures PNG.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs"
            / "experiments"
            / "figures.tex"
        ),
        help="Fichier LaTeX généré.",
    )

    parser.add_argument(
        "--latex-figures-dir",
        type=str,
        default="figures/experiments",
        help=(
            "Chemin des figures tel qu'il apparaîtra "
            "dans le projet Overleaf."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    figures_dir = args.figures_dir
    output_path = args.output

    if not figures_dir.is_absolute():
        figures_dir = PROJECT_ROOT / figures_dir

    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    validate_figures(figures_dir)

    latex_content = generate_latex_content(
        args.latex_figures_dir
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write(latex_content)

    print("=== Blocs LaTeX des figures générés ===")
    print(f"Figures vérifiées : {len(FIGURE_DEFINITIONS)}")
    print(f"Dossier source    : {figures_dir.resolve()}")
    print(f"Fichier LaTeX     : {output_path.resolve()}")
    print(
        "Chemin Overleaf   : "
        f"{args.latex_figures_dir}"
    )


if __name__ == "__main__":
    main()