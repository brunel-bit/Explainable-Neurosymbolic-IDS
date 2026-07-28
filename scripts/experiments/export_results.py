from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scripts.run_explainable_pipeline import PROJECT_ROOT


COMMANDS = [
    "scripts.run_batch_experiments",
    "scripts.experiments.generate_latex_table",
    "scripts.experiments.compute_statistics",
    "scripts.experiments.generate_figures",
    "scripts.experiments.generate_latex_figures",
    "scripts.experiments.generate_discussion",
]


def run_module(module_name: str) -> None:
    print()
    print("=" * 72)
    print(f"Exécution : python -m {module_name}")
    print("=" * 72)

    subprocess.run(
        [sys.executable, "-m", module_name],
        cwd=PROJECT_ROOT,
        check=True,
    )


def verify_outputs() -> list[Path]:
    experiments_dir = PROJECT_ROOT / "outputs" / "experiments"

    expected_outputs = [
        experiments_dir / "batch_summary.csv",
        experiments_dir / "batch_summary.json",
        experiments_dir / "results_table.tex",
        experiments_dir / "statistics_summary.json",
        experiments_dir / "statistics_table.tex",
        experiments_dir / "figures.tex",
        experiments_dir / "results_discussion.tex",
        experiments_dir / "figures" / "classification_results.png",
        experiments_dir / "figures" / "prediction_probabilities.png",
        experiments_dir / "figures" / "semantic_coverage.png",
        experiments_dir / "figures" / "class_comparison.png",
        experiments_dir / "figures" / "confusion_matrix.png",
    ]

    missing_outputs = [
        path
        for path in expected_outputs
        if not path.exists()
    ]

    if missing_outputs:
        missing_text = "\n".join(
            f"- {path.resolve()}"
            for path in missing_outputs
        )
        raise FileNotFoundError(
            "Certains artefacts attendus sont absents :\n"
            f"{missing_text}"
        )

    return expected_outputs


def main() -> None:
    print("=== Export complet des résultats expérimentaux ===")

    for module_name in COMMANDS:
        run_module(module_name)

    generated_outputs = verify_outputs()

    print()
    print("=" * 72)
    print("EXPORT TERMINÉ AVEC SUCCÈS")
    print("=" * 72)
    print(f"Artefacts vérifiés : {len(generated_outputs)}")

    for path in generated_outputs:
        print(f"- {path.resolve()}")


if __name__ == "__main__":
    main()