from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from src.detection.preprocessing import (
    load_nsl_kdd,
    map_attack_categories,
    preprocess_features,
)
from src.integration.comat_adapter import (
    shap_json_to_comat,
)
from src.xai.shap_explainer import ShapExplainer
from src.xai.shap_export import (
    export_explanation_to_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "NSL-KDD"
    / "KDDTrain+.txt"
)

DEFAULT_TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "NSL-KDD"
    / "KDDTest+.txt"
)


def sanitize_case_id(case_id: str) -> str:
    """Nettoie l’identifiant utilisé dans les noms de fichiers."""
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


def normalize_label(label: Any) -> str:
    """Normalise une étiquette pour permettre une comparaison robuste."""
    return str(label).strip().lower()


def available_labels(y_test: Any) -> list[str]:
    """Retourne la liste ordonnée des classes présentes dans le test."""
    unique_labels: list[str] = []
    seen: set[str] = set()

    for value in y_test:
        label = str(value)

        if label not in seen:
            seen.add(label)
            unique_labels.append(label)

    return unique_labels


def find_first_label_index(
    y_test: Any,
    target_label: str,
    occurrence: int = 1,
) -> int:
    """
    Trouve l’index de la n-ième occurrence d’une classe réelle.

    occurrence=1 sélectionne le premier exemple de la classe.
    occurrence=2 sélectionne le deuxième, etc.
    """
    if occurrence < 1:
        raise ValueError(
            "--occurrence doit être supérieure ou égale à 1."
        )

    normalized_target = normalize_label(target_label)

    matching_indices = [
        index
        for index, label in enumerate(y_test)
        if normalize_label(label) == normalized_target
    ]

    if not matching_indices:
        labels = ", ".join(available_labels(y_test))

        raise ValueError(
            f"La classe '{target_label}' est absente du jeu de test. "
            f"Classes disponibles : {labels}"
        )

    if occurrence > len(matching_indices):
        raise ValueError(
            f"La classe '{target_label}' possède seulement "
            f"{len(matching_indices)} occurrence(s), mais "
            f"--occurrence={occurrence} a été demandé."
        )

    return matching_indices[occurrence - 1]


def resolve_instance_index(
    y_test: Any,
    sample_index: int | None,
    target_label: str | None,
    occurrence: int,
) -> int:
    """
    Détermine l’index à expliquer.

    Un seul mode est permis :
    - --sample-index
    - --target-label
    """
    if sample_index is not None and target_label is not None:
        raise ValueError(
            "Utilise soit --sample-index, soit --target-label, "
            "mais pas les deux simultanément."
        )

    if target_label is not None:
        return find_first_label_index(
            y_test=y_test,
            target_label=target_label,
            occurrence=occurrence,
        )

    resolved_index = (
        sample_index
        if sample_index is not None
        else 0
    )

    if resolved_index < 0 or resolved_index >= len(y_test):
        raise IndexError(
            f"Index invalide : {resolved_index}. "
            f"Le jeu de test contient {len(y_test)} instances."
        )

    return resolved_index


def infer_case_id(
    instance_index: int,
    target_label: str | None,
) -> str:
    """Construit un identifiant par défaut."""
    if target_label:
        normalized_label = re.sub(
            r"[^a-zA-Z0-9_-]+",
            "_",
            target_label.strip().lower(),
        ).strip("_")

        return f"{normalized_label}_{instance_index}"

    return f"case_{instance_index}"


def build_output_paths(
    case_id: str,
) -> tuple[Path, Path]:
    """Construit les chemins SHAP et COMAT."""
    shap_output_path = (
        PROJECT_ROOT
        / "outputs"
        / "shap"
        / f"instance_{case_id}.json"
    )

    comat_output_path = (
        PROJECT_ROOT
        / "outputs"
        / "comat"
        / f"input_{case_id}.json"
    )

    return shap_output_path, comat_output_path


def run_export(
    sample_index: int | None,
    target_label: str | None,
    occurrence: int,
    case_id: str | None,
    top_k: int,
    train_path: Path,
    test_path: Path,
) -> dict[str, Any]:
    """Calcule SHAP pour une instance réelle et exporte le format COMAT."""
    if not train_path.exists():
        raise FileNotFoundError(
            f"Jeu d’entraînement introuvable : "
            f"{train_path.resolve()}"
        )

    if not test_path.exists():
        raise FileNotFoundError(
            f"Jeu de test introuvable : "
            f"{test_path.resolve()}"
        )

    train_df, test_df = load_nsl_kdd(
        str(train_path),
        str(test_path),
    )

    train_df = map_attack_categories(train_df)
    test_df = map_attack_categories(test_df)

    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
        feature_names,
    ) = preprocess_features(
        train_df,
        test_df,
    )

    instance_index = resolve_instance_index(
        y_test=y_test,
        sample_index=sample_index,
        target_label=target_label,
        occurrence=occurrence,
    )

    resolved_case_id = sanitize_case_id(
        case_id
        if case_id
        else infer_case_id(
            instance_index=instance_index,
            target_label=target_label,
        )
    )

    shap_output_path, comat_output_path = (
        build_output_paths(resolved_case_id)
    )

    explainer = ShapExplainer()

    explanation = explainer.explain_instance(
        X_test[
            instance_index : instance_index + 1
        ],
        top_k=top_k,
    )

    true_label = str(
        y_test.iloc[instance_index]
    )

    predicted_label = str(
        explanation["prediction"]
    )

    is_correct = (
        normalize_label(predicted_label)
        == normalize_label(true_label)
    )

    shap_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    export_explanation_to_json(
        explanation=explanation,
        output_path=str(shap_output_path),
        instance_id=instance_index,
        true_label=true_label,
    )

    shap_json_to_comat(
        shap_json_path=shap_output_path,
        output_path=comat_output_path,
    )

    return {
        "case_id": resolved_case_id,
        "instance_index": instance_index,
        "true_label": true_label,
        "prediction": predicted_label,
        "probability": float(
            explanation["probability"]
        ),
        "correct": is_correct,
        "top_features": explanation.get(
            "top_features",
            [],
        ),
        "shap_output_path": shap_output_path,
        "comat_output_path": comat_output_path,
    }


def print_result(
    result: dict[str, Any],
) -> None:
    """Affiche un résumé de l’export."""
    print("=" * 68)
    print("=== Export d’un cas SHAP → COMAT ===")
    print(f"Cas         : {result['case_id']}")
    print(
        f"Index       : {result['instance_index']}"
    )
    print(
        f"Classe réelle : {result['true_label']}"
    )
    print(
        f"Prédiction    : {result['prediction']}"
    )
    print(
        f"Correct       : {result['correct']}"
    )
    print(
        f"Probabilité   : "
        f"{result['probability']:.4f}"
    )

    print()
    print("Principales caractéristiques SHAP")
    print("-" * 68)

    for feature in result["top_features"]:
        feature_name = str(
            feature.get("feature", "")
        )

        feature_value = float(
            feature.get("value", 0.0)
        )

        shap_value = float(
            feature.get("shap", 0.0)
        )

        direction = feature.get(
            "direction",
            "",
        )

        print(
            f"{feature_name:<38}"
            f"{feature_value:>10.4f}"
            f"{shap_value:>11.4f}"
            f"  {direction}"
        )

    print()
    print(
        f"SHAP enregistré  : "
        f"{result['shap_output_path'].resolve()}"
    )

    print(
        f"COMAT enregistré : "
        f"{result['comat_output_path'].resolve()}"
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sélectionne une instance NSL-KDD réelle, "
            "calcule son explication SHAP et l’exporte "
            "vers le format COMAT."
        )
    )

    selection_group = parser.add_mutually_exclusive_group()

    selection_group.add_argument(
        "--sample-index",
        type=int,
        default=None,
        help=(
            "Index exact de l’instance dans le jeu de test."
        ),
    )

    selection_group.add_argument(
        "--target-label",
        type=str,
        default=None,
        help=(
            "Classe réelle à sélectionner automatiquement, "
            "par exemple Normal, DoS, Probe, R2L ou U2R."
        ),
    )

    parser.add_argument(
        "--occurrence",
        type=int,
        default=1,
        help=(
            "Numéro de l’occurrence à sélectionner lorsque "
            "--target-label est utilisé. Valeur par défaut : 1."
        ),
    )

    parser.add_argument(
        "--case-id",
        type=str,
        default=None,
        help=(
            "Identifiant du cas utilisé dans les fichiers de sortie."
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help=(
            "Nombre de caractéristiques SHAP exportées. "
            "Valeur par défaut : 10."
        ),
    )

    parser.add_argument(
        "--train-path",
        type=Path,
        default=DEFAULT_TRAIN_PATH,
        help="Chemin vers KDDTrain+.txt.",
    )

    parser.add_argument(
        "--test-path",
        type=Path,
        default=DEFAULT_TEST_PATH,
        help="Chemin vers KDDTest+.txt.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    train_path = args.train_path

    if not train_path.is_absolute():
        train_path = PROJECT_ROOT / train_path

    test_path = args.test_path

    if not test_path.is_absolute():
        test_path = PROJECT_ROOT / test_path

    result = run_export(
        sample_index=args.sample_index,
        target_label=args.target_label,
        occurrence=args.occurrence,
        case_id=args.case_id,
        top_k=args.top_k,
        train_path=train_path,
        test_path=test_path,
    )

    print_result(result)


if __name__ == "__main__":
    main()