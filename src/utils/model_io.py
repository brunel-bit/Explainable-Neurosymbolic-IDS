"""
Utilities for saving and loading trained models.
"""

from pathlib import Path

import joblib


def save_model(
    model,
    preprocessor,
    feature_names,
    output_dir: str | Path = "models",
) -> None:
    """
    Save the trained model and preprocessing objects.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        model,
        output_dir / "random_forest.joblib",
    )

    joblib.dump(
        preprocessor,
        output_dir / "preprocessor.joblib",
    )

    joblib.dump(
        feature_names,
        output_dir / "feature_names.joblib",
    )

    print(f"Objects saved in: {output_dir.resolve()}")