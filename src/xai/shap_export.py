"""
Utilities for exporting SHAP explanations.
"""

import json
from pathlib import Path
from typing import Any


def export_explanation_to_json(
    explanation: dict[str, Any],
    output_path: str | Path,
    *,
    instance_id: str | int | None = None,
    true_label: str | None = None,
) -> Path:
    """
    Export one SHAP explanation to a JSON file.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    exported_data = {
        "instance_id": instance_id,
        "true_label": true_label,
        "prediction": explanation["prediction"],
        "predicted_index": explanation["predicted_index"],
        "probability": explanation["probability"],
        "correct": (
            explanation["prediction"] == true_label
            if true_label is not None
            else None
        ),
        "top_features": explanation["top_features"],
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            exported_data,
            file,
            indent=4,
            ensure_ascii=False,
        )

    return output_path