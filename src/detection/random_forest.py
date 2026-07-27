"""
Random Forest utilities for intrusion detection.
"""

from sklearn.ensemble import RandomForestClassifier


RANDOM_FOREST_CONFIGS = {
    "rf_base": {
        "n_estimators": 200,
        "max_depth": 15,
        "min_samples_leaf": 1,
        "class_weight": "balanced_subsample",
    },
    "rf_balanced": {
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_leaf": 2,
        "class_weight": "balanced",
    },
    "rf_deep": {
        "n_estimators": 300,
        "max_depth": 25,
        "min_samples_leaf": 1,
        "class_weight": "balanced_subsample",
    },
}


def build_random_forest(
    config_name: str = "rf_balanced",
    random_state: int = 42,
) -> RandomForestClassifier:
    """
    Build a configured Random Forest classifier.
    """

    if config_name not in RANDOM_FOREST_CONFIGS:
        available_configs = ", ".join(RANDOM_FOREST_CONFIGS)

        raise ValueError(
            f"Unknown configuration: {config_name}. "
            f"Available configurations: {available_configs}"
        )

    config = RANDOM_FOREST_CONFIGS[config_name]

    return RandomForestClassifier(
        **config,
        random_state=random_state,
        n_jobs=-1,
    )