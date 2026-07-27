import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    matthews_corrcoef,
)

from src.detection.preprocessing import (
    load_nsl_kdd,
    map_attack_categories,
    preprocess_features,
)
from src.detection.random_forest import (
    RANDOM_FOREST_CONFIGS,
    build_random_forest,
)


train_df, test_df = load_nsl_kdd(
    "data/raw/NSL-KDD/KDDTrain+.txt",
    "data/raw/NSL-KDD/KDDTest+.txt",
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
) = preprocess_features(train_df, test_df)

results = []

for config_name in RANDOM_FOREST_CONFIGS:
    print(f"\nEntraînement : {config_name}")

    model = build_random_forest(config_name=config_name)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0,
    )

    results.append(
        {
            "configuration": config_name,
            "accuracy": accuracy_score(y_test, y_pred),
            "balanced_accuracy": balanced_accuracy_score(
                y_test,
                y_pred,
            ),
            "macro_f1": f1_score(
                y_test,
                y_pred,
                average="macro",
            ),
            "mcc": matthews_corrcoef(y_test, y_pred),
            "recall_R2L": report["R2L"]["recall"],
            "recall_U2R": report["U2R"]["recall"],
        }
    )

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by=["macro_f1", "balanced_accuracy"],
    ascending=False,
)

print("\nComparaison des modèles :")
print(results_df.round(4).to_string(index=False))

results_df.to_csv(
    "outputs/random_forest_comparison.csv",
    index=False,
)

print(
    "\nRésultats enregistrés dans "
    "outputs/random_forest_comparison.csv"
)