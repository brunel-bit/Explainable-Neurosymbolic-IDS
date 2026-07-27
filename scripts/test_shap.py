
instance_index = 0

from src.xai.shap_export import export_explanation_to_json

from src.detection.preprocessing import (
    load_nsl_kdd,
    map_attack_categories,
    preprocess_features,
)

from src.xai.shap_explainer import ShapExplainer


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

explainer = ShapExplainer()

result = explainer.explain_instance(
    X_test[instance_index : instance_index + 1],
    top_k=10,
)

true_label = y_test.iloc[instance_index]
is_correct = result["prediction"] == true_label

print("=" * 60)

print("Prediction :", result["prediction"])

print(f"Probability : {result['probability']:.4f}")

print()

print("Top SHAP Features")

print("-" * 60)

print("True label :", true_label)
print("Prediction :", result["prediction"])
print("Correct    :", is_correct)
print(f"Probability : {result['probability']:.4f}")

output_path = export_explanation_to_json(
    explanation=result,
    output_path="outputs/shap/instance_0.json",
    instance_id=instance_index,
    true_label=str(true_label),
)

print()
print(f"Explanation saved to: {output_path}")

for feature in result["top_features"]:
    print(
        f"{feature['feature']:<35}"
        f"{feature['value']:>8.2f}"
        f"{feature['shap']:>12.4f}"
        f"  {feature['direction']}"
    )