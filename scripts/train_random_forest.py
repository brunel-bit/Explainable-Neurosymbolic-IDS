from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
)

from src.detection.preprocessing import (
    load_nsl_kdd,
    map_attack_categories,
    preprocess_features,
)
from src.detection.random_forest import build_random_forest



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

model = build_random_forest()

print("Entraînement du Random Forest...")
model.fit(X_train, y_train)

print("Prédiction sur le jeu de test...")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)

print("\nAccuracy :")
print(f"{accuracy:.4f}")

print("\nBalanced accuracy :")
print(f"{balanced_accuracy:.4f}")

print("\nMatthews Correlation Coefficient :")
print(f"{mcc:.4f}")

print("\nRapport de classification :")
print(classification_report(y_test, y_pred, digits=4))

print("\nMatrice de confusion :")
print(confusion_matrix(y_test, y_pred))

print("\nClasses du modèle :")
print(model.classes_)

print("\nNombre de variables utilisées :")
print(len(feature_names))


from src.utils.model_io import save_model
print(model.classes_)
save_model(model, preprocessor, feature_names)