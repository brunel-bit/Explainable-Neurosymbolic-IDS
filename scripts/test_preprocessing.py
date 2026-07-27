from src.detection.preprocessing import (
    load_nsl_kdd,
    map_attack_categories,
    preprocess_features,
)


train_df, test_df = load_nsl_kdd(
    "data/raw/NSL-KDD/KDDTrain+.txt",
    "data/raw/NSL-KDD/KDDTest+.txt",
)

train_df = map_attack_categories(train_df)
test_df = map_attack_categories(test_df)

EXPECTED_CLASSES = {"Normal", "DoS", "Probe", "R2L", "U2R"}

assert set(train_df["label"].unique()) == EXPECTED_CLASSES
assert set(test_df["label"].unique()) == EXPECTED_CLASSES

assert train_df["label"].isna().sum() == 0
assert test_df["label"].isna().sum() == 0

(
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor,
    feature_names,
) = preprocess_features(train_df, test_df)

print("Dimensions avant encodage :")
print("Train :", train_df.shape)
print("Test  :", test_df.shape)

print("\nDimensions après encodage :")
print("X_train :", X_train.shape)
print("X_test  :", X_test.shape)
print("y_train :", y_train.shape)
print("y_test  :", y_test.shape)

print("\nType de X_train :")
print(type(X_train))

print("\nNombre de variables après encodage :")
print(len(feature_names))

print("\nPremières variables :")
for feature_name in feature_names[:20]:
    print("-", feature_name)

print("\nRépartition des classes d'entraînement :")
print(y_train.value_counts())

print("\nValidation du prétraitement réussie.")