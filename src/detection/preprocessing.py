"""
Preprocessing utilities for the Explainable-Neurosymbolic-IDS framework.
"""

from pathlib import Path

import pandas as pd
from scipy.sparse import spmatrix
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


NSL_KDD_COLUMNS = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label",
    "difficulty",
]

CATEGORICAL_COLUMNS = [
    "protocol_type",
    "service",
    "flag",
]

TARGET_COLUMN = "label"

COLUMNS_TO_DROP = [
    "difficulty",
]


def load_nsl_kdd(
    train_path: str | Path,
    test_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the NSL-KDD training and testing datasets.
    """

    train_df = pd.read_csv(
        train_path,
        names=NSL_KDD_COLUMNS,
        header=None,
    )

    test_df = pd.read_csv(
        test_path,
        names=NSL_KDD_COLUMNS,
        header=None,
    )

    return train_df, test_df


ATTACK_CATEGORY_MAP = {
    "normal": "Normal",

    "back": "DoS",
    "land": "DoS",
    "neptune": "DoS",
    "pod": "DoS",
    "smurf": "DoS",
    "teardrop": "DoS",
    "mailbomb": "DoS",
    "apache2": "DoS",
    "processtable": "DoS",
    "udpstorm": "DoS",
    "worm": "DoS",

    "satan": "Probe",
    "ipsweep": "Probe",
    "nmap": "Probe",
    "portsweep": "Probe",
    "mscan": "Probe",
    "saint": "Probe",

    "guess_passwd": "R2L",
    "ftp_write": "R2L",
    "imap": "R2L",
    "phf": "R2L",
    "multihop": "R2L",
    "warezmaster": "R2L",
    "warezclient": "R2L",
    "spy": "R2L",
    "xlock": "R2L",
    "xsnoop": "R2L",
    "snmpguess": "R2L",
    "snmpgetattack": "R2L",
    "httptunnel": "R2L",
    "sendmail": "R2L",
    "named": "R2L",

    "buffer_overflow": "U2R",
    "loadmodule": "U2R",
    "rootkit": "U2R",
    "perl": "U2R",
    "sqlattack": "U2R",
    "xterm": "U2R",
    "ps": "U2R",
}


def map_attack_categories(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Map detailed NSL-KDD attack labels to five broad categories.

    The returned DataFrame is a copy of the original one.
    """

    mapped_df = dataframe.copy()

    mapped_df["label"] = mapped_df["label"].map(ATTACK_CATEGORY_MAP)

    unknown_labels = mapped_df["label"].isna()

    if unknown_labels.any():
        original_unknown = dataframe.loc[unknown_labels, "label"].unique()

        raise ValueError(
            "Unknown NSL-KDD labels found: "
            f"{sorted(original_unknown.tolist())}"
        )

    return mapped_df

def split_features_target(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate input features from the target label.

    The target column is stored in y, while the difficulty column
    is removed from the model inputs.
    """

    columns_to_remove = [TARGET_COLUMN, *COLUMNS_TO_DROP]

    X = dataframe.drop(columns=columns_to_remove).copy()
    y = dataframe[TARGET_COLUMN].copy()

    return X, y


def build_preprocessor() -> ColumnTransformer:
    """
    Build the preprocessing transformer for NSL-KDD.

    Categorical columns are one-hot encoded.
    Numerical columns are kept unchanged.
    """

    categorical_encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=True,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                categorical_encoder,
                CATEGORICAL_COLUMNS,
            ),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    return preprocessor     


def preprocess_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[
    spmatrix,
    spmatrix,
    pd.Series,
    pd.Series,
    ColumnTransformer,
    list[str],
]:
    """
    Prepare NSL-KDD features for machine-learning models.

    The preprocessor is fitted only on the training data and then
    applied to the testing data to prevent data leakage.
    """

    X_train_raw, y_train = split_features_target(train_df)
    X_test_raw, y_test = split_features_target(test_df)

    preprocessor = build_preprocessor()

    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    feature_names = preprocessor.get_feature_names_out().tolist()

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
        feature_names,
    )