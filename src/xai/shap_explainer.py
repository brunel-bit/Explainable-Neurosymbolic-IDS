"""
SHAP explanation utilities.
"""

from pathlib import Path

import joblib
import numpy as np
import shap


class ShapExplainer:
    """
    Compute SHAP explanations for the trained Random Forest.
    """

    def __init__(
        self,
        model_path: str | Path = "models/random_forest.joblib",
        feature_path: str | Path = "models/feature_names.joblib",
    ) -> None:
        """
        Load the trained model, feature names, and SHAP explainer.
        """

        self.model = joblib.load(model_path)
        self.feature_names = joblib.load(feature_path)

        self.explainer = shap.TreeExplainer(self.model)

    def explain_instance(
        self,
        instance: np.ndarray,
        top_k: int = 10,
    ) -> dict:
        """
        Explain one preprocessed network connection.

        Parameters
        ----------
        instance:
            A two-dimensional NumPy array with shape (1, n_features).

        top_k:
            Number of most important SHAP features to return.

        Returns
        -------
        dict
            Prediction, probability, predicted class index, and top features.
        """

        if not isinstance(instance, np.ndarray):
            instance = np.asarray(instance)

        if instance.ndim == 1:
            instance = instance.reshape(1, -1)

        if instance.shape[0] != 1:
            raise ValueError(
                "explain_instance expects exactly one observation. "
                f"Received shape: {instance.shape}"
            )

        if instance.shape[1] != len(self.feature_names):
            raise ValueError(
                "The number of values in the instance does not match "
                "the number of feature names. "
                f"Instance: {instance.shape[1]}, "
                f"feature names: {len(self.feature_names)}"
            )

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        shap_values = self.explainer(instance)

        prediction = self.model.predict(instance)[0]
        probabilities = self.model.predict_proba(instance)[0]

        predicted_index = int(np.argmax(probabilities))

        values = shap_values.values[0, :, predicted_index]

        order = np.argsort(np.abs(values))[::-1]

        features = []

        for index in order[:top_k]:
            shap_value = float(values[index])

            if shap_value > 0:
                direction = "supports_prediction"
            elif shap_value < 0:
                direction = "opposes_prediction"
            else:
                direction = "neutral"

            features.append(
                {
                    "feature": str(self.feature_names[index]),
                    "value": float(instance[0, index]),
                    "shap": shap_value,
                    "absolute_shap": abs(shap_value),
                    "direction": direction,
                }
            )

        return {
            "prediction": str(prediction),
            "predicted_index": predicted_index,
            "probability": float(probabilities[predicted_index]),
            "top_features": features,
        }