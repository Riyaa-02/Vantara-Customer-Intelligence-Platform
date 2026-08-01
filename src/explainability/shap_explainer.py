import numpy as np
import shap
import torch

from src.models.ann_service import (
    feature_names,
    model,
    preprocess_customer,
)


def create_background_data(sample_size: int = 50) -> torch.Tensor:
    """
    Create a neutral background dataset for SHAP.

    Since the ANN numeric inputs are standardized, zero represents
    the average value for those features. Country one-hot features
    also remain zero in this neutral baseline.
    """

    background = np.zeros(
        (sample_size, len(feature_names)),
        dtype=np.float32,
    )

    return torch.tensor(
        background,
        dtype=torch.float32,
    )


def explain_customer(
    customer_data: dict,
    top_n: int = 5,
) -> dict:
    """
    Generate SHAP explanations for one customer.

    Parameters
    ----------
    customer_data:
        Raw customer features accepted by the ANN service.

    top_n:
        Number of highest-impact features to return separately.

    Returns
    -------
    dict
        Top feature contributions and all feature contributions.
    """

    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    processed_input = preprocess_customer(customer_data)

    input_tensor = torch.tensor(
        processed_input,
        dtype=torch.float32,
    )

    background_tensor = create_background_data()

    explainer = shap.DeepExplainer(
        model,
        background_tensor,
    )

    shap_values = explainer.shap_values(
        input_tensor,
    )

    values = np.asarray(shap_values).squeeze()

    if values.ndim != 1:
        values = values.reshape(-1)

    explanation_rows = []

    for feature, shap_value, processed_value in zip(
        feature_names,
        values,
        processed_input[0],
    ):
        shap_value = float(shap_value)

        if shap_value > 0:
            impact = "Increases churn risk"
        elif shap_value < 0:
            impact = "Decreases churn risk"
        else:
            impact = "No meaningful impact"

        explanation_rows.append(
            {
                "feature": feature,
                "shap_value": round(shap_value, 6),
                "processed_value": round(
                    float(processed_value),
                    6,
                ),
                "absolute_impact": round(
                    abs(shap_value),
                    6,
                ),
                "impact": impact,
            }
        )

    explanation_rows.sort(
        key=lambda row: row["absolute_impact"],
        reverse=True,
    )

    return {
        "top_features": explanation_rows[:top_n],
        "all_features": explanation_rows,
    }