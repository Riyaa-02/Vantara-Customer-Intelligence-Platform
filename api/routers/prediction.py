import io

import pandas as pd
from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    UploadFile,
)
from fastapi.responses import StreamingResponse
from pydantic import ValidationError
from sqlalchemy.orm import Session

from api.database.database import get_db
from api.database.models import Prediction
from api.schemas.prediction_schema import CustomerInput
from api.schemas.response_schema import (
    PredictionHistoryResponse,
    PredictionResponse,
)
from src.explainability.shap_explainer import explain_customer
from src.models.ann_service import score_customer

router = APIRouter(tags=["Prediction"])


# ------------------------------------------------------------------
# Single-customer prediction endpoint
# ------------------------------------------------------------------

@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict churn for one customer",
)
def predict_churn(
    customer: CustomerInput,
    db: Session = Depends(get_db),
) -> PredictionResponse:
    """Generate and store a churn prediction for one customer."""

    try:
        result = score_customer(
            customer.model_dump()
        )

        prediction_record = Prediction(
            customer_id=None,
            churn_prediction=result["churn_prediction"],
            churn_probability=result["churn_probability"],
            risk_level=result["risk_level"],
            prediction_type="single",
        )

        db.add(prediction_record)
        db.commit()
        db.refresh(prediction_record)

        return PredictionResponse(
            churn_prediction=result["churn_prediction"],
            churn_probability=result["churn_probability"],
            risk_level=result["risk_level"],
        )

    except ValueError as error:
        db.rollback()

        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except HTTPException:
        db.rollback()
        raise

    except Exception as error:
        db.rollback()

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {error!s}",
        ) from error


# ------------------------------------------------------------------
# Single-customer prediction with SHAP explanation
# ------------------------------------------------------------------

@router.post(
    "/predict/explain",
    summary="Predict churn and explain the result",
)
def predict_churn_with_explanation(
    customer: CustomerInput,
    db: Session = Depends(get_db),
) -> dict:
    """
    Generate a churn prediction and return the most influential
    SHAP features for the customer.
    """

    try:
        customer_data = customer.model_dump()

        prediction_result = score_customer(
            customer_data
        )

        explanation_result = explain_customer(
            customer_data
        )

        prediction_record = Prediction(
            customer_id=None,
            churn_prediction=prediction_result[
                "churn_prediction"
            ],
            churn_probability=prediction_result[
                "churn_probability"
            ],
            risk_level=prediction_result[
                "risk_level"
            ],
            prediction_type="explain",
        )

        db.add(prediction_record)
        db.commit()
        db.refresh(prediction_record)

        return {
            "churn_prediction": prediction_result[
                "churn_prediction"
            ],
            "churn_probability": prediction_result[
                "churn_probability"
            ],
            "risk_level": prediction_result[
                "risk_level"
            ],
            "top_features": explanation_result[
                "top_features"
            ],
        }

    except ValueError as error:
        db.rollback()

        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except HTTPException:
        db.rollback()
        raise

    except Exception as error:
        db.rollback()

        raise HTTPException(
            status_code=500,
            detail=(
                "Prediction explanation failed: "
                f"{error!s}"
            ),
        ) from error


# ------------------------------------------------------------------
# Batch prediction endpoint
# ------------------------------------------------------------------

@router.post(
    "/predict/batch",
    summary="Predict churn for customers in a CSV file",
)
async def predict_churn_batch(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    """
    Accept a CSV file, score every customer, store successful predictions,
    and return a downloadable CSV.

    The uploaded CSV must contain the same fields required by CustomerInput.
    """

    filename = file.filename or ""

    if not filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported.",
        )

    try:
        file_content = await file.read()

        if not file_content:
            raise HTTPException(
                status_code=400,
                detail="The uploaded CSV file is empty.",
            )

        try:
            dataframe = pd.read_csv(
                io.BytesIO(file_content)
            )

        except Exception as error:
            raise HTTPException(
                status_code=400,
                detail=f"Unable to read CSV file: {error!s}",
            ) from error

        if dataframe.empty:
            raise HTTPException(
                status_code=400,
                detail=(
                    "The uploaded CSV does not contain "
                    "any customer rows."
                ),
            )

        required_columns = set(
            CustomerInput.model_fields.keys()
        )

        uploaded_columns = set(
            dataframe.columns
        )

        missing_columns = (
            required_columns - uploaded_columns
        )

        if missing_columns:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Missing required CSV columns: "
                    + ", ".join(
                        sorted(missing_columns)
                    )
                ),
            )

        churn_predictions = []
        churn_probabilities = []
        risk_levels = []
        row_errors = []

        database_records = []

        for row_number, row in dataframe.iterrows():
            try:
                row_data = row.to_dict()

                row_data = {
                    key: (
                        None
                        if pd.isna(value)
                        else value
                    )
                    for key, value in row_data.items()
                }

                validated_customer = CustomerInput(
                    **row_data
                )

                result = score_customer(
                    validated_customer.model_dump()
                )

                churn_predictions.append(
                    result["churn_prediction"]
                )

                churn_probabilities.append(
                    result["churn_probability"]
                )

                risk_levels.append(
                    result["risk_level"]
                )

                row_errors.append("")

                prediction_record = Prediction(
                    customer_id=None,
                    churn_prediction=result[
                        "churn_prediction"
                    ],
                    churn_probability=result[
                        "churn_probability"
                    ],
                    risk_level=result["risk_level"],
                    prediction_type="batch",
                )

                database_records.append(
                    prediction_record
                )

            except ValidationError as error:
                churn_predictions.append(None)
                churn_probabilities.append(None)
                risk_levels.append("Error")

                row_errors.append(
                    f"Row {row_number + 2}: "
                    f"{error.errors()}"
                )

            except ValueError as error:
                churn_predictions.append(None)
                churn_probabilities.append(None)
                risk_levels.append("Error")

                row_errors.append(
                    f"Row {row_number + 2}: "
                    f"{error!s}"
                )

            except HTTPException as error:
                churn_predictions.append(None)
                churn_probabilities.append(None)
                risk_levels.append("Error")

                row_errors.append(
                    f"Row {row_number + 2}: "
                    f"{error.detail}"
                )

            except Exception as error:
                churn_predictions.append(None)
                churn_probabilities.append(None)
                risk_levels.append("Error")

                row_errors.append(
                    f"Row {row_number + 2}: "
                    f"{error!s}"
                )

        if database_records:
            db.add_all(database_records)
            db.commit()

        dataframe["churn_prediction"] = (
            churn_predictions
        )

        dataframe["churn_probability"] = (
            churn_probabilities
        )

        dataframe["risk_level"] = (
            risk_levels
        )

        dataframe["prediction_error"] = (
            row_errors
        )

        output_stream = io.StringIO()

        dataframe.to_csv(
            output_stream,
            index=False,
        )

        output_stream.seek(0)

        response_headers = {
            "Content-Disposition": (
                'attachment; '
                'filename="batch_churn_predictions.csv"'
            )
        }

        return StreamingResponse(
            iter([output_stream.getvalue()]),
            media_type="text/csv",
            headers=response_headers,
        )

    except HTTPException:
        db.rollback()
        raise

    except Exception as error:
        db.rollback()

        raise HTTPException(
            status_code=500,
            detail=(
                "Batch prediction failed: "
                f"{error!s}"
            ),
        ) from error

    finally:
        await file.close()


# ------------------------------------------------------------------
# Prediction-history endpoint
# ------------------------------------------------------------------

@router.get(
    "/predictions",
    response_model=list[PredictionHistoryResponse],
    summary="Retrieve saved prediction history",
)
def get_prediction_history(
    limit: int = 100,
    db: Session = Depends(get_db),
) -> list[Prediction]:
    """Return the most recently stored predictions."""

    if limit < 1 or limit > 1000:
        raise HTTPException(
            status_code=400,
            detail="Limit must be between 1 and 1000.",
        )

    predictions = (
        db.query(Prediction)
        .order_by(Prediction.created_at.desc())
        .limit(limit)
        .all()
    )

    return predictions