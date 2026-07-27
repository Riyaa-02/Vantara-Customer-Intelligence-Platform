import joblib
from fastapi import APIRouter

router = APIRouter(tags=["Metadata"])


@router.get("/metadata")
def metadata():
    features = joblib.load("models/ann_feature_names.pkl")
    countries = joblib.load("models/ann_country_values.pkl")

    return {
        "total_features": len(features),
        "supported_countries": countries,
        "model": "ANN",
        "version": "1.0.0"
    }