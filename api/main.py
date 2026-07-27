"""
======================================================================
VANTARA CUSTOMER INTELLIGENCE PLATFORM
DAY 16 : FASTAPI APPLICATION
======================================================================
"""

from fastapi import FastAPI

from api.routers.health import router as health_router
from api.routers.metadata import router as metadata_router
from api.routers.prediction import router as prediction_router

app = FastAPI(
    title="Vantara Customer Intelligence Platform",
    description="AI-powered Customer Churn Prediction API",
    version="1.0.0"
)

app.include_router(health_router)
app.include_router(metadata_router)
app.include_router(prediction_router)


@app.get("/")
def root():
    return {
        "project": "Vantara Customer Intelligence Platform",
        "version": "1.0.0",
        "status": "Running"
    }