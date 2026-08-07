"""
======================================================================
VANTARA CUSTOMER INTELLIGENCE PLATFORM
FASTAPI APPLICATION
======================================================================
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.database.database import Base, engine
from api.routers.dashboard import router as dashboard_router
from api.routers.health import router as health_router
from api.routers.metadata import router as metadata_router
from api.routers.prediction import router as prediction_router

app = FastAPI(
    title="Vantara Customer Intelligence Platform",
    description="Customer behaviour and churn prediction API",
    version="1.0.0",
)

app.include_router(dashboard_router)

Base.metadata.create_all(bind=engine)


# Allow the HTML/CSS/JavaScript dashboard to call FastAPI.
# Live Server commonly uses ports 5500 or 5501.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:5501",
        "http://localhost:5501",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(health_router)
app.include_router(metadata_router)
app.include_router(prediction_router)


@app.get("/")
def root() -> dict[str, str]:
    """Return basic API information."""

    return {
        "project": "Vantara Customer Intelligence Platform",
        "version": "1.0.0",
        "status": "Running",
    }