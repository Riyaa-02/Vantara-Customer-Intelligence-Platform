from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


sample_customer = {
    "Recency": 30,
    "Frequency": 8,
    "Monetary_Total": 1200,
    "Monetary_Average": 150,
    "Total_Quantity": 60,
    "Average_Basket_Size": 7,
    "Average_Order_Value": 150,
    "Unique_Products": 15,
    "Average_Purchase_Gap": 20,
    "Customer_Lifespan_Days": 365,
    "Return_Rate": 0.05,
    "Price_Sensitivity": 0.30,
    "Predicted_CLV": 4500,
    "Engagement_Score": 80,
    "Country": "United Kingdom",
}


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["api"] == "running"


def test_metadata_endpoint():

    response = client.get("/metadata")

    assert response.status_code == 200

    data = response.json()

    assert "total_features" in data
    assert "supported_countries" in data
    assert data["model"] == "ANN"


def test_prediction_history_endpoint():

    response = client.get("/predictions")

    assert response.status_code == 200

    assert isinstance(response.json(), list)


def test_prediction_history_invalid_limit():

    response = client.get("/predictions?limit=0")

    assert response.status_code == 400


def test_predict_endpoint():

    response = client.post(
        "/predict",
        json=sample_customer,
    )

    assert response.status_code == 200

    data = response.json()

    assert "churn_prediction" in data
    assert "churn_probability" in data
    assert "risk_level" in data


def test_predict_explain_endpoint():

    response = client.post(
        "/predict/explain",
        json=sample_customer,
    )

    assert response.status_code == 200

    data = response.json()

    assert "top_features" in data
    assert "risk_level" in data