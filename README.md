# Vantara Customer Intelligence Platform

## Overview

The Vantara Customer Intelligence Platform is an AI-powered analytics solution that helps businesses understand customer behavior and improve retention through machine learning and deep learning.

The platform predicts customer churn, estimates Customer Lifetime Value (CLV), segments customers into meaningful groups, and provides explainable AI insights using SHAP and LIME. It also includes a FastAPI backend, PostgreSQL database integration, and an interactive Streamlit dashboard for business users.

---

## Key Features

- Customer Churn Prediction
- Customer Lifetime Value (CLV) Prediction
- Customer Segmentation
- Explainable AI using SHAP
- LIME and Partial Dependence Plot (PDP) Analysis
- Deep Learning Models (ANN, LSTM, Autoencoder)
- FastAPI REST API
- PostgreSQL Prediction History
- Batch CSV Prediction
- Interactive Streamlit Dashboard

---

## Technologies Used

### Programming Language

- Python 3.12

### Machine Learning

- Scikit-learn
- XGBoost
- LightGBM

### Deep Learning

- PyTorch

### Explainable AI

- SHAP
- LIME

### Backend

- FastAPI
- SQLAlchemy
- PostgreSQL

### Frontend

- Streamlit

### Data Processing

- Pandas
- NumPy

### Visualization

- Matplotlib
- Seaborn

---

## Project Structure

```
Vantara Customer Intelligence Platform/
│
├── api/
├── frontend/
├── src/
├── data/
├── models/
├── reports/
├── notebooks/
├── requirements.txt
├── README.md
└── config.yaml
```

---

## Installation

Clone the repository

```bash
git clone <repository-url>
```

Move into the project directory

```bash
cd "Vantara Customer Intelligence Platform"
```

Create virtual environment

```bash
python -m venv venv
```

Activate environment

Windows

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the project root.

```
DATABASE_URL=postgresql+psycopg2://postgres:YOUR_PASSWORD@localhost:5432/vantara_db
```

---

## Run FastAPI

```bash
uvicorn api.main:app --reload
```

Swagger Documentation

```
http://127.0.0.1:8000/docs
```

---

## Run Streamlit Dashboard

```bash
streamlit run frontend/dashboard.py
```

---

## API Endpoints

| Method | Endpoint | Description |
|---------|----------|-------------|
| GET | /health | Health Check |
| GET | /metadata | Model Metadata |
| POST | /predict | Single Prediction |
| POST | /predict/explain | Prediction with SHAP Explanation |
| POST | /predict/batch | Batch CSV Prediction |
| GET | /predictions | Prediction History |

---

## Dashboard Features

- Customer Churn Prediction
- Customer Explanation
- Batch Prediction
- Prediction History
- Customer Segmentation
- Churn Leaderboard
- Revenue & CLV Analytics

---

## Machine Learning Models

### Classical Models

- Logistic Regression
- Random Forest
- Decision Tree
- XGBoost
- LightGBM
- Support Vector Machine

### Deep Learning Models

- Artificial Neural Network (ANN)
- Long Short-Term Memory (LSTM)
- Autoencoder

---

## Explainable AI

The platform provides transparent model predictions using:

- SHAP
- LIME
- Partial Dependence Plot (PDP)

Business users can understand why a customer is predicted to churn through feature importance and plain-language explanations.

---

## Database

Prediction history is stored in PostgreSQL, including:

- Customer inputs
- Prediction
- Churn probability
- Risk level
- Timestamp

---

## Future Improvements

- Docker Deployment
- Cloud Deployment (AWS/Azure)
- CI/CD Pipeline
- Real-time Prediction Service

---

## Author

**Riya Khairnar**

Computer Engineering Graduate

AI • Machine Learning • Data Analytics