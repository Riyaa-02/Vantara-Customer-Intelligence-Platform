# Vantara Customer Intelligence Platform

## Overview

The Vantara Customer Intelligence Platform is an end-to-end AI-powered customer analytics solution designed to help businesses understand customer behaviour, identify churn risk, estimate Customer Lifetime Value (CLV), segment customers, and generate explainable business insights.

The platform combines machine learning, deep learning, customer segmentation, explainable AI, REST APIs, PostgreSQL persistence, and an interactive Streamlit dashboard into a reproducible and containerized analytics system.

---

## Key Features

- Customer Churn Prediction
- Customer Lifetime Value (CLV) Prediction
- Customer Segmentation
- Customer Risk Analysis
- SHAP Explainability
- LIME and Partial Dependence Plot (PDP) Analysis
- Customer-ID Based Explanation
- Batch CSV Prediction
- Prediction History
- Revenue and CLV Analytics
- Churn Risk Leaderboard
- FastAPI REST API
- PostgreSQL Database Integration
- Interactive Streamlit Dashboard
- Docker Compose Deployment
- Automated Data Pipeline
- Automated Testing and Code Quality Checks

---

## Technology Stack

### Programming Language
- Python 3.12

### Data Processing
- Pandas
- NumPy
- OpenPyXL

### Machine Learning
- Scikit-learn
- XGBoost
- LightGBM

### Deep Learning
- PyTorch

### Explainable AI
- SHAP
- LIME
- Partial Dependence Plots

### Backend
- FastAPI
- SQLAlchemy
- Uvicorn

### Database
- PostgreSQL

### Frontend
- Streamlit

### Visualization
- Matplotlib
- Seaborn

### Testing and Code Quality
- Pytest
- Pytest-Cov
- Ruff

### Deployment
- Docker
- Docker Compose

---

## Project Structure

```text
Vantara Customer Intelligence Platform/
|
|-- api/
|   |-- database/
|   |-- routers/
|   `-- main.py
|
|-- config/
|   `-- config.yaml
|
|-- data/
|   |-- raw/
|   |-- interim/
|   |-- processed/
|   `-- final/
|
|-- docs/
|   `-- diagrams/
|
|-- frontend/
|   |-- pages/
|   `-- dashboard.py
|
|-- models/
|-- models_artifacts/
|-- notebooks/
|-- outputs/
|-- src/
|-- tests/
|
|-- run_pipeline.py
|-- docker-compose.yml
|-- Dockerfile
|-- requirements.txt
|-- README.md
`-- .gitignore
```

---

## Data Pipeline

The production data pipeline processes the raw Online Retail II dataset through cleaning and customer-level feature engineering.

Pipeline flow:

```text
Raw Excel Dataset
        |
        v
Dataset Combination
        |
        v
Data Cleaning
        |
        v
Feature Engineering
        |
        v
Processed Customer Feature Table
```

Run the complete production pipeline with a single command:

```bash
python run_pipeline.py
```

The pipeline performs the required processing without notebook intervention and uses structured logging to report execution progress.

---

## Configuration

Project configuration is maintained in:

```text
config/config.yaml
```

The configuration contains data paths, modelling parameters, random seed, train/validation/test settings, churn window, target metrics, feature configuration, and artifact paths.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Riyaa-02/Vantara-Customer-Intelligence-Platform
```

### 2. Move into the project directory

```bash
cd "Vantara Customer Intelligence Platform"
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

Windows:

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the project root.

Example:

```env
DATABASE_URL=postgresql+psycopg2://postgres:YOUR_PASSWORD@localhost:5432/vantara_db
```

Do not commit the `.env` file or database credentials to GitHub.

---

## Running the Application Locally

### Start FastAPI

```bash
uvicorn api.main:app --reload
```

FastAPI is available at:

```text
http://127.0.0.1:8000
```

Swagger/OpenAPI documentation:

```text
http://127.0.0.1:8000/docs
```

### Start Streamlit

Open another terminal and run:

```bash
streamlit run frontend/dashboard.py
```

The dashboard is available at:

```text
http://localhost:8501
```

---

## Docker Deployment

The complete application can also be started using Docker Compose.

```bash
docker compose up --build
```

The Docker deployment starts:

- PostgreSQL database
- FastAPI backend
- Streamlit dashboard

Services:

```text
FastAPI:    http://localhost:8000
Swagger:    http://localhost:8000/docs
Streamlit:  http://localhost:8501
PostgreSQL: localhost:5432
```

Check running containers:

```bash
docker compose ps
```

Stop the application:

```bash
docker compose down
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | API health check |
| GET | `/metadata` | Model metadata |
| POST | `/predict` | Single customer churn prediction |
| POST | `/predict/explain` | Prediction with SHAP explanation |
| POST | `/predict/batch` | Batch CSV prediction |
| GET | `/predictions` | Prediction history |
| GET | `/dashboard/summary` | Dashboard analytics summary |

Interactive API documentation is available through Swagger at `/docs`.

---

## Dashboard

The Streamlit dashboard provides business-facing access to the customer intelligence platform.

Major dashboard capabilities include:

- Executive Overview
- Customer Churn Prediction
- Customer Explanation
- Customer-ID Lookup
- SHAP Feature Explanations
- Batch Prediction
- Prediction History
- Customer Segmentation
- Churn Risk Leaderboard
- Revenue Analytics
- Customer Lifetime Value Analytics
- System/API Status

---

## Machine Learning Models

Multiple classical machine-learning models were evaluated for churn prediction.

Models include:

- Logistic Regression
- Random Forest
- Decision Tree
- XGBoost
- LightGBM
- Support Vector Machine

Model selection was based on comparative evaluation rather than relying on a single algorithm.

---

## Deep Learning Models

The project also evaluates deep-learning approaches:

- Artificial Neural Network (ANN)
- Long Short-Term Memory (LSTM)
- Autoencoder

The Autoencoder is used for customer anomaly analysis, while the LSTM supports sequential purchase-behaviour modelling.

---

## Customer Segmentation

Customer segmentation is performed using clustering techniques including:

- K-Means
- Gaussian Mixture Model (GMM)

The resulting clusters are translated into business-readable customer profiles such as active, champion, dormant, and at-risk customer groups.

---

## Explainable AI

The platform uses explainable AI techniques to make model predictions understandable.

Implemented methods include:

- SHAP
- LIME
- Partial Dependence Plots (PDP)

The dashboard allows users to inspect customer-level churn explanations and understand which features contribute most strongly to an individual prediction.

---

## PostgreSQL Database

PostgreSQL is used for persistent application data.

The database supports:

- Customer profiles
- Customer segments
- Prediction history
- Churn probability
- Risk level
- Prediction type
- Prediction timestamp

SQLAlchemy is used for database integration.

---

## Testing

Run the automated test suite with:

```bash
pytest --cov=src --cov-report=term-missing
```

Final verified result:

```text
49 tests passed
77% source-code coverage
```

The project therefore exceeds the required 70% source-code coverage threshold.

---

## Code Quality

Ruff is used for Python linting and code-quality validation.

Run:

```bash
ruff check api src frontend tests
```

Final verification:

```text
All checks passed!
```

---

## Performance Verification

The application was locally benchmarked during final validation.

### API

Measured health-endpoint p95 latency:

```text
7.38 ms
```

Target:

```text
< 400 ms
```

### Dashboard Data API

Measured dashboard-summary p95 latency:

```text
6.46 ms
```

Both measurements were within the project performance targets in the local test environment.

---

## Documentation

Project documentation includes:

- Feature documentation
- EDA report
- Preprocessing report
- Model comparison outputs
- Architecture diagram
- Entity Relationship (ER) diagram
- Workflow diagram
- Explainability outputs

Diagrams are located under:

```text
docs/diagrams/
```

---

## Reproducibility

The project provides:

- `requirements.txt` for Python dependencies
- `config/config.yaml` for project configuration
- `run_pipeline.py` for single-command data processing
- Docker configuration for containerized execution
- Automated tests
- Fixed modelling random seed
- Structured pipeline logging

These components support reproducible execution from a clean project environment.

---

## Known Limitation

The CLV regression model achieved an R² of approximately 0.588 against the project target of 0.60.

The result is retained rather than artificially optimized because it represents the performance obtained from the available dataset and modelling pipeline. The small difference from the target is documented as a project limitation.

---

## Future Improvements

Potential future improvements include:

- Cloud deployment using AWS or Azure
- CI/CD automation
- Real-time streaming customer analytics
- Automated model retraining
- Model and data drift monitoring
- Additional customer behavioural features

---

## Author

**Riya Khairnar**

Computer Engineering Graduate

AI • Machine Learning • Data Analytics