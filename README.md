# Vantara Customer Intelligence Platform

## Overview

The Vantara Customer Intelligence Platform is an end-to-end machine learning project developed using the Online Retail II dataset. It analyzes customer purchasing behaviour to generate actionable business insights through predictive analytics, customer segmentation, explainable AI, and an interactive dashboard.

The platform helps businesses identify customers who are likely to churn, estimate Customer Lifetime Value (CLV), discover meaningful customer segments, detect unusual purchasing behaviour, and visualize insights through a user-friendly web application.

---

## Problem Statement

Businesses generate large volumes of customer transaction data, but transforming this data into meaningful business decisions remains a challenge. This project addresses that challenge by building a complete machine learning pipeline that converts raw transactional data into actionable insights for customer retention and business growth.

The platform focuses on:

- Customer Churn Prediction
- Customer Lifetime Value (CLV) Prediction
- Customer Segmentation
- Business Insights
- Explainable AI
- Interactive Data Visualization

---

## Dataset

This project uses the **Online Retail II** dataset from the UCI Machine Learning Repository.

**Reference**

Chen, D. (2012). *Online Retail II Dataset*. UCI Machine Learning Repository.

https://doi.org/10.24432/C5CG6D

---

## Technologies Used

### Programming Language

- Python

### Libraries & Frameworks

- Pandas
- NumPy
- Scikit-learn
- PyTorch
- XGBoost
- LightGBM
- SHAP
- LIME
- Matplotlib
- Streamlit
- FastAPI

---

## Development Workflow 

The project was developed in multiple stages.

### Day 1 – Dataset Collection
- Downloaded and organized the Online Retail II dataset.

### Day 2 – Data Cleaning
- Cleaned the dataset by handling missing values, duplicates, and inconsistent records.

### Day 3 – Exploratory Data Analysis
- Performed statistical analysis and data visualization to understand customer purchasing behaviour.

### Day 4 – Feature Engineering
- Created business-oriented features to improve model performance.

### Day 5 – Data Preprocessing
- Scaled data and prepared datasets for machine learning models.

### Day 6 – Customer Segmentation
- Built customer segments using K-Means clustering.

### Day 7 – Churn Prediction
- Developed and evaluated machine learning models for customer churn prediction.

### Day 8 – Customer Lifetime Value Prediction
- Built predictive models to estimate customer lifetime value.

### Day 9 – Business Insights
- Generated customer insights and cluster-based business recommendations.

### Day 10 – Model Comparison & Dashboard Preparation
- Compared multiple machine learning models and prepared data for visualization.

### Day 11 – Artificial Neural Network (ANN)
- Implemented an ANN model for churn prediction.

### Day 12 – LSTM Purchase Prediction
- Developed an LSTM model for purchase behaviour forecasting.

### Day 13 – Anomaly Detection
- Built an Autoencoder model to detect unusual customer behaviour.

### Day 14 – SHAP Explainability
- Applied SHAP to understand feature importance and model predictions.

### Day 15 – LIME Explainability
- Generated local prediction explanations using LIME and created Partial Dependence Plots.

### Day 16 – FastAPI Deployment
- Prepared trained models and artifacts for API deployment.

---

## Features

- Customer Churn Prediction
- Customer Lifetime Value (CLV) Prediction
- Customer Segmentation
- Explainable AI using SHAP
- Local Prediction Explanations using LIME
- Partial Dependence Plots
- Autoencoder-based Anomaly Detection
- LSTM Purchase Forecasting
- Interactive Streamlit Dashboard
- FastAPI Prediction API

---

## Project Structure

```text
Vantara Customer Intelligence Platform
│
├── api/                    FastAPI backend
├── config/                 Configuration files
├── data/                   Raw, processed, and final datasets
├── docs/                   Project documentation
├── frontend/               Frontend resources
├── models/                 Trained machine learning models
├── outputs/                Reports and visualizations
├── src/                    Source code modules
├── app.py                  Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## Running the Project

### 1. Clone the repository

```bash
git clone https://github.com/Riyaa-02/Vantara-Customer-Intelligence-Platform.git
cd Vantara-Customer-Intelligence-Platform
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Execute the project pipeline

```bash
python day1_download.py
python day2_clean.py
python day3_eda.py
python day4_features.py
python day5_preprocessing.py
...
python day16_prepare_api_artifacts.py
```

### 4. Launch the Streamlit Dashboard

```bash
streamlit run app.py
```

### 5. Run the FastAPI Server

```bash
uvicorn api.main:app --reload
```

---

## Explainable AI

The project includes Explainable AI techniques to improve model transparency and interpretability.

- **SHAP** for global feature importance and prediction explanations.
- **LIME** for local explanations of individual predictions.
- **Partial Dependence Plots** for analyzing the influence of important features.

---

## Project Outcomes

- Built an end-to-end customer analytics pipeline using the Online Retail II dataset.
- Implemented customer segmentation using K-Means clustering.
- Developed predictive models for customer churn and Customer Lifetime Value (CLV).
- Applied SHAP and LIME to improve model interpretability.
- Developed a FastAPI backend for model inference.
- Built an interactive Streamlit dashboard for visualization.

## Future Improvements

- Docker containerization
- Cloud deployment
- Real-time prediction API
- Automated model retraining
- User authentication and role management

---

## Author

**Riya Khairnar**

Bachelor of Engineering (Computer Engineering)

Mumbai, Maharashtra, India

GitHub: https://github.com/Riyaa-02

---

## Acknowledgements

- UCI Machine Learning Repository for providing the Online Retail II dataset.
- Open-source Python community for the libraries and frameworks used in this project.