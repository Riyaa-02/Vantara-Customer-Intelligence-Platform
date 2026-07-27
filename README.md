# Vantara Customer Intelligence Platform

## Overview

The Vantara Customer Intelligence Platform is an end-to-end machine learning project developed to analyze customer purchasing behaviour using the Online Retail II dataset. The project combines customer analytics, predictive modelling, customer segmentation, explainable AI, and an interactive dashboard to demonstrate how machine learning can support business decision-making.

The objective of the project is to help businesses identify customers who are likely to churn, estimate customer lifetime value (CLV), discover customer segments, detect unusual purchasing behaviour, and present insights through an easy-to-use web application.

---

## Problem Statement

Businesses collect large amounts of customer transaction data, but converting that data into meaningful business decisions is often challenging. This project addresses that problem by building a complete analytics pipeline that transforms raw transaction data into actionable insights.

The platform focuses on:

- Customer churn prediction
- Customer Lifetime Value (CLV) prediction
- Customer segmentation
- Business insight generation
- Explainable AI
- Interactive visualization

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

### Libraries

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

## Project Workflow

The project was developed in multiple stages.

### Day 1
- Dataset collection

### Day 2
- Data cleaning

### Day 3
- Exploratory Data Analysis

### Day 4
- Feature Engineering

### Day 5
- Data preprocessing

### Day 6
- Baseline machine learning models
- Customer segmentation

### Day 7
- Hyperparameter tuning

### Day 8
- Customer Lifetime Value prediction

### Day 9
- Business insights and clustering analysis

### Day 10
- Dashboard data preparation
- Model comparison

### Day 11
- Artificial Neural Network (ANN)

### Day 12
- LSTM purchase prediction

### Day 13
- Autoencoder-based anomaly detection

### Day 14
- SHAP explainability

### Day 15
- LIME explainability and Partial Dependence Plots

### Day 16
- FastAPI deployment artifacts

---

## Features

- Customer Churn Prediction
- Customer Lifetime Value Prediction
- Customer Segmentation
- Explainable AI using SHAP
- Local Prediction Explanation using LIME
- Partial Dependence Plots
- Autoencoder-based anomaly detection
- LSTM purchase forecasting
- Interactive Streamlit dashboard
- FastAPI prediction service

---

## Project Structure

```
api/                    FastAPI backend
data/                   Dataset and processed files
docs/                   Project documentation
frontend/               Frontend resources
models/                 Trained models
outputs/                Reports and visualizations
src/                    Source modules
app.py                  Streamlit dashboard
```

---

## Running the Project

Install dependencies

```bash
pip install -r requirements.txt
```

Run the project pipeline

```bash
python day1_download.py
python day2_clean.py
...
python day16_prepare_api_artifacts.py
```

Launch the dashboard

```bash
streamlit run app.py
```

Run the FastAPI server

```bash
uvicorn api.main:app --reload
```

---

## Explainable AI

To improve model transparency, the project includes:

- SHAP for global feature importance
- LIME for local prediction explanations
- Partial Dependence Plots for feature analysis

---

## Future Improvements

- Docker deployment
- Cloud deployment
- Real-time prediction API
- Continuous model retraining
- User authentication

---

## Author

**Riya Khairnar**

Bachelor of Engineering (Computer Engineering)

Mumbai, India