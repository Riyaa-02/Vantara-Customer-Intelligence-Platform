"""
==============================================================
VANTARA CUSTOMER INTELLIGENCE PLATFORM
DAY 4 - FEATURE ENGINEERING
==============================================================
"""

import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

print("=" * 70)
print("DAY 4 : FEATURE ENGINEERING")
print("=" * 70)

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

os.makedirs("data/processed", exist_ok=True)
os.makedirs("docs", exist_ok=True)

# ------------------------------------------------------------
# Load cleaned dataset
# ------------------------------------------------------------

df = pd.read_csv("data/interim/cleaned_data.csv")

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["Total"] = df["Quantity"] * df["Price"]

print(f"\nLoaded {len(df):,} transactions")

# ------------------------------------------------------------
# Churn Definition
# ------------------------------------------------------------

CUTOFF_DATE = pd.Timestamp("2011-06-01")
FUTURE_WINDOW = 90

print("\nCutoff Date :", CUTOFF_DATE.date())
print("Future Window :", FUTURE_WINDOW, "days")

df_before = df[df["InvoiceDate"] < CUTOFF_DATE].copy()
df_after = df[df["InvoiceDate"] >= CUTOFF_DATE].copy()

print(f"Historical Transactions : {len(df_before):,}")
print(f"Future Transactions     : {len(df_after):,}")

customers = df_before.groupby("Customer ID")

features = pd.DataFrame(index=customers.size().index)
features.index.name = "Customer ID"

print("\nBuilding customer features...")

# ============================================================
# RFM FEATURES
# ============================================================

print("Creating RFM Features...")

last_purchase = customers["InvoiceDate"].max()

features["Recency"] = (
    CUTOFF_DATE - last_purchase
).dt.days

features["Frequency"] = customers["Invoice"].nunique()

features["Monetary_Total"] = customers["Total"].sum()

features["Monetary_Average"] = customers["Total"].mean()

# ============================================================
# PURCHASE FEATURES
# ============================================================

print("Creating Purchase Features...")

features["Total_Quantity"] = customers["Quantity"].sum()

features["Average_Basket_Size"] = customers["Quantity"].mean()

features["Average_Order_Value"] = (
    features["Monetary_Total"] /
    features["Frequency"]
)

features["Unique_Products"] = (
    customers["StockCode"]
    .nunique()
)

# ------------------------------------------------------------
# Average Purchase Gap
# ------------------------------------------------------------

def average_gap(group):

    dates = (
        group["InvoiceDate"]
        .sort_values()
        .drop_duplicates()
    )

    if len(dates) < 2:
        return 0

    gaps = dates.diff().dt.days.dropna()

    return gaps.mean()

features["Average_Purchase_Gap"] = (
    df_before
    .groupby("Customer ID")
    .apply(average_gap, include_groups=False)
)

# ============================================================
# CUSTOMER PROFILE
# ============================================================

print("Creating Customer Profile...")

first_purchase = customers["InvoiceDate"].min()

last_purchase = customers["InvoiceDate"].max()

features["Customer_Lifespan_Days"] = (
    last_purchase -
    first_purchase
).dt.days

features["Country"] = (
    customers["Country"]
    .agg(lambda x: x.mode().iloc[0])
)

# ============================================================
# RETURN FEATURES
# ============================================================

print("Creating Return Features...")

total_transactions = customers.size()

returned = (
    df_before[df_before["Quantity"] < 0]
    .groupby("Customer ID")
    .size()
)

features["Return_Rate"] = (
    returned /
    total_transactions *
    100
).fillna(0)

# ============================================================
# PRICE SENSITIVITY
# ============================================================

features["Price_Sensitivity"] = (
    customers["Price"]
    .var()
    .fillna(0)
)

# ============================================================
# REVENUE FEATURES
# ============================================================

features["Historical_Revenue"] = (
    features["Monetary_Total"]
)

daily_value = (
    features["Historical_Revenue"] /
    (features["Customer_Lifespan_Days"] + 1)
)

features["Predicted_CLV"] = (
    daily_value * 365
)

clip_value = (
    features["Predicted_CLV"]
    .quantile(0.99)
)

features["Predicted_CLV"] = (
    features["Predicted_CLV"]
    .clip(upper=clip_value)
)

# ============================================================
# ENGAGEMENT SCORE
# ============================================================

print("Creating Engagement Score...")

rfm = features[
    [
        "Recency",
        "Frequency",
        "Monetary_Total"
    ]
].copy()

rfm["Recency"] = (
    rfm["Recency"].max()
    - rfm["Recency"]
)

scaler = MinMaxScaler(
    feature_range=(0,100)
)

scaled = scaler.fit_transform(rfm)

features["Engagement_Score"] = (
    scaled.mean(axis=1)
)
# ============================================================
# CHURN LABEL CREATION
# ============================================================

print("\n" + "=" * 70)
print("CREATING CHURN LABEL")
print("=" * 70)

active_window_end = CUTOFF_DATE + pd.Timedelta(days=90)

future_customers = (
    df_after[
        df_after["InvoiceDate"] <= active_window_end
    ]["Customer ID"]
    .unique()
)

features["Churn"] = (
    ~features.index.isin(future_customers)
).astype(int)

print(f"Customers : {len(features):,}")
print(f"Churned   : {features['Churn'].sum():,}")
print(f"Active    : {(features['Churn']==0).sum():,}")
print(f"Churn Rate: {features['Churn'].mean()*100:.2f}%")

# ============================================================
# LEAKAGE CHECKS
# ============================================================

print("\n" + "=" * 70)
print("LEAKAGE VALIDATION")
print("=" * 70)

assert df_before["InvoiceDate"].max() < CUTOFF_DATE
print("PASS : Historical features use only past data")

assert (
    df_after[
        df_after["InvoiceDate"] <= active_window_end
    ]["InvoiceDate"].min()
    >= CUTOFF_DATE
)
print("PASS : Churn labels use future window only")

assert features["Recency"].min() >= 0
print("PASS : Recency values are valid")

# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

print("\n" + "=" * 70)
print("TRAIN / VALIDATION / TEST SPLIT")
print("=" * 70)

X = features.drop(columns=["Churn"])
y = features["Churn"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=RANDOM_SEED,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=RANDOM_SEED,
    stratify=y_temp
)

train = pd.concat([X_train, y_train], axis=1)
validation = pd.concat([X_val, y_val], axis=1)
test = pd.concat([X_test, y_test], axis=1)

# ============================================================
# SAVE FILES
# ============================================================

features.to_csv(
    "data/processed/customer_features.csv"
)

train.to_csv(
    "data/processed/train.csv"
)

validation.to_csv(
    "data/processed/validation.csv"
)

test.to_csv(
    "data/processed/test.csv"
)

print("\nDatasets Saved Successfully")

# ============================================================
# FEATURE DOCUMENTATION
# ============================================================

documentation = """
VANTARA CUSTOMER INTELLIGENCE PLATFORM

FEATURE DOCUMENTATION

Customer ID
Unique customer identifier.

Recency
Days since customer's last purchase before cutoff date.

Frequency
Number of unique invoices placed by customer.

Monetary_Total
Total historical revenue generated.

Monetary_Average
Average transaction value.

Average_Order_Value
Average revenue per invoice.

Total_Quantity
Total items purchased.

Average_Basket_Size
Average quantity purchased per transaction.

Unique_Products
Number of distinct products purchased.

Average_Purchase_Gap
Average days between purchases.

Customer_Lifespan_Days
Days between first and last purchase.

Country
Most frequent purchasing country.

Return_Rate
Percentage of returned transactions.

Price_Sensitivity
Variance of product prices purchased.

Historical_Revenue
Historical revenue before cutoff.

Predicted_CLV
Simple annualized estimate of customer value.

Engagement_Score
Composite score based on Recency, Frequency and Monetary.

Churn
Target variable.

1 = Customer did not purchase during next 90 days.

0 = Customer purchased again within 90 days.
"""

with open(
    "docs/feature_documentation.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write(documentation)

print("Feature documentation saved.")

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FEATURE SUMMARY")
print("=" * 70)

print(features.describe())

print("\nFeature Columns")

for col in features.columns:
    print("-", col)

print("\n" + "=" * 70)
print("DAY 4 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nGenerated Files")

print("data/processed/customer_features.csv")
print("data/processed/train.csv")
print("data/processed/validation.csv")
print("data/processed/test.csv")
print("docs/feature_documentation.txt")