"""
============================================================
VANTARA CUSTOMER INTELLIGENCE PLATFORM
DAY 3 - EXPLORATORY DATA ANALYSIS
============================================================

Purpose:
    - Understand customer purchasing behaviour
    - Generate business insights
    - Create visualizations
    - Save EDA report for documentation

Input:
    data/interim/cleaned_data.csv

Outputs:
    docs/
        monthly_sales.png
        country_revenue.png
        top_products.png
        top_customers.png
        quantity_distribution.png
        price_distribution.png
        hypotheses.txt
        eda_report.txt
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Create docs folder
# ============================================================

os.makedirs("docs", exist_ok=True)

# ============================================================
# Load Dataset
# ============================================================

print("=" * 60)
print("DAY 3 : EXPLORATORY DATA ANALYSIS")
print("=" * 60)

df = pd.read_csv("data/interim/cleaned_data.csv")

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["Total"] = df["Quantity"] * df["Price"]

print("\nDataset Loaded Successfully\n")

# ============================================================
# Dataset Overview
# ============================================================

transactions = len(df)
customers = df["Customer ID"].nunique()
products = df["StockCode"].nunique()
countries = df["Country"].nunique()

start_date = df["InvoiceDate"].min().date()
end_date = df["InvoiceDate"].max().date()

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)

print(f"Transactions : {transactions:,}")
print(f"Customers    : {customers:,}")
print(f"Products     : {products:,}")
print(f"Countries    : {countries:,}")
print(f"Date Range   : {start_date} to {end_date}")

# ============================================================
# Missing Values
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUE SUMMARY")
print("=" * 60)

missing = df.isnull().sum()

print(missing[missing > 0])

# ============================================================
# Statistical Summary
# ============================================================

print("\n" + "=" * 60)
print("STATISTICAL SUMMARY")
print("=" * 60)

print(df[["Quantity", "Price", "Total"]].describe())

# ============================================================
# Revenue Summary
# ============================================================

print("\n" + "=" * 60)
print("REVENUE SUMMARY")
print("=" * 60)

total_revenue = df["Total"].sum()
average_order = df["Total"].mean()
median_order = df["Total"].median()

print(f"Total Revenue : £{total_revenue:,.2f}")
print(f"Average Order : £{average_order:,.2f}")
print(f"Median Order  : £{median_order:,.2f}")

# ============================================================
# Country Analysis
# ============================================================

print("\n" + "=" * 60)
print("TOP COUNTRIES")
print("=" * 60)

country_stats = (
    df.groupby("Country")
      .agg(
          Revenue=("Total", "sum"),
          Orders=("Invoice", "nunique"),
          Customers=("Customer ID", "nunique")
      )
      .sort_values("Revenue", ascending=False)
)

print(country_stats.head(10))

plt.figure(figsize=(12,6))

country_stats.head(10)["Revenue"].plot(
    kind="bar"
)

plt.title("Top 10 Countries by Revenue")
plt.xlabel("Country")
plt.ylabel("Revenue (£)")
plt.tight_layout()

plt.savefig(
    "docs/country_revenue.png",
    dpi=200
)

plt.close()

# ============================================================
# Product Analysis
# ============================================================

print("\n" + "=" * 60)
print("TOP PRODUCTS")
print("=" * 60)

top_products = (
    df.groupby("Description_Clean")
      .agg(
          Quantity=("Quantity", "sum"),
          Revenue=("Total", "sum")
      )
      .sort_values("Revenue", ascending=False)
)

print(top_products.head(10))

plt.figure(figsize=(14,6))

top_products.head(10)["Revenue"].plot(
    kind="bar"
)

plt.title("Top 10 Products by Revenue")
plt.ylabel("Revenue (£)")
plt.tight_layout()

plt.savefig(
    "docs/top_products.png",
    dpi=200
)

plt.close()

# ============================================================
# Customer Analysis
# ============================================================

print("\n" + "=" * 60)
print("TOP CUSTOMERS")
print("=" * 60)

top_customers = (
    df.groupby("Customer ID")
      .agg(
          Revenue=("Total", "sum"),
          Orders=("Invoice", "nunique")
      )
      .sort_values("Revenue", ascending=False)
)

print(top_customers.head(10))

plt.figure(figsize=(12,6))

top_customers.head(10)["Revenue"].plot(
    kind="bar"
)

plt.title("Top 10 Customers by Revenue")
plt.ylabel("Revenue (£)")
plt.tight_layout()

plt.savefig(
    "docs/top_customers.png",
    dpi=200
)

plt.close()

# ============================================================
# Monthly Sales Trend
# ============================================================

print("\n" + "=" * 60)
print("MONTHLY SALES TREND")
print("=" * 60)

monthly_sales = (
    df.groupby(df["InvoiceDate"].dt.to_period("M"))["Total"]
      .sum()
)

print(monthly_sales.tail(12))

plt.figure(figsize=(14,6))

monthly_sales.plot(kind="bar")

plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Revenue (£)")
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "docs/monthly_sales.png",
    dpi=200
)

plt.close()

# ============================================================
# Returns Analysis
# ============================================================

print("\n" + "=" * 60)
print("RETURNS ANALYSIS")
print("=" * 60)

returns = df[df["Quantity"] < 0]

return_transactions = len(returns)
return_percent = (
    return_transactions / len(df)
) * 100

print(f"Returned Transactions : {return_transactions:,}")
print(f"Return Percentage     : {return_percent:.2f}%")

# ============================================================
# Quantity Distribution
# ============================================================

plt.figure(figsize=(10,6))

plt.hist(
    df["Quantity"],
    bins=50
)

plt.title("Quantity Distribution")
plt.xlabel("Quantity")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    "docs/quantity_distribution.png",
    dpi=200
)

plt.close()

# ============================================================
# Price Distribution
# ============================================================

plt.figure(figsize=(10,6))

plt.hist(
    df["Price"],
    bins=50
)

plt.title("Price Distribution")
plt.xlabel("Price (£)")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    "docs/price_distribution.png",
    dpi=200
)

plt.close()

# ============================================================
# Business Insights
# ============================================================

print("\n" + "=" * 60)
print("BUSINESS INSIGHTS")
print("=" * 60)

top_country = country_stats.index[0]
top_country_revenue = country_stats.iloc[0]["Revenue"]

country_share = (
    top_country_revenue /
    total_revenue
) * 100

top_customer = top_customers.index[0]
top_customer_revenue = top_customers.iloc[0]["Revenue"]

top_product = top_products.index[0]
top_product_revenue = top_products.iloc[0]["Revenue"]

peak_month = monthly_sales.idxmax()
peak_revenue = monthly_sales.max()

print(f"Top Country : {top_country}")
print(f"Revenue Contribution : {country_share:.2f}%")

print()

print(f"Top Customer : {top_customer}")
print(f"Revenue : £{top_customer_revenue:,.2f}")

print()

print(f"Top Product : {top_product}")
print(f"Revenue : £{top_product_revenue:,.2f}")

print()

print(f"Peak Sales Month : {peak_month}")
print(f"Revenue : £{peak_revenue:,.2f}")

print()

print(f"Return Rate : {return_percent:.2f}%")

# ============================================================
# Generate EDA Report
# ============================================================

report = f"""
============================================================
VANTARA CUSTOMER INTELLIGENCE PLATFORM
EDA REPORT
============================================================

DATASET SUMMARY
-----------------------
Transactions : {transactions:,}
Customers    : {customers:,}
Products     : {products:,}
Countries    : {countries:,}

Date Range
-----------------------
{start_date} to {end_date}

REVENUE
-----------------------
Total Revenue : £{total_revenue:,.2f}

Average Order : £{average_order:,.2f}

Median Order  : £{median_order:,.2f}

TOP COUNTRY
-----------------------
{top_country}

Revenue Share : {country_share:.2f}%

TOP CUSTOMER
-----------------------
Customer ID : {top_customer}

Revenue : £{top_customer_revenue:,.2f}

TOP PRODUCT
-----------------------
{top_product}

Revenue : £{top_product_revenue:,.2f}

RETURNS
-----------------------
Transactions : {return_transactions:,}

Return Rate : {return_percent:.2f}%

PEAK SALES MONTH
-----------------------
{peak_month}

Revenue : £{peak_revenue:,.2f}

BUSINESS INSIGHTS
-----------------------
1. Revenue is concentrated in a few countries.

2. High-value customers contribute a significant portion of revenue.

3. Sales show seasonal variation.

4. Returns should be used as a predictive feature.

5. Revenue distribution is highly skewed.

6. Top-selling products should be prioritized for inventory planning.
"""

with open(
    "docs/eda_report.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write(report)

print("\nEDA Report Saved")

# ============================================================
# Hypotheses
# ============================================================

hypotheses = """
HYPOTHESES FOR MACHINE LEARNING

H1. Customers with high Recency are more likely to churn.

H2. Customers with low Frequency are more likely to churn.

H3. High Monetary customers deserve retention priority.

H4. Return behaviour influences churn probability.

H5. Seasonal shoppers behave differently from regular customers.

H6. Customer location influences purchasing behaviour.

H7. Purchase history can predict future customer value.
"""

with open(
    "docs/hypotheses.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write(hypotheses)

print("Hypotheses Saved")

# ============================================================
# Completed
# ============================================================

print("\n" + "=" * 60)
print("DAY 3 COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nGenerated Files:")

print("docs/monthly_sales.png")
print("docs/country_revenue.png")
print("docs/top_products.png")
print("docs/top_customers.png")
print("docs/quantity_distribution.png")
print("docs/price_distribution.png")
print("docs/eda_report.txt")
print("docs/hypotheses.txt")