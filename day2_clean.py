"""
Day 2 - Data Cleaning
"""

import pandas as pd

print("=" * 60)
print("DAY 2 : DATA CLEANING")
print("=" * 60)

# =====================================================
# Load Dataset
# =====================================================

INPUT_FILE = "data/raw/online_retail_ii_combined.csv"
OUTPUT_FILE = "data/interim/cleaned_data.csv"

df = pd.read_csv(INPUT_FILE)

original_rows = len(df)

print(f"Loaded {original_rows:,} rows")

# =====================================================
# Validate Required Columns
# =====================================================

required_columns = [
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country"
]

missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

print("✓ Required columns verified")

# =====================================================
# Safe Type Conversion
# =====================================================

df["Customer ID"] = pd.to_numeric(df["Customer ID"], errors="coerce")
df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"],
    errors="coerce"
)

# Remove rows with invalid essential values

before = len(df)

df = df.dropna(
    subset=[
        "Customer ID",
        "Quantity",
        "Price",
        "InvoiceDate"
    ]
).copy()

print(f"Removed invalid numeric/date rows : {before-len(df):,}")

# =====================================================
# Customer ID Validation
# =====================================================

null_rate = df["Customer ID"].isna().mean()

if null_rate > 0.30:
    raise ValueError(
        f"Customer ID null rate too high ({null_rate:.1%})"
    )

df["Customer ID"] = df["Customer ID"].astype(int)

print(f"Customer ID null rate : {null_rate:.2%}")

# =====================================================
# Remove Exact Duplicates
# =====================================================

before = len(df)

df = df.drop_duplicates().copy()

duplicates_removed = before - len(df)

print(f"Duplicates removed : {duplicates_removed:,}")

# =====================================================
# Remove Invalid Prices
# =====================================================

before = len(df)

df = df[df["Price"] > 0].copy()

bad_price_removed = before - len(df)

print(f"Rows with invalid price removed : {bad_price_removed:,}")

# =====================================================
# IQR Outlier Detection
# (Flag only, do NOT remove)
# =====================================================

Q1_qty = df["Quantity"].quantile(0.25)
Q3_qty = df["Quantity"].quantile(0.75)

IQR_qty = Q3_qty - Q1_qty

qty_lower = Q1_qty - 1.5 * IQR_qty
qty_upper = Q3_qty + 1.5 * IQR_qty

Q1_price = df["Price"].quantile(0.25)
Q3_price = df["Price"].quantile(0.75)

IQR_price = Q3_price - Q1_price

price_lower = Q1_price - 1.5 * IQR_price
price_upper = Q3_price + 1.5 * IQR_price

df["Qty_Outlier"] = (
    (df["Quantity"] < qty_lower)
    | (df["Quantity"] > qty_upper)
)

df["Price_Outlier"] = (
    (df["Price"] < price_lower)
    | (df["Price"] > price_upper)
)

print(f"Quantity Outliers : {df['Qty_Outlier'].sum():,}")
print(f"Price Outliers    : {df['Price_Outlier'].sum():,}")

# =====================================================
# Standardize Product Description
# =====================================================

description_lookup = (
    df.dropna(subset=["Description"])
      .groupby("StockCode")["Description"]
      .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
      .to_dict()
)

df["Description_Clean"] = (
    df["StockCode"]
    .map(description_lookup)
    .str.strip()
    .str.upper()
)

print(f"Unique Products : {len(description_lookup):,}")

# =====================================================
# Create Total Sales Column
# =====================================================

df["Total"] = df["Quantity"] * df["Price"]

# =====================================================
# Save Clean Dataset
# =====================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 60)
print("DATA CLEANING SUMMARY")
print("=" * 60)

print(f"Original Rows           : {original_rows:,}")
print(f"Final Rows              : {len(df):,}")
print(f"Duplicates Removed      : {duplicates_removed:,}")
print(f"Invalid Prices Removed  : {bad_price_removed:,}")
print(f"Unique Customers        : {df['Customer ID'].nunique():,}")
print(f"Unique Products         : {df['StockCode'].nunique():,}")
print(f"Returns Retained        : {(df['Quantity'] < 0).sum():,}")
print(f"Quantity Outliers       : {df['Qty_Outlier'].sum():,}")
print(f"Price Outliers          : {df['Price_Outlier'].sum():,}")

print("\nClean dataset saved successfully.")
print(f"Location : {OUTPUT_FILE}")