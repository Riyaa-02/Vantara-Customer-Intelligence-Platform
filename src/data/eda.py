"""
Reusable exploratory-data-analysis functions for Vantara.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def prepare_eda_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare cleaned retail data for exploratory analysis.
    """
    prepared_df = df.copy()

    prepared_df["InvoiceDate"] = pd.to_datetime(
        prepared_df["InvoiceDate"],
        errors="coerce",
    )

    prepared_df["Total"] = (
        prepared_df["Quantity"] * prepared_df["Price"]
    )

    return prepared_df


def calculate_dataset_overview(df: pd.DataFrame) -> dict:
    """
    Calculate basic dataset-level information.
    """
    return {
        "transactions": len(df),
        "customers": df["Customer ID"].nunique(),
        "products": df["StockCode"].nunique(),
        "countries": df["Country"].nunique(),
        "start_date": df["InvoiceDate"].min().date(),
        "end_date": df["InvoiceDate"].max().date(),
    }


def calculate_revenue_summary(df: pd.DataFrame) -> dict:
    """
    Calculate overall revenue statistics.
    """
    return {
        "total_revenue": float(df["Total"].sum()),
        "average_order": float(df["Total"].mean()),
        "median_order": float(df["Total"].median()),
    }


def calculate_country_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate revenue, order, and customer statistics by country.
    """
    return (
        df.groupby("Country")
        .agg(
            Revenue=("Total", "sum"),
            Orders=("Invoice", "nunique"),
            Customers=("Customer ID", "nunique"),
        )
        .sort_values("Revenue", ascending=False)
    )


def calculate_product_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate quantity and revenue statistics by product.
    """
    return (
        df.groupby("Description_Clean")
        .agg(
            Quantity=("Quantity", "sum"),
            Revenue=("Total", "sum"),
        )
        .sort_values("Revenue", ascending=False)
    )


def calculate_customer_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate revenue and order statistics by customer.
    """
    return (
        df.groupby("Customer ID")
        .agg(
            Revenue=("Total", "sum"),
            Orders=("Invoice", "nunique"),
        )
        .sort_values("Revenue", ascending=False)
    )


def calculate_monthly_sales(df: pd.DataFrame) -> pd.Series:
    """
    Calculate total revenue for each calendar month.
    """
    return (
        df.groupby(
            df["InvoiceDate"].dt.to_period("M")
        )["Total"]
        .sum()
    )


def calculate_return_summary(df: pd.DataFrame) -> dict:
    """
    Calculate returned-transaction count and percentage.
    """
    returned_transactions = int((df["Quantity"] < 0).sum())

    return_percentage = (
        returned_transactions / len(df) * 100
        if len(df) > 0
        else 0.0
    )

    return {
        "return_transactions": returned_transactions,
        "return_percentage": return_percentage,
    }


def calculate_business_insights(
    country_stats: pd.DataFrame,
    product_stats: pd.DataFrame,
    customer_stats: pd.DataFrame,
    monthly_sales: pd.Series,
    total_revenue: float,
    return_percentage: float,
) -> dict:
    """
    Calculate the principal business-insight values.
    """
    top_country = country_stats.index[0]
    top_country_revenue = float(
        country_stats.iloc[0]["Revenue"]
    )

    country_share = (
        top_country_revenue / total_revenue * 100
        if total_revenue != 0
        else 0.0
    )

    return {
        "top_country": top_country,
        "top_country_revenue": top_country_revenue,
        "country_share": country_share,
        "top_customer": customer_stats.index[0],
        "top_customer_revenue": float(
            customer_stats.iloc[0]["Revenue"]
        ),
        "top_product": product_stats.index[0],
        "top_product_revenue": float(
            product_stats.iloc[0]["Revenue"]
        ),
        "peak_month": monthly_sales.idxmax(),
        "peak_revenue": float(monthly_sales.max()),
        "return_percentage": return_percentage,
    }


def save_bar_chart(
    data: pd.Series,
    output_path: str | Path,
    title: str,
    xlabel: str,
    ylabel: str,
    figsize: tuple[int, int] = (12, 6),
    rotation: int = 0,
) -> None:
    """
    Save a bar chart to disk.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=figsize)
    data.plot(kind="bar")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_histogram(
    data: pd.Series,
    output_path: str | Path,
    title: str,
    xlabel: str,
    bins: int = 50,
) -> None:
    """
    Save a histogram to disk.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def create_eda_report(
    overview: dict,
    revenue: dict,
    returns: dict,
    insights: dict,
) -> str:
    """
    Create the text used in the EDA report.
    """
    return f"""
============================================================
VANTARA CUSTOMER INTELLIGENCE PLATFORM
EDA REPORT
============================================================

DATASET SUMMARY
-----------------------
Transactions : {overview['transactions']:,}
Customers    : {overview['customers']:,}
Products     : {overview['products']:,}
Countries    : {overview['countries']:,}

Date Range
-----------------------
{overview['start_date']} to {overview['end_date']}

REVENUE
-----------------------
Total Revenue : £{revenue['total_revenue']:,.2f}
Average Order : £{revenue['average_order']:,.2f}
Median Order  : £{revenue['median_order']:,.2f}

TOP COUNTRY
-----------------------
{insights['top_country']}
Revenue Share : {insights['country_share']:.2f}%

TOP CUSTOMER
-----------------------
Customer ID : {insights['top_customer']}
Revenue : £{insights['top_customer_revenue']:,.2f}

TOP PRODUCT
-----------------------
{insights['top_product']}
Revenue : £{insights['top_product_revenue']:,.2f}

RETURNS
-----------------------
Transactions : {returns['return_transactions']:,}
Return Rate : {returns['return_percentage']:.2f}%

PEAK SALES MONTH
-----------------------
{insights['peak_month']}
Revenue : £{insights['peak_revenue']:,.2f}

BUSINESS INSIGHTS
-----------------------
1. Revenue is concentrated in a few countries.
2. High-value customers contribute a significant portion of revenue.
3. Sales show seasonal variation.
4. Returns should be used as a predictive feature.
5. Revenue distribution is highly skewed.
6. Top-selling products should be prioritized for inventory planning.
"""


def create_hypotheses_text() -> str:
    """
    Return the machine-learning hypotheses identified during EDA.
    """
    return """
HYPOTHESES FOR MACHINE LEARNING

H1. Customers with high Recency are more likely to churn.

H2. Customers with low Frequency are more likely to churn.

H3. High Monetary customers deserve retention priority.

H4. Return behaviour influences churn probability.

H5. Seasonal shoppers behave differently from regular customers.

H6. Customer location influences purchasing behaviour.

H7. Purchase history can predict future customer value.
"""


def save_text_file(
    text: str,
    output_path: str | Path,
) -> None:
    """
    Save text content using UTF-8 encoding.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def run_eda(
    input_file: str | Path,
    output_directory: str | Path = "docs",
) -> dict:
    """
    Run the complete EDA pipeline and save all required artifacts.
    """
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_file)
    df = prepare_eda_data(df)

    overview = calculate_dataset_overview(df)
    revenue = calculate_revenue_summary(df)
    country_stats = calculate_country_statistics(df)
    product_stats = calculate_product_statistics(df)
    customer_stats = calculate_customer_statistics(df)
    monthly_sales = calculate_monthly_sales(df)
    returns = calculate_return_summary(df)

    insights = calculate_business_insights(
        country_stats=country_stats,
        product_stats=product_stats,
        customer_stats=customer_stats,
        monthly_sales=monthly_sales,
        total_revenue=revenue["total_revenue"],
        return_percentage=returns["return_percentage"],
    )

    save_bar_chart(
        country_stats.head(10)["Revenue"],
        output_directory / "country_revenue.png",
        "Top 10 Countries by Revenue",
        "Country",
        "Revenue (£)",
    )

    save_bar_chart(
        product_stats.head(10)["Revenue"],
        output_directory / "top_products.png",
        "Top 10 Products by Revenue",
        "Product",
        "Revenue (£)",
        figsize=(14, 6),
    )

    save_bar_chart(
        customer_stats.head(10)["Revenue"],
        output_directory / "top_customers.png",
        "Top 10 Customers by Revenue",
        "Customer ID",
        "Revenue (£)",
    )

    save_bar_chart(
        monthly_sales,
        output_directory / "monthly_sales.png",
        "Monthly Sales Trend",
        "Month",
        "Revenue (£)",
        figsize=(14, 6),
        rotation=45,
    )

    save_histogram(
        df["Quantity"],
        output_directory / "quantity_distribution.png",
        "Quantity Distribution",
        "Quantity",
    )

    save_histogram(
        df["Price"],
        output_directory / "price_distribution.png",
        "Price Distribution",
        "Price (£)",
    )

    report = create_eda_report(
        overview,
        revenue,
        returns,
        insights,
    )

    save_text_file(
        report,
        output_directory / "eda_report.txt",
    )

    save_text_file(
        create_hypotheses_text(),
        output_directory / "hypotheses.txt",
    )

    return {
        "dataframe": df,
        "overview": overview,
        "revenue": revenue,
        "country_stats": country_stats,
        "product_stats": product_stats,
        "customer_stats": customer_stats,
        "monthly_sales": monthly_sales,
        "returns": returns,
        "insights": insights,
    }