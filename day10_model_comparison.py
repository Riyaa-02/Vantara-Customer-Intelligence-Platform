# ==========================================================
# VANTARA CUSTOMER INTELLIGENCE PLATFORM
# DAY 10 - CONSOLIDATED MODEL COMPARISON
# ==========================================================

import os
import pandas as pd


# ==========================================================
# File Paths
# ==========================================================

BASELINE_PATH = (
    "outputs/reports/"
    "baseline_test_results.csv"
)

RF_XGB_PATH = (
    "outputs/reports/"
    "day7_test_comparison.csv"
)

LGBM_SVM_PATH = (
    "outputs/reports/"
    "day8_test_comparison.csv"
)

CLV_PATH = (
    "outputs/reports/"
    "day8_clv_regression_results.csv"
)

CLUSTERING_PATH = (
    "outputs/reports/"
    "day9_clustering_comparison.csv"
)

FINAL_CHURN_PATH = (
    "outputs/reports/"
    "day10_churn_model_comparison.csv"
)

FINAL_SUMMARY_PATH = (
    "outputs/reports/"
    "day10_week2_summary.txt"
)


os.makedirs(
    "outputs/reports",
    exist_ok=True
)


# ==========================================================
# 1. Load Churn Model Results
# ==========================================================

print("\nLoading churn model results...")

baseline_df = pd.read_csv(
    BASELINE_PATH
)

rf_xgb_df = pd.read_csv(
    RF_XGB_PATH
)

lgbm_svm_df = pd.read_csv(
    LGBM_SVM_PATH
)


# ==========================================================
# 2. Standardize Columns
# ==========================================================

required_columns = [
    "Model",
    "Accuracy",
    "Precision",
    "Recall",
    "F1_Score",
    "ROC_AUC"
]


baseline_df = baseline_df[
    required_columns
]

rf_xgb_df = rf_xgb_df[
    required_columns
]

lgbm_svm_df = lgbm_svm_df[
    required_columns
]


# ==========================================================
# 3. Combine All Churn Models
# ==========================================================

comparison_df = pd.concat(
    [
        baseline_df,
        rf_xgb_df,
        lgbm_svm_df
    ],
    ignore_index=True
)


# ==========================================================
# 4. Rank Models
# ==========================================================

comparison_df[
    "ROC_AUC_Rank"
] = (
    comparison_df[
        "ROC_AUC"
    ]
    .rank(
        ascending=False,
        method="min"
    )
    .astype(int)
)


comparison_df[
    "Recall_Rank"
] = (
    comparison_df[
        "Recall"
    ]
    .rank(
        ascending=False,
        method="min"
    )
    .astype(int)
)


comparison_df[
    "F1_Rank"
] = (
    comparison_df[
        "F1_Score"
    ]
    .rank(
        ascending=False,
        method="min"
    )
    .astype(int)
)


comparison_df[
    "Overall_Rank_Score"
] = (

    comparison_df[
        "ROC_AUC_Rank"
    ]

    +

    comparison_df[
        "Recall_Rank"
    ]

    +

    comparison_df[
        "F1_Rank"
    ]

)


comparison_df = (
    comparison_df
    .sort_values(
        by=[
            "Overall_Rank_Score",
            "ROC_AUC",
            "Recall"
        ],
        ascending=[
            True,
            False,
            False
        ]
    )
    .reset_index(
        drop=True
    )
)


comparison_df[
    "Final_Rank"
] = (
    comparison_df.index + 1
)


# ==========================================================
# 5. Display Churn Model Comparison
# ==========================================================

print(
    "\nConsolidated Churn Model Comparison"
)

print("-" * 110)


display_columns = [
    "Final_Rank",
    "Model",
    "Accuracy",
    "Precision",
    "Recall",
    "F1_Score",
    "ROC_AUC"
]


print(
    comparison_df[
        display_columns
    ]
    .round(4)
    .to_string(
        index=False
    )
)


# ==========================================================
# 6. Select Recommended Production Model
# ==========================================================

print(
    "\nSelecting recommended "
    "production model..."
)


best_overall_model = (
    comparison_df
    .iloc[0]
)


best_auc_model = (
    comparison_df
    .sort_values(
        "ROC_AUC",
        ascending=False
    )
    .iloc[0]
)


best_recall_model = (
    comparison_df
    .sort_values(
        "Recall",
        ascending=False
    )
    .iloc[0]
)


print(
    "Best Overall Model:",
    best_overall_model["Model"]
)

print(
    "Best ROC-AUC Model:",
    best_auc_model["Model"]
)

print(
    "Best Recall Model:",
    best_recall_model["Model"]
)


# ==========================================================
# 7. Load CLV Regression Results
# ==========================================================

print(
    "\nLoading CLV regression results..."
)


clv_df = pd.read_csv(
    CLV_PATH
)


print(
    "\nCLV Regression Results"
)

print("-" * 60)


print(
    clv_df
    .round(4)
    .to_string(
        index=False
    )
)


# ==========================================================
# 8. Load Clustering Results
# ==========================================================

print(
    "\nLoading clustering results..."
)


clustering_df = pd.read_csv(
    CLUSTERING_PATH
)


print(
    "\nClustering Model Comparison"
)

print("-" * 75)


print(
    clustering_df
    .round(4)
    .to_string(
        index=False
    )
)


# ==========================================================
# 9. Create Week 2 Summary
# ==========================================================

test_clv_row = (
    clv_df[
        clv_df[
            "Dataset"
        ] == "Test"
    ]
    .iloc[0]
)


best_cluster_row = (
    clustering_df
    .sort_values(
        by=[
            "Silhouette_Score",
            "Davies_Bouldin_Score"
        ],
        ascending=[
            False,
            True
        ]
    )
    .iloc[0]
)


summary_text = f"""
VANTARA CUSTOMER INTELLIGENCE PLATFORM
WEEK 2 CHECKPOINT SUMMARY
=====================================

CHURN CLASSIFICATION
--------------------
Total Models Compared: {len(comparison_df)}

Recommended Overall Model:
{best_overall_model["Model"]}

Recommended Model Metrics:
Accuracy  : {best_overall_model["Accuracy"]:.4f}
Precision : {best_overall_model["Precision"]:.4f}
Recall    : {best_overall_model["Recall"]:.4f}
F1 Score  : {best_overall_model["F1_Score"]:.4f}
ROC-AUC   : {best_overall_model["ROC_AUC"]:.4f}

Best ROC-AUC Model:
{best_auc_model["Model"]}
ROC-AUC: {best_auc_model["ROC_AUC"]:.4f}

Best Recall Model:
{best_recall_model["Model"]}
Recall: {best_recall_model["Recall"]:.4f}


CLV REGRESSION
--------------
Test MAE  : {test_clv_row["MAE"]:.2f}
Test RMSE : {test_clv_row["RMSE"]:.2f}
Test R2   : {test_clv_row["R2"]:.4f}


CUSTOMER SEGMENTATION
---------------------
Selected Model:
{best_cluster_row["Model"]}

Number of Clusters:
{int(best_cluster_row["Clusters"])}

Silhouette Score:
{best_cluster_row["Silhouette_Score"]:.4f}

Davies-Bouldin Score:
{best_cluster_row["Davies_Bouldin_Score"]:.4f}


WEEK 2 STATUS
-------------
Completed:
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- LightGBM
- SVM
- CLV Regression
- K-Means Clustering
- Gaussian Mixture Model
- Business-readable customer segments
- Consolidated model comparison

Next Phase:
Week 3 - Deep Learning and Explainability
"""


# ==========================================================
# 10. Save Final Outputs
# ==========================================================

comparison_df.to_csv(
    FINAL_CHURN_PATH,
    index=False
)


with open(
    FINAL_SUMMARY_PATH,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        summary_text
    )


# ==========================================================
# 11. Final Summary
# ==========================================================

print(
    "\nWeek 2 Checkpoint Summary"
)

print(
    summary_text
)


print(
    "\nSaved:"
)

print(
    FINAL_CHURN_PATH
)

print(
    FINAL_SUMMARY_PATH
)


print(
    "\nDay 10 Consolidated Model Comparison "
    "completed successfully."
)