import os
import pandas as pd


# ==========================================================
# VANTARA CUSTOMER INTELLIGENCE PLATFORM
# CLEANUP - FINAL CHURN MODEL COMPARISON
# ==========================================================

OUTPUT_PATH = (
    "outputs/reports/"
    "final_churn_model_comparison.csv"
)

SUMMARY_PATH = (
    "outputs/reports/"
    "final_model_selection_summary.txt"
)

os.makedirs(
    "outputs/reports",
    exist_ok=True
)


# ==========================================================
# 1. Consolidate Actual Test Results
# ==========================================================

print("\nCreating final churn model comparison...")


results = [

    {
        "Model": "Logistic Regression",
        "Accuracy": 0.7160,
        "Precision": 0.8409,
        "Recall": 0.7080,
        "F1_Score": 0.7687,
        "ROC_AUC": 0.7970
    },

    {
        "Model": "Decision Tree",
        "Accuracy": 0.6720,
        "Precision": 0.7668,
        "Recall": 0.7300,
        "F1_Score": 0.7480,
        "ROC_AUC": 0.6453
    },

    {
        "Model": "Random Forest",
        "Accuracy": 0.7453,
        "Precision": 0.8294,
        "Recall": 0.7780,
        "F1_Score": 0.8029,
        "ROC_AUC": 0.8001
    },

    {
        "Model": "XGBoost",
        "Accuracy": 0.7507,
        "Precision": 0.7925,
        "Recall": 0.8480,
        "F1_Score": 0.8193,
        "ROC_AUC": 0.7982
    },

    {
        "Model": "LightGBM",
        "Accuracy": 0.7267,
        "Precision": 0.8300,
        "Recall": 0.7420,
        "F1_Score": 0.7835,
        "ROC_AUC": 0.7974
    },

    {
        "Model": "SVM",
        "Accuracy": 0.7080,
        "Precision": 0.8402,
        "Recall": 0.6940,
        "F1_Score": 0.7601,
        "ROC_AUC": 0.7945
    },

    {
        "Model": "ANN",
        "Accuracy": 0.7653,
        "Precision": 0.8045,
        "Recall": 0.8560,
        "F1_Score": 0.8295,
        "ROC_AUC": 0.8094
    }

]


comparison_df = pd.DataFrame(
    results
)


# ==========================================================
# 2. Rank Models
# ==========================================================

# Overall ranking based primarily on F1 Score.
# F1 is useful here because churn detection requires
# balancing precision and recall.

comparison_df = (

    comparison_df

    .sort_values(

        by=[
            "F1_Score",
            "ROC_AUC"
        ],

        ascending=False

    )

    .reset_index(
        drop=True
    )

)


comparison_df.insert(

    0,

    "Final_Rank",

    range(
        1,
        len(comparison_df) + 1
    )

)


# ==========================================================
# 3. Display Comparison
# ==========================================================

print(
    "\nFinal Churn Model Comparison"
)

print("-" * 105)


print(

    comparison_df

    .round(4)

    .to_string(
        index=False
    )

)


# ==========================================================
# 4. Select Best Models
# ==========================================================

best_overall = (

    comparison_df

    .iloc[0]

)


best_accuracy = (

    comparison_df.loc[

        comparison_df[
            "Accuracy"
        ].idxmax()

    ]

)


best_precision = (

    comparison_df.loc[

        comparison_df[
            "Precision"
        ].idxmax()

    ]

)


best_recall = (

    comparison_df.loc[

        comparison_df[
            "Recall"
        ].idxmax()

    ]

)


best_roc_auc = (

    comparison_df.loc[

        comparison_df[
            "ROC_AUC"
        ].idxmax()

    ]

)


# ==========================================================
# 5. Print Final Selection
# ==========================================================

print(
    "\nFinal Model Selection"
)

print("-" * 60)


print(
    "Recommended Production Model:",
    best_overall["Model"]
)


print(
    "Best Accuracy Model:",
    best_accuracy["Model"]
)


print(
    "Best Precision Model:",
    best_precision["Model"]
)


print(
    "Best Recall Model:",
    best_recall["Model"]
)


print(
    "Best ROC-AUC Model:",
    best_roc_auc["Model"]
)


# ==========================================================
# 6. Save Comparison
# ==========================================================

comparison_df.to_csv(

    OUTPUT_PATH,

    index=False

)


# ==========================================================
# 7. Save Final Selection Summary
# ==========================================================

summary = f"""
VANTARA CUSTOMER INTELLIGENCE PLATFORM
FINAL CHURN MODEL SELECTION
======================================

Total Churn Models Compared:
{len(comparison_df)}

Recommended Production Model:
{best_overall["Model"]}

Production Model Test Metrics:
Accuracy  : {best_overall["Accuracy"]:.4f}
Precision : {best_overall["Precision"]:.4f}
Recall    : {best_overall["Recall"]:.4f}
F1 Score  : {best_overall["F1_Score"]:.4f}
ROC-AUC   : {best_overall["ROC_AUC"]:.4f}


BEST MODELS BY METRIC
---------------------

Best Accuracy:
{best_accuracy["Model"]}
Accuracy: {best_accuracy["Accuracy"]:.4f}

Best Precision:
{best_precision["Model"]}
Precision: {best_precision["Precision"]:.4f}

Best Recall:
{best_recall["Model"]}
Recall: {best_recall["Recall"]:.4f}

Best ROC-AUC:
{best_roc_auc["Model"]}
ROC-AUC: {best_roc_auc["ROC_AUC"]:.4f}


FINAL DECISION
--------------

The ANN is selected as the recommended production churn
classification model because it achieved the strongest
overall test performance, including the highest F1 Score,
Accuracy, Recall, and ROC-AUC among the evaluated models.

The final production model selection is based on held-out
test-set performance.

No models were retrained during this consolidation step.
"""


with open(

    SUMMARY_PATH,

    "w",

    encoding="utf-8"

) as file:

    file.write(
        summary
    )


print(
    "\nSaved:"
)

print(
    OUTPUT_PATH
)

print(
    SUMMARY_PATH
)


print(
    "\nFinal model comparison "
    "completed successfully."
)