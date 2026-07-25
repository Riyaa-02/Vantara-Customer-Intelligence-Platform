# ==========================================================
# VANTARA CUSTOMER INTELLIGENCE PLATFORM
# DAY 12 - LSTM PURCHASE TIMING PREDICTION
# ==========================================================

import os
import random
import copy
import numpy as np
import pandas as pd
import torch

from torch import nn
from torch.utils.data import (
    TensorDataset,
    DataLoader
)

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ==========================================================
# Configuration
# ==========================================================

SEED = 42

SEQUENCE_LENGTH = 5

PURCHASE_WINDOW_DAYS = 30

BATCH_SIZE = 64

MAX_EPOCHS = 50

PATIENCE = 7

LEARNING_RATE = 0.001


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


os.makedirs(
    "models",
    exist_ok=True
)

os.makedirs(
    "outputs/reports",
    exist_ok=True
)


# ==========================================================
# 1. Load Transaction Data
# ==========================================================

print("\nLoading transaction data...")

df = pd.read_csv(
    "data/interim/cleaned_data.csv"
)


df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"]
)


print(
    "Transaction Shape:",
    df.shape
)


# ==========================================================
# 2. Create Invoice-Level Purchase Data
# ==========================================================

print(
    "\nCreating invoice-level "
    "purchase sequences..."
)


orders = (

    df.groupby(
        [
            "Customer ID",
            "Invoice",
            "InvoiceDate"
        ]
    )

    .agg(

        Order_Amount=(
            "Total",
            "sum"
        ),

        Total_Quantity=(
            "Quantity",
            "sum"
        ),

        Unique_Products=(
            "StockCode",
            "nunique"
        )

    )

    .reset_index()

)


orders = orders.sort_values(

    [
        "Customer ID",
        "InvoiceDate"
    ]

)


print(
    "Invoice-Level Shape:",
    orders.shape
)


# ==========================================================
# 3. Calculate Purchase Gaps
# ==========================================================

orders[
    "Previous_Purchase_Date"
] = (

    orders

    .groupby(
        "Customer ID"
    )[
        "InvoiceDate"
    ]

    .shift(1)

)


orders[
    "Gap_Days"
] = (

    orders[
        "InvoiceDate"
    ]

    -

    orders[
        "Previous_Purchase_Date"
    ]

).dt.days


orders[
    "Gap_Days"
] = (

    orders[
        "Gap_Days"
    ]

    .fillna(0)

    .clip(
        lower=0
    )

)


# ==========================================================
# 4. Create Sequential Samples
# ==========================================================

print(
    "\nCreating LSTM sequences..."
)


feature_columns = [

    "Order_Amount",

    "Total_Quantity",

    "Unique_Products",

    "Gap_Days"

]


sequences = []

targets = []

target_dates = []


for customer_id, customer_data in (

    orders.groupby(
        "Customer ID"
    )

):


    customer_data = (

        customer_data

        .sort_values(
            "InvoiceDate"
        )

        .reset_index(
            drop=True
        )

    )


    if len(
        customer_data
    ) <= SEQUENCE_LENGTH:

        continue


    for i in range(

        SEQUENCE_LENGTH,

        len(
            customer_data
        )

    ):


        sequence = (

            customer_data

            .iloc[
                i - SEQUENCE_LENGTH:i
            ]

            [
                feature_columns
            ]

            .values

        )


        previous_date = (

            customer_data

            .iloc[
                i - 1
            ][
                "InvoiceDate"
            ]

        )


        next_date = (

            customer_data

            .iloc[
                i
            ][
                "InvoiceDate"
            ]

        )


        next_gap = (

            next_date

            -

            previous_date

        ).days


        target = int(

            next_gap
            <= PURCHASE_WINDOW_DAYS

        )


        sequences.append(
            sequence
        )

        targets.append(
            target
        )

        target_dates.append(
            next_date
        )


X = np.array(
    sequences,
    dtype=np.float32
)

y = np.array(
    targets,
    dtype=np.float32
)

dates = np.array(
    target_dates
)


print(
    "Sequence Dataset Shape:",
    X.shape
)

print(
    "Target Shape:",
    y.shape
)


# ==========================================================
# 5. Chronological Train / Validation / Test Split
# ==========================================================

print(
    "\nCreating chronological splits..."
)


sorted_indices = np.argsort(
    dates
)


X = X[
    sorted_indices
]

y = y[
    sorted_indices
]


train_end = int(
    len(X) * 0.70
)

validation_end = int(
    len(X) * 0.85
)


X_train = X[
    :train_end
]

y_train = y[
    :train_end
]


X_validation = X[
    train_end:validation_end
]

y_validation = y[
    train_end:validation_end
]


X_test = X[
    validation_end:
]

y_test = y[
    validation_end:
]


print(
    "Training Set   :",
    X_train.shape
)

print(
    "Validation Set :",
    X_validation.shape
)

print(
    "Testing Set    :",
    X_test.shape
)


# ==========================================================
# 6. Scale Sequence Features
# ==========================================================

print(
    "\nScaling sequence features..."
)


scaler = StandardScaler()


train_reshaped = (

    X_train

    .reshape(
        -1,
        X_train.shape[
            2
        ]
    )

)


scaler.fit(
    train_reshaped
)


def scale_sequences(
    data
):

    original_shape = (
        data.shape
    )


    scaled = scaler.transform(

        data.reshape(

            -1,

            original_shape[
                2
            ]

        )

    )


    return scaled.reshape(
        original_shape
    )


X_train = scale_sequences(
    X_train
)

X_validation = scale_sequences(
    X_validation
)

X_test = scale_sequences(
    X_test
)


# ==========================================================
# 7. Convert to PyTorch Tensors
# ==========================================================

X_train_tensor = torch.tensor(
    X_train,
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train,
    dtype=torch.float32
).reshape(
    -1,
    1
)


X_validation_tensor = torch.tensor(
    X_validation,
    dtype=torch.float32
)

y_validation_tensor = torch.tensor(
    y_validation,
    dtype=torch.float32
).reshape(
    -1,
    1
)


X_test_tensor = torch.tensor(
    X_test,
    dtype=torch.float32
)

y_test_tensor = torch.tensor(
    y_test,
    dtype=torch.float32
).reshape(
    -1,
    1
)


train_dataset = TensorDataset(

    X_train_tensor,

    y_train_tensor

)


train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True

)


# ==========================================================
# 8. Build LSTM Model
# ==========================================================

class LSTMModel(
    nn.Module
):


    def __init__(
        self,
        input_size
    ):


        super().__init__()


        self.lstm = nn.LSTM(

            input_size=input_size,

            hidden_size=64,

            num_layers=1,

            batch_first=True

        )


        self.dropout = nn.Dropout(
            0.30
        )


        self.output = nn.Linear(
            64,
            1
        )


    def forward(
        self,
        x
    ):


        lstm_output, _ = (

            self.lstm(
                x
            )

        )


        last_output = (

            lstm_output[
                :,
                -1,
                :
            ]

        )


        last_output = (

            self.dropout(
                last_output
            )

        )


        return self.output(
            last_output
        )


model = LSTMModel(

    input_size=len(
        feature_columns
    )

)


# ==========================================================
# 9. Loss and Optimizer
# ==========================================================

loss_function = (
    nn.BCEWithLogitsLoss()
)


optimizer = (

    torch.optim.Adam(

        model.parameters(),

        lr=LEARNING_RATE

    )

)


# ==========================================================
# 10. Train With Early Stopping
# ==========================================================

print(
    "\nTraining LSTM model..."
)


best_validation_loss = (
    float(
        "inf"
    )
)

best_model_state = None

patience_counter = 0


training_history = []


for epoch in range(

    1,

    MAX_EPOCHS + 1

):


    model.train()


    total_training_loss = 0


    for (
        batch_X,
        batch_y
    ) in train_loader:


        optimizer.zero_grad()


        outputs = model(
            batch_X
        )


        loss = loss_function(

            outputs,

            batch_y

        )


        loss.backward()


        optimizer.step()


        total_training_loss += (

            loss.item()

            *

            batch_X.size(
                0
            )

        )


    training_loss = (

        total_training_loss

        /

        len(
            train_dataset
        )

    )


    # Validation

    model.eval()


    with torch.no_grad():


        validation_outputs = (

            model(
                X_validation_tensor
            )

        )


        validation_loss = (

            loss_function(

                validation_outputs,

                y_validation_tensor

            )

            .item()

        )


    training_history.append(

        {

            "Epoch":
                epoch,

            "Train_Loss":
                training_loss,

            "Validation_Loss":
                validation_loss

        }

    )


    print(

        f"Epoch {epoch:03d} | "

        f"Train Loss: "
        f"{training_loss:.4f} | "

        f"Validation Loss: "
        f"{validation_loss:.4f}"

    )


    if (

        validation_loss

        < best_validation_loss

    ):


        best_validation_loss = (

            validation_loss

        )


        best_model_state = (

            copy.deepcopy(

                model.state_dict()

            )

        )


        patience_counter = 0


    else:


        patience_counter += 1


    if (

        patience_counter

        >= PATIENCE

    ):


        print(

            "\nEarly stopping triggered."

        )


        break


# ==========================================================
# 11. Restore Best Model
# ==========================================================

if best_model_state is not None:

    model.load_state_dict(

        best_model_state

    )


print(
    "\nLSTM training completed."
)


# ==========================================================
# 12. Evaluation Function
# ==========================================================

def evaluate_model(

    dataset_name,

    X_tensor,

    y_true

):


    model.eval()


    with torch.no_grad():


        logits = model(
            X_tensor
        )


        probabilities = (

            torch.sigmoid(
                logits
            )

            .numpy()

            .ravel()

        )


    predictions = (

        probabilities

        >= 0.5

    ).astype(int)


    return {

        "Dataset":
            dataset_name,

        "Accuracy":
            accuracy_score(
                y_true,
                predictions
            ),

        "Precision":
            precision_score(
                y_true,
                predictions,
                zero_division=0
            ),

        "Recall":
            recall_score(
                y_true,
                predictions,
                zero_division=0
            ),

        "F1_Score":
            f1_score(
                y_true,
                predictions,
                zero_division=0
            ),

        "ROC_AUC":
            roc_auc_score(
                y_true,
                probabilities
            )

    }


# ==========================================================
# 13. Evaluate Model
# ==========================================================

print(
    "\nEvaluating LSTM..."
)


validation_results = (

    evaluate_model(

        "Validation",

        X_validation_tensor,

        y_validation

    )

)


test_results = (

    evaluate_model(

        "Test",

        X_test_tensor,

        y_test

    )

)


results_df = pd.DataFrame(

    [

        validation_results,

        test_results

    ]

)


print(
    "\nLSTM Model Results"
)

print("-" * 75)


print(

    results_df

    .round(4)

    .to_string(
        index=False
    )

)


# ==========================================================
# 14. Save Results
# ==========================================================

results_df.to_csv(

    "outputs/reports/"
    "day12_lstm_results.csv",

    index=False

)


pd.DataFrame(

    training_history

).to_csv(

    "outputs/reports/"
    "day12_lstm_training_history.csv",

    index=False

)


# ==========================================================
# 15. Save Model
# ==========================================================

torch.save(

    model.state_dict(),

    "models/"
    "lstm_purchase_model.pth"

)


import joblib


joblib.dump(

    scaler,

    "models/"
    "lstm_sequence_scaler.pkl"

)


joblib.dump(

    feature_columns,

    "models/"
    "lstm_feature_names.pkl"

)


print(
    "\nModel saved successfully."
)


print(
    "\nDay 12 LSTM completed successfully."
)