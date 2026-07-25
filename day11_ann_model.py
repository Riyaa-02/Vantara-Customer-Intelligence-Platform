# ==========================================================
# VANTARA CUSTOMER INTELLIGENCE PLATFORM
# DAY 11 - ANN CHURN MODEL USING PYTORCH
# ==========================================================

import os
import random
import joblib
import numpy as np
import pandas as pd
import torch

from torch import nn
from torch.utils.data import (
    TensorDataset,
    DataLoader
)

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
BATCH_SIZE = 32
MAX_EPOCHS = 100
PATIENCE = 10
LEARNING_RATE = 0.001


# ==========================================================
# Reproducibility
# ==========================================================

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
# 1. Load Data
# ==========================================================

print("\nLoading datasets...")

X_train = pd.read_csv(
    "data/final/X_train.csv"
).astype("float32")

X_validation = pd.read_csv(
    "data/final/X_validation.csv"
).astype("float32")

X_test = pd.read_csv(
    "data/final/X_test.csv"
).astype("float32")

y_train = pd.read_csv(
    "data/final/y_train.csv"
).squeeze().astype("float32")

y_validation = pd.read_csv(
    "data/final/y_validation.csv"
).squeeze().astype("float32")

y_test = pd.read_csv(
    "data/final/y_test.csv"
).squeeze().astype("float32")


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
# 2. Convert to PyTorch Tensors
# ==========================================================

X_train_tensor = torch.tensor(
    X_train.values,
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train.values,
    dtype=torch.float32
).reshape(-1, 1)


X_validation_tensor = torch.tensor(
    X_validation.values,
    dtype=torch.float32
)

y_validation_tensor = torch.tensor(
    y_validation.values,
    dtype=torch.float32
).reshape(-1, 1)


X_test_tensor = torch.tensor(
    X_test.values,
    dtype=torch.float32
)

y_test_tensor = torch.tensor(
    y_test.values,
    dtype=torch.float32
).reshape(-1, 1)


# ==========================================================
# 3. Create DataLoader
# ==========================================================

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
# 4. Build ANN Model
# ==========================================================

class ANNModel(nn.Module):

    def __init__(
        self,
        input_size
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                input_size,
                128
            ),

            nn.BatchNorm1d(
                128
            ),

            nn.ReLU(),

            nn.Dropout(
                0.30
            ),

            nn.Linear(
                128,
                64
            ),

            nn.BatchNorm1d(
                64
            ),

            nn.ReLU(),

            nn.Dropout(
                0.30
            ),

            nn.Linear(
                64,
                32
            ),

            nn.ReLU(),

            nn.Dropout(
                0.20
            ),

            nn.Linear(
                32,
                1
            )

        )


    def forward(
        self,
        x
    ):

        return self.network(
            x
        )


model = ANNModel(
    X_train.shape[1]
)


# ==========================================================
# 5. Loss and Optimizer
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
# 6. Early Stopping Setup
# ==========================================================

best_validation_loss = (
    float("inf")
)

patience_counter = 0

best_model_state = None


training_history = []


# ==========================================================
# 7. Train ANN
# ==========================================================

print(
    "\nTraining ANN model..."
)


for epoch in range(
    1,
    MAX_EPOCHS + 1
):


    model.train()

    total_train_loss = 0


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


        total_train_loss += (
            loss.item()
            * batch_X.size(0)
        )


    train_loss = (
        total_train_loss
        / len(
            train_dataset
        )
    )


    # ------------------------------------------------------
    # Validation Loss
    # ------------------------------------------------------

    model.eval()


    with torch.no_grad():

        validation_outputs = model(
            X_validation_tensor
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
                train_loss,

            "Validation_Loss":
                validation_loss

        }

    )


    print(

        f"Epoch {epoch:03d} | "

        f"Train Loss: "
        f"{train_loss:.4f} | "

        f"Validation Loss: "
        f"{validation_loss:.4f}"

    )


    # ------------------------------------------------------
    # Early Stopping
    # ------------------------------------------------------

    if (
        validation_loss
        < best_validation_loss
    ):


        best_validation_loss = (
            validation_loss
        )


        best_model_state = {

            key:
                value.clone()

            for (
                key,
                value
            )

            in model.state_dict().items()

        }


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
# 8. Restore Best Model
# ==========================================================

if (
    best_model_state
    is not None
):

    model.load_state_dict(
        best_model_state
    )


print(
    "\nANN training completed."
)


# ==========================================================
# 9. Evaluation Function
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
                predictions
            ),

        "Recall":
            recall_score(
                y_true,
                predictions
            ),

        "F1_Score":
            f1_score(
                y_true,
                predictions
            ),

        "ROC_AUC":
            roc_auc_score(
                y_true,
                probabilities
            )

    }


# ==========================================================
# 10. Evaluate Validation and Test Sets
# ==========================================================

print(
    "\nEvaluating ANN "
    "on validation dataset..."
)


validation_results = evaluate_model(

    "Validation",

    X_validation_tensor,

    y_validation.values

)


print(
    "\nEvaluating ANN "
    "on test dataset..."
)


test_results = evaluate_model(

    "Test",

    X_test_tensor,

    y_test.values

)


results_df = pd.DataFrame(

    [

        validation_results,

        test_results

    ]

)


print(
    "\nANN Model Results"
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
# 11. Save Results
# ==========================================================

results_df.to_csv(

    "outputs/reports/"
    "day11_ann_results.csv",

    index=False

)


# ==========================================================
# 12. Save Training History
# ==========================================================

history_df = pd.DataFrame(
    training_history
)


history_df.to_csv(

    "outputs/reports/"
    "day11_ann_training_history.csv",

    index=False

)


# ==========================================================
# 13. Save Model
# ==========================================================

torch.save(

    model.state_dict(),

    "models/"
    "ann_churn_model.pth"

)


joblib.dump(

    list(
        X_train.columns
    ),

    "models/"
    "ann_feature_names.pkl"

)


print(
    "\nModel saved successfully."
)


print(
    "\nDay 11 ANN completed successfully."
)