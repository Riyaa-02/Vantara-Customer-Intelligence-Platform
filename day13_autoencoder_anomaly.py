# ==========================================================
# DAY 13 - AUTOENCODER ANOMALY DETECTION
# ==========================================================

import os
import copy
import random
import numpy as np
import pandas as pd
import torch

from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------
# Configuration
# ----------------------------------------------------------

SEED = 42
BATCH_SIZE = 64
MAX_EPOCHS = 100
PATIENCE = 10
LEARNING_RATE = 0.001

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

os.makedirs("models", exist_ok=True)
os.makedirs("outputs/reports", exist_ok=True)
os.makedirs("data/final", exist_ok=True)

# ----------------------------------------------------------
# 1. Load Customer Features
# ----------------------------------------------------------

print("\nLoading customer features...")

df = pd.read_csv(
    "data/processed/customer_features.csv"
)

print("Dataset Shape:", df.shape)

# ----------------------------------------------------------
# 2. Select Behaviour Features
# ----------------------------------------------------------

feature_columns = [
    "Recency",
    "Frequency",
    "Monetary_Total",
    "Monetary_Average",
    "Total_Quantity",
    "Average_Basket_Size",
    "Average_Order_Value",
    "Unique_Products",
    "Average_Purchase_Gap",
    "Customer_Lifespan_Days",
    "Return_Rate",
    "Price_Sensitivity",
    "Engagement_Score"
]

missing_columns = [
    column
    for column in feature_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )

X = df[feature_columns].copy()

# ----------------------------------------------------------
# 3. Handle Extreme Values
# ----------------------------------------------------------

print("\nPreparing anomaly detection features...")

for column in feature_columns:

    lower = X[column].quantile(0.01)
    upper = X[column].quantile(0.99)

    X[column] = X[column].clip(
        lower=lower,
        upper=upper
    )

# ----------------------------------------------------------
# 4. Scale Features
# ----------------------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    X
).astype(np.float32)

X_tensor = torch.tensor(
    X_scaled,
    dtype=torch.float32
)

# ----------------------------------------------------------
# 5. Train / Validation Split
# ----------------------------------------------------------

split_index = int(
    len(X_tensor) * 0.80
)

X_train = X_tensor[:split_index]
X_validation = X_tensor[split_index:]

train_dataset = TensorDataset(
    X_train,
    X_train
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# ----------------------------------------------------------
# 6. Build Autoencoder
# ----------------------------------------------------------

class Autoencoder(nn.Module):

    def __init__(self, input_size):

        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 8),
            nn.ReLU(),

            nn.Linear(8, 4),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),

            nn.Linear(8, input_size)
        )

    def forward(self, x):

        encoded = self.encoder(x)

        reconstructed = self.decoder(
            encoded
        )

        return reconstructed


model = Autoencoder(
    len(feature_columns)
)

loss_function = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# ----------------------------------------------------------
# 7. Train With Early Stopping
# ----------------------------------------------------------

print("\nTraining Autoencoder...")

best_validation_loss = float("inf")
best_model_state = None
patience_counter = 0

training_history = []

for epoch in range(
    1,
    MAX_EPOCHS + 1
):

    model.train()

    total_loss = 0

    for batch_X, _ in train_loader:

        optimizer.zero_grad()

        reconstructed = model(
            batch_X
        )

        loss = loss_function(
            reconstructed,
            batch_X
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item()
            * batch_X.size(0)
        )

    train_loss = (
        total_loss
        / len(X_train)
    )

    model.eval()

    with torch.no_grad():

        validation_reconstructed = model(
            X_validation
        )

        validation_loss = loss_function(
            validation_reconstructed,
            X_validation
        ).item()

    training_history.append({
        "Epoch": epoch,
        "Train_Loss": train_loss,
        "Validation_Loss": validation_loss
    })

    print(
        f"Epoch {epoch:03d} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Validation Loss: "
        f"{validation_loss:.4f}"
    )

    if validation_loss < best_validation_loss:

        best_validation_loss = (
            validation_loss
        )

        best_model_state = copy.deepcopy(
            model.state_dict()
        )

        patience_counter = 0

    else:

        patience_counter += 1

    if patience_counter >= PATIENCE:

        print(
            "\nEarly stopping triggered."
        )

        break

# ----------------------------------------------------------
# 8. Restore Best Model
# ----------------------------------------------------------

if best_model_state is not None:

    model.load_state_dict(
        best_model_state
    )

print(
    "\nAutoencoder training completed."
)

# ----------------------------------------------------------
# 9. Calculate Reconstruction Error
# ----------------------------------------------------------

print(
    "\nCalculating reconstruction errors..."
)

model.eval()

with torch.no_grad():

    reconstructed = model(
        X_tensor
    )

    reconstruction_error = torch.mean(
        (
            X_tensor
            - reconstructed
        ) ** 2,
        dim=1
    ).numpy()

df[
    "Anomaly_Score"
] = reconstruction_error

# ----------------------------------------------------------
# 10. Define Anomaly Threshold
# ----------------------------------------------------------

threshold = np.percentile(
    reconstruction_error,
    95
)

df[
    "Is_Anomaly"
] = (
    df["Anomaly_Score"]
    >= threshold
).astype(int)

# ----------------------------------------------------------
# 11. Summary
# ----------------------------------------------------------

total_anomalies = int(
    df["Is_Anomaly"].sum()
)

anomaly_percentage = (
    total_anomalies
    / len(df)
    * 100
)

print(
    "\nAnomaly Detection Summary"
)

print("-" * 50)

print(
    "Reconstruction Error Threshold:",
    round(threshold, 4)
)

print(
    "Total Customers:",
    len(df)
)

print(
    "Anomalous Customers:",
    total_anomalies
)

print(
    "Anomaly Percentage:",
    round(
        anomaly_percentage,
        2
    ),
    "%"
)

# ----------------------------------------------------------
# 12. Save Outputs
# ----------------------------------------------------------

df.to_csv(
    "data/final/"
    "customer_anomaly_results.csv",
    index=False
)

pd.DataFrame(
    training_history
).to_csv(
    "outputs/reports/"
    "day13_autoencoder_training_history.csv",
    index=False
)

pd.DataFrame([
    {
        "Threshold":
            threshold,

        "Total_Customers":
            len(df),

        "Anomalous_Customers":
            total_anomalies,

        "Anomaly_Percentage":
            anomaly_percentage
    }
]).to_csv(
    "outputs/reports/"
    "day13_anomaly_summary.csv",
    index=False
)

torch.save(
    model.state_dict(),
    "models/"
    "autoencoder_anomaly_model.pth"
)

import joblib

joblib.dump(
    scaler,
    "models/"
    "autoencoder_scaler.pkl"
)

joblib.dump(
    feature_columns,
    "models/"
    "autoencoder_feature_names.pkl"
)

print(
    "\nOutputs saved successfully."
)

print(
    "\nDay 13 Autoencoder "
    "completed successfully."
)