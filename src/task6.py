# train and test only augmented data 
# Task 6: Use VAE Decoder to Generate Synthetic Data, Retrain Models
import os
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import json
from itertools import product
import time
import random
import csv
from joblib import load

#predict model
models=["LSTM","CNN-LSTM","Transformer"]  #"CNN-LSTM" "Transformer" "LSTM"
dropout_options = [0]
learning_rates = [0.001]
batch_sizes = [16]
epochs_list = [50]

#vae
input_dim=10*10+5+219
hidden_dim = 128
latent_dim = 20
vae_batch_sizes = 32


sequences_per_country = 30
test_size = 5
val_size = 3

final_trainer=False
final_train_epochs=5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def set_seed(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed=11
set_seed(seed)

# -------------------------------
# Configuration
# -------------------------------
# data dir
result_dir_task4 = "results/task4"
timestamp_task4 = "1743690723"
task4_dir=os.path.join(result_dir_task4, "models", timestamp_task4)

# vae model dir
result_dir_task5 = "results/task5"
timestamp_task5 = "1743702214"

vae_path = os.path.join(result_dir_task5,"models",timestamp_task5,"vae_best.pth")
vae_sclaer_path = os.path.join(result_dir_task5,"models",timestamp_task5,"vae_scaler.joblib")

result_dir = "results/task6"
os.makedirs(result_dir, exist_ok=True)
timestamp = int(time.time())
result_dir = "results/task6"
os.makedirs(result_dir, exist_ok=True)
models_dir = os.path.join(result_dir, "models")
os.makedirs(models_dir, exist_ok=True)
time_models_dir = os.path.join(models_dir, str(timestamp))
os.makedirs(time_models_dir, exist_ok=True)
data_dir=os.path.join(time_models_dir,"data")
os.makedirs(data_dir, exist_ok=True)
params_tune_dir=os.path.join(time_models_dir,"params_tune")
os.makedirs(params_tune_dir, exist_ok=True)
graphs_dir=os.path.join(time_models_dir,"outputs")
os.makedirs(graphs_dir, exist_ok=True)

# -------------------------------
# Load original data (Task 4)
# -------------------------------
features_sclaer_path=os.path.join(task4_dir, "vae_scaler.joblib")
data_dir_task4 = os.path.join(task4_dir, "data")
X_train = np.load(os.path.join(data_dir_task4, "X_train.npy")) 
y_train = np.load(os.path.join(data_dir_task4, "y_train.npy")) 
X_val = np.load(os.path.join(data_dir_task4, "X_val.npy")) 
y_val = np.load(os.path.join(data_dir_task4, "y_val.npy")) 
X_test = np.load(os.path.join(data_dir_task4, "X_test.npy")) 
y_test = np.load(os.path.join(data_dir_task4, "y_test.npy")) 

# Optional: Load test country labels for visualizations
test_country_labels = pd.read_csv(os.path.join(task4_dir, "outputs", "test_actuals.csv"))["country"].tolist()

# ---------------------- VAE Model ----------------------

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, latent_dim=20):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        # Decoder output activation is Sigmoid to output values in [0,1]
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

vae = VAE(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device) # 10*229+5
vae.load_state_dict(torch.load(vae_path))
vae.eval()

n_synthetic = X_train.shape[0]  # Match original size
z_samples = torch.randn(n_synthetic, latent_dim).to(device)

vae_scaler = load(vae_sclaer_path)

with torch.no_grad():
    synthetic_data = vae.decode(z_samples).cpu().numpy()

# Inverse transform to get back to original scale
synthetic_data_unscaled = vae_scaler.inverse_transform(synthetic_data)

# Split into input/output
print(synthetic_data_unscaled.shape)
# synthetic_input = synthetic_data_unscaled[:, :10]+synthetic_data_unscaled[:, 15:]
# synthetic_output = synthetic_data_unscaled[:, 10:15]  # shape: (n_synthetic, 5)
print(X_train.shape)

n_samples = synthetic_data_unscaled.shape[0]

numerical_flat = synthetic_data_unscaled[:, 0:100]  # Shape: (4818, 100)
y_train_reconstructed = synthetic_data_unscaled[:, 100:105]  # Shape: (4818, 5)
country_data_reconstructed = synthetic_data_unscaled[:, 105:324]  # Shape: (4818, 219)

numerical_data_reconstructed = numerical_flat.reshape(4818, 10, 10)  # Shape: (4818, 10, 10)
country_data_expanded = np.tile(country_data_reconstructed[:, np.newaxis, :], (1, 10, 1))  # Shape: (4818, 10, 219)
X_train_reconstructed = np.concatenate((numerical_data_reconstructed, country_data_expanded), axis=2)  # Shape: (4818, 10, 229)

# Verify shapes
print(f"X_train_reconstructed shape: {X_train_reconstructed.shape}")  # Expected: (4818, 10, 229)
print(f"y_train_reconstructed shape: {y_train_reconstructed.shape}")  # Expected: (4818, 5)

# -------------------------------
# Combine Original and Synthetic
# -------------------------------
scaler_all = load(features_sclaer_path)

def scale_sequences(sequences):
    num_features_total = sequences.shape[2]  # 229
    flat = sequences.reshape(-1, num_features_total)
    flat_scaled = scaler_all.transform(flat)
    return flat_scaled.reshape(sequences.shape)

X_train_reconstructed_scaled = scale_sequences(X_train_reconstructed)
X_augmented = np.concatenate([X_train, X_train_reconstructed_scaled], axis=0)
y_augmented = np.concatenate([y_train, y_train_reconstructed], axis=0)

print("Original training data:", X_train.shape)
print("Synthetic data:", X_train_reconstructed.shape)
print("Augmented data:", X_augmented.shape)

# -------------------------------
# GDP Dataset
# -------------------------------
class GDPDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# -------------------------------
# Model Definitions
# -------------------------------
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, verbose=True, path="best_model.pth"):
        """
        Early stopping to stop training when validation loss stops improving.

        Args:
            patience (int): Number of epochs to wait before stopping if no improvement.
            min_delta (float): Minimum change in loss to qualify as an improvement.
            verbose (bool): If True, prints early stopping updates.
            path (str): Path to save the best model.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.path = path  # File path to save best model
        self.best_loss = float("inf")  # Initialize with a very high loss
        self.counter = 0  # Counts epochs without improvement
        self.early_stop = False  # Flag to signal early stopping
        self.best_model_state = None  # Store the best model weights in memory

    def __call__(self, val_loss, model):
        """
        Checks if validation loss improved, otherwise increments counter.

        Args:
            val_loss (float): The current validation loss.
            model (torch.nn.Module): The model being trained.
        """
        if val_loss < self.best_loss - self.min_delta:  # Check for improvement
            self.best_loss = val_loss
            self.counter = 0  # Reset counter

            # Save the best model state (both to disk and memory)
            self.best_model_state = model.state_dict()
            torch.save(model.state_dict(), self.path, _use_new_zipfile_serialization=False)

            if self.verbose:
                print(f"Validation loss improved! Model saved to {self.path}")

        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience} (No improvement)")

            if self.counter >= self.patience:  # Stop if patience exceeded
                self.early_stop = True
                if self.verbose:
                    print(f"Early stopping triggered after {self.patience} epochs of no improvement.")

class LSTMModel(nn.Module):
    def __init__(self, hidden_size=128, num_layers=2, dropout_rate=0.3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=229, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, 5)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out

class CNNLSTMModel(nn.Module):
    def __init__(self, input_size=229, output_size=5, hidden_size=128, num_layers=1, dropout_rate=0):
        super(CNNLSTMModel, self).__init__()
        self.conv = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2) 
        self.lstm = nn.LSTM(64, hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, features, seq_len)
        x = self.conv(x)
        x = self.bn(x)  # Apply BatchNorm
        x = nn.functional.relu(x)
        x = self.pool(x)
        x = x.permute(0, 2, 1)  # (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
    
class TransformerModel(nn.Module):
    def __init__(self, input_size=229, output_size=5, d_model=64, nhead=8, num_layers=2, dropout_rate=0.2):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_size, d_model) 
        self.pos_encoder = PositionalEncoding(d_model, max_len=10, dropout=dropout_rate)

        # Transformer Encoder with Dropout
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=4 * d_model,  # Usually 4x d_model for FFN
            dropout=dropout_rate,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Fully Connected Output Layer with Dropout
        self.fc = nn.Sequential(
            nn.LayerNorm(d_model),  # Layer Normalization before FC
            nn.Dropout(dropout),  
            nn.Linear(d_model, output_size)
        )

    def forward(self, x):
        x = self.embedding(x)  # Linear embedding
        # x = self.pos_encoder(x)
        x = self.transformer(x)  # Transformer Encoder
        x = x[:, -1, :]  # Get last time step
        return self.fc(x)  # Fully Connected Layer

# Training function with early stopping
def train_model(model, train_loader, val_loader, epochs=100, lr=0.001, model_path=None):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    early_stopping = EarlyStopping(patience=10, min_delta=0.001, path=model_path)

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation phase
        if val_loader is not None:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    preds = model(xb)
                    loss = criterion(preds, yb)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            val_losses.append(val_loss)

            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            early_stopping(val_loss, model)

            if early_stopping.early_stop:
                print("Early stopping triggered. Restoring best model weights...")
                model.load_state_dict(early_stopping.best_model_state)  # Restore Best Model
                break

        else:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}")

    if val_loader is not None:
        return train_losses, val_losses, early_stopping.best_loss, early_stopping.path
    else:
        return train_losses, None, None, model_path
    
def evaluate_model(model, loader, test_countries):
    model.to(device)
    model.eval()
    actuals, predictions, country_labels = [], [], []

    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(loader):
            X, y = X.to(device), y.to(device)
            pred = model(X).cpu().numpy()

            predictions.append(pred)
            actuals.append(y.cpu().numpy())

            # Ensure batch-country mapping is correct
            batch_start = batch_idx * loader.batch_size
            batch_end = batch_start + len(y)  # Handle cases where batch < batch_size
            country_labels.extend(test_countries[batch_start:batch_end])

    predictions = np.concatenate(predictions)
    actuals = np.concatenate(actuals)

    #  Denormalize GDP values 
    predictions = np.expm1(predictions)
    actuals = np.expm1(actuals)

    #  Compute Metrics 
    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100

    return actuals, predictions, country_labels, [mse, mae, mape]

def save_training_history(history, plot_path):
    epochs = range(1, len(history["train_losses"]) + 1)

    plt.figure(figsize=(8, 5))

    # Plot training loss
    plt.plot(epochs, history["train_losses"], label="Train Loss", color="blue")
    
    # Plot validation loss only if available
    if "val_losses" in history and history["val_losses"]:
        plt.plot(epochs, history["val_losses"], label="Validation Loss", color="orange")

        # Annotate last values
        plt.annotate(f"{history['train_losses'][-1]:.2f}", xy=(epochs[-1], history["train_losses"][-1]),
                     xytext=(epochs[-1] - 5, history["train_losses"][-1] + 0.05),
                     arrowprops=dict(arrowstyle="->", color="blue"), fontsize=10, color="blue")

        plt.annotate(f"{history['val_losses'][-1]:.2f}", xy=(epochs[-1], history["val_losses"][-1]),
                     xytext=(epochs[-1] - 5, history["val_losses"][-1] + 0.05),
                     arrowprops=dict(arrowstyle="->", color="orange"), fontsize=10, color="orange")

    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()

    print(f"Training history saved to {plot_path}")

def save_predict_results(country_labels, values, csv_path, label="predicted_gdp"):
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(["country", label])

        for country, val in zip(country_labels, values):
            formatted_val = "[" + ", ".join(f"{v:.2f}" for v in val) + "]"
            writer.writerow([country, formatted_val])

MODEL_MAPPING = {
    "LSTM": LSTMModel,
    "CNN-LSTM": CNNLSTMModel,
    "Transformer": TransformerModel
}

# -------------------------------
# Train and Evaluate on Augmented Data
# -------------------------------
# Hyperparameter grid search
lstm_preds = None
cnn_lstm_preds = None
transformer_preds = None

for name in models:
    best_model_info = {"model": None, "val_loss": float("inf"), "params": {}, "grid_search_params":{}, "path": None}
    model_class=  MODEL_MAPPING.get(name)
    param_grid = product(dropout_options, learning_rates, batch_sizes, epochs_list)

    for dropout, lr, batch_size, epochs in param_grid:
    # for dropout, lr, batch_size, epochs in param_grid:
        set_seed(seed)
        print(f"Training {name} with Dropout={dropout}, LR={lr}, Batch={batch_size}, Epochs={epochs}")
        train_loader = DataLoader(GDPDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(GDPDataset(X_val, y_val), batch_size=batch_size)

        model = model_class(dropout_rate=dropout)
        
        model_path = f"best_model_{name}_drop{int(dropout*10)}_lr{int(lr*10000)}_bs{batch_size}_ep{epochs}.pth"
        tune_model_path=os.path.join(params_tune_dir, model_path)

        train_losses, val_losses, val_loss, saved_path = train_model(
            model, train_loader, val_loader,
            lr=lr, epochs=epochs, model_path=tune_model_path
        )

        if val_loss < best_model_info["val_loss"]:
            best_model_info = {
                "model": name,
                "val_loss": val_loss,
                "params": {
                    "dropout": dropout,
                    "lr": lr,
                    "batch_size": batch_size,
                    "epochs": epochs
                },
                "grid_search_params": {
                    "dropout_options": dropout_options,
                    "learning_rates": learning_rates,
                    "batch_sizes": batch_sizes,
                    "epochs_list": epochs_list
                },
                "path": saved_path
            }

            history={
                'batch_size': batch_size, 'epochs': epochs, 'lr': lr, 
                'train_losses': train_losses, 'val_losses': val_losses,
            }

    best_model_info_path=f"{time_models_dir}/best_model_info_{name}.json"
    with open(best_model_info_path, "w") as json_file:
        json.dump(best_model_info, json_file, indent=4)

    with open(best_model_info_path, "r") as file:
        config=json.load(file)

    best_params=config["params"]
    model_path=config["path"]

    save_training_history(history, plot_path=f"{graphs_dir}/{name}_best_model_training_graph")
    final_model_path=os.path.join(time_models_dir, f"{name}.pth")

    if final_trainer==True:
        
        print(f"Training Final {name} with Dropout={best_params['dropout']}, LR={best_params['lr']}, Batch={best_params['batch_size']}, Epochs={best_params['epochs']}")

        X_final_train = np.concatenate([X_train, X_val], axis=0)
        y_final_train = np.concatenate([y_train, y_val], axis=0)
        final_train_loader = DataLoader(GDPDataset(X_final_train, y_final_train), batch_size=best_params['batch_size'], shuffle=False)
        
        model=model_class(dropout_rate=best_params["dropout"])
        train_losses, val_losses, val_loss, model_path = train_model(
                model, train_loader=final_train_loader, val_loader=None,
                lr=best_params["lr"], epochs=final_train_epochs, model_path=final_model_path
        )

        torch.save(model.state_dict(), model_path, _use_new_zipfile_serialization=False)

        history={
            'batch_size': best_params['batch_size'], 'epochs': final_train_epochs, 'lr': best_params["lr"], 
            'train_losses': train_losses, 'val_losses': val_losses,
        }
        save_training_history(history, plot_path=f"{time_models_dir}/{name}_final_model_training_graph")

    model=model_class(dropout_rate=best_params["dropout"])
    model.load_state_dict(torch.load(model_path))

    torch.save(model.state_dict(), final_model_path, _use_new_zipfile_serialization=False)

    test_loader = DataLoader(GDPDataset(X_test, y_test), batch_size=best_params['batch_size'] ,shuffle=False)
    actuals, predictions, country_labels, results = evaluate_model(model, test_loader, test_country_labels)
    predictions_csv_path = os.path.join(graphs_dir, f"{name}_test_predictions.csv")
    save_predict_results(country_labels, predictions, predictions_csv_path, label="predicted_gdp")


    if name == "LSTM":
        lstm_preds = predictions
    elif name == "CNN-LSTM":
        cnn_lstm_preds = predictions
    elif name == "Transformer":
        transformer_preds = predictions
        
    # Save results to a CSV file
    os.makedirs(result_dir, exist_ok=True)
    result_path = f"{result_dir}/results.csv"
    file_exists = os.path.isfile(result_path)
    header = ["model_path", "params", "mse", "mae", "mape"]
    results=[final_model_path, str(best_params)] + results
    print(results)

    with open(result_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(results)

actuals_csv_path = os.path.join(graphs_dir, f"test_actuals.csv")
save_predict_results(country_labels, actuals, actuals_csv_path, label="actual_gdp")

# # -------------------------------
# # Load Original Results from Task 4
# # -------------------------------
# with open("results/task4/results.csv", "r") as f:
#     reader = csv.reader(f)
#     next(reader)  # skip header
#     for row in reader:
#         model_name = os.path.basename(row[0]).replace(".pth", "")
#         if model_name in results:
#             results[model_name]["mape_original"] = float(row[-1])
#             results[model_name]["mae_original"] = float(row[-2])
#             results[model_name]["mse_original"] = float(row[-3])

# # -------------------------------
# # Generate Comparison Table
# # -------------------------------
# table_path = os.path.join(vae_output_dir, "performance_comparison.csv")

# with open(table_path, "w", newline='') as f:
#     writer = csv.writer(f)
#     writer.writerow(["Model", "MAPE (Original)", "MAPE (Augmented)", "MAE (Original)", "MAE (Augmented)", "MSE (Original)", "MSE (Augmented)"])
#     for model_name, data in results.items():
#         writer.writerow([
#             model_name,
#             f"{data.get('mape_original', 'N/A'):.2f}",
#             f"{data['mape']:.2f}",
#             f"{data.get('mae_original', 'N/A'):.2f}",
#             f"{data['mae']:.2f}",
#             f"{data.get('mse_original', 'N/A'):.2f}",
#             f"{data['mse']:.2f}"
#         ])

# print("\n📊 Performance comparison table saved to:", table_path)