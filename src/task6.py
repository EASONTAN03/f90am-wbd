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

seed=11

#predict model
models=["LSTM","CNN-LSTM","Transformer"]  #"CNN-LSTM" "Transformer" "LSTM"
batch_sizes = [16]
epochs_list = [50]
learning_rates = [0.0001]
dropout_options = [0]

# vae
hidden_dim = 256
latent_dim = 20

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
set_seed(seed)

df = pd.read_csv(r"data/final_impute_world_bank_data_dev.csv")
country_features = len(df["country"].unique())
input_dim = 10 * 10 + country_features + 5     # 10*10=100, plus 5 plus country_features (e.g., 219) equals 324

timestamp_task4 = "1744209367"
result_dir_task4 = "results/task4"
task4_model_dir = os.path.join(result_dir_task4, "models", timestamp_task4)
task4_data_dir = os.path.join(task4_model_dir, "data")

timestamp_task5 = "1744210542"
result_dir_task5 = "results/task5"
vae_path = os.path.join(result_dir_task5,"models",timestamp_task5,"vae.pth")
vae_sclaer_path = os.path.join(result_dir_task5,"models",timestamp_task5,"vae_scaler.joblib")

timestamp = int(time.time())
result_dir = "results/task6"
os.makedirs(result_dir, exist_ok=True)
models_dir = os.path.join(result_dir, "models")
os.makedirs(models_dir, exist_ok=True)
time_models_dir = os.path.join(models_dir, str(timestamp))
os.makedirs(time_models_dir, exist_ok=True)
params_tune_dir=os.path.join(time_models_dir,"params_tune")
os.makedirs(params_tune_dir, exist_ok=True)
output_dir=os.path.join(time_models_dir,"outputs")
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

features_sclaer_path=os.path.join(task4_model_dir, "scaler.joblib")
scaler = load(features_sclaer_path)
vae_scaler = load(vae_sclaer_path)

X_train = np.load(os.path.join(task4_data_dir, "X_train.npy")) 
y_train = np.load(os.path.join(task4_data_dir, "y_train.npy")) 
X_val = np.load(os.path.join(task4_data_dir, "X_val.npy")) 
y_val = np.load(os.path.join(task4_data_dir, "y_val.npy")) 
X_test = np.load(os.path.join(task4_data_dir, "X_test.npy")) 
y_test = np.load(os.path.join(task4_data_dir, "y_test.npy")) 
test_country_labels = np.load(os.path.join(task4_data_dir, "test_country_labels.npy")) 

def scale_sequences(task4_scaler,sequences):
    num_features_total = sequences.shape[2]  # 229
    flat = sequences.reshape(-1, num_features_total)
    flat_scaled = task4_scaler.transform(flat)
    return flat_scaled.reshape(sequences.shape)


class EarlyStopping():
    def __init__(self, patience=5, min_delta=0, verbose=True, path="best_model.pth"):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.path = path  # File path to save best model
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, model):
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
                    print(f"Early stopping triggered after {self.patience}")

class GDPDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

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

def plot_gdp_predictions_multi(actuals, lstm_preds, cnn_lstm_preds, transformer_preds, country_labels, selected_countries=None, save_dir=time_models_dir):
    for selected_country in selected_countries:
        # Get indices for the selected country
        country_indices = [i for i, country in enumerate(country_labels) if country == selected_country]
        if not country_indices:
            print(f"No data found for {selected_country}. Skipping...")
            continue  # Skip if no data for this country

        # Extract relevant actual and predicted values
        actual_values = actuals[country_indices].flatten()

        plt.figure(figsize=(8, 5))
        plt.plot(actual_values, label="Actual GDP", marker='o', linestyle='-', color='black')

        if lstm_preds is not None:
            lstm_values = lstm_preds[country_indices].flatten()
            plt.plot(lstm_values, label="LSTM Predictions", marker='x', linestyle='--', color='blue')
        if cnn_lstm_preds is not None:
            cnn_lstm_values = cnn_lstm_preds[country_indices].flatten()
            plt.plot(cnn_lstm_values, label="CNN-LSTM Predictions", marker='s', linestyle='--', color='red')
        if transformer_preds is not None:
            transformer_values = transformer_preds[country_indices].flatten()
            plt.plot(transformer_values, label="Transformer Predictions", marker='d', linestyle='--', color='green')

        plt.title(f"GDP Predictions vs Actual for {selected_country}")
        plt.xlabel("Sequence Index")
        plt.ylabel("GDP (after inverse log transform)")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f"{selected_country}_gdp_predictions.png"))
        plt.close()

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

# VAE
vae = VAE(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device) # 10*229+5
vae.load_state_dict(torch.load(vae_path))

vae.eval()
n_synthetic = X_train.shape[0]  # Match original size
z_samples = torch.randn(n_synthetic, latent_dim).to(device)
vae_scaler = load(vae_sclaer_path)
with torch.no_grad():
    synthetic_data = vae.decode(z_samples).cpu().numpy()

synthetic_data_unscaled = vae_scaler.inverse_transform(synthetic_data)
print(synthetic_data_unscaled.shape)
print(X_train.shape)

n_samples = synthetic_data_unscaled.shape[0]
numerical_flat = synthetic_data_unscaled[:, 0:100]  # Shape: (4818, 100)
country_data_reconstructed = synthetic_data_unscaled[:, 100:319]  #  Shape: (4818, 219)
y_train_reconstructed = synthetic_data_unscaled[:, 319:] 

numerical_data_reconstructed = numerical_flat.reshape(n_samples, 10, 10)
country_data_expanded = np.tile(country_data_reconstructed[:, np.newaxis, :], (1, 10, 1))  # (n_samples, 10, 219)
X_train_reconstructed = np.concatenate((numerical_data_reconstructed, country_data_expanded), axis=2)  # (n_samples, 10, 229)

# Verify shapes
print(f"X_train_reconstructed shape: {X_train_reconstructed.shape}")  # Expected: (4818, 10, 229)
print(f"y_train_reconstructed shape: {y_train_reconstructed.shape}")  # Expected: (4818, 5)

X_augmented = np.concatenate([X_train, X_train_reconstructed], axis=0)
y_augmented = np.concatenate([y_train, y_train_reconstructed], axis=0)

X_augmented_scale = scale_sequences(scaler, X_augmented)
X_val_scale = scale_sequences(scaler, X_val)
X_test_scale = scale_sequences(scaler, X_test)

print("Original training data:", X_train[0][0])
print("Synthetic data:", X_train_reconstructed[0][0])
print("Augmented data:", X_augmented[-1][-1])

print("Original training data:", X_train.shape)
print("Synthetic data:", X_train_reconstructed.shape)
print("Augmented data:", X_augmented.shape)

MODEL_MAPPING = {
    "LSTM": LSTMModel,
    "CNN-LSTM": CNNLSTMModel,
    "Transformer": TransformerModel
}

# Hyperparameter grid search
aug_lstm_preds = None
aug_cnn_lstm_preds = None
aug_transformer_preds = None

for name in models:
    best_model_info = {"model": None, "val_loss": float("inf"), "params": {}, "grid_search_params":{}, "path": None}
    model_class=  MODEL_MAPPING.get(name)
    param_grid = product(dropout_options, learning_rates, batch_sizes, epochs_list)

    for dropout, lr, batch_size, epochs in param_grid:
    # for dropout, lr, batch_size, epochs in param_grid:
        set_seed(seed)
        print(f"Training {name} with Dropout={dropout}, LR={lr}, Batch={batch_size}, Epochs={epochs}")
        train_loader = DataLoader(GDPDataset(X_augmented_scale, y_augmented), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(GDPDataset(X_val_scale, y_val), batch_size=batch_size)

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

    save_training_history(history, plot_path=f"{output_dir}/{name}_best_model_training_graph")
    final_model_path=os.path.join(time_models_dir, f"{name}.pth")

    model=model_class(dropout_rate=best_params["dropout"])
    model.load_state_dict(torch.load(model_path))

    torch.save(model.state_dict(), final_model_path, _use_new_zipfile_serialization=False)

    test_loader = DataLoader(GDPDataset(X_test_scale, y_test), batch_size=best_params['batch_size'] ,shuffle=False)
    actuals, predictions, country_labels, results = evaluate_model(model, test_loader, test_country_labels)
    predictions_csv_path = os.path.join(output_dir, f"{name}_test_predictions.csv")
    save_predict_results(country_labels, predictions, predictions_csv_path, label="predicted_gdp")


    if name == "LSTM":
        aug_lstm_preds = predictions
    elif name == "CNN-LSTM":
        aug_cnn_lstm_preds = predictions
    elif name == "Transformer":
        aug_transformer_preds = predictions
        
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

selected_countries = ["United States", "China", "Russian Federation", "Brazil", "Switzerland", "Denmark"] 
plot_gdp_predictions_multi(aug_lstm_preds, aug_cnn_lstm_preds, aug_transformer_preds, country_labels, selected_countries, output_dir)
task6_result_df = pd.read_csv(result_path)
print(task6_result_df.tail(3))