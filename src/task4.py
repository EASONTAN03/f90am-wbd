# Re-import after reset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import time
import os
import csv
import json
import numpy as np
import pandas as pd
import random
from itertools import product
from sklearn.metrics import mean_absolute_percentage_error

models=["Transformer"]  #"CNN-LSTM" "Transformer" "LSTM"
dropout_options = [0]
learning_rates = [0.001]
batch_sizes = [16]
epochs_list = [50]

# dropout_options = [0,0.1,0.2,0.3,0.4,0.5]
# learning_rates = [0.001]
# batch_sizes = [8]
# epochs_list = [30]

seed=11

sequences_per_country = 30
test_size = 5
val_size = 3

final_trainer=False
final_train_epochs=5

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(seed)

timestamp = int(time.time())
result_dir = "results/task4"
os.makedirs(result_dir, exist_ok=True)
models_dir = os.path.join(result_dir, "models")
os.makedirs(models_dir, exist_ok=True)
time_models_dir = os.path.join(models_dir, str(timestamp))
os.makedirs(time_models_dir, exist_ok=True)
data_dir=os.path.join(time_models_dir,"data")
os.makedirs(data_dir, exist_ok=True)
params_tune_dir=os.path.join(time_models_dir,"params_tune")
os.makedirs(params_tune_dir, exist_ok=True)
graphs_dir=os.path.join(time_models_dir,"output_graphs")
os.makedirs(graphs_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

features = [
        "Population_total", "Life_expectancy", "Literacy_rate", "Unemployment_rate",
        "Energy_use", "Fertility_rate", "Poverty_ratio", "Primary_school_enrolment_rate", "Exports_2017$"
    ]
target = "GDPpc_2017$"

# ---------------------- Data Preparation ----------------------
df = pd.read_csv(r"data/final_impute_world_bank_data_dev.csv")
df["date"] = pd.to_datetime(df["date"], format="%Y")
df = df.sort_values(["country", "date"]).reset_index(drop=True)

# Define features and target

#  One-hot encode country names 
df_countries = pd.get_dummies(df["country"], prefix="Country").astype(int)

# Scale GDP before log transformation
df[target] = np.log1p(df[target])

# Concatenate one-hot encoded countries
df_scaled = pd.concat([df, df_countries], axis=1)
features = features + list(df_countries.columns)

# ---------------------- Sequence Creation ----------------------
def create_sequences(data, input_length=10, output_length=5):
    """Generate sliding window sequences."""
    X, y = [], []
    
    for i in range(len(data) - input_length - output_length + 1):
        X.append(data[i:i+input_length, :]) 
        y.append(data[i+input_length:i+input_length+output_length, -1])  # GDP target
    return np.array(X), np.array(y)

def save_npy(X, result_dir, var_name):
    np.save(os.path.join(result_dir, f"{var_name}.npy"), X)

# ----------------------- Preparing Train, Val, Test data ----------------------
all_X_train, all_y_train, all_X_val, all_y_val, all_X_test, all_y_test = [], [], [], [], [], []
test_country_labels = []

for country in df["country"].unique():
    country_data = df_scaled[df_scaled["country"] == country][features + [target]].values
    #  Ensure at least 15 years of data exist for a full sequence 
    if len(country_data) < 15:
        print(f"⚠ Skipping {country} needs at least 15).")
        continue

    #  Create sequences for the country 
    X, y = create_sequences(country_data)
    train_size = len(X) - test_size - val_size

    X_train, y_train = X[:train_size], y[:train_size]  
    X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]  
    X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]  # ✅ Last 5 sequences → Test Set

    #  Append Data 
    all_X_train.append(X_train)
    all_y_train.append(y_train)
    all_X_val.append(X_val)
    all_y_val.append(y_val)
    all_X_test.append(X_test)
    all_y_test.append(y_test)

    #  Ensure Each Country Has Test Data 
    test_country_labels.extend([country] * len(X_test))

# Convert lists to numpy arrays
X_train = np.concatenate(all_X_train, axis=0)
y_train = np.concatenate(all_y_train, axis=0)
X_val = np.concatenate(all_X_val, axis=0)
y_val = np.concatenate(all_y_val, axis=0)
X_test = np.concatenate(all_X_test, axis=0)
y_test = np.concatenate(all_y_test, axis=0)

save_npy(X_train, data_dir, "X_train")
save_npy(y_train, data_dir, "y_train")
save_npy(X_val, data_dir, "X_val")
save_npy(y_val, data_dir, "y_val")
save_npy(X_test, data_dir, "X_test")
save_npy(y_test, data_dir, "y_test")

# ---------------------- Verify Data Splitting ----------------------
print(f"Total Training Sequences: {sum(len(x) for x in all_X_train)}")
print(f"Total Validation Sequences: {sum(len(x) for x in all_X_val)}")
print(f"Total Test Sequences: {sum(len(x) for x in all_X_test)}")

# ---------------------- Feature Scaling After Splitting ----------------------
num_features = len(features)  

# Flatten all training sequences to (n_samples*seq_len, n_features)
train_features = np.concatenate([
    x[:, :, :-1].reshape(-1, num_features)  # Use num_features instead of len(features)
    for x in all_X_train
])

scaler_X = StandardScaler().fit(train_features)  # Fit only once on training data

# Function to scale features while keeping GDP unchanged
def scale_sequences(sequences):
    features = sequences[:, :, :-1].reshape(-1, num_features)
    gdp = sequences[:, :, -1:]  # Keep GDP unchanged
    features_scaled = scaler_X.transform(features).reshape(sequences.shape[0], sequences.shape[1], num_features)
    return np.concatenate([features_scaled, gdp], axis=-1)

# Apply scaling
X_train = scale_sequences(np.concatenate(all_X_train))
X_val = scale_sequences(np.concatenate(all_X_val))
X_test = scale_sequences(np.concatenate(all_X_test))

# Verify shapes
print(f"Shapes after scaling:")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

class GDPDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
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

def plot_gdp_predictions_multi(actuals, lstm_preds, cnn_lstm_preds, transformer_preds, country_labels, selected_countries, save_dir=time_models_dir):
    """
    Plots predicted vs actual GDP for multiple countries across different models.
    
    Params:
        actuals (numpy array): Actual GDP values
        lstm_preds (numpy array): LSTM model predictions
        cnn_lstm_preds (numpy array): CNN-LSTM model predictions
        transformer_preds (numpy array): Transformer model predictions
        country_labels (list): Country names corresponding to each sequence
        selected_countries (list): List of countries to plot
    """
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

MODEL_MAPPING = {
    "LSTM": LSTMModel,
    "CNN-LSTM": CNNLSTMModel,
    "Transformer": TransformerModel
}

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

selected_countries = ["Brazil", "United States", "Germany"]  # Add more countries

plot_gdp_predictions_multi(actuals, lstm_preds, cnn_lstm_preds, transformer_preds, country_labels, selected_countries, graphs_dir)
