# ---------------------- Imports ----------------------
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import random
import time
import os
import csv
from itertools import product

batch_sizes = [16]
epochs_list= [50]
learning_rates=[0.001]

# batch_sizes = [16,32,64]
# epochs_list= [50,100,150]
# learning_rates=[0.001,0.01,0.1]

# ---------------------- Set Seed for Reproducibility ----------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ---------------------- Activate CUDA ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------- Data Preparation ----------------------
df = pd.read_csv(r"data\final_impute_world_bank_data_dev.csv")

df["date"] = pd.to_datetime(df["date"], format="%Y")
df = df.sort_values(["country", "date"]).reset_index(drop=True)


# Define features and target
features = [
    "Population_total", "Life_expectancy", "Literacy_rate", "Unemployment_rate",
    "Energy_use", "Fertility_rate", "Poverty_ratio", "Primary_school_enrolment_rate", "Exports_2017$"
]
target = "GDPpc_2017$"

# **One-hot encode country names**
df_countries = pd.get_dummies(df["country"], prefix="Country").astype(int)

# Initialize scalers
scaler_X = StandardScaler()

# Transform all data
df_scaled = df.copy()
df_scaled[features] = scaler_X.fit_transform(df[features])
df_scaled[target] = np.log1p(df[target])

# Initialize Standard Scaler for numerical features
scaler_X = StandardScaler()

# Apply transformations
df_scaled = df.copy()
df_scaled[features] = scaler_X.fit_transform(df[features])
df_scaled[target] = np.log1p(df[target])

# Concatenate one-hot encoded countries with scaled numerical features
df_scaled = pd.concat([df_scaled, df_countries], axis=1)
df_scaled.to_csv('data/task3_world_bank_data_dev.csv', index=False)
# print(df_scaled.head())

# **Update feature list to include one-hot encoded countries**
features = features + list(df_countries.columns)
# print(f"Total Features: {len(features)}")

timestamp = int(time.time())
result_dir = "results/task4"
os.makedirs(result_dir, exist_ok=True)

# ---------------------- Sequence Creation ----------------------
def create_sequences(data, input_length=10, output_length=5):
    """Generate sliding window sequences."""
    X, y = [], []
    
    for i in range(len(data) - input_length - output_length + 1):
        X.append(data[i:i+input_length, :])  # Features EXCEPT GDP
        y.append(data[i+input_length:i+input_length+output_length, -1])  # GDP target
    return np.array(X), np.array(y)

def save_npy(X, result_dir, var_name):
    np.save(os.path.join(result_dir, f"{timestamp}_{var_name}.npy"), X)

# ---------------------- Rolling Cross Validation ----------------------
all_X_train, all_y_train = [], []
all_X_val, all_y_val = [], []
all_X_test, all_y_test = [], []

for country in df["country"].unique():
    country_data = df_scaled[df_scaled['country'] == country][features + [target]].values
    if len(country_data) < 15:  
        continue  # Skip if not enough data

    # Create sequences for the country
    X, y = create_sequences(country_data)

    # **Rolling Split within the SAME COUNTRY**
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))

    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
    X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

    # Append to lists
    all_X_train.append(X_train)
    all_y_train.append(y_train)
    all_X_val.append(X_val)
    all_y_val.append(y_val)
    all_X_test.append(X_test)
    all_y_test.append(y_test)

print(X_train)

# Convert lists to numpy arrays
X_train = np.concatenate(all_X_train, axis=0)
y_train = np.concatenate(all_y_train, axis=0)
X_val = np.concatenate(all_X_val, axis=0)
y_val = np.concatenate(all_y_val, axis=0)
X_test = np.concatenate(all_X_test, axis=0)
y_test = np.concatenate(all_y_test, axis=0)

data_dir=os.path.join(result_dir,"data")
os.makedirs(data_dir, exist_ok=True)

# print(X_train,y_train)

save_npy(X_train, data_dir, "X_train")
save_npy(y_train, data_dir, "y_train")
save_npy(X_val, data_dir, "X_val")
save_npy(y_val, data_dir, "y_val")
save_npy(X_test, data_dir, "X_test")
save_npy(y_test, data_dir, "y_test")

np.load(f"{data_dir}/{timestamp}_X_train.npy")
np.load(f"{data_dir}/{timestamp}_y_train.npy")
np.load(f"{data_dir}/{timestamp}_X_val.npy")
np.load(f"{data_dir}/{timestamp}_y_val.npy")
np.load(f"{data_dir}/{timestamp}_X_test.npy")
np.load(f"{data_dir}/{timestamp}_y_test.npy")

print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")


# ---------------------- DataLoaders ----------------------
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
        Args:
            patience (int): Number of epochs to wait before stopping if no improvement.
            min_delta (float): Minimum change in loss to qualify as an improvement.
            verbose (bool): If True, prints early stopping updates.
            path (str): Path to save the best model.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.path = path
        self.best_loss = float("inf")  # Set to infinity to ensure the first epoch updates it
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, model):
        """Checks if validation loss improved, otherwise increments counter."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.path)  # Save best model
            if self.verbose:
                print(f"Validation loss improved! Model saved to {self.path}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience} (No improvement)")

            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"Early stopping triggered after {self.patience} epochs of no improvement.")

# ---------------------- Model Architectures ----------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size=229, output_size=5, hidden_size=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class CNNLSTMModel(nn.Module):
    def __init__(self, input_size=229, output_size=5, hidden_size=128, num_layers=2):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.lstm = nn.LSTM(64, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pool(nn.functional.relu(self.conv(x))) #Apply CNN+Pool
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class TransformerModel(nn.Module):
    def __init__(self, input_size=229, output_size=5, d_model=64, nhead=4, num_layers=3):
        super().__init__()
        self.embedding = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, output_size)
    
    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        return self.fc(x[:, -1, :])

def train_model(model, train_loader, val_loader, epochs, lr=0.001, model_path="best_model.pth"):
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    early_stopping = EarlyStopping(patience=10, min_delta=0.001, path=model_path)

    train_losses, val_losses = [], []
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)  # ✅ Move data to CUDA
            optimizer.zero_grad()
            y_pred = model(X)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_losses.append(train_loss)

        if val_loader is not None:
            val_loss = 0
            model.eval()
            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(device), y.to(device)  # ✅ Move data to CUDA
                    y_pred = model(X)
                    val_loss += criterion(y_pred, y).item()
            val_losses.append(val_loss)

            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

    return train_losses, val_losses
        

def evaluate_model(model, loader):
    model.to(device)
    model.eval()
    actuals, predictions = [], []
    
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)  # ✅ Move data to CUDA
            pred = model(X).cpu().numpy()
            predictions.append(pred)
            actuals.append(y.cpu().numpy())  # ✅ Ensure y is moved back to CPU
    
    predictions = np.concatenate(predictions)
    actuals = np.concatenate(actuals)

    # **Denormalize**
    predictions = np.expm1(predictions)
    actuals = np.expm1(actuals)

    print(len(predictions), len(actuals))
    print(type(predictions), type(actuals))


    # **Compute Metrics**
    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
    print(f"{name} Performance:\nMSE: {mse:.4f}\nMAE: {mae:.4f}\nMAPE: {mape:.4f}%")

    return actuals, predictions, [mse, mae, mape]

def save_training_history(history, file_path, output_dir):
    output_dir = f"{output_dir}/results_graphs"
    os.makedirs(output_dir, exist_ok=True)
    
    epochs = range(1, len(history["train_losses"]) + 1)
    
    plt.figure(figsize=(8, 5))
    
    # Loss Plot
    plt.plot(epochs, history["train_losses"], label='Train Loss')
    plt.plot(epochs, history["test_losses"], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.annotate(f"{history['train_losses'][-1]:.2f}", xy=(epochs[-1], history["train_losses"][-1]),
                 xytext=(epochs[-1] - 5, history["train_losses"][-1] + 0.05),
                 arrowprops=dict(arrowstyle="->", color="blue"), fontsize=10, color="blue")
    
    plt.annotate(f"{history['test_losses'][-1]:.2f}", xy=(epochs[-1], history["test_losses"][-1]),
                 xytext=(epochs[-1] - 5, history["test_losses"][-1] + 0.05),
                 arrowprops=dict(arrowstyle="->", color="orange"), fontsize=10, color="orange")
    
    plot_path = os.path.join(output_dir, f"{file_path}_training_history.png")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    
    print(f"Training history saved to {plot_path}")

# ---------------------- Run Training & Evaluation ----------------------
models = {
    "LSTM": LSTMModel(),
    "CNN-LSTM": CNNLSTMModel(),
    "Transformer": TransformerModel()
}

models_dir = os.path.join(result_dir, "models")
time_models_dir = os.path.join(models_dir, str(timestamp))
os.makedirs(models_dir, exist_ok=True)
os.makedirs(time_models_dir, exist_ok=True)

X_final_train = np.concatenate([X_train, X_val], axis=0)
y_final_train = np.concatenate([y_train, y_val], axis=0)

for name, model in models.items():
    best_loss= float("inf")
    best_param=None
    history = {}
    for batch_size, num_epochs, lr in product(batch_sizes, epochs_list, learning_rates):
        train_loader = DataLoader(GDPDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(GDPDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

        print(f"\nTraining {name}...")
        model_path = os.path.join(time_models_dir, f"{name}.pth")
        model.to(device)
        train_losses, val_losses=train_model(model, train_loader, val_loader, num_epochs, lr=lr, model_path=model_path)

        val_loss=min(val_losses)
        if val_loss < best_loss:
            best_loss = val_loss
            best_params = {'batch_size': batch_size, 'epochs': num_epochs, 'learning_rate': lr}
    
    print(f"\Training final {name}...")
    test_loader = DataLoader(GDPDataset(X_test, y_test), batch_size=best_params['batch_size'], shuffle=False)
    final_train_loader = DataLoader(GDPDataset(X_final_train, y_final_train), batch_size=best_params['batch_size'], shuffle=False)
            
    model.to(device)
    train_losses, val_losses=train_model(model, final_train_loader, test_loader, epochs=best_params['epochs'], lr=best_params['learning_rate'], model_path=model_path)

    history={
            'batch_size': batch_size, 'epochs': num_epochs, 'lr': lr, 
            'train_losses': train_losses, 'test_losses': val_losses,
    }

    save_training_history(history, file_path=f"{str(timestamp)}_{name}", output_dir=result_dir)

    model.load_state_dict(torch.load(model_path))
    actuals, predictions, results = evaluate_model(model, test_loader)

    os.makedirs(result_dir, exist_ok=True)
    result_path = f"{result_dir}/results.csv"
    file_exists = os.path.isfile(result_path)

    header = ["model_path", "params", "mse", "mae", "mape"]
    results=[model_path, str(best_params)] + results

    print(results)

    with open(result_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write header if the file does not exist
        if not file_exists:
            writer.writerow(header)
        # Write the statistics
        writer.writerow(results)
