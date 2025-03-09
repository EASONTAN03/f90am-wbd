import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df=pd.read_csv(r'data\final_impute_world_bank_data_dev.csv')

# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Normalize features using MinMaxScaler (excluding 'country' and 'date')
scaler = MinMaxScaler()
feature_cols = df.columns[2:]  # Exclude 'country' and 'date'
df[feature_cols] = scaler.fit_transform(df[feature_cols])

# Function to create input-output sequences
def create_sequences(data, country_col, target_col, input_window=10, output_window=5):
    sequences = []
    targets = []
    
    grouped = data.groupby(country_col)  # Group by country
    for _, group in grouped:
        group = group.sort_values(by="date")  # Ensure chronological order
        values = group.drop(columns=[country_col, "date"]).values  # Convert to numpy array
        target_idx = group.columns.get_loc(target_col)  # Get index of target column
        
        for i in range(len(values) - input_window - output_window + 1):
            input_seq = values[i : i + input_window]  # 10-year input window
            output_seq = values[i + input_window : i + input_window + output_window, target_idx]  # GDP forecast
            sequences.append(input_seq)
            targets.append(output_seq)
    
    return np.array(sequences), np.array(targets)

# Create sequences
X, y = create_sequences(df, country_col="country", target_col="GDPpc_2017$")

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape)
# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

# Create DataLoaders
batch_size = 32
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=batch_size, shuffle=False)

test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=batch_size, shuffle=False)

class EarlyStopping():
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# Define LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

# Define CNN-LSTM Model
class CNN_LSTM_Model(nn.Module):
    def __init__(self, input_dim, num_filters, hidden_dim, output_dim):
        super(CNN_LSTM_Model, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=num_filters, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.lstm = nn.LSTM(num_filters, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pool(torch.relu(self.conv1d(x)))
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

# Define Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 10, model_dim))
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(model_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x) + self.positional_encoding
        x = self.transformer_encoder(x)
        out = self.fc(x[:, -1, :])
        return out

# Model Initialization
input_dim = 10
hidden_dim = 64
num_filters = 32
model_dim = 64
num_heads = 4
num_layers = 2
output_dim = 5

lstm_model = LSTMModel(input_dim, hidden_dim, output_dim).to(device)
cnn_lstm_model = CNN_LSTM_Model(input_dim, num_filters, hidden_dim, output_dim).to(device)
transformer_model = TransformerModel(input_dim, model_dim, num_heads, num_layers, output_dim).to(device)

# Training Function
def train_model(model, train_loader, batch_sizes, epochs_list, learning_rates):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    early_stopping = EarlyStopping(patience=10, min_delta=0.001)

    bes_loss = 0
    best_params = None


    train_losses, val_losses = [], []

    for batch_size, num_epochs, lr in product(batch_sizes, epochs_list, learning_rates):
            train_dataset = TensorDataset(train_X_tensor, train_y_tensor)
            val_dataset = TensorDataset(train_X_tensor, train_y_tensor)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)  # Store training loss

        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}')

    model.eval()
    val_loss, correct_val, total_val = 0, 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1}/{num_epochs} completed.")

    early_stopping(val_loss)
    if early_stopping.early_stop:
        print("Early stopping triggered")
        break



    # Update best model if the current one is better
    if val_loss < best_loss:
        best_loss = val_loss
        best_params = {'epochs': num_epochs, 'learning_rate': lr}
    

# Train Models
optimizer_lstm = optim.Adam(lstm_model.parameters(), lr=0.001)
optimizer_cnn_lstm = optim.Adam(cnn_lstm_model.parameters(), lr=0.001)
optimizer_transformer = optim.Adam(transformer_model.parameters(), lr=0.001)

print("Training LSTM...")
train_model(lstm_model, train_loader, val_loader)

print("Training CNN-LSTM...")
train_model(cnn_lstm_model, train_loader, val_loader)

print("Training Transformer...")
train_model(transformer_model, train_loader, val_loader)

print("Training complete.")
