# task5_vae_merged.py (with Early Stopping)

import os
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import product
from joblib import dump

# ---------------------- Setup ----------------------
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed = 11
set_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------- Hyperparameter Tuning ----------------------
hidden_dims = [128]
latent_dims = [20]
learning_rates = [0.01]
batch_sizes = [32]
epochs_list = [50]
# batch_size = 32
# epochs = 50
# learning_rate = 0.001
# latent_dim = 20


# ---------------------- Load Data ----------------------
result_dir_task4 = "results/task4"
timestamp_task4 = "1743690723"
timestamp = int(time.time())
result_dir = "results/task5"
os.makedirs(result_dir, exist_ok=True)
models_dir = os.path.join(result_dir, "models")
os.makedirs(models_dir, exist_ok=True)
time_models_dir = os.path.join(models_dir, str(timestamp))
os.makedirs(time_models_dir, exist_ok=True)
data_dir = os.path.join(time_models_dir, "data")
os.makedirs(data_dir, exist_ok=True)
output_dir = os.path.join(time_models_dir, "outputs")
os.makedirs(output_dir, exist_ok=True)

data_dir_task4 = os.path.join(result_dir_task4, "models", timestamp_task4, "data")
X_train = np.load(os.path.join(data_dir_task4, "X_train.npy"))  # (n_samples, 10, 229)
y_train = np.load(os.path.join(data_dir_task4, "y_train.npy"))  # (n_samples, 5)


input_dim = 10 * 10 + 5 + 219

# Flatten input: (n_samples, 10, 229) → (n_samples, 2290)
numerical_data = X_train[:, :, :10]  # shape: (1045, 10, 10)
numerical_flat = numerical_data.reshape(X_train.shape[0], -1)  # shape: (1045, 100)

# Extract country encoding from the first time step (assumed constant for each country)
country_data = X_train[:, 0, 10:]  # shape: (1045, 219)
flattened_data = np.concatenate((numerical_flat, y_train, country_data), axis=1)  # shape: (1045, 324)

print(f"Flattened data shape: {flattened_data.shape}, Expected input_dim: {input_dim}")

scaler=MinMaxScaler()
flattened_data_scaled = scaler.fit_transform(flattened_data)

dump(scaler, os.path.join(time_models_dir, "vae_scaler.joblib"))
dataset = torch.tensor(flattened_data_scaled, dtype=torch.float32)

# ---------------------- Early Stopping Class ----------------------
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
        self.path = path
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model_state = model.state_dict()
            torch.save(model.state_dict(), self.path, _use_new_zipfile_serialization=False)
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

# # ---------------------- VAE Model ----------------------
# class VAE(nn.Module):
#     def __init__(self, input_dim, hidden_dim=512, latent_dim=20):
#         super(VAE, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.BatchNorm1d(hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.BatchNorm1d(hidden_dim // 2),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(hidden_dim // 2, hidden_dim // 4),
#             nn.ReLU()
#         )
#         self.fc_mu = nn.Linear(hidden_dim // 4, latent_dim)
#         self.fc_logvar = nn.Linear(hidden_dim // 4, latent_dim)
#         self.decoder = nn.Sequential(
#             nn.Linear(latent_dim, hidden_dim // 4),
#             nn.ReLU(),
#             nn.Linear(hidden_dim // 4, hidden_dim // 2),
#             nn.BatchNorm1d(hidden_dim // 2),
#             nn.ReLU(),
#             nn.Linear(hidden_dim // 2, hidden_dim),
#             nn.BatchNorm1d(hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, input_dim)
#         )
    
#     def encode(self, x):
#         h = self.encoder(x)
#         return self.fc_mu(h), self.fc_logvar(h)
    
#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std
    
#     def decode(self, z):
#         return self.decoder(z)
    
#     def forward(self, x):
#         mu, logvar = self.encode(x)
#         z = self.reparameterize(mu, logvar)
#         recon_x = self.decode(z)
#         return recon_x, mu, logvar

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
    
# ---------------------- Loss Function ----------------------
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    elbo = -(recon_loss + kl_div)
    return elbo, recon_loss, kl_div

# ---------------------- Training Function with Early Stopping ----------------------
def train_vae(model, train_loader, epochs, lr, early_stopping):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_history = {'elbo_loss': [], 'recon_loss': [], 'kl_loss': []}

    for epoch in range(epochs):
        model.train()
        total_recon_loss = 0
        total_kl_div = 0
        total_elbo = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(batch)
            elbo, recon_loss, kl_div = vae_loss(recon_batch, batch, mu, logvar)
            loss = recon_loss + kl_div
            loss.backward()
            optimizer.step()
            total_recon_loss += recon_loss.item()
            total_kl_div += kl_div.item()
            total_elbo += elbo.item()
        
        avg_recon_loss = total_recon_loss / len(train_loader.dataset)
        avg_kl_div = total_kl_div / len(train_loader.dataset)
        avg_elbo = total_elbo / len(train_loader.dataset)
        
        train_history['elbo_loss'].append(avg_elbo)
        train_history['recon_loss'].append(avg_recon_loss)
        train_history['kl_loss'].append(avg_kl_div)

        
        print(f"Epoch {epoch+1}/{epochs} | Recon Loss: {avg_recon_loss:.4f} | KL Div: {avg_kl_div:.4f} | ELBO: {avg_elbo:.4f}")
    
        # Early stopping check
        early_stopping(avg_elbo, model)
        if early_stopping.early_stop:
            print(f"Training stopped early at epoch {epoch+1}")
            break

    # Load the best model state back into the model
    model.load_state_dict(early_stopping.best_model_state)
    return train_history, early_stopping.best_loss

def save_training_history(train_history, output_dir):
    plot_path = os.path.join(output_dir, f"training_history.png")
    epochs_trained = len(train_history['elbo_loss'])
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs_trained+1), train_history['recon_loss'], label='Reconstruction Loss')
    plt.plot(range(1, epochs_trained+1), train_history['kl_loss'], label='KL Divergence')
    plt.plot(range(1, epochs_trained+1), train_history['elbo_loss'], label='ELBO')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('VAE Training Metrics')
    plt.legend()
    plt.grid()
    plt.savefig(plot_path)
    plt.close()
    print(f"Training metrics plot saved at: {plot_path}")

# ---------------------- Visualization ----------------------
def visualize_latent_space(model, data_loader, save_path):
    model.eval()
    latent_vectors = []
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            mu, logvar = model.encode(batch)
            z = model.reparameterize(mu, logvar)
            latent_vectors.append(z.cpu().numpy())
    latent_vectors = np.vstack(latent_vectors)
    tsne = TSNE(n_components=3, random_state=seed, perplexity=30)
    latent_3d = tsne.fit_transform(latent_vectors)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(latent_3d[:, 0], latent_3d[:, 1], latent_3d[:, 2], s=10, alpha=0.6)
    ax.set_title("Latent Space Visualization (t-SNE)")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_zlabel("Component 3")
    plt.savefig(save_path)
    plt.close()
    return latent_vectors

# ---------------------- Grid Search ----------------------
best_elbo_loss = float('inf')
best_config = {
    "params": None,
    "grid_search_params": {
        "hidden_dims": hidden_dims,
        "latent_dims": latent_dims,
        "learning_rates": learning_rates,
        "batch_sizes": batch_sizes,
        "epochs_list": epochs_list
    }
}
best_model = None
best_latents = None
best_history = None

param_grid = product(hidden_dims, latent_dims, learning_rates, batch_sizes, epochs_list)

for hidden_dim, latent_dim, lr, batch_size, epochs in param_grid:
    set_seed(seed)
    print(f"\nTraining VAE | Hidden: {hidden_dim} | Latent: {latent_dim} | LR: {lr} | Batch: {batch_size} | Epochs: {epochs}")
    
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    vae = VAE(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    
    # Initialize EarlyStopping
    early_stopping = EarlyStopping(
        patience=20,  # Wait 10 epochs before stopping
        min_delta=0.001,  # Minimum improvement required
        verbose=True,
        path=os.path.join(time_models_dir, "vae_best.pth")
    )
    
    train_history, elbo_loss = train_vae(vae, train_loader, epochs, lr, early_stopping)

    if elbo_loss < best_elbo_loss:
        best_elbo_loss = elbo_loss
        best_config["params"] = {
            "hidden_dim": hidden_dim,
            "latent_dim": latent_dim,
            "lr": lr,
            "batch_size": batch_size,
            "epochs": epochs
        }
        best_model = vae
        best_history = train_history
        best_latents = visualize_latent_space(vae, train_loader, os.path.join(output_dir, "best_latent_tsne.png"))

# ---------------------- Save Final Outputs ----------------------
best_model_config_path=os.path.join(time_models_dir, "best_model_info.json")
with open(best_model_config_path, "w") as f:
    json.dump(best_config, f, indent=4)
save_training_history(best_history, output_dir)

latent_df = pd.DataFrame(best_latents)
latent_df.to_csv(os.path.join(output_dir, "vae_latent_vectors.csv"), index=False)

train_loader = DataLoader(dataset, batch_size=best_config["params"]["batch_size"])
reconstructed = []
best_model.eval()
with torch.no_grad():
    for batch in train_loader:
        batch = batch.to(device)
        recon, _, _ = best_model(batch)
        reconstructed.append(recon.cpu().numpy())
reconstructed = np.concatenate(reconstructed, axis=0)
reconstructed_df = pd.DataFrame(reconstructed)
reconstructed_df.to_csv(os.path.join(output_dir, "vae_reconstructed_data.csv"), index=False)

# Additional Visualizations
plt.figure(figsize=(15, 10))
n_cols = 5
n_rows = int(np.ceil(best_config["params"]["latent_dim"] / n_cols))
for i in range(best_config["params"]["latent_dim"]):
    plt.subplot(n_rows, n_cols, i+1)
    plt.hist(best_latents[:, i], bins=30, alpha=0.7)
    plt.title(f"Latent dim {i+1}")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "latent_dimensions_histograms.png"))
plt.close()

plt.figure(figsize=(8, 6))
plt.scatter(best_latents[:, 0], best_latents[:, 1], alpha=0.6)
plt.xlabel("Latent Dimension 1")
plt.ylabel("Latent Dimension 2")
plt.title("Pairwise Scatter: Latent Dimensions 1 vs 2")
plt.savefig(os.path.join(output_dir, "latent_pairwise_scatter.png"))
plt.close()

print(f"\n✅ Best model saved with config: {best_config}")
print(f"Outputs saved in: {output_dir}")
print(f"Final Elbo Loss: {best_history['elbo_loss'][-1]:.4f}")