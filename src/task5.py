# task 5 do not remove one-hot encoding
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from joblib import dump
import time
import pandas as pd

seed=11
dataset_timestamp = "1744209367"

batch_size = 32
epochs = 200
learning_rate = 0.001
hidden_dim = 256
latent_dim = 20

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

task4_data_dir = f"results/task4/models/{dataset_timestamp}/data"

X_train = np.load(os.path.join(task4_data_dir, "X_train.npy"))  # shape: (n_samples, 10, 229)
y_train = np.load(os.path.join(task4_data_dir, "y_train.npy"))  # shape: (n_samples, 5)

timestamp = int(time.time())
result_dir = "results/task5"
os.makedirs(result_dir, exist_ok=True)
models_dir = os.path.join(result_dir, "models")
os.makedirs(models_dir, exist_ok=True)
time_models_dir = os.path.join(models_dir, str(timestamp))
os.makedirs(time_models_dir, exist_ok=True)
output_dir=os.path.join(time_models_dir,"outputs")
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(r"data/final_impute_world_bank_data_dev.csv")
country_features = len(df["country"].unique())

input_dim = 10 * 10 + country_features + 5     # 10*10=100, plus 5 plus country_features (e.g., 219) equals 324
print(X_train[0])
numerical_data = X_train[:, :, :10]  # shape: (1045, 10, 10)
numerical_flat = numerical_data.reshape(X_train.shape[0], -1)  # shape: (1045, 100)

# Extract country encoding from the first time step (assumed constant for each country)
country_data = X_train[:, 0, 10:]  # shape: (1045, 219)

# Concatenate numerical data, y_train (GDP target), and country_data along axis=1
flattened_data = np.concatenate((numerical_flat, country_data,  y_train), axis=1)  # shape: (1045, 324)
print(f"Flattened data shape: {flattened_data.shape}, Expected input_dim: {input_dim}")

scaler = MinMaxScaler()
flattened_data_scaled = scaler.fit_transform(flattened_data)
dump(scaler, os.path.join(time_models_dir, "vae_scaler.joblib"))

# ---------------------- Dataset for VAE ----------------------
class VAEDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

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

# ---------------------- Loss Function ----------------------
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    elbo = -(recon_loss + kl_div)
    return recon_loss, kl_div, elbo

# ---------------------- Training Function ----------------------
def train_vae(model, train_loader, epochs=50, lr=0.001):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_recon_losses = []
    train_kl_divs = []
    train_elbos = []
    
    for epoch in range(epochs):
        model.train()
        total_recon_loss = 0
        total_kl_div = 0
        total_elbo = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(batch)
            recon_loss, kl_div, elbo = vae_loss(recon_batch, batch, mu, logvar)
            loss = recon_loss + kl_div
            loss.backward()
            optimizer.step()
            total_recon_loss += recon_loss.item()
            total_kl_div += kl_div.item()
            total_elbo += elbo.item()
        
        avg_recon_loss = total_recon_loss / len(train_loader.dataset)
        avg_kl_div = total_kl_div / len(train_loader.dataset)
        avg_elbo = total_elbo / len(train_loader.dataset)
        
        train_recon_losses.append(avg_recon_loss)
        train_kl_divs.append(avg_kl_div)
        train_elbos.append(avg_elbo)
        
        print(f"Epoch {epoch+1}/{epochs} | Recon Loss: {avg_recon_loss:.4f} | KL Div: {avg_kl_div:.4f} | ELBO: {avg_elbo:.4f}")
    
    return train_recon_losses, train_kl_divs, train_elbos

# ---------------------- Visualization ----------------------
def visualize_latent_space(model, data_loader, output_dir):
    model.eval()
    latent_vectors = []
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            mu, logvar = model.encode(batch)
            z = model.reparameterize(mu, logvar)
            latent_vectors.append(z.cpu().numpy())
    latent_vectors = np.concatenate(latent_vectors, axis=0)
    tsne = TSNE(n_components=3, random_state=42)
    latent_3d = tsne.fit_transform(latent_vectors)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(latent_3d[:, 0], latent_3d[:, 1], latent_3d[:, 2], c='b', marker='o')
    ax.set_title("3D t-SNE of VAE Latent Space")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_zlabel("t-SNE 3")
    plt.savefig(os.path.join(output_dir, "vae_latent_3d.png"))
    plt.show()

    plt.figure(figsize=(15, 10))
    n_cols = 5
    n_rows = int(np.ceil(latent_dim / n_cols))
    for i in range(latent_dim):
        plt.subplot(n_rows, n_cols, i+1)
        plt.hist(latent_vectors[:, i], bins=30, alpha=0.7)
        plt.title(f"Latent dim {i+1}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "latent_dimensions_histograms.png"))
    plt.show()

    # -------------------------
    # Pairwise Latent Scatter
    # -------------------------
    plt.figure(figsize=(8, 6))
    plt.scatter(latent_vectors[:, 0], latent_vectors[:, 1], alpha=0.6)
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.title("Pairwise Scatter: Latent Dimensions 1 vs 2")
    plt.savefig(os.path.join(output_dir, "latent_pairwise_scatter.png"))
    plt.show()

# ---------------------- Main Execution ----------------------
# Use the scaled flattened data for VAE training.
train_dataset = VAEDataset(flattened_data_scaled)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

vae = VAE(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
recon_losses, kl_divs, elbos = train_vae(vae, train_loader, epochs=epochs, lr=learning_rate)

model_path = os.path.join(time_models_dir, f"vae.pth")
torch.save(vae.state_dict(), model_path)
print(f"VAE model saved at: {model_path}")

visualize_latent_space(vae, train_loader, output_dir)
    
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs+1), recon_losses, label='Reconstruction Loss')
plt.plot(range(1, epochs+1), kl_divs, label='KL Divergence')
plt.plot(range(1, epochs+1), elbos, label='ELBO')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('VAE Training Metrics')
plt.legend()
plt.grid()
metrics_plot_path = os.path.join(output_dir, f"vae_training_metrics.png")
plt.savefig(metrics_plot_path)
plt.show()
print(f"Training metrics plot saved at: {metrics_plot_path}")

