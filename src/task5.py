# ---------------------- Imports ----------------------
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
import os
import time

# ---------------------- Set Seed for Reproducibility ----------------------
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ---------------------- Activate CUDA ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------- Data Preparation ----------------------
result_dir = "results/task4"
timestamp = "1741748824"
data_dir = os.path.join(result_dir, "data")

X_train = np.load(f"{data_dir}/{timestamp}_X_train.npy")  # Shape: (1045, 10, 229)
y_train = np.load(f"{data_dir}/{timestamp}_y_train.npy")  # Shape: (1045, 5)

# Feature breakdown
numerical_features = 10  # 10 numerical features (e.g., including GDP)
country_features = 229 - numerical_features  # 219

# Extract components
numerical_data = X_train[:, :, :numerical_features]  # Shape: (1045, 10, 10)
country_data = X_train[:, 0, numerical_features:]    # Shape: (1045, 219)

# Flatten numerical data
numerical_flat = numerical_data.reshape(X_train.shape[0], -1)  # Shape: (1045, 100)

# Concatenate: numerical_data, y_train (GDP output), country_data
flattened_data = np.concatenate((numerical_flat, y_train, country_data), axis=1)  # Shape: (1045, 324)
input_dim = 10 * numerical_features + 5 + country_features  # 100 + 5 + 219 = 324


# Extract components
numerical_data = X_train[:, :, :numerical_features]  # Shape: (1045, 10, 10)

# Flatten numerical data
numerical_flat = numerical_data.reshape(X_train.shape[0], -1)  # Shape: (1045, 100)

# Concatenate: numerical_data, y_train (GDP output), country_data
flattened_data = np.concatenate((numerical_flat, y_train), axis=1)  # Shape: (1045, 324)
input_dim = 10 * numerical_features + 5   # 100 + 5 + 219 = 324


print(f"Flattened data shape: {flattened_data.shape}, Expected dimension: {input_dim}")

# ---------------------- Dataset ----------------------
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
def visualize_latent_space(model, data_loader, result_dir, timestamp):
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
    plot_path = f"{result_dir}/vae_latent_3d_{timestamp}.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Latent space visualization saved at: {plot_path}")

# ---------------------- Main Execution ----------------------
if __name__ == "__main__":
    batch_size = 32
    epochs = 50
    learning_rate = 0.001
    latent_dim = 20
    
    timestamp = int(time.time())
    result_dir = "results/task5"
    os.makedirs(result_dir, exist_ok=True)
    
    train_dataset = VAEDataset(flattened_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    vae = VAE(input_dim=input_dim, hidden_dim=128, latent_dim=latent_dim)
    recon_losses, kl_divs, elbos = train_vae(vae, train_loader, epochs=epochs, lr=learning_rate)
    
    model_path = f"{result_dir}/vae_{timestamp}.pth"
    torch.save(vae.state_dict(), model_path)
    print(f"VAE model saved at: {model_path}")
    
    visualize_latent_space(vae, train_loader, result_dir, timestamp)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs+1), recon_losses, label='Reconstruction Loss')
    plt.plot(range(1, epochs+1), kl_divs, label='KL Divergence')
    plt.plot(range(1, epochs+1), elbos, label='ELBO')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('VAE Training Metrics')
    plt.legend()
    plt.grid()
    metrics_plot_path = f"{result_dir}/vae_metrics_{timestamp}.png"
    plt.savefig(metrics_plot_path)
    plt.close()
    print(f"Training metrics plot saved at: {metrics_plot_path}")