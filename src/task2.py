import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from itertools import product
import time
import ast

df=pd.read_csv('data/task2_world_bank_data_dev.csv')

seed=11

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(seed)

batch_sizes = [32]
epochs_list = [30, 50, 70]
learning_rates = [0.001, 0.01, 0.1]
dropout_rate = [0,0.1,0.2,0.3,0.4]
latent_list = [4, 5, 6]

grid_params = {'batch_sizes': batch_sizes, 'epochs_list': epochs_list, 'learning_rates': learning_rates, "dropout_rate": dropout_rate, "latent_list": latent_list}

# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

timestamp = int(time.time())
result_dir = f"results/task2"
os.makedirs(result_dir, exist_ok=True)
models_dir = os.path.join(result_dir, "models")
os.makedirs(models_dir, exist_ok=True)
timestamp_dir = os.path.join(models_dir, str(timestamp))
os.makedirs(timestamp_dir, exist_ok=True)
output_dir = os.path.join(timestamp_dir, "outputs")
os.makedirs(output_dir, exist_ok=True)

def preprocess_data(df, countries):
    aggregated_data = []
    country_names = []
    
    for country in countries:
        country_df = df.loc[df["country"] == country].copy()  # Avoid SettingWithCopyWarning
        
        indicator = "sequence"
        
        # Convert string representations of lists to actual lists
        if isinstance(country_df[indicator].iloc[0], str):  
            country_df.loc[:, indicator] = country_df[indicator].apply(lambda x: ast.literal_eval(x))

        # Convert tuples to lists
        if isinstance(country_df[indicator].iloc[0], tuple):  
            country_df.loc[:, indicator] = country_df[indicator].apply(list)

        if not isinstance(country_df[indicator].iloc[0], (list, np.ndarray)):
            raise ValueError(f"Indicator {indicator} is not a sequence but {type(country_df[indicator].iloc[0])}")

        # Convert sequences into NumPy array
        stacked_values = np.vstack(country_df[indicator].values)

        # Compute statistics for each time step
        stats_dict = {
            f"{indicator}_mean": np.mean(stacked_values, axis=0),
            f"{indicator}_median": np.median(stacked_values, axis=0),
            f"{indicator}_max": np.max(stacked_values, axis=0),
            f"{indicator}_min": np.min(stacked_values, axis=0),
            f"{indicator}_var": np.var(stacked_values, axis=0),
            f"{indicator}_kurtosis": np.apply_along_axis(lambda x: pd.Series(x).kurt(), 0, stacked_values),  # ✅ FIXED
        }

        aggregated_data.append(np.hstack(list(stats_dict.values())))  # Flatten into a single row
        country_names.append(country)

    return np.array(aggregated_data), country_names

class EarlyStopping():
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False
        self.best_model=None

    def __call__(self, model, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_model = model
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, dropout):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return latent, reconstructed
    
def train_autoencoder(data, batch_size, num_epochs, lr, latent_dim, dropout):
    input_dim = data.shape[1]
    train_dataset = TensorDataset(torch.tensor(data, dtype=torch.float32).to(device))    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    model = Autoencoder(input_dim, latent_dim, dropout).to(device)
    early_stopping = EarlyStopping(patience=20, min_delta=0.001)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    train_losses = []

    for epoch in range(num_epochs):
        running_loss = 0
        for batch in train_loader:
            batch = batch[0].to(device)
            optimizer.zero_grad()
            latent, reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)  # Store training loss
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}")
        
        scheduler.step(train_loss)

        early_stopping(model, running_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    return train_losses, early_stopping.best_model

# Clustering function
def cluster_countries(latent_data, n_clusters=4):
    kmeans = KMeans(n_clusters=n_clusters, random_state=11)
    clusters = kmeans.fit_predict(latent_data)

    # Compute clustering metrics
    silhouette = silhouette_score(latent_data, clusters)
    davies_bouldin = davies_bouldin_score(latent_data, clusters)

    # Compute clustering metrics
    silhouette = silhouette_score(latent_data, clusters)
    davies_bouldin = davies_bouldin_score(latent_data, clusters)
    calinski_harabasz = calinski_harabasz_score(latent_data, clusters)
    wcss = kmeans.inertia_  # Sum of squared distances of samples to their closest cluster center

    print(f"Silhouette Score: {silhouette}")
    print(f"Davies-Bouldin Index: {davies_bouldin}")
    print(f"Calinski-Harabasz Index: {calinski_harabasz}")
    print(f"WCSS (Within-Cluster Sum of Squares): {wcss}")

    # Save results to CSV
    performance_results = {
        "model_path": [cluster_model_path],  # Wrap scalars in lists
        "Silhouette Score": [silhouette],
        "Davies-Bouldin Index": [davies_bouldin],
        "Calinski-Harabasz Index": [calinski_harabasz], 
        "WCSS": [wcss]
    }

    return kmeans, clusters, performance_results

def save_training_history(history, plot_dir):
    plot_path = os.path.join(plot_dir, f"training_history_batch{history['batch_size']}_lr{history['lr']}.png")
    epochs = range(1, len(history["train_losses"]) + 1)

    # Create a single plot for both accuracy & loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["train_losses"], label="Train Loss", linestyle="-", marker="s", color="red")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy / Loss")
    plt.legend()
    plt.grid()
    plt.title(f"Training History (Batch {history['batch_size']}, LR {history['lr']})")
    plt.savefig(plot_path)
    plt.close()
    print(f"Training history plot saved at: {plot_path}")

def visualize_latent(latent_variance, plot_dir):
    # Print variance values
    plot_path = os.path.join(plot_dir, f"latent_variance.png")
    print("\nLatent Variance for Each Dimension:")
    for i, var in enumerate(latent_variance):
        print(f"Latent Dimension {i+1}: {var:.6f}")

    # Plot variance distribution
    plt.figure(figsize=(8, 5))
    sns.barplot(x=np.arange(1, len(latent_variance) + 1), y=latent_variance, palette="viridis")
    plt.xlabel("Latent Dimension")
    plt.ylabel("Variance")
    plt.title("Variance of Latent Representations")
    plt.xticks(range(1, len(latent_variance) + 1))
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Save plot
    plt.savefig(plot_path)
    plt.close
    print(f"Latent variance plot saved at: {plot_path}")

def visualize_clusters(latent_data, clusters, country_names, plot_dir):
    plot_path = os.path.join(plot_dir, f"tsne.png")

    tsne = TSNE(n_components=2, random_state=seed)
    tsne_results = tsne.fit_transform(latent_data)
    
    df_viz = pd.DataFrame({
        "TSNE1": tsne_results[:, 0],
        "TSNE2": tsne_results[:, 1],
        "Cluster": clusters,
        "Country": country_names
    })
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="TSNE1", y="TSNE2", hue="Cluster", data=df_viz, palette="viridis", legend="full")
    for i, country in enumerate(df_viz["Country"]):
        plt.text(df_viz["TSNE1"].iloc[i], df_viz["TSNE2"].iloc[i], country, fontsize=8)
    plt.title("Clusters of Countries Based on Economic Indicators")
    plt.savefig(plot_path)
    plt.close()
    print(f"TSNE of clusters plot saved at: {plot_path}")


countries = list(sorted(df['country'].unique()))
data, country_names = preprocess_data(df, countries)
print(data)
print(country_names)

best_loss = float("inf")
best_silhouette = -1
best_params = None
history = {}  # Store loss and accuracy per epoch
final_clusters=None
final_latent_data=None
final_performance=None

cluster_model_path = os.path.join(timestamp_dir,f"clusters_{timestamp}.pkl") 

for batch_size, num_epochs, lr, latent_dim, dropout in product(batch_sizes, epochs_list, learning_rates, latent_list, dropout_rate):
    print(f"Training with batch_size={batch_size}, epochs={num_epochs}, learning_rate={lr}, 'latent_dim={latent_dim}")
    set_seed(seed)
    train_losses, trained_autoencoder = train_autoencoder(data, batch_size, num_epochs, lr, latent_dim, dropout)

    latent_data = trained_autoencoder.encoder(torch.tensor(data, dtype=torch.float32).to(device))
    latent_data = latent_data.cpu().detach().numpy()
    latent_variance=np.var(latent_data, axis=0)
    print("\nLatent Variance for Each Dimension:")
    for i, var in enumerate(latent_variance):
        print(f"Latent Dimension {i+1}: {var:.6f}")

    min_variance_threshold = 1
    num_dead_neurons = np.sum(latent_variance < min_variance_threshold)
    if num_dead_neurons > 0:
        print(f"❌ Skipping due to {num_dead_neurons} dead neurons (variance < {min_variance_threshold})")
        continue
    try:
        kmeans, clusters, performance_results = cluster_countries(latent_data)
    except ValueError as e:
        continue

    if performance_results['Silhouette Score'][0]>best_silhouette:
    # Store training history for plotting
        joblib.dump(kmeans, cluster_model_path)
        print(f"KMeans clustering model saved at {cluster_model_path}")
        final_clusters=clusters
        final_performance=performance_results
        final_latent_data=latent_data
        best_silhouette = performance_results['Silhouette Score'][0]
        best_params = {'batch_size': batch_size, 'epochs': num_epochs, 'learning_rate': lr, "dropout_rate": dropout, "latent_dim": latent_dim}
        best_model = trained_autoencoder
        history={
            'batch_size': batch_size, 'epochs': num_epochs, 'lr': lr, 'train_losses': train_losses
        }


model_path = os.path.join(timestamp_dir, f"autoencoder_{timestamp}.pth")
torch.save(best_model.state_dict(), model_path)

save_training_history(history, output_dir)
visualize_latent(np.var(final_latent_data, axis=0), output_dir)

# Perform clustering
final_performance["best_params"] = [best_params]
final_performance["grid_params"] = [grid_params]
if final_performance is not None:
    print("Final Performance Metrics:")
    print(final_performance)
else:
    print("No final performance metrics available (no valid model found).")
performance_df = pd.DataFrame(final_performance)
performance_csv_path = f"{result_dir}/clustering_performance.csv"
# Check if the CSV file exists before writing
if os.path.exists(performance_csv_path):
    performance_df.to_csv(performance_csv_path, mode='a', header=False, index=False)  # Append without headers
else:
    performance_df.to_csv(performance_csv_path, index=False)  # Write with headers

print(f"Clustering performance results saved at {performance_csv_path}")

# Save clustering assignments
cluster_assignments_df = pd.DataFrame({
    "country": country_names,
    "cluster": final_clusters
})
cluster_assignments_csv_path = os.path.join(output_dir, "cluster_assignments.csv")
if os.path.exists(cluster_assignments_csv_path):
    cluster_assignments_df.to_csv(cluster_assignments_csv_path, mode='a', header=False, index=False)
else:
    cluster_assignments_df.to_csv(cluster_assignments_csv_path, index=False)
print(f"Cluster assignments saved at: {cluster_assignments_csv_path}")

# Visualize clusters
visualize_clusters(final_latent_data, final_clusters, country_names, output_dir)

