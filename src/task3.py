import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from itertools import product
import time
import csv

normalised_imputed_df=pd.read_csv('data/normalised_imputed_world_bank_data_dev.csv')

torch.manual_seed(11)
np.random.seed(11)

# Ensure GPU usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Early Stopping Class
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

# Define model class
class GDPClassifier(nn.Module):
    def __init__(self, input_size):
        super(GDPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 4)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return self.softmax(x)

# Load and preprocess data
def load_and_preprocess_data(df):
    avg_gdp = df.groupby('country')['GDPpc_2017$'].mean().reset_index()
    avg_gdp['GDP_class'] = pd.qcut(avg_gdp['GDPpc_2017$'], q=4, labels=["Under-developed", "Developing", "Emerging", "Developed"])

    features = df.columns.difference(['country', 'date', 'GDPpc_2017$'])
    aggregated_data = df.groupby('country')[features].mean().reset_index()
    merged_data = aggregated_data.merge(avg_gdp[['country', 'GDP_class']], on='country')

    le = LabelEncoder()
    merged_data['label'] = le.fit_transform(merged_data['GDP_class'])

    X = merged_data.drop(['country', 'GDP_class', 'label'], axis=1).values
    y = merged_data['label'].values

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Convert to torch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y, dtype=torch.long, device=device)

    return X_tensor, y_tensor, le

def train_model(input_shape, train_loader, val_loader, batch_size, num_epochs, lr):
    # Split dataset (train 70%, validation 15%, test 15%)
    print(f"Training with batch_size={batch_size}, epochs={num_epochs}, learning_rate={lr}")
    
    model = GDPClassifier(input_size=input_shape).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    early_stopping = EarlyStopping(patience=10, min_delta=0.001)
    
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct_train, total_train = 0, 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_loss /= len(train_loader)
        train_accuracy = correct_train / total_train

        train_losses.append(train_loss)  # Store training loss
        train_accuracies.append(train_accuracy)  # Store training accuracy

        # Validation evaluation
        correct_val, total_val = 0, 0
        running_loss, val_loss = 0, 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = correct_val / total_val

        val_losses.append(val_loss)  # Store validation loss
        val_accuracies.append(val_accuracy)  # Store validation accuracy

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    
    params = {'batch_size': batch_size, 'epochs': num_epochs, 'learning_rate': lr}

    # Store training history for plotting
    history={
        'batch_size': batch_size, 'epochs': num_epochs, 'lr': lr, 
        'train_losses': train_losses, 'test_losses': val_losses,
        'train_accuracies': train_accuracies, 'test_accuracies': val_accuracies
    }

    return model, params, history  # Return history for plotting

def train_model_cv(X_tensor, y_tensor, batch_sizes, epochs_list, learning_rates, k_folds=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_accuracy = 0
    best_params = None
    history = {}  # Store loss and accuracy per epoch
    
    # Split dataset (train 70%, validation 15%, test 15%)
    # Split 85% Train, 15% Test
    train_X_tensor, test_X_tensor, train_y_tensor, test_y_tensor = train_test_split(
        X_tensor, y_tensor, test_size=0.15, random_state=11, shuffle=True
    )    
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=11)

    for batch_size, num_epochs, lr in product(batch_sizes, epochs_list, learning_rates):
        print(f"Training with batch_size={batch_size}, epochs={num_epochs}, learning_rate={lr}")

        fold_accuracies = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(train_X_tensor)):
            print(f"Fold {fold+1}/{k_folds}")

            # Create train and validation datasets
            train_dataset = Subset(TensorDataset(train_X_tensor, train_y_tensor), train_idx)
            val_dataset = Subset(TensorDataset(train_X_tensor, train_y_tensor), val_idx)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            # Initialize model
            model = GDPClassifier(input_size=X_tensor.shape[1]).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)

            # Early stopping mechanism
            early_stopping = EarlyStopping(patience=10, min_delta=0.001)

            train_losses = []
            val_losses = []
            train_accuracies = []
            val_accuracies = []

            for epoch in range(num_epochs):
                model.train()
                running_loss, correct_train, total_train = 0.0, 0, 0

                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)
                    total_train += labels.size(0)
                    correct_train += (predicted == labels).sum().item()

                train_loss = running_loss / len(train_loader)
                train_accuracy = correct_train / total_train
                train_losses.append(train_loss)
                train_accuracies.append(train_accuracy)

                # Validation step
                model.eval()
                val_loss, correct_val, total_val = 0, 0, 0
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()

                        _, predicted = torch.max(outputs, 1)
                        total_val += labels.size(0)
                        correct_val += (predicted == labels).sum().item()

                val_loss /= len(val_loader)
                val_accuracy = correct_val / total_val
                val_losses.append(val_loss)
                val_accuracies.append(val_accuracy)

                print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")

                early_stopping(val_loss)
                if early_stopping.early_stop:
                    print("Early stopping triggered")
                    break

            # Store fold history
            fold_accuracies.append(val_accuracies[-1])  # Store last validation accuracy for this fold

        # Compute average accuracy across folds
        avg_accuracy = sum(fold_accuracies) / k_folds

        # Update best model if the current one is better
        if avg_accuracy > best_accuracy:
            best_accuracy = avg_accuracy
            best_params = {'batch_size': batch_size, 'epochs': num_epochs, 'learning_rate': lr}
    
    #train best model
    train_dataset = TensorDataset(train_X_tensor, train_y_tensor)
    train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
    test_dataset = TensorDataset(test_X_tensor, test_y_tensor)
    test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'], shuffle=True)

    model,params,history = train_model(input_shape=X_tensor.shape[1], train_loader=train_loader, val_loader=test_loader, batch_size=best_params['batch_size'], num_epochs=best_params['epochs'], lr=best_params['learning_rate'])
    results=evaluate(model, TensorDataset(test_X_tensor, test_y_tensor), batch_size, le)

    return model, params, history, results  # Return history for plotting

def save_training_history(history, file_path, output_dir):
    """Saves training history to a pickle file and plots accuracy & loss in a single graph."""

    output_dir=f"{output_dir}/training_graphs"
    # Ensure the directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over multiple history dictionaries
    epochs = range(1, len(history["train_losses"]) + 1)

    # Create a single plot for both accuracy & loss
    plt.figure(figsize=(10, 6))

    # Plot accuracy
    plt.plot(epochs, history["train_accuracies"], label="Train Accuracy", linestyle="-", marker="o", color="blue")
    plt.plot(epochs, history["train_losses"], label="Train Loss", linestyle="-", marker="s", color="red")
    
    try:
        plt.plot(epochs, history["test_accuracies"], label="Test Accuracy", linestyle="--", marker="o", color="dodgerblue")
        plt.plot(epochs, history["test_losses"], label="Test Loss", linestyle="--", marker="s", color="darkred")
    except Exception as e:
        pass

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy / Loss")
    plt.legend()
    plt.grid()
    plt.title(f"Training History (Batch {history['batch_size']}, LR {history['lr']})")

    # Save the plot
    plot_filename = f"{file_path}_batch{history['batch_size']}_lr{history['lr']}.png"
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path)
    plt.close()

    print(f"Training history plot saved at: {plot_path}")

def evaluate(model, test_dataset, batch_size, le):
    
    model.eval()
    y_true, y_pred = [], []
    all_outputs = []    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Ensure same device
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy())

    print(classification_report(y_true, y_pred, target_names=le.classes_))

    # Compute confusion matrix and metrics
    cm = confusion_matrix(y_true, y_pred)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - (fp + tp + fn)

    # Compute per-class precision, recall, F1-score, and accuracy
    precision = np.diag(cm) / np.sum(cm, axis=0)  # TP / (TP + FP)
    recall = np.diag(cm) / np.sum(cm, axis=1)  # TP / (TP + FN)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)  # Avoid division by zero
    accuracy = np.sum(np.diag(cm)) / np.sum(cm)  # Overall accuracy

    precision = np.nanmean(precision)  # Mean precision across classes
    recall = np.nanmean(recall)  # Mean recall across classes
    f1_score = np.nanmean(f1_score)  # Mean F1-score across classes

    return [tp.tolist(), fp.tolist(), tn.tolist(), fn.tolist(), accuracy, precision, recall, f1_score]

if __name__ == "__main__":
    # Set seed for reproducibility
    torch.manual_seed(11)
    np.random.seed(11)

    timestamp = int(time.time())
    result_dir = "results/task3"
    os.makedirs(result_dir, exist_ok=True)

    df_task3 = normalised_imputed_df.copy()
    X_tensor, y_tensor, le = load_and_preprocess_data(df_task3)

    batch_sizes = [32]
    epochs_list = [150]
    learning_rates = [0.01,0.1]

    model, params, history, results = train_model_cv(X_tensor, y_tensor, batch_sizes, epochs_list, learning_rates)
    save_training_history(history, file_path=str(timestamp), output_dir=result_dir)

    model_dir=f"{result_dir}/model"
    os.makedirs(model_dir, exist_ok=True)
    model_path = f"{model_dir}/{timestamp}.pth"
    torch.save(model.state_dict(), model_path)

    # Save results
    os.makedirs(result_dir, exist_ok=True)
    result_path = f"{result_dir}/results.csv"
    file_exists = os.path.isfile(result_path)
    header = ["model_path", "model", "tp", "fp", "tn", "fn", "accuracy", "precision", "recall", "f1-score"]

    results=[model_path] + results

    print(results)
    with open(result_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write header if the file does not exist
        if not file_exists:
            writer.writerow(header)
        # Write the statistics
        writer.writerow(results)

