import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from itertools import product
import time
import csv
import json
import seaborn as sns

imputed_df=pd.read_csv('data/final_impute_world_bank_data_dev.csv')

seed=11

batch_sizes=[16, 32]
epochs_list=[50, 100, 150]
learning_rates=[0.0001] 
scoring_metric="loss" #"f1" "loss" "accuracy"

k_folds=5

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(seed)

timestamp = int(time.time())
result_dir = "results/task3"
os.makedirs(result_dir, exist_ok=True)
models_dir = os.path.join(result_dir, "models")
os.makedirs(models_dir, exist_ok=True)
time_models_dir = os.path.join(models_dir, str(timestamp))
os.makedirs(time_models_dir, exist_ok=True)
output_dir = os.path.join(time_models_dir, "outputs")
os.makedirs(output_dir, exist_ok=True)
params_tune_dir=os.path.join(time_models_dir,"params_tune")
os.makedirs(params_tune_dir, exist_ok=True)
graphs_dir=os.path.join(time_models_dir,"outputs")
os.makedirs(graphs_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load and preprocess data
def load_and_preprocess_data(df):
    log_transform_indicators = [
        "GDPpc_2017$", "Population_total", "Energy_use", "Exports_2017$"
    ]

    for indicator in log_transform_indicators:
        df[indicator] = np.log1p(df[indicator])  # log1p to avoid log(0) issues


    avg_gdp = df.groupby('country')['GDPpc_2017$'].mean().reset_index()
    avg_gdp['GDP_class'] = pd.qcut(avg_gdp['GDPpc_2017$'], q=4, 
                                  labels=["Under-developed", "Developing", "Emerging", "Developed"])

    features = df.columns.difference(['country', 'date', 'GDPpc_2017$'])
    aggregated_data = df.groupby('country')[features].mean().reset_index()
    merged_data = aggregated_data.merge(avg_gdp[['country', 'GDP_class']], on='country')

    le = LabelEncoder()
    merged_data['label'] = le.fit_transform(merged_data['GDP_class'])

    X = merged_data.drop(['country', 'GDP_class', 'label'], axis=1).values
    y = merged_data['label'].values

    # Handle class imbalance
    class_counts = np.bincount(y)
    class_weights = 1.0 / class_counts
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    return X, y, le, class_weights

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

class GDPClassifier(nn.Module):
    def __init__(self, input_size):
        super(GDPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.fc3 = nn.Linear(64, 4)
        self.dropout = nn.Dropout(0.4)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Training function with early stopping
def train_model(input_shape, train_loader, val_loader, epochs, lr, class_weights, model_path=None):
    model = GDPClassifier(input_size=input_shape).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    early_stopping = EarlyStopping(patience=10, min_delta=0.001, path=model_path)
    
    val_f1s = []
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(epochs):
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

        if val_loader is not None:
            # Validation evaluation
            model.eval()
            val_loss = 0
            correct_val, total_val = 0, 0
            all_preds, all_labels = [], []
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)
                    all_preds.append(predicted)
                    all_labels.append(labels)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()

            val_loss /= len(val_loader)
            val_accuracy = correct_val / total_val

            all_val_preds = torch.cat(all_preds)
            all_val_labels = torch.cat(all_labels)
            val_f1 = f1_score(all_val_labels.cpu().numpy(), all_val_preds.cpu().numpy(), average="weighted")
            val_f1s.append(val_f1)

            val_losses.append(val_loss)  # Store validation loss
            val_accuracies.append(val_accuracy)  # Store validation accuracy

            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")

            early_stopping(val_loss, model)

            if early_stopping.early_stop:
                print("Early stopping triggered. Restoring best model weights...")
                model.load_state_dict(early_stopping.best_model_state)  # Restore Best Model
                break
            else:
                print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}")
    if val_loader is not None:
        return train_losses, train_accuracies, val_losses, val_accuracies, val_f1s, early_stopping.best_loss, early_stopping.path
    else:
        return train_losses, train_accuracies, None, None, None, None, model_path
def evaluate_model(model, test_loader, le):
    model.to(device)
    model.eval()
    y_true, y_pred = [], []
    all_outputs = []    

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

    return cm, [tp.tolist(), fp.tolist(), tn.tolist(), fn.tolist(), accuracy, precision, recall, f1_score]
def save_training_history(history, plot_path):
    plot_path = os.path.join(plot_path, f"training_history.png")

    epochs = range(1, len(history["train_losses"]) + 1)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
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
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_accuracies"], label='Train Accuracy')
    plt.plot(epochs, history["test_accuracies"], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.annotate(f"{history['train_accuracies'][-1]:.2f}", xy=(epochs[-1], history["train_accuracies"][-1]),
                 xytext=(epochs[-1] - 5, history["train_accuracies"][-1] + 0.02),
                 arrowprops=dict(arrowstyle="->", color="blue"), fontsize=10, color="blue")
    plt.annotate(f"{history['test_accuracies'][-1]:.2f}", xy=(epochs[-1], history["test_accuracies"][-1]),
                 xytext=(epochs[-1] - 5, history["test_accuracies"][-1] + 0.02),
                 arrowprops=dict(arrowstyle="->", color="orange"), fontsize=10, color="orange")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.show()
    
    print(f"Training history saved to {plot_path}")
    
def save_cm_figure(cm, plot_path):
    plot_path = os.path.join(plot_path, f"confusion_matrix.png")

    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(plot_path)
    plt.show()
    print(f"Confusion matrix saved to {plot_path}")

df_task3 = imputed_df.copy()
X, y, le, class_weights = load_and_preprocess_data(df_task3)

# K-Fold CV Training
train_X, test_X, train_y, test_y = train_test_split(
    X, y, test_size=0.2, random_state=seed, shuffle=True, stratify=y
)

scaler = StandardScaler()
train_X = scaler.fit_transform(train_X)  # Fit on training data
test_X = scaler.transform(test_X)  # Transform validation data using the same scaler

kfold = KFold(n_splits=k_folds, shuffle=True, random_state=seed)

best_model_info = {"accuracy": -1, "loss": float("inf"), "f1": -1, "params": {}, "grid_search_params":{}, "path": None}
param_grid = product(batch_sizes, epochs_list, learning_rates)

for batch_size, epochs, lr in param_grid:
    set_seed(seed)
    print(f"Training in {kfold} Fold CV with LR={lr}, Batch={batch_size}, Epochs={epochs}")

    model_path = f"best_model_lr{int(lr*10000)}_bs{batch_size}_ep{epochs}.pth"
    tune_model_path=os.path.join(params_tune_dir, model_path)

    fold_accuracies = []
    fold_losses = []
    fold_f1s = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_X)):
        print(f"Fold {fold+1}/{k_folds}")

        X_train, X_val = train_X[train_idx], train_X[val_idx]
        y_train, y_val = train_y[train_idx], train_y[val_idx]

        # Create train and validation datasets
        train_X_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
        val_X_tensor = torch.tensor(X_val, dtype=torch.float32, device=device)
        train_y_tensor = torch.tensor(y_train, dtype=torch.long, device=device)
        val_y_tensor = torch.tensor(y_val, dtype=torch.long, device=device)
        
        train_dataset = TensorDataset(train_X_tensor, train_y_tensor)
        val_dataset = TensorDataset(val_X_tensor, val_y_tensor)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Initialize model
        train_losses, train_accuracies, val_losses, val_accuracies, val_f1s, val_loss, saved_path=train_model(train_X_tensor.shape[1], train_loader, val_loader, epochs, lr, class_weights, model_path=tune_model_path)

        # Store fold history
        fold_accuracies.append(max(val_accuracies))  
        fold_losses.append(val_loss)
        fold_f1s.append(max(val_f1s))  # Store last validation accuracy for this fold

    # Compute average accuracy across folds
    avg_f1 = sum(fold_f1s) / k_folds
    avg_acc = sum(fold_accuracies) / k_folds
    avg_loss =sum(fold_losses) / k_folds

    # Update best model if the current one is better
    if (scoring_metric == "accuracy" and avg_acc > best_model_info["accuracy"]) or \
            (scoring_metric == "loss" and avg_loss < best_model_info["loss"]) or \
                (scoring_metric == "f1" and avg_f1 > best_model_info["f1"]):
        best_model_info = {
            "accuracy": avg_acc,
            "loss": avg_loss,
            "f1": avg_f1,
            "params": {
                "lr": lr,
                "batch_size": batch_size,
                "epochs": epochs
            },
            "grid_search_params": {
                "learning_rates": learning_rates,
                "batch_sizes": batch_sizes,
                "epochs_list": epochs_list
            },
            "path": saved_path
        }

best_model_info_path=f"{time_models_dir}/best_model_info.json"
with open(best_model_info_path, "w") as json_file:
    json.dump(best_model_info, json_file, indent=4)

with open(best_model_info_path, "r") as file:
    config=json.load(file)

best_params=config["params"]

final_model_path=os.path.join(time_models_dir, f"gdp_classifier.pth")

#train best model
train_X_tensor = torch.tensor(train_X, dtype=torch.float32, device=device)
test_X_tensor = torch.tensor(test_X, dtype=torch.float32, device=device)
train_y_tensor = torch.tensor(train_y, dtype=torch.long, device=device)
test_y_tensor = torch.tensor(test_y, dtype=torch.long, device=device)

train_dataset = TensorDataset(train_X_tensor, train_y_tensor)
train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
test_dataset = TensorDataset(test_X_tensor, test_y_tensor)
test_loader= DataLoader(test_dataset, batch_size=best_params['batch_size'], shuffle=True)

train_losses, train_accuracies, test_losses, test_accuracies, test_f1s, test_loss, model_path=train_model(train_X_tensor.shape[1], train_loader, test_loader, best_params['epochs'], best_params['lr'], class_weights, model_path=final_model_path)
history={
        'batch_size': batch_size, 'epochs': epochs, 'lr': lr, 
        'train_losses': train_losses, 'test_losses': test_losses,
        'train_accuracies': train_accuracies, 'test_accuracies': test_accuracies,
    }

model = GDPClassifier(input_size=test_X_tensor.shape[1])
model.load_state_dict(torch.load(model_path))
cm, results=evaluate_model(model, test_loader, le)

save_training_history(history, plot_path=output_dir)
save_cm_figure(cm, plot_path=output_dir)

# Save results
os.makedirs(result_dir, exist_ok=True)
result_path = f"{result_dir}/results.csv"
file_exists = os.path.isfile(result_path)
header = ["model_path", "params", "tp", "fp", "tn", "fn", "accuracy", "precision", "recall", "f1-score"]

results=[model_path, best_params] + results

with open(result_path, mode='a', newline='') as file:
    writer = csv.writer(file)
    # Write header if the file does not exist
    if not file_exists:
        writer.writerow(header)
    # Write the statistics
    writer.writerow(results)

task3_result_df = pd.read_csv(result_path)
print(task3_result_df.tail(1))