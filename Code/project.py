from sklearn.datasets import load_breast_cancer
import pandas as pd

# Load the dataset as a pandas DataFrame
cancer_data = load_breast_cancer(as_frame=True)
df = cancer_data.frame

# View the first few rows
print(df.head())

# Check dataset dimensions and features
print(df.info())

# Check class distribution (Malignant vs Benign)
# Note: In sklearn, 0 = Malignant, 1 = Benign (or vice versa depending on target_names)
print(cancer_data.target_names)
print(df['target'].value_counts())

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, ConfusionMatrixDisplay
)

# Set random seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ==========================================
# 1. DATA PREPARATION & SPLITTING (Breast Cancer Dataset)
# ==========================================
from sklearn.datasets import load_breast_cancer

# Load the real Breast Cancer Wisconsin dataset
cancer_data = load_breast_cancer()
X = cancer_data.data
y = cancer_data.target  # Note: 0 = Malignant, 1 = Benign

# Split into train+val (80%) and test (20%) using stratification
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y
)

# Standardize features (fit on train_val, transform test)
scaler = StandardScaler()
X_train_val = scaler.fit_transform(X_train_val)
X_test = scaler.transform(X_test)

# Convert test set to PyTorch Tensors
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# ==========================================
# 2. NEURAL NETWORK ARCHITECTURE
# ==========================================
class BalancedClassifier(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.3):
        super(BalancedClassifier, self).__init__()
        # Moderately complex architecture with 2 hidden layers
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            
            nn.Linear(32, 1) # Raw logits output
        )

    def forward(self, x):
        return self.network(x)

def l1_regularization(model, l1_lambda):
    """Calculates L1 penalty over linear layer weights."""
    l1_loss = torch.tensor(0.0)
    for name, param in model.named_parameters():
        if 'weight' in name:
            l1_loss += torch.norm(param, 1)
    return l1_lambda * l1_loss


# ==========================================
# 3. EARLY STOPPING CLASS
# ==========================================
class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False
        self.best_state = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = model.state_dict().copy()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# ==========================================
# 4. 5-FOLD CROSS-VALIDATION TRAINING
# ==========================================
def train_and_validate_cv(X_data, y_data, n_splits=5, epochs=150, batch_size=64):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    fold_histories = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_data, y_data)):
        print(f"--- Training Fold {fold + 1}/{n_splits} ---")
        
        # Prepare Data Loaders
        X_tr, y_tr = X_data[train_idx], y_data[train_idx]
        X_va, y_va = X_data[val_idx], y_data[val_idx]
        
        train_ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1))
        val_ds = TensorDataset(torch.tensor(X_va, dtype=torch.float32), torch.tensor(y_va, dtype=torch.float32).unsqueeze(1))
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        
        # Initialize Model, Weighted Loss Function, and Optimizer
        model = BalancedClassifier(input_dim=X_data.shape[1], dropout_rate=0.3)
        
        # Calculate class weights for imbalance: pos_weight = negative / positive
        # Calculate class weights for imbalance: pos_weight = count(negative) / count(positive)
        # Ensure y_tr is treated correctly as binary flags
        num_negatives = (y_tr == 0).sum()
        num_positives = (y_tr == 1).sum()
        pos_weight = torch.tensor([num_negatives / max(1, num_positives)], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # Adam Optimizer incorporates L2 regularization via weight_decay
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4) # L2 regularized
        early_stopping = EarlyStopping(patience=15)
        
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            # Training Phase
            model.train()
            running_train_loss = 0.0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                logits = model(batch_x)
                loss = criterion(logits, batch_y) + l1_regularization(model, l1_lambda=1e-5) # Add L1
                loss.backward()
                optimizer.step()
                running_train_loss += loss.item() * batch_x.size(0)
            
            epoch_train_loss = running_train_loss / len(train_loader.dataset)
            
            # Validation Phase
            model.eval()
            running_val_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    logits = model(batch_x)
                    loss = criterion(logits, batch_y)
                    running_val_loss += loss.item() * batch_x.size(0)
                    
            epoch_val_loss = running_val_loss / len(val_loader.dataset)
            
            history['train_loss'].append(epoch_train_loss)
            history['val_loss'].append(epoch_val_loss)
            
            # Check Early Stopping
            early_stopping(epoch_val_loss, model)
            if early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                model.load_state_dict(early_stopping.best_state)
                break
                
        fold_histories.append((model, history))
        
    return fold_histories

# Execute Cross-Validation
fold_histories = train_and_validate_cv(X_train_val, y_train_val)


# ==========================================
# 5. TEST EVALUATION & METRICS
# ==========================================
# Use the best model from fold 1 (or perform ensemble prediction across all fold models)
best_model = fold_histories[0][0]
best_history = fold_histories[0][1]

best_model.eval()
with torch.no_grad():
    test_logits = best_model(X_test_tensor)
    test_probs = torch.sigmoid(test_logits).numpy()
    y_pred = (test_probs >= 0.5).astype(int).flatten()

# Calculate Performance Metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"\n--- Final Test Set Evaluation ---")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-Score : {f1:.4f}")

# ==========================================
# CO4 COMPARATIVE ANALYSIS
# ==========================================
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Train standard machine learning baselines on the exact same splits
lr_model = LogisticRegression(class_weight='balanced', random_state=SEED, max_iter=1000)
lr_model.fit(X_train_val, y_train_val)
lr_pred = lr_model.predict(X_test)

rf_model = RandomForestClassifier(class_weight='balanced', random_state=SEED)
rf_model.fit(X_train_val, y_train_val)
rf_pred = rf_model.predict(X_test)

# Baseline metrics
lr_acc = accuracy_score(y_test, lr_pred)
lr_prec = precision_score(y_test, lr_pred)
lr_rec = recall_score(y_test, lr_pred)
lr_f1 = f1_score(y_test, lr_pred)

rf_acc = accuracy_score(y_test, rf_pred)
rf_prec = precision_score(y_test, rf_pred)
rf_rec = recall_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)

print(f"\n--- CO4: Comparative Analysis Table ---")
print(f"{'Model':<25} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
print("-" * 75)
print(f"{'Logistic Regression':<25} | {lr_acc:.4f}     | {lr_prec:.4f}     | {lr_rec:.4f}     | {lr_f1:.4f}")
print(f"{'Random Forest':<25} | {rf_acc:.4f}     | {rf_prec:.4f}     | {rf_rec:.4f}     | {rf_f1:.4f}")
print(f"{'Proposed BalancedNN':<25} | {acc:.4f}     | {prec:.4f}     | {rec:.4f}     | {f1:.4f}")

# ==========================================
# 6. BIAS-VARIANCE DECOMPOSITION ANALYSIS
# ==========================================
# Calculating Bias and Variance empirically using test set predictions
final_train_loss = best_history['train_loss'][-1]
final_val_loss = best_history['val_loss'][-1]

# Empirical Bias estimate (Training Error)
bias_sq = final_train_loss 
# Empirical Variance estimate (Generalization Gap = Validation Loss - Training Loss)
variance = max(0, final_val_loss - final_train_loss)

print(f"\n--- Bias-Variance Diagnostics ---")
print(f"Estimated Empirical Bias (Train Loss): {bias_sq:.4f}")
print(f"Estimated Variance (Val-Train Gap)  : {variance:.4f}")
if final_train_loss > 0.4:
    print("Diagnosis: Model exhibits high bias (Underfitting). Consider increasing model capacity.")
elif variance > 0.15:
    print("Diagnosis: Model exhibits high variance (Overfitting). Increase regularization/dropout.")
else:
    print("Diagnosis: Model is well-balanced with low bias and low variance.")


# ==========================================
# 7. VISUALIZATIONS
# ==========================================
plt.figure(figsize=(16, 5))

# Plot 1: Train vs Validation Loss
plt.subplot(1, 3, 1)
plt.plot(best_history['train_loss'], label='Training Loss', linewidth=2)
plt.plot(best_history['val_loss'], label='Validation Loss', linewidth=2, linestyle='--')
plt.title('Loss Curves (Early Stopping Active)', fontsize=12)
plt.xlabel('Epochs')
plt.ylabel('BCE Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Evaluation Metrics Bar Chart
plt.subplot(1, 3, 2)
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [acc, prec, rec, f1]
bars = plt.bar(metrics, values, color=['#2b5c8f', '#d95f02', '#7570b3', '#1b9e77'])
plt.ylim(0, 1.1)
plt.title('Performance Metrics (Imbalanced Test Set)', fontsize=12)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f'{yval:.3f}', ha='center', va='bottom')
plt.grid(axis='y', alpha=0.3)

# Plot 3: Confusion Matrix
plt.subplot(1, 3, 3)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Negative (0)', 'Positive (1)'],
            yticklabels=['Negative (0)', 'Positive (1)'])
plt.title('Confusion Matrix', fontsize=12)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

plt.tight_layout()
plt.show()