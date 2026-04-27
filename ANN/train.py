import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score, RocCurveDisplay
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import joblib

from ucimlrepo import fetch_ucirepo

heart_disease = fetch_ucirepo(id=45)

X = heart_disease.data.features
y = heart_disease.data.targets

# =============================================================================
# STEP 1 — EXPLORE
# =============================================================================
df = X.copy()
df['target'] = y

print("Shape:", df.shape)
print("\nColumn types:")
print(df.info())
print("\nDescriptive stats:")
print(df.describe())
print("\nMissing values:")
print(df.isnull().sum())
print("\nTarget distribution (raw):")
print(df['target'].value_counts())

df['target'].value_counts().plot(kind='bar', color=['steelblue', 'salmon'])
plt.title("Raw Target Distribution")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.show()

# =============================================================================
# STEP 2 — CLEAN
# =============================================================================
df['target'] = (df['target'] > 0).astype(int)
print("\nTarget after binarizing:")
print(df['target'].value_counts())

suspicious_cols = ['trestbps', 'chol', 'thalach']
for col in suspicious_cols:
    zero_count = (df[col] == 0).sum()
    if zero_count > 0:
        print(f"Warning: {zero_count} zero values in '{col}' — treating as missing")
        df[col] = df[col].replace(0, np.nan)

continuous_cols = ['trestbps', 'chol', 'thalach', 'oldpeak', 'age']
df[continuous_cols] = SimpleImputer(strategy='median').fit_transform(df[continuous_cols])

categorical_cols = ['ca', 'thal', 'cp', 'restecg', 'slope']
df[categorical_cols] = SimpleImputer(strategy='most_frequent').fit_transform(df[categorical_cols])

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# =============================================================================
# STEP 3 — ENCODE CATEGORICALS
# =============================================================================
ohe_cols = ['cp', 'restecg', 'slope', 'thal']
df = pd.get_dummies(df, columns=ohe_cols, drop_first=True)

print("\nShape after encoding:", df.shape)

# =============================================================================
# STEP 4 — SPLIT FEATURES AND TARGET
# =============================================================================
X = df.drop(columns=['target'])
y = df['target']

# =============================================================================
# STEP 5 & 6 — TRAIN/TEST SPLIT + SCALE
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
y_train_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
y_test_t  = torch.tensor(y_test.values,  dtype=torch.float32).unsqueeze(1)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)

# =============================================================================
# STEP 7 — BUILD MLP
# =============================================================================
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),  # wider first layer = more capacity to learn feature combos
            nn.ReLU(),
            nn.Dropout(0.4),            # slightly more dropout to compensate for larger layer
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),          # added third layer for more gradual compression
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP(X_train_t.shape[1]).to(device)
print(model)

# =============================================================================
# STEP 8 — TRAIN
# =============================================================================
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)  # lower lr = smaller steps, less overshooting

train_losses, val_losses = [], []
train_accs,   val_accs   = [], []
best_val_loss = float('inf')
patience, patience_counter = 15, 0  # more patience = more time to find a better minimum

for epoch in range(100):
    # --- Training ---
    model.train()
    batch_losses, batch_correct = [], 0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(Xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())
        batch_correct += ((out >= 0.5) == yb).sum().item()

    train_losses.append(np.mean(batch_losses))
    train_accs.append(batch_correct / len(X_train_t))

    # --- Validation ---
    model.eval()
    with torch.no_grad():
        val_out = model(X_test_t.to(device))
        val_loss = criterion(val_out, y_test_t.to(device)).item()
        val_acc = ((val_out >= 0.5) == y_test_t.to(device)).float().mean().item()

    val_losses.append(val_loss)
    val_accs.append(val_acc)

    # --- Early stopping ---
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d} | Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

# Restore best weights
model.load_state_dict(best_weights)

torch.save(model.state_dict(), "ANN/models/heart_mlp.pth")
joblib.dump(scaler, "ANN/models/scaler.pkl")

# Plot training curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses,   label='Val Loss')
plt.title("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Acc')
plt.plot(val_accs,   label='Val Acc')
plt.title("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("ANN/plots/training_curves.png", dpi=150, bbox_inches='tight')
plt.show()



# =============================================================================
# STEP 9 — EVALUATE
# =============================================================================
model.eval()
with torch.no_grad():
    y_pred_prob = model(X_test_t.to(device)).cpu().numpy().flatten()

y_pred = (y_pred_prob >= 0.5).astype(int)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("AUC-ROC:", roc_auc_score(y_test, y_pred_prob))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Disease', 'Disease'],
            yticklabels=['No Disease', 'Disease'])
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("ANN/plots/confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.show()


RocCurveDisplay.from_predictions(y_test, y_pred_prob)
plt.title("ROC Curve")
plt.savefig("ANN/plots/roc_curve.png", dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# STEP 10 — CROSS VALIDATE
# =============================================================================
def train_fold(X_tr, y_tr, X_val, y_val, input_dim, epochs=50):
    m = MLP(input_dim).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=0.001)
    crit = nn.BCELoss()
    loader = DataLoader(
        TensorDataset(
            torch.tensor(X_tr, dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
        ), batch_size=32, shuffle=True
    )
    m.train()
    for _ in range(epochs):
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            crit(m(Xb), yb).backward()
            opt.step()
    m.eval()
    with torch.no_grad():
        probs = m(torch.tensor(X_val, dtype=torch.float32).to(device)).cpu().numpy().flatten()
    return roc_auc_score(y_val, probs)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
X_all = np.concatenate([X_train, X_test])
y_all = np.concatenate([y_train.values, y_test.values])

cv_scores = []
for fold, (tr_idx, val_idx) in enumerate(kfold.split(X_all, y_all)):
    score = train_fold(X_all[tr_idx], y_all[tr_idx],
                       X_all[val_idx], y_all[val_idx],
                       input_dim=X_train.shape[1])
    cv_scores.append(score)
    print(f"Fold {fold+1} AUC: {score:.3f}")

print(f"\nMean AUC: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")

joblib.dump({
    "train_losses": train_losses,
    "val_losses":   val_losses,
    "train_accs":   train_accs,
    "val_accs":     val_accs,
    "y_test":       y_test.tolist(),
    "y_pred_prob":  y_pred_prob.tolist()
}, "ANN/plots/training_data.pkl")

# =============================================================================
# STEP 11 — TUNE (starting points)
# =============================================================================
# Things to try if results are underwhelming:
# - Increase/decrease layer sizes (e.g. 128 -> 64 -> 32)
# - Adjust Dropout rate (try 0.2 or 0.4)
# - Lower learning rate (e.g. 0.0005)
# - Increase patience in early stopping
# - Add a third hidden layer
# - Try nn.LeakyReLU() instead of nn.ReLU()