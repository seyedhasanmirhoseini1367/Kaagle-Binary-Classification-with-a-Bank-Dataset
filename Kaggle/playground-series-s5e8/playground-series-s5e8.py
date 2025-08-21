import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

# ======================
# Load Data
# ======================


train_path = r"D:\Kaggle\playground-series-s5e8\train.csv"
test_path = r"D:\Kaggle\playground-series-s5e8\test.csv"
sub_path = r"D:\Kaggle\playground-series-s5e8\my_submission.csv"

# Load data
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# Features and target
X = train.drop(['id', 'y'], axis=1)
y = train['y']
test = test.drop(['id'], axis=1)

s = train[['education']]

e = LabelEncoder()
s.loc[:, 'edu_encoder'] = e.fit_transform(s['education'])

cat_train = X.select_dtypes(include='object').columns.tolist()

for col in cat_train:
    num_classes = len(X[col].unique())
    embedding_dim = min(50, (num_classes // 2) + 1)
    embedding = nn.Embedding(num_classes, embedding_dim)

num_classes = len(s['education'].unique())
embedding_dim = min(50, (num_classes // 2) + 1)

embedding = nn.Embedding(num_classes, embedding_dim)

embedding[]

# Separate categorical and numeric
cat_cols = X.select_dtypes(include='object').columns
num_cols = X.select_dtypes(exclude='object').columns

# Label encode categorical columns
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    test[col] = le.transform(test[col])  # use same mapping
    encoders[col] = le  # store encoder if needed later

# Scale numeric columns
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])
test[num_cols] = scaler.transform(test[num_cols])

# Mark categoricals as category dtype (for LightGBM)
for col in cat_cols:
    X[col] = X[col].astype('category')
    test[col] = test[col].astype('category')

# Train-test split
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

# Encode target (if not already numeric)
target_le = LabelEncoder()
ytrain = target_le.fit_transform(ytrain)
ytest = target_le.transform(ytest)


# ======================
# Train LightGBM
# ======================

def roc_auc(n_leaves, lr):
    model = lgb.LGBMClassifier(
        num_leaves=n_leaves,
        n_estimators=500,
        learning_rate=lr,
        max_depth=-1,
        random_state=42,
        verbose=-1
    )

    model.fit(xtrain, ytrain, categorical_feature='auto')

    # Predict probabilities instead of class labels
    y_pred_proba = model.predict_proba(xtest)[:, 1]  # probability for class "1"

    # Compute ROC AUC
    auc = roc_auc_score(ytest, y_pred_proba)
    # print("ROC AUC:", auc)
    return auc


for n in [32, 64, 128]:
    for lr in [0.5, 0.05, 0.005]:
        print(f'num_Leaves, Learning_rate ({n}, {lr}): {roc_auc(n, lr)}')

# ======================
# Preprocessing
# ======================
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

# Encode categoricals
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))
    label_encoders[col] = le

# Scale numeric
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# Train/Val split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# ======================
# Torch Dataset
# ======================
class TabDataset(torch.utils.data.Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = None if y is None else torch.tensor(y.values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx]


train_ds = TabDataset(X_train, y_train)
val_ds = TabDataset(X_val, y_val)
test_ds = TabDataset(X_test)

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=256, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=256, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False)


# ======================
# Transformer Model
# ======================
class TabTransformer(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=2, num_classes=1):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.input_layer(x)
        # Add sequence dimension → treat each feature as a token
        x = x.unsqueeze(1)
        x = self.transformer(x)
        x = x.mean(dim=1)  # pooling
        out = self.fc(x)
        return self.sigmoid(out)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TabTransformer(input_dim=X.shape[1]).to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# ======================
# Training
# ======================
def train_model(model, train_loader, val_loader, epochs=10):
    best_auc = 0
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device).unsqueeze(1)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device).unsqueeze(1)
                preds = model(xb)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(yb.cpu().numpy())
        val_auc = roc_auc_score(val_targets, val_preds)
        print(f"Epoch {epoch + 1}/{epochs}, Val ROC-AUC: {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), "best_tabtransformer.pth")


train_model(model, train_loader, val_loader, epochs=15)

# ======================
# Inference for Kaggle
# ======================
model.load_state_dict(torch.load("best_tabtransformer.pth"))
model.eval()

test_preds = []
with torch.no_grad():
    for xb in test_loader:
        xb = xb.to(device)
        preds = model(xb)
        test_preds.extend(preds.cpu().numpy())

submission[target_col] = test_preds
submission.to_csv("tabtransformer_submission.csv", index=False)
print("Submission saved!")
