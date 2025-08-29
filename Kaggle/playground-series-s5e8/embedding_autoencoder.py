
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random

# ====================== Load Data ======================

train_path = r"D:\datasets\playground-series-s5e8\train.csv"
test_path = r"D:\datasets\playground-series-s5e8\test.csv"
sub_path = r"D:\datasets\playground-series-s5e8\my_submission.csv"


# reproducibility
def set_seed(seed=42):
    random.seed(seed);
    np.random.seed(seed);
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

set_seed(42)


train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
submission_df = pd.read_csv(sub_path)

# ====================== Split X/y & basic preprocessing ======================
target_col = "y"
id_col = "id" if "id" in train_df.columns else None

y = train_df[target_col]
X = train_df.drop(columns=[target_col] + ([id_col] if id_col in train_df.columns else []))
X_test_raw = test_df.drop(columns=[id_col] if id_col in test_df.columns else [])

# Identify categorical vs numerical columns
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.columns.difference(cat_cols).tolist()

# Label-encode categoricals
label_encoders = {}
X_cat = pd.DataFrame(index=X.index)
X_test_cat = pd.DataFrame(index=X_test_raw.index)
for c in cat_cols:
    le = LabelEncoder()
    X_cat[c] = le.fit_transform(X[c].astype(str))
    X_test_cat[c] = le.transform(X_test_raw[c].astype(str))
    label_encoders[c] = le

# Scale numerical columns
scaler = StandardScaler()
X_num = pd.DataFrame(scaler.fit_transform(X[num_cols]), columns=num_cols, index=X.index)
X_test_num = pd.DataFrame(scaler.transform(X_test_raw[num_cols]), columns=num_cols, index=X_test_raw.index)

# ====================== Train/Val split (Stratified) ======================

X_cat_tr, X_cat_val, X_num_tr, X_num_val, y_tr, y_val = train_test_split(
    X_cat, X_num, y, test_size=0.2, stratify=y, random_state=42
)

# ====================== Numeric Autoencoder (unsupervised) to compress numericals ======================
class NumericAutoencoder(nn.Module):
    def __init__(self, in_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, in_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def encode(self, x):
        with torch.no_grad():
            return self.encoder(x)


class NumOnlyDS(Dataset):
    def __init__(self, Xn):
        self.Xn = torch.tensor(Xn.values, dtype=torch.float32)

    def __len__(self): return len(self.Xn)

    def __getitem__(self, i): return self.Xn[i]

# latent_dim for numeric = number of numeric columns
latent_dim_num = len(num_cols)
ae = NumericAutoencoder(in_dim=len(num_cols), latent_dim=latent_dim_num)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ae = ae.to(device)

ae_train_loader = DataLoader(NumOnlyDS(X_num_tr), batch_size=512, shuffle=True)
ae_val_loader = DataLoader(NumOnlyDS(X_num_val), batch_size=512, shuffle=False)

ae_crit = nn.MSELoss()
ae_opt = torch.optim.Adam(ae.parameters(), lr=1e-3)

AE_EPOCHS = 50
best_ae_val = float("inf")
for epoch in range(AE_EPOCHS):
    ae.train()
    tr_loss = 0.0
    for xb in ae_train_loader:
        xb = xb.to(device)
        ae_opt.zero_grad()
        x_hat, _ = ae(xb)
        loss = ae_crit(x_hat, xb)
        loss.backward()
        ae_opt.step()
        tr_loss += loss.item()
    tr_loss /= len(ae_train_loader)

    ae.eval()
    va_loss = 0.0
    with torch.no_grad():
        for xb in ae_val_loader:
            xb = xb.to(device)
            x_hat, _ = ae(xb)
            loss = ae_crit(x_hat, xb)
            va_loss += loss.item()
    va_loss /= len(ae_val_loader)
    print(f"[AE] Epoch {epoch + 1}/{AE_EPOCHS}  train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}")

    if va_loss < best_ae_val:
        best_ae_val = va_loss
        torch.save(ae.state_dict(), "best_numeric_ae.pth")

# Encode numericals to latent space
ae.load_state_dict(torch.load("best_numeric_ae.pth", map_location=device))
ae.eval()
with torch.no_grad():
    Z_tr = ae.encode(torch.tensor(X_num_tr.values, dtype=torch.float32, device=device)).cpu().numpy()
    Z_val = ae.encode(torch.tensor(X_num_val.values, dtype=torch.float32, device=device)).cpu().numpy()
    Z_tst = ae.encode(torch.tensor(X_test_num.values, dtype=torch.float32, device=device)).cpu().numpy()

Z_tr_df = pd.DataFrame(Z_tr, index=X_num_tr.index, columns=[f"z{i}" for i in range(Z_tr.shape[1])])
Z_val_df = pd.DataFrame(Z_val, index=X_num_val.index, columns=[f"z{i}" for i in range(Z_val.shape[1])])
Z_tst_df = pd.DataFrame(Z_tst, index=X_test_num.index, columns=[f"z{i}" for i in range(Z_tst.shape[1])])


# ====================== Tabular model with cat-embeddings + numeric-latents ======================
class TabularDS(Dataset):
    def __init__(self, X_cat_df, Z_num_df, y=None):
        self.X_cat = torch.tensor(X_cat_df.values, dtype=torch.long)
        self.X_num = torch.tensor(Z_num_df.values, dtype=torch.float32)
        self.y = None if y is None else torch.tensor(y.values, dtype=torch.float32)

    def __len__(self): return len(self.X_cat)

    def __getitem__(self, i):
        if self.y is None:
            return self.X_cat[i], self.X_num[i]
        return self.X_cat[i], self.X_num[i], self.y[i]


train_ds = TabularDS(X_cat_tr, Z_tr_df, y_tr)
val_ds = TabularDS(X_cat_val, Z_val_df, y_val)
test_ds = TabularDS(X_test_cat, Z_tst_df, None)

train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False)

# Embedding dims
embed_dim = 32
emb_dims = [min(embed_dim, (len(label_encoders[c].classes_) + 1) // 2) for c in cat_cols]


class TabularNN(nn.Module):
    def __init__(self, cat_dims, num_latent_dim, embed_dim_list):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(cat_dim, embed_dim_list[i])
            for i, cat_dim in enumerate(cat_dims)
        ])
        self.embed_dropout = nn.Dropout(0.2)

        total_in = sum(embed_dim_list) + num_latent_dim

        self.net = nn.Sequential(
            nn.Linear(total_in, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x_cat, x_num_lat):
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat(embs, dim=1)
        x = self.embed_dropout(x)
        x = torch.cat([x, x_num_lat], dim=1)
        return self.net(x)


model = TabularNN(
    cat_dims=[len(label_encoders[c].classes_) for c in cat_cols],
    num_latent_dim=latent_dim_num,
    embed_dim_list=emb_dims
).to(device)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

import torch

EPOCHS = 50
best_auc = 0.0

for epoch in range(EPOCHS):
    # ----------- Training -----------
    model.train()
    train_loss = 0.0

    for Xc, Xn, yb in train_loader:
        Xc, Xn, yb = Xc.to(device), Xn.to(device), yb.to(device)

        optimizer.zero_grad()
        preds = model(Xc, Xn).squeeze()

        loss = criterion(preds, yb.float())
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    avg_train_loss = train_loss / len(train_loader)

    # ----------- Validation -----------
    model.eval()
    val_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for Xc, Xn, yb in val_loader:
            Xc, Xn, yb = Xc.to(device), Xn.to(device), yb.to(device)

            preds = model(Xc, Xn).squeeze()
            loss = criterion(preds, yb.float())
            val_loss += loss.item()

            all_preds.append(preds.cpu())
            all_labels.append(yb.cpu())

    avg_val_loss = val_loss / len(val_loader)

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    auc = roc_auc_score(all_labels, all_preds)

    if auc > best_auc:
        best_auc = auc
        torch.save(model.state_dict(), "best_model.pth")

    print(f"Epoch [{epoch + 1}/{EPOCHS}] | "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Val Loss: {avg_val_loss:.4f} | "
          f"AUC: {auc:.4f} | Best AUC: {best_auc:.4f}")

model.eval()
all_preds = []

with torch.no_grad():
    for Xc, Xn in test_loader:
        Xc, Xn = Xc.to(device), Xn.to(device)
        outputs = model(Xc, Xn)
        preds = outputs.squeeze().cpu().numpy()
        all_preds.extend(preds)

submission_df['y'] = all_preds
submission_df.to_csv(sub_path, index=False)

