import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os

# ---------------------------------------------------------
# 1. Load NPZ data
# ---------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "electric_fields.npz")
data = np.load(data_path)
 
# Load REAL and IMAG components
Ex_r = data["Ex_real"]   # (m, H, W)
Ex_i = data["Ex_imag"]
Ey_r = data["Ey_real"]
Ey_i = data["Ey_imag"]
Ez_r = data["Ez_real"]
Ez_i = data["Ez_imag"]

X = data["X"]
Y = data["Y"]
Z = data["Z"]

lam_values = data["lam_values"]
m_values = data["m_values"]
k_values = data["k_values"]

num_lam = Ex_r.shape[0]
H, W = X.shape
num_points = H * W

print("Loaded data:")
print(" Ex_r:", Ex_r.shape)
print(" Grid:", X.shape, Y.shape, Z.shape)
print(" λ values:", lam_values.shape)
print(" num_points =", num_points)

# ---------------------------------------------------------
# 2. Flatten grid + build dataset
# ---------------------------------------------------------
X_flat = X.reshape(-1).astype(np.float32)
Y_flat = Y.reshape(-1).astype(np.float32)
Z_flat = Z.reshape(-1).astype(np.float32)

inputs_list = []
targets_list = []

for i in range(num_lam):
    lam_val = lam_values[i]
    lam_col = np.full((num_points,), lam_val, dtype=np.float32)

    # Inputs: (x, y, z, λ)
    inp = np.column_stack([X_flat, Y_flat, Z_flat, lam_col])

    # Targets: 6 channels (Ex_r, Ex_i, Ey_r, Ey_i, Ez_r, Ez_i)
    tar = np.column_stack([
        Ex_r[i].reshape(-1), Ex_i[i].reshape(-1),
        Ey_r[i].reshape(-1), Ey_i[i].reshape(-1),
        Ez_r[i].reshape(-1), Ez_i[i].reshape(-1),
    ]).astype(np.float32)

    inputs_list.append(inp)
    targets_list.append(tar)

# Stack all data
X_all = torch.tensor(np.vstack(inputs_list), dtype=torch.float32)
Y_all = torch.tensor(np.vstack(targets_list), dtype=torch.float32)

print("X_all:", X_all.shape)  # (N, 4)
print("Y_all:", Y_all.shape)  # (N, 6)

# ---------------------------------------------------------
# 3. Train/test split based on wavelength index
# ---------------------------------------------------------
num_train_lam = 80  # first 80 λ values for training

sample_lam_indices = np.repeat(np.arange(num_lam), num_points)
train_mask = torch.tensor(sample_lam_indices < num_train_lam, dtype=torch.bool)
test_mask = ~train_mask

X_train, Y_train = X_all[train_mask], Y_all[train_mask]
X_test,  Y_test  = X_all[test_mask],  Y_all[test_mask]

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

train_ds = TensorDataset(X_train, Y_train)
test_ds  = TensorDataset(X_test,  Y_test)

train_loader = DataLoader(train_ds, batch_size=4096, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=4096, shuffle=False)

# ---------------------------------------------------------
# 4. Neural network model
# ---------------------------------------------------------
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 6) 
        )

    def forward(self, x):
        return self.layers(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = MLP().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ---------------------------------------------------------
# 5. Training loop
# ---------------------------------------------------------
epochs = 200
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()

        pred = model(xb)
        loss = criterion(pred, yb)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}/{epochs} - Loss = {epoch_loss/len(train_loader):.6f}")

# ---------------------------------------------------------
# 6. Test evaluation
# ---------------------------------------------------------
model.eval()
test_losses = []

with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        test_losses.append(criterion(pred, yb).item())

print("\n✅ Test MSE:", np.mean(test_losses))

# ---------------------------------------------------------
# 7. Save model
# ---------------------------------------------------------
torch.save(model.state_dict(), "Efield_predictor_mlp.pt")
print("✅ Saved model as Efield_predictor_mlp.pt")