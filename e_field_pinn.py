import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os

# =============================================================
# 1. Load NPZ data
# =============================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "electric_fields.npz")
data = np.load(data_path)

# Load REAL & IMAG components
Ex_r = data["Ex_real"]
Ex_i = data["Ex_imag"]
Ey_r = data["Ey_real"]
Ey_i = data["Ey_imag"]
Ez_r = data["Ez_real"]
Ez_i = data["Ez_imag"]

X = data["X"]
Y = data["Y"]
Z = data["Z"]

lam_values = data["lam_values"]
m_vals = data["m_values"]
k_vals = data["k_values"]

num_lam = Ex_r.shape[0]
H, W = X.shape
num_points = H * W

# =============================================================
# 2. Build dataset
# =============================================================
Xf = X.reshape(-1).astype(np.float32)
Yf = Y.reshape(-1).astype(np.float32)
Zf = Z.reshape(-1).astype(np.float32)

inputs_list = []
targets_list = []
params_list = []  # store (n^2 k0^2) values for PDE

for i in range(num_lam):

    lam = lam_values[i]
    m = m_vals[i]
    k = k_vals[i]

    # Complex refractive index
    n_complex = m + 1j * k
    k0 = 2 * np.pi / lam
    helm_coeff = (n_complex**2 * k0**2)  # complex
    helm_real = np.real(helm_coeff).astype(np.float32)
    helm_imag = np.imag(helm_coeff).astype(np.float32)

    lam_col = np.full((num_points,), lam, dtype=np.float32)
    helm_real_col = np.full((num_points,), helm_real, dtype=np.float32)
    helm_imag_col = np.full((num_points,), helm_imag, dtype=np.float32)

    helm_coeff_pair = np.column_stack([helm_real_col, helm_imag_col])

    inp = np.column_stack([Xf, Yf, Zf, lam_col])
    tar = np.column_stack([
        Ex_r[i].reshape(-1), Ex_i[i].reshape(-1),
        Ey_r[i].reshape(-1), Ey_i[i].reshape(-1),
        Ez_r[i].reshape(-1), Ez_i[i].reshape(-1)
    ]).astype(np.float32)

    inputs_list.append(inp)
    targets_list.append(tar)
    params_list.append(helm_coeff_pair)


X_all = torch.tensor(np.vstack(inputs_list), dtype=torch.float32)
Y_all = torch.tensor(np.vstack(targets_list), dtype=torch.float32)
helm_all = torch.tensor(np.concatenate(params_list), dtype=torch.float32)


# =============================================================
# 3. Train/test split
# =============================================================
num_train_lam = 80
sample_lam_index = np.repeat(np.arange(num_lam), num_points)
train_mask = torch.tensor(sample_lam_index < num_train_lam)
test_mask = ~train_mask

X_train = X_all[train_mask]
Y_train = Y_all[train_mask]
helm_train = helm_all[train_mask]

X_test = X_all[test_mask]
Y_test = Y_all[test_mask]
helm_test = helm_all[test_mask]

train_ds = TensorDataset(X_train, Y_train, helm_train)
test_ds  = TensorDataset(X_test, Y_test, helm_test)

train_loader = DataLoader(train_ds, batch_size=4096, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=4096)

# =============================================================
# 4. Neural Network
# =============================================================
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
model = MLP().to(device)

# =============================================================
# 5. PDE LOSS: Helmholtz equation
# =============================================================
def laplacian(outputs, inputs):
    """
    Compute ΔE using autograd.
    outputs: (batch, 6)
    inputs:  (batch, 4) -> (x, y, z, lambda)
    """
    grads = torch.autograd.grad(
        outputs, inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True
    )[0]  # shape: (batch, 4)

    # Second derivatives
    lap = 0.0
    for i in range(3):  # only x,y,z
        grad_i = grads[:, i]
        second = torch.autograd.grad(
            grad_i, inputs,
            grad_outputs=torch.ones_like(grad_i),
            create_graph=True,
            retain_graph=True
        )[0][:, i]
        lap += second

    return lap


# =============================================================
# 6. Training Loop With PDE Loss
# =============================================================
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

lambda_pde = 1e-3   # weight of PDE loss

epochs = 15
for epoch in range(epochs):

    model.train()
    total_loss = 0

    for xb, yb, helm in train_loader:
        xb = xb.to(device).requires_grad_(True)
        yb = yb.to(device)
        helm = helm.to(device)

        optimizer.zero_grad()

        pred = model(xb)  # (batch,6)

        # Data loss
        loss_data = criterion(pred, yb)

        # PDE loss: full complex Helmholtz PDE
        pde_loss = 0.0

        # --- Compute radius to determine inside/outside sphere ---
        r = torch.sqrt(xb[:,0]**2 + xb[:,1]**2 + xb[:,2]**2)
        inside = (r <= 0.5)
        outside = ~inside

        # --- Compute free-space k0^2 ---
        lam = xb[:, 3]
        k0 = 2 * torch.pi / lam
        k0_sq = k0**2

        # --- Helmholtz coefficients from dataset ---
        a = helm[:, 0]   # real part
        b = helm[:, 1]   # imag part

        # --- Effective coefficients ---
        a_eff = torch.zeros_like(a)
        b_eff = torch.zeros_like(b)

        # Inside sphere: actual material coefficient
        a_eff[inside] = a[inside]
        b_eff[inside] = b[inside]

        # Outside sphere: free space (n = 1 + i0)
        a_eff[outside] = k0_sq[outside]
        b_eff[outside] = 0.0


        # Loop over field components in pairs:
        # (0,1) = Ex_real, Ex_imag
        # (2,3) = Ey_real, Ey_imag
        # (4,5) = Ez_real, Ez_imag
        for idx_r in [0, 2, 4]:
            idx_i = idx_r + 1

            # Extract predicted real/imag fields
            E_r = pred[:, idx_r]
            E_i = pred[:, idx_i]

            # Compute Laplacian(E_real) and Laplacian(E_imag)
            lap_r = laplacian(E_r, xb)
            lap_i = laplacian(E_i, xb)

            # Complex Helmholtz PDE split into real and imaginary equations:
            #   lap_r = a*E_r - b*E_i
            #   lap_i = a*E_i + b*E_r

            pde_r = lap_r - (a_eff * E_r - b_eff * E_i)
            pde_i = lap_i - (a_eff * E_i + b_eff * E_r)

            # Accumulate PDE residual
            pde_loss += torch.mean(pde_r**2) + torch.mean(pde_i**2)

        loss = loss_data + lambda_pde * pde_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}  Loss={total_loss/len(train_loader):.6f}, PDE Loss={pde_loss.item():.6f}, Data Loss={loss_data.item():.6f}")

# =============================================================
# 7. Evaluation
# =============================================================
model.eval()
test_losses = []
with torch.no_grad():
    for xb, yb, _ in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        test_losses.append(criterion(pred, yb).item())

print("\nTest MSE:", np.mean(test_losses))

# =============================================================
# 8. Save model
# =============================================================
torch.save(model.state_dict(), "Efield_predictor_mlp_pde.pt")
print("✅ Saved: Efield_predictor_pinn.pt")
