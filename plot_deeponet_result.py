import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import miepython.field as fields

# =========================================================
# 1. DeepONet model (same as training)
# =========================================================

class BranchNet(nn.Module):
    def __init__(self, param_dim=1, width=128, latent_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(param_dim, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, latent_dim)
        )

    def forward(self, lam):
        return self.net(lam)


class TrunkNet(nn.Module):
    def __init__(self, coord_dim=3, width=128, latent_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(coord_dim, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, latent_dim)
        )

    def forward(self, xyz):
        return self.net(xyz)


class DeepONet(nn.Module):
    def __init__(self, latent_dim=128, output_dim=6):
        super().__init__()
        self.branch = BranchNet(param_dim=1, latent_dim=latent_dim)
        self.trunk  = TrunkNet(coord_dim=3, latent_dim=latent_dim)

        # Final linear layer maps inner-product → field components
        self.out = nn.Linear(latent_dim, output_dim)

    def forward(self, coords, params):
        """
        coords: (N,3)   = (x,y,z)
        params: (N,1)   = λ
        """
        b = self.branch(params)   # (N, latent)
        t = self.trunk(coords)    # (N, latent)

        # Elementwise product = DeepONet operator learning
        prod = b * t              # (N, latent)

        return self.out(prod)     # -> (N,6)

# =========================================================
# 2. Load trained DeepONet model + data
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "models/Efield_predictor_deeponet.pt")
data_path = os.path.join(script_dir, "inputs/electric_fields.npz")

print("Loading DeepONet model:", model_path)
model = DeepONet().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

data = np.load(data_path)

X = data["X"]
Y = data["Y"]
Z = data["Z"]

H, W = X.shape
num_points = H * W

# =========================================================
# 3. Choose parameter value (λ) and build inputs
# =========================================================
chosen_lam = 6.0
print(f"\nPlotting prediction for λ = {chosen_lam}")

# flatten coords
coords = np.column_stack([
    X.reshape(-1).astype(np.float32),
    Y.reshape(-1).astype(np.float32),
    Z.reshape(-1).astype(np.float32)
])

# parameter column
params = np.full((num_points,1), chosen_lam, dtype=np.float32)

coords_tensor = torch.tensor(coords, dtype=torch.float32).to(device)
params_tensor = torch.tensor(params, dtype=torch.float32).to(device)

# =========================================================
# 4. Predict with DeepONet
# =========================================================
with torch.no_grad():
    pred = model(coords_tensor, params_tensor).cpu().numpy()

Ex_r_pred = pred[:,0].reshape(H, W)
Ex_i_pred = pred[:,1].reshape(H, W)
Ey_r_pred = pred[:,2].reshape(H, W)
Ey_i_pred = pred[:,3].reshape(H, W)
Ez_r_pred = pred[:,4].reshape(H, W)
Ez_i_pred = pred[:,5].reshape(H, W)

# =========================================================
# 5. Analytic solution using miepython
# =========================================================
chosen_m = 2 * chosen_lam
chosen_k = 5 * chosen_lam

E_xyz, _ = fields.eh_near_cartesian(
    lambda0=chosen_lam,
    d_sphere=1.0,
    m_sphere=chosen_m + 1j * chosen_k,
    n_env=1.0,
    x=X, y=Y, z=Z
)

Ex_true, Ey_true, Ez_true = E_xyz
Ex_r_an, Ex_i_an = Ex_true.real, Ex_true.imag
Ey_r_an, Ey_i_an = Ey_true.real, Ey_true.imag
Ez_r_an, Ez_i_an = Ez_true.real, Ez_true.imag

norm_pred = np.sqrt(
    Ex_r_pred**2 + Ex_i_pred**2 +
    Ey_r_pred**2 + Ey_i_pred**2 +
    Ez_r_pred**2 + Ez_i_pred**2
)

norm_true = np.sqrt(
    np.abs(Ex_true)**2 +
    np.abs(Ey_true)**2 +
    np.abs(Ez_true)**2
)

# =========================================================
# 6. Plot 3×7 comparison (Model / Analytic / Error)
# =========================================================
components_pred = [
    Ex_r_pred, Ex_i_pred,
    Ez_r_pred, Ez_i_pred,
    norm_pred
]

components_true = [
    Ex_r_an, Ex_i_an,
    Ez_r_an, Ez_i_an,
    norm_true
]

titles = [
    "Ex Real", "Ex Imag",
    "Ez Real", "Ez Imag",
    "|E| Norm"
]

fig, axes = plt.subplots(2, 5, figsize=(20, 8), constrained_layout=True)

# --------------------------------------------------
# Compute global scale (per row)
# --------------------------------------------------
# Row 1: model predictions
vals_pred = np.concatenate([c.ravel() for c in components_pred])
vmin_pred, vmax_pred = vals_pred.min(), vals_pred.max()

# Row 2: analytic solution
vals_true = np.concatenate([c.ravel() for c in components_true])
vmin_true, vmax_true = vals_true.min(), vals_true.max()

# --------------------------------------------------
# Row 1 — DeepONet prediction
# --------------------------------------------------
for i in range(5):
    im0 = axes[0, i].imshow(
        components_pred[i],
        origin="lower",
        cmap="viridis",
        vmin=vmin_pred,
        vmax=vmax_pred
    )
    axes[0, i].set_title(f"{titles[i]}\nModel")
    axes[0, i].axis("off")

# shared colorbar (row 1)
fig.colorbar(
    im0,
    ax=axes[0, :],
    location="right",
    shrink=0.85,
    pad=0.02
)

# --------------------------------------------------
# Row 2 — Analytic field
# --------------------------------------------------
for i in range(5):
    im1 = axes[1, i].imshow(
        components_true[i],
        origin="lower",
        cmap="viridis",
        vmin=vmin_true,
        vmax=vmax_true
    )
    axes[1, i].set_title(f"{titles[i]}\nAnalytic")
    axes[1, i].axis("off")

# shared colorbar (row 2)
fig.colorbar(
    im1,
    ax=axes[1, :],
    location="right",
    shrink=0.85,
    pad=0.02
)

# Relative error plot for |E| norm
eps = 1e-10  # small constant to avoid division by zero
error =   np.abs(norm_pred - norm_true)

threshold = 1e-2 * norm_true.max()
mask = norm_true > threshold

log_rel_error_masked = np.full_like(norm_true, np.nan)
log_rel_error_masked[mask] = np.log10(
    np.abs(norm_pred[mask] - norm_true[mask]) / norm_true[mask]
)

avg_log_rel_error = np.nanmean(log_rel_error_masked)


fig, axs = plt.subplots(1, 2, figsize=(10, 8), constrained_layout=True)

im = axs[0].imshow(
    log_rel_error_masked,
    origin="lower",
    cmap="inferno"
)

axs[0].set_title(f"Log Relative Error of |E|  \n(Avg: {avg_log_rel_error:.2f})")
fig.colorbar(im, ax = axs[0], label=r"$\log_{10}(| |E|_{pred} - |E|_{true} | / |E|_{true})$", shrink = 0.75)

im2 = axs[1].imshow(
    error,
    origin="lower",
    cmap="inferno"
)

axs[1].set_title("Absolute Error of |E|")
fig.colorbar(im2, ax = axs[1], label=r"$| |E|_{pred} - |E|_{true} |$", shrink = 0.75)

plt.show()
