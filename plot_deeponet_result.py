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
model_path = os.path.join(script_dir, "Efield_predictor_deeponet.pt")
data_path = os.path.join(script_dir, "electric_fields.npz")

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
    Ey_r_pred, Ey_i_pred,
    Ez_r_pred, Ez_i_pred,
    norm_pred
]

components_true = [
    Ex_r_an, Ex_i_an,
    Ey_r_an, Ey_i_an,
    Ez_r_an, Ez_i_an,
    norm_true
]

titles = [
    "Ex Real", "Ex Imag",
    "Ey Real", "Ey Imag",
    "Ez Real", "Ez Imag",
    "‖E‖ Norm"
]

plt.figure(figsize=(28, 10))

for i in range(7):
    # row 1 — DeepONet prediction
    plt.subplot(3, 7, i+1)
    plt.title(f"{titles[i]}\nModel")
    plt.imshow(components_pred[i], origin="lower", cmap="viridis")
    plt.axis("off")
    plt.colorbar(fraction=0.046, pad=0.04)

    # row 2 — analytic field
    plt.subplot(3, 7, i+1+7)
    plt.title(f"{titles[i]}\nAnalytic")
    plt.imshow(components_true[i], origin="lower", cmap="viridis")
    plt.axis("off")
    plt.colorbar(fraction=0.046, pad=0.04)

    # row 3 — absolute error
    plt.subplot(3, 7, i+1+14)
    plt.title(f"{titles[i]}\nError")
    plt.imshow(np.abs(components_pred[i] - components_true[i]),
               origin="lower", cmap="magma")
    plt.axis("off")
    plt.colorbar(fraction=0.046, pad=0.04)

plt.show()