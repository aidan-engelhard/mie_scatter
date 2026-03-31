import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import miepython.field as fields

print(fields.eh_near_cartesian)
print(fields.__file__)

# ---------------------------------------------------------
# 1. Define the model architecture (must match training)
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
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.layers(x)


# ---------------------------------------------------------
# 2. Load model
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP().to(device)

# Find folder where this script lives
script_dir = os.path.dirname(os.path.abspath(__file__))

# Auto-locate model + data inside this folder
model_path = os.path.join(script_dir, "Efield_predictor_mlp.pt")
data_path  = os.path.join(script_dir, "intensities.npz")

print("Loading model from:", model_path)

model.load_state_dict(torch.load(model_path, map_location=device))

# ---------------------------------------------------------
# 3. Load dataset / grid
# ---------------------------------------------------------
data = np.load(data_path)

u = np.linspace(-1.5, 1.5, 101)
X, Z = np.meshgrid(u, u, indexing="xy")
Y = np.zeros_like(X)

m_values = data["m_values"]
E_true = data["E"]   # shape (100,101,101)

H, W = X.shape
num_points = H * W

# Flatten grid
Xf = X.reshape(-1).astype(np.float32)
Yf = Y.reshape(-1).astype(np.float32)
Zf = Z.reshape(-1).astype(np.float32)

# ---------------------------------------------------------
# 4. Choose an m-value to visualize
# ---------------------------------------------------------
chosen_lam = 7.0
chosen_m = 2 * chosen_lam
chosen_k = 5 * chosen_lam

print(f"\nPlotting prediction for lam = {chosen_lam}")

lam_column = np.full((num_points,), chosen_lam, dtype=np.float32)

# Build input features (x,y,z,m)
inputs = np.column_stack([Xf, Yf, Zf, lam_column])
inputs = torch.tensor(inputs, dtype=torch.float32).to(device)

# ---------------------------------------------------------
# 5. Run model prediction and analytical solution
# ---------------------------------------------------------
with torch.no_grad():
    pred = model(inputs).cpu().numpy().reshape(H, W)

E_xyz, H_xyz = fields.eh_near_cartesian(
        lambda0=chosen_lam,
        d_sphere=1.0,
        m_sphere=chosen_m + chosen_k * 1j,
        n_env=1.0,
        x=X,
        y=Y,
        z=Z,
    )

# --- Compute intensity |E|² ---
Ex, Ey, Ez = E_xyz[0], E_xyz[1], E_xyz[2]
Intensity = np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2

error_map = np.abs(pred - Intensity)

# ---------------------------------------------------------
# 6. Plot results
# ---------------------------------------------------------
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.title(r"Predicted Intensity ($\lambda$" f"={chosen_lam})")
plt.imshow(pred, cmap="viridis", origin="lower")
plt.colorbar()

# If chosen_m exists in the dataset, show ground truth
plt.subplot(1,3,2)
plt.title(r"Analytic Solution ($\lambda$" f"={chosen_lam})")
plt.imshow(Intensity, cmap="viridis", origin="lower")
plt.colorbar()

# --- Error plot ---
plt.subplot(1, 3, 3)
plt.title("Absolute Error |Pred − True|")
plt.imshow(error_map, cmap="viridis", origin="lower")
plt.colorbar()

plt.tight_layout()
plt.show()