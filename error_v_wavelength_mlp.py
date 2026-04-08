import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import miepython.field as fields

# ---------------------------------------------------------
# 1. Model architecture (match the 6-output training model)
# ---------------------------------------------------------
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, 256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 6)   # predict 6 channels
        )
    def forward(self, x):
        return self.layers(x)

# ---------------------------------------------------------
# 2. Load model + dataset
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "models/Efield_predictor_mlp.pt")
data_path  = os.path.join(script_dir, "inputs/electric_fields.npz")

print("Loading model from:", model_path)
model = MLP().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))

data = np.load(data_path)

# Load real/imag components (true fields)
Ex_r_true = data["Ex_real"]
Ex_i_true = data["Ex_imag"]
Ey_r_true = data["Ey_real"]
Ey_i_true = data["Ey_imag"]
Ez_r_true = data["Ez_real"]
Ez_i_true = data["Ez_imag"]

X = data["X"]
Y = data["Y"]
Z = data["Z"]
lam_values = data["lam_values"]

H, W = X.shape
num_points = H * W

# ---------------------------------------------------------
# 3. Select wavelength and prepare inputs
# ---------------------------------------------------------
lams = np.linspace(1.0, 20.0, 20)
avg_log_rel_errors = []

for chosen_lam in lams:
    chosen_m = 2 * chosen_lam
    chosen_k = 5 * chosen_lam

    Xf = X.reshape(-1).astype(np.float32)
    Yf = Y.reshape(-1).astype(np.float32)
    Zf = Z.reshape(-1).astype(np.float32)

    lam_col = np.full((num_points,), chosen_lam, dtype=np.float32)
    inputs = np.column_stack([Xf, Yf, Zf, lam_col])

    inputs = torch.tensor(inputs, dtype=torch.float32).to(device)

    # ---------------------------------------------------------
    # 4. Predict using neural model
    # ---------------------------------------------------------
    with torch.no_grad():
        pred = model(inputs).cpu().numpy()

    # reshape to images
    Ex_r_pred = pred[:, 0].reshape(H, W)
    Ex_i_pred = pred[:, 1].reshape(H, W)
    Ey_r_pred = pred[:, 2].reshape(H, W)
    Ey_i_pred = pred[:, 3].reshape(H, W)
    Ez_r_pred = pred[:, 4].reshape(H, W)
    Ez_i_pred = pred[:, 5].reshape(H, W)

    # ---------------------------------------------------------
    # 5. Compute analytic fields for comparison
    # ---------------------------------------------------------
    E_xyz, _ = fields.eh_near_cartesian(
        lambda0=chosen_lam,
        d_sphere=1.0,
        m_sphere=chosen_m + 1j * chosen_k,
        n_env=1.0,
        x=X, y=Y, z=Z
    )

    Ex_true = E_xyz[0]
    Ey_true = E_xyz[1]
    Ez_true = E_xyz[2]

    Ex_r_an = Ex_true.real
    Ex_i_an = Ex_true.imag
    Ey_r_an = Ey_true.real
    Ey_i_an = Ey_true.imag
    Ez_r_an = Ez_true.real
    Ez_i_an = Ez_true.imag

    # Norm
    norm_pred = np.sqrt(Ex_r_pred**2 + Ex_i_pred**2 +
                        Ey_r_pred**2 + Ey_i_pred**2 +
                        Ez_r_pred**2 + Ez_i_pred**2)

    norm_true = np.abs(Ex_true)**2 + np.abs(Ey_true)**2 + np.abs(Ez_true)**2
    norm_true = np.sqrt(norm_true)

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

    avg_log_rel_errors.append(avg_log_rel_error)

# Plot average log-relative error vs wavelength
plt.figure(figsize=(8, 5))

plt.plot(lams, avg_log_rel_errors, marker='o')
plt.xlabel("Wavelength (λ)")   
plt.ylabel("Average Log-Relative Error")
plt.show()