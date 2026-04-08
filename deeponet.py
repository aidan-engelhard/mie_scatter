import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os
import csv

# ---------------------------------------------------------
# Load NPZ data
# ---------------------------------------------------------
R = 0.5 # radius of the sphere (from dataset description)
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "inputs/electric_fields.npz")
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
# Flatten grid + build dataset
# ---------------------------------------------------------
X_flat = X.reshape(-1).astype(np.float32)
Y_flat = Y.reshape(-1).astype(np.float32)
Z_flat = Z.reshape(-1).astype(np.float32)

coords_list = []
params_list = []
targets_list = []
helm_list = []  # store (n^2 k0^2) values for PDE

for i in range(num_lam):
    lam_val = lam_values[i]
    m = m_values[i]
    k = k_values[i]

    # Complex refractive index
    n_complex = m + 1j * k
    k0 = 2 * np.pi / lam_val
    helm_coeff = (n_complex**2 * k0**2)  # complex
    helm_real = np.real(helm_coeff).astype(np.float32)
    helm_imag = np.imag(helm_coeff).astype(np.float32)

    lam_col = np.full((num_points,), lam_val, dtype=np.float32)
    helm_real_col = np.full((num_points,), helm_real, dtype=np.float32)
    helm_imag_col = np.full((num_points,), helm_imag, dtype=np.float32)

    helm_coeff_pair = np.column_stack([helm_real_col, helm_imag_col])

    # Inputs: (x, y, z), (λ)
    coords = np.column_stack([X_flat, Y_flat, Z_flat])
    params = lam_col.reshape(-1,1)

    # Targets: 6 channels (Ex_r, Ex_i, Ey_r, Ey_i, Ez_r, Ez_i)
    tar = np.column_stack([
        Ex_r[i].reshape(-1), Ex_i[i].reshape(-1),
        Ey_r[i].reshape(-1), Ey_i[i].reshape(-1),
        Ez_r[i].reshape(-1), Ez_i[i].reshape(-1),
    ]).astype(np.float32)
    
    coords_list.append(coords)
    params_list.append(params)
    targets_list.append(tar)
    helm_list.append(helm_coeff_pair)

# Stack all data
coords_all = torch.tensor(np.vstack(coords_list), dtype=torch.float32)
params_all = torch.tensor(np.vstack(params_list), dtype=torch.float32)
Y_all = torch.tensor(np.vstack(targets_list), dtype=torch.float32)
helm_all = torch.tensor(np.concatenate(helm_list), dtype=torch.float32)


# ---------------------------------------------------------
# Train/test split based on wavelength index
# ---------------------------------------------------------
num_train_lam = 80  # first 80 λ values for training

sample_lam_indices = np.repeat(np.arange(num_lam), num_points)
train_mask = torch.tensor(sample_lam_indices < num_train_lam, dtype=torch.bool)
test_mask = ~train_mask

coords_train, params_train, Y_train, helm_train = coords_all[train_mask], params_all[train_mask], Y_all[train_mask], helm_all[train_mask]
coords_test, params_test, Y_test, helm_test = coords_all[test_mask], params_all[test_mask], Y_all[test_mask], helm_all[test_mask]

train_ds = TensorDataset(coords_train, params_train, Y_train, helm_train)
test_ds  = TensorDataset(coords_test, params_test, Y_test, helm_test)

train_loader = DataLoader(train_ds, batch_size=4096, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=4096, shuffle=False)

# ---------------------------------------------------------
# DeepONet architecture:
# ---------------------------------------------------------

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = DeepONet().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Helper function for Laplacian
def laplacian(outputs, inputs):
    """
    Compute ΔE using autograd.
    outputs: (batch, 6)
    inputs:  (batch, 3) -> (x, y, z)
    """
    grads = torch.autograd.grad(
        outputs, inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True
    )[0]  # shape: (batch, 3)

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

# Helper function to sample points on sphere boundary
def sample_sphere_boundary(N, R, device):
    theta = torch.rand(N, device=device) * 2 * np.pi
    phi = torch.acos(2*torch.rand(N, device=device) - 1)

    x = R * torch.sin(phi) * torch.cos(theta)
    y = R * torch.sin(phi) * torch.sin(theta)
    z = R * torch.cos(phi)

    return torch.stack([x, y, z], dim=1)


# Helper function to get boundary pairs (inside/outside) and normals
def get_boundary_pairs(coords_b, R, eps=1e-3):
    """
    coords_b: (Nb, 3) points ON the boundary
    R: scatterer radius
    eps: small offset

    Returns:
        coords_in, coords_out, normals
    """

    # radial distance
    r = torch.norm(coords_b, dim=1, keepdim=True)

    # outward normal
    n_hat = coords_b / r

    # shift slightly inside / outside
    coords_in = coords_b - eps * n_hat
    coords_out = coords_b + eps * n_hat

    return coords_in, coords_out, n_hat

# Helper function to compute normal derivative dE/dn
def normal_derivative(E, coords, normals):
    """
    E: (N, 6) field outputs
    coords: (N, 3)
    normals: (N, 3)
    """

    grads = torch.autograd.grad(
        E, coords,
        grad_outputs=torch.ones_like(E),
        create_graph=True,
        retain_graph=True
    )[0]                         # (N, 3)

    # scalar normal derivative for each field component
    dEdn = torch.sum(grads * normals, dim=1, keepdim=True)
    return dEdn

# Helper function to compute boundary loss
def boundary_loss(model, coords_b, params_b, R):
    """
    Enforces:
      E_in = E_out
      dE/dn_in = dE/dn_out
    """

    coords_b.requires_grad_(True)

    # create inside / outside points
    coords_in, coords_out, normals = get_boundary_pairs(coords_b, R)

    coords_in.requires_grad_(True)
    coords_out.requires_grad_(True)

    # predict fields
    E_in = model(coords_in, params_b)
    E_out = model(coords_out, params_b)

    # field continuity
    loss_E = torch.mean((E_in - E_out)**2)

    # normal derivative continuity
    dEin_dn = normal_derivative(E_in, coords_in, normals)
    dEout_dn = normal_derivative(E_out, coords_out, normals)

    loss_dE = torch.mean((dEin_dn - dEout_dn)**2)

    return loss_E, loss_dE

# ---------------------------------------------------------
# Training loop
# ---------------------------------------------------------
# weights for loss terms
lambda_data = 0.0
lambda_pde = 1.0  
lambda_E_bc = 1.0
lambda_dE_bc = 1.0

epochs = 100
Nb = 1000 # number of boundary points per epoch
loss_history = []
data_loss_history = []
pde_loss_history = []
loss_E_bc_history = []
loss_dE_bc_history = []

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0

    for coordb, paramb, yb, helm in train_loader:
        coordb, paramb, yb, helm = coordb.to(device).requires_grad_(True), paramb.to(device), yb.to(device), helm.to(device)
        optimizer.zero_grad()

        pred = model(coordb, paramb)
        data_loss = criterion(pred, yb)

        # PDE loss: full complex Helmholtz PDE
        pde_loss = 0.0

        # --- Compute radius to determine inside/outside sphere ---
        r = torch.sqrt(coordb[:,0]**2 + coordb[:,1]**2 + coordb[:,2]**2)
        inside = (r <= R)
        outside = ~inside

        # --- Compute free-space k0^2 ---
        lam = paramb[:, 0]
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
            lap_r = laplacian(E_r, coordb)
            lap_i = laplacian(E_i, coordb)

            # Complex Helmholtz PDE split into real and imaginary equations:
            #   lap_r = a*E_r - b*E_i
            #   lap_i = a*E_i + b*E_r

            pde_r = lap_r - (a_eff * E_r - b_eff * E_i)
            pde_i = lap_i - (a_eff * E_i + b_eff * E_r)

            # Accumulate PDE residual
            pde_loss += torch.mean(pde_r**2) + torch.mean(pde_i**2)

        
        # --- Boundary loss ---
        coords_boundary = sample_sphere_boundary(Nb, R, device)
        
        Nb = coords_boundary.shape[0]

        # pick one physical configuration (e.g. first in batch)
        param_single = paramb[0:1]                 # (1, param_dim)

        # repeat for all boundary points
        params_boundary = param_single.repeat(Nb, 1)  # (Nb, param_dim)


        loss_E_bc, loss_dE_bc = boundary_loss(
            model, coords_boundary, params_boundary, R
        )


        loss = lambda_data*data_loss + lambda_pde * pde_loss + lambda_E_bc * loss_E_bc + lambda_dE_bc * loss_dE_bc
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        loss_history.append(loss.item())
        data_loss_history.append(data_loss.item())      
        pde_loss_history.append(pde_loss.item())
        loss_E_bc_history.append(loss_E_bc.item())
        loss_dE_bc_history.append(loss_dE_bc.item())

    if epoch % 10 == 0:
        print(f"Epoch {epoch}/{epochs} - Loss = {epoch_loss/len(train_loader):.6f}", 
                f"Data: {data_loss.item():.6f}", 
                f"PDE: {pde_loss.item():.6f}",
                f"BC_E: {loss_E_bc.item():.6f}", 
                f"BC_dE: {loss_dE_bc.item():.6f}")

# Save training losses to CSV
with open("training_losses.csv", mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "loss", "data_loss", "pde_loss", "loss_E_bc", "loss_dE_bc"])  # header

    for epoch, (loss, data_loss, pde_loss, loss_E_bc, loss_dE_bc) in enumerate(zip(loss_history, data_loss_history, pde_loss_history, loss_E_bc_history, loss_dE_bc_history)):
        writer.writerow([epoch, loss, data_loss, pde_loss, loss_E_bc, loss_dE_bc])


# ---------------------------------------------------------
# 6. Test evaluation
# ---------------------------------------------------------
model.eval()
test_losses = []

with torch.no_grad():
    for coordb, paramb, yb, hem in test_loader:
        coordb, paramb, yb, hem = coordb.to(device), paramb.to(device), yb.to(device), hem.to(device)
        pred = model(coordb, paramb)
        test_losses.append(criterion(pred, yb).item())

print("\n✅ Test MSE:", np.mean(test_losses))

# ---------------------------------------------------------
# 7. Save model
# ---------------------------------------------------------
torch.save(model.state_dict(), "models/Efield_predictor_deeponet_physics_only.pt")
print("✅ Saved model as Efield_predictor_deeponet_physics_only.pt")