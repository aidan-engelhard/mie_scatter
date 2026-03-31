import numpy as np
import matplotlib.pyplot as plt
import os
import miepython.field as fields

# --- Grid configuration ---
u = np.linspace(-1.5, 1.5, 101)
X, Z = np.meshgrid(u, u, indexing="xy")
Y = np.zeros_like(X)

# --- wavelength and permittivity values ---
num_samples = 100
lam_vals = np.linspace(5.0, 10.0, num_samples)
m_vals = 2 * lam_vals
k_vals = 5 * lam_vals

# --- Storage array for NN training ---
E_data = np.zeros((len(m_vals), X.shape[0], X.shape[1]), dtype=np.float32)

for i in range(num_samples):
    print(f"Computing {i+1}/{len(m_vals)}")

    # --- Compute fields ---
    E_xyz, H_xyz = fields.eh_near_cartesian(
        lambda0=lam_vals[i],
        d_sphere=1.0,
        m_sphere=m_vals[i] + k_vals[i]*1j,
        n_env=1.0,
        x=X,
        y=Y,
        z=Z,
    )

    # --- Compute intensity |E|² ---
    Ex, Ey, Ez = E_xyz[0], E_xyz[1], E_xyz[2]
    Intensity = np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2

    # --- Store for NN training ---
    E_data[i] = Intensity.astype(np.float64)

    print(f"Max Intensity: {np.max(Intensity)}")

# --- Save final NPZ arrays (everything in one file) ---
np.savez_compressed(
    "intensities.npz",
    E=E_data,
    X=X.astype(np.float32),
    Y=Y.astype(np.float32),
    Z=Z.astype(np.float32),
    lam_values = lam_vals.astype(np.float32),
    m_values=m_vals.astype(np.float32),
    k_values = k_vals.astype(np.float32),

)

print("\n✅ Finished generating training data!")
print("Saved:")