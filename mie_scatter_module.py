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

# --- Storage arrays for REAL and IMAG components ---
Ex_real = np.zeros((num_samples, X.shape[0], X.shape[1]), dtype=np.float32)
Ex_imag = np.zeros_like(Ex_real)

Ey_real = np.zeros_like(Ex_real)
Ey_imag = np.zeros_like(Ex_real)

Ez_real = np.zeros_like(Ex_real)
Ez_imag = np.zeros_like(Ex_real)

for i in range(num_samples):
    print(f"Computing {i+1}/{num_samples}")

    # --- Compute fields ---
    E_xyz, H_xyz = fields.eh_near_cartesian(
        lambda0=lam_vals[i],
        d_sphere=1.0,
        m_sphere=m_vals[i] + 1j * k_vals[i],
        n_env=1.0,
        x=X,
        y=Y,
        z=Z,
    )

    Ex, Ey, Ez = E_xyz[0], E_xyz[1], E_xyz[2]

    # --- Save REAL components ---
    Ex_real[i] = Ex.real
    Ey_real[i] = Ey.real
    Ez_real[i] = Ez.real

    # --- Save IMAG components ---
    Ex_imag[i] = Ex.imag
    Ey_imag[i] = Ey.imag
    Ez_imag[i] = Ez.imag

# --- Save final NPZ arrays ---
np.savez_compressed(
    "electric_fields.npz",
    Ex_real=Ex_real,
    Ex_imag=Ex_imag,
    Ey_real=Ey_real,
    Ey_imag=Ey_imag,
    Ez_real=Ez_real,
    Ez_imag=Ez_imag,
    X=X.astype(np.float32),
    Y=Y.astype(np.float32),
    Z=Z.astype(np.float32),
    lam_values=lam_vals.astype(np.float32),
    m_values=m_vals.astype(np.float32),
    k_values=k_vals.astype(np.float32),
)

print("\n✅ Finished generating REAL/IMAG E-field data!")
print("Saved: electric_fields_real_imag.npz")