import numpy as np
import pyvista as pv
import miepython.field as fields

# -----------------------
# 1. Define 3D grid
# -----------------------
N = 41  # lower resolution for interactive 3D plotting
u = np.linspace(-2, 2, N)

X, Y, Z = np.meshgrid(u, u, u, indexing="ij")

# -----------------------
# 2. Compute analytic fields
# -----------------------
lam = 7.0
m_val = 2 * lam
k_val = 5 * lam

E_xyz, H_xyz = fields.eh_near_cartesian(
    lambda0=lam,
    d_sphere=1.0,
    m_sphere=m_val + 1j * k_val,
    n_env=1.0,
    x=X, y=Y, z=Z
)

Ex = E_xyz[0]
Ey = E_xyz[1]
Ez = E_xyz[2]

# Magnitude
E_norm = np.sqrt(np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2)

# -----------------------
# 3. Build PyVista grid
# -----------------------

# --- Create grid ---
grid = pv.StructuredGrid(X, Y, Z)

# Store vector components
grid["vectors_real"] = np.column_stack([
    Ex.real.ravel(order="F"),
    Ey.real.ravel(order="F"),
    Ez.real.ravel(order="F")
])

grid["vectors_imag"] = np.column_stack([
    Ex.imag.ravel(order="F"),
    Ey.imag.ravel(order="F"),
    Ez.imag.ravel(order="F")
])

# Optional scalar for coloring
grid["E_norm"] = E_norm.ravel(order="F")

# --- Visualize vector field ---
plotter = pv.Plotter()

# Add arrows (vector field)
plotter.add_arrows(
    grid.points,
    grid["vectors_imag"],
    mag=0.2,            # scale arrow size
    opacity=0.8,
)

# Optional: add semi-transparent magnitude volume
# plotter.add_volume(grid, scalars="E_norm", opacity="sigmoid")

plotter.add_axes()
plotter.show()
