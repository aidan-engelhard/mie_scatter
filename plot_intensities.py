import numpy as np
import matplotlib.pyplot as plt
import os

# ---------------------------------------------------------
# Load intensities.npz from the same directory as the script
# ---------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "intensities.npz")

data = np.load(data_path)

E = data["E"]                # shape: (100, 101, 101)
m_values = data["m_values"]  # shape: (100,)
lam_values = data["lam_values"]
k_values = data["k_values"]

num_m, H, W = E.shape

print(f"Loaded E-field dataset: {E.shape}")
print(f"Loaded lambda values: {lam_values.shape}")

# ---------------------------------------------------------
# Create 10×10 grid plot
# ---------------------------------------------------------
fig, axes = plt.subplots(10, 10, figsize=(12, 12))
fig.suptitle("Electric Field Intensity for All m Values (|E|²)", fontsize=18)

# Color scale normalization
vmin = np.min(E)
vmax = np.max(E)

index = 0
for row in range(10):
    for col in range(10):
        ax = axes[row, col]
        
        img = E[index]
        lam_val = lam_values[index]

        ax.imshow(img, cmap="viridis", origin="lower")
        ax.set_title(r"$\lambda$ = " f"{lam_val:.5f}", fontsize=7)
        ax.axis("off")

        index += 1

# Add a shared colorbar
cbar = fig.colorbar(axes[0,0].images[0], ax=axes.ravel().tolist(), shrink=0.6)
cbar.set_label("|E|² Intensity", fontsize=12)

#plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()