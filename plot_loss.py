import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# Load CSV
# --------------------------------------------------
df = pd.read_csv("training_losses.csv")

epochs = df["epoch"]

# --------------------------------------------------
# Plot all losses on one log-scale plot
# --------------------------------------------------
plt.figure(figsize=(9, 6))

plt.plot(epochs, df["loss"], label="Total loss")
# plt.plot(epochs, df["data_loss"], label="Data loss")
plt.plot(epochs, df["pde_loss"], label="PDE loss")
plt.plot(epochs, df["loss_E_bc"], label="BC: E continuity")
plt.plot(epochs, df["loss_dE_bc"], label="BC: dE/dn continuity")

plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("All training losses (log scale)")
plt.legend()
plt.grid(True, which="both", linewidth=0.5)

plt.tight_layout()
plt.savefig("loss_all_log.png", dpi=300)
plt.show()