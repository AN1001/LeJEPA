"""
Generates graphs from the data in experiments/main/train_log.json.

Generates graphs for loss, val acc, std min/med/max, lr, erank and cos
as epoch changes.
"""

import matplotlib.pyplot as plt
import json

with open("./experiments/main/train_log.json") as f:
    d = json.load(f)

data_keys = ("epoch", "sig_loss", "sim_loss", "erank", "cos_cent",
             "cos", "lr", "std_min", "std_med", "std_max", "val_acc")
data = {key: [] for key in data_keys}

# Extract data
for epoch_data in d:
    for k in data_keys:
        if k in epoch_data:
            data[k].append(epoch_data[k])


def _save_basic(x, y, label, f_name):
    """Plots and saves a basic graph"""
    fig, ax = plt.subplots()

    ax.plot(x, y)

    ax.set_title(label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(label)

    fig.savefig(f"./figures/{f_name}.png", bbox_inches="tight")

# Plot and save effective rank graph, mp cos and lr
x = data["epoch"]
_save_basic(x, data["erank"], "Effective Rank", "erank")
_save_basic(x, data["cos"], "Mean Pairwise Cosine", "cos")
_save_basic(x, data["lr"], "Learning Rate", "lr")
_save_basic(range(0, 600, 15), data["val_acc"], "Validation Accuracy", "val_acc")


# Plot and save loss graph
lam = 0.01
loss = [lam*sig+(1-lam)*sim for sim, sig in zip(data["sim_loss"], data["sig_loss"])]
fig, ax = plt.subplots()

ax.plot(x, loss)

ax.set_title("Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")

fig.savefig(f"./figures/loss.png", bbox_inches="tight")


# Plot and save std graph
fig, ax = plt.subplots()

ax.fill_between(data["epoch"], data["std_min"], data["std_max"], alpha=0.3)
ax.plot(data["epoch"], data["std_med"])

ax.set_title("Min/Med/Max std across Dimensions")
ax.set_xlabel("Epoch")
ax.set_ylabel("Min/Med/Max std Across Dimensions")

fig.savefig(f"./figures/std.png", bbox_inches="tight")
