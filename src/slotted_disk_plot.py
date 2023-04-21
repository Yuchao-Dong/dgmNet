import torch
import numpy as np
import matplotlib.pyplot as plt
from util import regular_sample, slotted_disk

# select
bounds = torch.Tensor([[-1.25, 1.25], [-1.25, 1.25]])  # x and y bounds
T = 1

# load model
model = torch.load("../models/slotted_disk.pt")

# plot attempted solution
n = 65
t, x = regular_sample(n=n, d=2, requires_grad=False, bounds=bounds, T=T)
fx = model(t, x).detach()
# global min and max for colorbar
hmin, hmax = torch.min(fx), torch.max(fx)
# initialze 2 by 3 plot
tplot = [0, 0.25 * T, 0.5 * T, 0.75 * T, T]
fig, axes = plt.subplots(2, 3, figsize=(11, 6))
# u0
axes[0, 0].imshow(
    np.flipud(slotted_disk(x[(t == 0).squeeze()]).numpy().reshape(n, n, order="F")),
    cmap="hot",
    vmin=hmin,
    vmax=hmax,
    interpolation="nearest",
    extent=bounds.flatten(),
)
axes[0, 0].set_title("u0")
# each subplot is a snapshot in time
for i in range(len(tplot)):
    idx = (t == tplot[i]).squeeze()
    j = i + 1
    im = axes[j // 3, j % 3].imshow(
        np.flipud(fx[idx].numpy().reshape(n, n, order="F")),
        cmap="hot",
        vmin=hmin,
        vmax=hmax,
        interpolation="nearest",
        extent=bounds.flatten(),
    )
    axes[j // 3, j % 3].set_title(f"t = {tplot[i]:.2f}")
# finish plot
fig.colorbar(im, ax=axes.ravel().tolist())
axes[1, 0].set_xlabel("x")
axes[1, 0].set_ylabel("y")
plt.show()
