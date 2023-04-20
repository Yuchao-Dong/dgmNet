import torch
import numpy as np
import matplotlib.pyplot as plt
from util import regular_sample, square2d

# select
a = 0  # horizontal velocity
bounds = torch.Tensor([[0, 1], [0, 1]])  # x and y bounds

# load model
model = torch.load("../models/advection2d.pt")

# plot attempted solution
n = 65
t, x = regular_sample(n=n, d=2, requires_grad=False, bounds=bounds)
# include constant velocity
x = torch.cat((x, a * torch.ones(len(t), 1)), dim=1)
fx = model(t, x).detach()
# global min and max for colorbar
hmin, hmax = torch.min(fx), torch.max(fx)
# initialze 2 by 3 plot
tplot = [0, 0.25, 0.5, 0.75, 1]
fig, axes = plt.subplots(2, 3, figsize=(11, 6))
# u0
axes[0, 0].imshow(
    np.flipud(square2d(x[(t == 0).squeeze()]).numpy().reshape(n, n, order="F")),
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
    axes[j // 3, j % 3].set_title(f"t = {tplot[i]}")
# finish plot
fig.colorbar(im, ax=axes.ravel().tolist())
fig.suptitle(f"a = {a}")
axes[1, 0].set_xlabel("x")
axes[1, 0].set_ylabel("y")
plt.show()
