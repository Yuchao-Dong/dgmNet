import torch
import numpy as np
import matplotlib.pyplot as plt
from util import regular_sample, slotted_disk

# select
bounds = torch.Tensor([[-1.25, 1.25], [-1.25, 1.25]])  # x and y bounds

# load models
model1 = torch.load("../models/slotted_disk_T0_T1.pt")
model2 = torch.load("../models/slotted_disk_T1_T2.pt")
model3 = torch.load("../models/slotted_disk_T2_T3.pt")
model4 = torch.load("../models/slotted_disk_T3_T4.pt")
model5 = torch.load("../models/slotted_disk_T4_T5.pt")
model6 = torch.load("../models/slotted_disk_T5_T6.pt")
model7 = torch.load("../models/slotted_disk_T6_T7.pt")


# combine into large model
def model(t: float, x: torch.Tensor):
    n = x.shape[0]
    if t > 6:
        ttensor = (t - 6) * torch.ones(n, 1)
        return model7(ttensor, x)
    elif t > 5:
        ttensor = (t - 5) * torch.ones(n, 1)
        return model6(ttensor, x)
    elif t > 4:
        ttensor = (t - 4) * torch.ones(n, 1)
        return model5(ttensor, x)
    elif t > 3:
        ttensor = (t - 3) * torch.ones(n, 1)
        return model4(ttensor, x)
    elif t > 2:
        ttensor = (t - 2) * torch.ones(n, 1)
        return model3(ttensor, x)
    elif t > 1:
        ttensor = (t - 1) * torch.ones(n, 1)
        return model2(ttensor, x)
    else:
        ttensor = t * torch.ones(n, 1)
        return model1(ttensor, x)


# plot attempted solution
n = 65
t, x = regular_sample(n=n, d=2, requires_grad=False, bounds=bounds, T=1)
x = x[(t == 0).squeeze()]
fx_0 = model(0, x).detach()
fx_pihalf = model(np.pi / 2, x).detach()
fx_pi = model(np.pi, x).detach()
fx_3pihalf = model(3 * np.pi / 2, x).detach()
fx_2pi = model(2 * np.pi, x).detach()
fx = [fx_0, fx_pihalf, fx_pi, fx_3pihalf, fx_2pi]
# global min and max for colorbar
hmin, hmax = torch.min(fx_2pi), torch.max(fx_2pi)
# initialze 2 by 3 plot
tplot = [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]
fig, axes = plt.subplots(2, 3, figsize=(11, 6))
# u0
axes[0, 0].imshow(
    np.flipud(slotted_disk(x).numpy().reshape(n, n, order="F")),
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
        np.flipud(fx[i].numpy().reshape(n, n, order="F")),
        cmap="hot",
        vmin=hmin,
        vmax=hmax,
        interpolation="nearest",
        extent=bounds.flatten(),
    )
    axes[j // 3, j % 3].set_title(f"t = {tplot[i]:.4f}")
# finish plot
fig.colorbar(im, ax=axes.ravel().tolist())
axes[1, 0].set_xlabel("x")
axes[1, 0].set_ylabel("y")
plt.show()
