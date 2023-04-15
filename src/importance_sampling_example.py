import torch
import numpy as np
import matplotlib.pyplot as plt
from dgm import DGMloss
from util import importance_resample

# load model
model = torch.load("../models/advection1d.pt")
criterion = DGMloss(u0_preset="square")

# importance test
nimp = 32  # square root of total number of sample points
t = torch.Tensor(
    np.repeat(np.linspace(0, 1, nimp), nimp).reshape(-1, 1)
).requires_grad_(True)
x = torch.Tensor(np.tile(np.linspace(0, 1, nimp), nimp).reshape(-1, 1)).requires_grad_(
    True
)
diffloss = torch.abs(criterion.differential_loss(model, t, x))
timp, ximp = importance_resample(t=t, x=x, importance=diffloss)

# plot
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
# uniform vs important points
ax1.plot(x.detach().numpy(), t.detach().numpy(), ".", alpha=1, label="previous points")
ax1.plot(
    ximp.detach().numpy(),
    timp.detach().numpy(),
    ".",
    alpha=0.5,
    label="importance resampled points",
)
ax1.set_ylabel("t")
ax1.legend()
ax1.title.set_text("Sample points")
# differential loss heatmap
hm = ax2.imshow(
    np.flipud(diffloss.detach().numpy().reshape(nimp, nimp)),
    cmap="hot",
    interpolation="nearest",
    extent=[0, 1, 0, 1],
    aspect="auto",
)
ax2.set_xlabel("x")
ax2.set_ylabel("t")
ax1.title.set_text("dfdt + L(f) loss evaluated at ")
plt.show()
