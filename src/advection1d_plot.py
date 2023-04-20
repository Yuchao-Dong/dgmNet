import torch
import numpy as np
import matplotlib.pyplot as plt
from util import u0Presets, regular_sample

# load model
model = torch.load('../models/advection1d.pt')

# plot attempted solution
t, x = regular_sample(n = 101, d=1, requires_grad = False)
fx = model(t, x).detach()
plt.plot(x[t == 0].numpy(), u0Presets().square(x[t == 0]).numpy(), label='u0')
plt.plot(x[t == 0].numpy(), fx[t == 0].numpy(), label='t=0')
plt.plot(x[t == 0.5].numpy(), fx[t == 0.5].numpy(), label='t=0.5')
plt.plot(x[t == 1].numpy(), fx[t == 1].numpy(), label='t=1')
plt.legend()
plt.show()