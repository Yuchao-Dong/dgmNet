from torch.utils.data import TensorDataset, DataLoader
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from dgm import DGMnet, DGMloss
from train import train

# define model
model = DGMnet(d=1, M=50, num_highway_layers=3)
criterion = DGMloss(u0_preset="square")

# train model
training_loss = train(
    model,
    criterion,
    num_epochs=1000,
    batch_size=1024,
    lr=0.01,
    num_lr_steps=2,
    gamma=0.5,
    save_path="models/advection1d.pt",
    print_every=100,
)

# plot error
plt.semilogy(training_loss)
plt.xlabel("iteration")
plt.ylabel("loss")
plt.show()

# plot attmepted solution
x = torch.Tensor(np.linspace(0, 1, 1000).reshape(-1, 1))
t = torch.Tensor(np.ones(1000).reshape(-1, 1))
plt.plot(
    x.numpy(),
    criterion.u0(x).detach().numpy(),
    label="ground truth t = 0",
)
plt.plot(x.numpy(), model(0 * t, x).detach().numpy(), label="predicted t = 0")
plt.plot(x.numpy(), model(0.1 * t, x).detach().numpy(), label="predicted t = 0.10")
plt.plot(x.numpy(), model(0.25 * t, x).detach().numpy(), label="predicted t = 0.25")
plt.plot(x.numpy(), model(0.5 * t, x).detach().numpy(), label="predicted t = 0.5")
plt.plot(x.numpy(), model(1 * t, x).detach().numpy(), label="predicted t = 1")
plt.legend()
plt.show()
