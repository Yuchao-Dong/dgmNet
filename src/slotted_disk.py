"""
train a DGM model to learn the rotation of a slotted disk in a vortex
we train 7 models in increments of 1 time unit where the initial condition of all
modles after the first is the previous model evaluated at T=1
model inputs:
    t, x, y
model ouputs:
    u
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from dgm import DGMnet
from util import l2norm, slotted_disk, random_sample

# define model
d = 2  # [x, y]
model = DGMnet(d=d, M=64, L=2, activation="relu")
# problem setup
u0 = slotted_disk
bounds = torch.Tensor([[-1.25, 1.25], [-1.25, 1.25]])  # x, y bounds
T = 2 * np.pi
velocity_scale = 1
# hyperparameters
num_epochs = 100
num_iterations = 10
print_every = 10
nint, nb, n0 = 1024, 1024, 1024  # sample sizes in each subdomain
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

Tlocal = 1
while Tlocal - 1 < T:
    print(f"training model from T = {Tlocal - 1:.0f} to {Tlocal:.0f}")
    # admin
    model_path = f"../models/slotted_disk_T{Tlocal - 1:.0f}_T{Tlocal:.0f}.pt"

    # train the model
    bounded_dims = [0, 1]  # only x and y have boundaries
    training_loss = []
    for epoch in range(num_epochs):
        # draw random sample
        t, x = random_sample(n=nint, d=d, subdomain="interior", bounds=bounds, T=1)
        tb, xb, xb_mirror = random_sample(
            n=nb,
            d=d,
            subdomain="boundary",
            bounds=bounds,
            bounded_dims=bounded_dims,
            T=1,
        )
        t0, x0 = random_sample(n=n0, d=d, subdomain="initial", bounds=bounds, T=1)

        for _ in range(num_iterations):
            # initialize loss
            Lr = Lb = L0 = torch.Tensor([0])

            # residual loss (dfdt = - a * dfdx)
            if nint > 0:
                fx = model(t, x)
                a = velocity_scale * -x[:, 1:2]  # -y
                b = velocity_scale * x[:, 0:1]  # +x
                dfdt = torch.autograd.grad(
                    fx, t, grad_outputs=torch.ones_like(fx), create_graph=True
                )[0]
                d_af_dx = torch.autograd.grad(
                    a * fx, x, grad_outputs=torch.ones_like(fx), create_graph=True
                )[0]
                d_bf_dy = torch.autograd.grad(
                    b * fx, x, grad_outputs=torch.ones_like(fx), create_graph=True
                )[0]
                Lr = dfdt + d_af_dx[:, 0:1] + d_bf_dy[:, 1:2]

            # boundary loss (periodic)
            if nb > 0:
                Lb = model(tb, xb)

            # initial loss (square)
            if n0 > 0:
                f0 = model(t0, x0)
                if isinstance(u0, torch.nn.Module):
                    L0 = f0 - u0(torch.ones(n0, 1), x0)
                else:
                    L0 = f0 - u0(x0)

            # combine
            loss = l2norm(Lr) + l2norm(Lb) + l2norm(L0)
            training_loss.append(loss.item())

            # backward step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # print update
        if print_every:
            if (epoch + 1) % print_every == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    # save model
    torch.save(model, model_path)

    # this model is now the initial condition for the next model
    u0 = torch.load(model_path)

    Tlocal += 1


# plot loss
plt.semilogy(training_loss)
plt.xlabel("iteration")
plt.ylabel("loss")
plt.show()
