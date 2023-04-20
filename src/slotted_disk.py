import torch
import numpy as np
import matplotlib.pyplot as plt
from dgm import DGMnet
from util import l2norm, slotted_disk, random_sample

# define model
d = 2  # [x, y, a]
model = DGMnet(d=d, M=64, L=2, activation="relu")
# problem setup
u0 = slotted_disk
bounds = torch.Tensor([[-1.25, 1.25], [-1.25, 1.25]])  # x, y, and a bounds
T = 1
velocity_scale = np.pi
# hyperparameters
num_epochs = 100
num_iterations = 10
print_every = 10
nint, nb, n0 = 1024, 1024, 1024  # sample sizes in each subdomain
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
# admin
model_path = "../models/slotted_disk.pt"

# train the model
bounded_dims = [0, 1]  # only x and y have boundaries
training_loss = []
for epoch in range(num_epochs):
    # draw random sample
    t, x = random_sample(n=nint, d=d, subdomain="interior", bounds=bounds, T=T)
    tb, xb, xb_mirror = random_sample(
        n=nb, d=d, subdomain="boundary", bounds=bounds, bounded_dims=bounded_dims, T=T
    )
    t0, x0 = random_sample(n=n0, d=d, subdomain="initial", bounds=bounds, T=T)

    for _ in range(num_iterations):
        # initialize loss
        Lr = Lb = L0 = torch.Tensor([0])

        # residual loss (dfdt = - a * dfdx)
        if nint > 0:
            fx = model(t, x)
            a = velocity_scale * -x[:, 1:2] # -y
            b = velocity_scale * x[:, 0:1] # +x
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
            L0 = f0 - u0(x0)

        # combine
        loss = l2norm(Lr) + l2norm(Lb) + l2norm(L0)
        training_loss.append(loss.item())

        # backward step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()

    # print update
    if print_every:
        if (epoch + 1) % print_every == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# save model
if model_path:
    torch.save(model, model_path)

# plot loss
plt.semilogy(training_loss)
plt.xlabel("iteration")
plt.ylabel("loss")
plt.show()
