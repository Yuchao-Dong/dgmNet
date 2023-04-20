import torch
import matplotlib.pyplot as plt
from dgm import DGMnet
from util import l2norm, square2d, random_sample

# define model
d = 3  # [x, y, a]
model = DGMnet(d=d, M=50, L=3, activation="relu")
# problem setup
u0 = square2d
b = 1  # vertical velocity
bounds = torch.Tensor([[0, 1], [0, 1], [-1, 1]])  # x, y, and a bounds
# hyperparameters
num_epochs = 100
num_iterations = 10
print_every = 10
nint, nb, n0 = 1024, 1024, 1024  # sample sizes in each subdomain
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
# admin
model_path = "../models/advection2d.pt"

# train the model
bounded_dims = [0, 1]  # only x and y have boundaries
training_loss = []
for epoch in range(num_epochs):
    # draw random sample
    t, x = random_sample(n=nint, d=d, subdomain="interior", bounds=bounds)
    tb, xb, xb_mirror = random_sample(
        n=nb, d=d, subdomain="boundary", bounds=bounds, bounded_dims=bounded_dims
    )
    t0, x0 = random_sample(n=n0, d=d, subdomain="initial", bounds=bounds)

    for _ in range(num_iterations):
        # initialize loss
        Lr = Lb = L0 = torch.Tensor([0])

        # residual loss (dfdt = - a * dfdx)
        if nint > 0:
            fx = model(t, x)
            dfdt = torch.autograd.grad(
                fx, t, grad_outputs=torch.ones_like(fx), create_graph=True
            )[0]
            dfdx = torch.autograd.grad(
                fx, x, grad_outputs=torch.ones_like(fx), create_graph=True
            )[0]
            Lr = dfdt + x[:, 2:3] * dfdx[:, 0:1] + b * dfdx[:, 1:2]

        # boundary loss (periodic)
        if nb > 0:
            fb = model(tb, xb)
            fb_mirror = model(tb, xb_mirror)
            Lb = fb - fb_mirror

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
