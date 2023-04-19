import torch
import matplotlib.pyplot as plt
from dgm import DGMnet
from util import u0Presets, random_sample, regular_sample

# define model
model = DGMnet(d = 1, M = 32, L = 2, activation='relu')
# problem setup
a = 1
u0 = u0Presets().square
# hyperparameters
num_epochs = 200
num_iterations = 10
print_every = 10
batch_size = (128, 128, 128)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

def l2norm(x):
    return torch.sqrt(torch.sum(torch.square(x)))

# train the model
training_loss = []
for epoch in range(num_epochs):
    # draw random sample
    t, x = random_sample(n = batch_size[0], subdomain = 'interior')
    tb, xb, xb_mirror = random_sample(n = batch_size[1], subdomain = 'boundary')
    t0, x0 = random_sample(n = batch_size[2], subdomain = 'initial')
    for _ in range(num_iterations):
        # residual loss (dfdt = - a * dfdx)
        fx = model(t, x)
        dfdt = torch.autograd.grad(fx, t, grad_outputs=torch.ones_like(fx), create_graph=True)[0]
        dfdx = torch.autograd.grad(fx, x, grad_outputs=torch.ones_like(fx), create_graph=True)[0]
        Lr = dfdt + a * dfdx

        # boundary loss (periodic)
        fb = model(tb, xb)
        fb_mirror = model(tb, xb_mirror)
        Lb = fb - fb_mirror

        # initial loss (square)
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

# plot loss
plt.semilogy(training_loss)
plt.xlabel('iteration')
plt.ylabel('loss')
plt.show()

# plot attempted solution
t, x = regular_sample(n = 101, d=1, requires_grad = False)
fx = model(t, x).detach()
plt.plot(x[:101].numpy(), u0(x[:101]).numpy(), label='u0')
plt.plot(x[:101].numpy(), fx[:101].numpy(), label='t=0')
plt.plot(x[t == 0.5].numpy(), fx[t == 0.5].numpy(), label='t=0.5')
plt.plot(x[-101:].numpy(), fx[-101:].numpy(), label='t=1')
plt.legend()
plt.show()

