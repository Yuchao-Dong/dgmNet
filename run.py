from numpy.random import rand
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ExponentialLR
import torch
from config import *
from dgm import DGM, DGMloss

# training data
# generate training data
x = torch.Tensor(rand(n).reshape(-1, 1))  # box size set to 1
t = torch.Tensor(rand(n).reshape(-1, 1))  # solving time set to 1
x.requires_grad = True
t.requires_grad = True
dataset = TensorDataset(t, x)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# define model, loss function, and optimizer
model = DGM(d=1, M=M)
criterion = DGMloss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=initial_learning_rate, weight_decay=0.001
)
scheduler = ExponentialLR(optimizer, gamma=learning_rate_decay)

# Train the model using the three datasets
training_loss = []
best_loss = 1e6
for epoch in range(num_epochs):
    for n, data in enumerate(dataloader):
        t_batch, x_batch = data

        # Forward pass
        loss = criterion(model, t_batch, x_batch)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # save loss
        training_loss.append(loss.item())

    # Print the loss every 100 epochs
    if (epoch + 1) % print_every == 0:
        print(
            f"Epoch [{epoch+1}/{num_epochs}], lr: {scheduler.get_last_lr()[0]:.0e}, Loss: {loss.item():.4f}"
        )
    scheduler.step()

# plot error
plt.loglog(training_loss)
plt.xlabel("iteration")
plt.ylabel("loss")
plt.show()

# plot attmepted solution
x = torch.Tensor(np.linspace(0, 1, 1000).reshape(-1, 1))
t = torch.Tensor(np.ones(1000).reshape(-1, 1))
plt.plot(
    x.numpy(),
    criterion.initial_condition(x).detach().numpy(),
    label="ground truth t = 0",
)
plt.plot(x.numpy(), model(0 * t, x).detach().numpy(), label="predicted t = 0")
plt.plot(x.numpy(), model(0.1 * t, x).detach().numpy(), label="predicted t = 0.10")
plt.plot(x.numpy(), model(0.25 * t, x).detach().numpy(), label="predicted t = 0.25")
plt.plot(x.numpy(), model(0.5 * t, x).detach().numpy(), label="predicted t = 0.5")
plt.plot(x.numpy(), model(1 * t, x).detach().numpy(), label="predicted t = 1")
plt.legend()
plt.show()
