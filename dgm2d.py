from numpy.random import rand, randint
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import torch
from util import DGM, DGMloss

def initial_condition(x):
    return 0.5 * np.tanh(10*(x - 0.5)) + 0.5

def boundary_condition(x):
    return np.where(x == 0, 0, np.where(x == 1, 1, np.nan))

# hyperparameters
num_epochs = 50 
batch_size = 50
learning_rate = 0.02
print_every = 10

# training data
n = 1000
# generate random numpy data
x = torch.Tensor(rand(n).reshape(-1, 1))
t = torch.Tensor(rand(n).reshape(-1, 1))
x.requires_grad = True
t.requires_grad = True
dataset = TensorDataset(t, x)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# define model, loss function, and optimizer
model = DGM(d = 1, M = 50)
criterion = DGMloss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)

# Train the model using the three datasets
training_loss = []
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
    if (epoch+1) % print_every == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# plot error
plt.plot(training_loss)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

# plot attmepted solution
x = torch.Tensor(np.linspace(0, 1, 100).reshape(-1,1))
t = torch.Tensor(np.ones(100).reshape(-1,1))
plt.plot(x.numpy(), criterion.initial_condition(x).detach().numpy(), label='ground truth t = 0')
plt.plot(x.numpy(), model(0 * t, x).detach().numpy(), label='predicted t = 0')
plt.plot(x.numpy(), model(0.1 * t, x).detach().numpy(), label='predicted t = 0.10')
plt.plot(x.numpy(), model(0.25 * t, x).detach().numpy(), label='predicted t = 0.25')
plt.plot(x.numpy(), model(0.5 * t, x).detach().numpy(), label='predicted t = 0.5')
plt.plot(x.numpy(), model(1 * t, x).detach().numpy(), label='predicted t = 1')
plt.legend()
plt.show()