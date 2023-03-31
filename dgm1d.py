import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from util import DGM

# Define the dataset
def g(x):
    return 0.5 * np.tanh(10*(x - 0.5)) + 0.5
x = np.random.rand(1000)
t = np.random.rand(1000)
X = np.array([t, x]).T
y = g(x)

X = torch.Tensor(X)
y = torch.Tensor(y.reshape(-1, 1))

# Define the batch size
batch_size = 80

# dataloader time
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define the neural network model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        M = 100
        fc1 = nn.Linear(2, M, bias = True)
        fc2 = nn.Linear(M, M, bias = True)
        fc3 = nn.Linear(M, 1, bias = True)
        activation = nn.Tanh()
        self.model = nn.Sequential(fc1, activation, fc2, activation, fc3)

    def forward(self, X):
        return self.model(X)

class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, model, X, target):
        prediction = model(X)
        target = torch.Tensor(g(X[..., 1].numpy()).reshape(-1, 1))
        return torch.sqrt(torch.sum(torch.square(prediction - target)))


# Instantiate the model and define the loss function
model = DGM(1, 50)
criterion = CustomLoss()

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)

# Train the model
training_loss = []
epochs = 100
for epoch in range(epochs):
    for n, data in enumerate(dataloader):
        X_batch, y_batch = data

        # Forward pass
        loss = criterion(model, X_batch, y_batch)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # save loss
        training_loss.append(loss.item())
    # Print the loss every 100 epochs
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Plot the predicted vs actual output
plt.plot(training_loss)
plt.show()

x = np.linspace(0, 1, 1000)
t = np.linspace(0, 1, 1000)
X = torch.Tensor(np.array([t, x]).T)
plt.plot(x, g(x))
plt.plot(x, model(X).detach().numpy())
plt.show()