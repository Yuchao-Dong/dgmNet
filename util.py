from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ExponentialLR
import torch
import numpy as np


def train(
    model,
    criterion,
    data_size=10000,
    batch_size=1000,
    num_epochs=10,
    initial_learning_rate=0.001,
    gamma=1,
):
    """
    args:
        hyperparameters
    returns:
        list of trianing loss history
    modifies:
        model
    """
    # generate training data
    x = torch.rand(data_size, 1)  # box size set to 1
    t = torch.rand(data_size, 1)  # solving time set to 1
    x.requires_grad = True
    t.requires_grad = True
    dataset = TensorDataset(t, x)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # define loss and optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = ExponentialLR(optimizer, gamma=gamma)

    # train the model
    training_loss = []
    for epoch in range(num_epochs):
        for n, data in enumerate(dataloader):
            t_batch, x_batch = data

            # Forward pass
            loss = criterion(f=model, t=t_batch, x=x_batch)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # save loss
            training_loss.append(loss.item())

        # Print the loss every some epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
        scheduler.step()
    return training_loss


def u0_presets(x, preset="square"):
    """
    args:
        x   tensor
        preset string
    returns:
        input function u0 evaluates at x
    """
    if preset == "guass":
        return torch.exp(-100 * torch.square(x - 0.5))
    if preset == "tanh":
        return 0.5 * torch.tanh(10 * (x - 0.5)) + 0.5
    if preset == "square":
        return torch.where(x > 0.3, 1.0, 0.0) - torch.where(x > 0.7, 1.0, 0.0)
    if preset == "composite":
        u0 = torch.zeros(x.shape)
        zeros = torch.zeros(x.shape)
        ones = torch.ones(x.shape)
        guass = (
            1
            / 6
            * (
                torch.exp(
                    -np.log(2) / 36 / 0.0025**2 * torch.square(x - 0.0025 - 0.15)
                )
                + torch.exp(
                    -np.log(2) / 36 / 0.0025**2 * torch.square(x + 0.0025 - 0.15)
                )
                + 4 * torch.exp(-np.log(2) / 36 / 0.0025**2 * torch.square(x - 0.15))
            )
        )
        square = 0.75
        cone = 1 - torch.abs(20 * (x - 0.55))
        sinus = (
            1
            / 6
            * (
                torch.sqrt(
                    torch.maximum(1 - (20 * torch.square(x - 0.75 - 0.0025)), zeros)
                )
                + torch.sqrt(
                    torch.maximum(1 - torch.square(20 * (x - 0.75 + 0.0025)), zeros)
                )
                + 4
                * torch.sqrt(torch.maximum(1 - torch.square(20 * (x - 0.75)), zeros))
            )
        )
        u0 = torch.where(
            torch.logical_and(torch.gt(x, 0.1 * ones), torch.gt(0.2 * ones, x)),
            guass,
            u0,
        )
        u0 = torch.where(
            torch.logical_and(torch.gt(x, 0.3 * ones), torch.gt(0.4 * ones, x)),
            square,
            u0,
        )
        u0 = torch.where(
            torch.logical_and(torch.gt(x, 0.5 * ones), torch.gt(0.6 * ones, x)),
            cone,
            u0,
        )
        u0 = torch.where(
            torch.logical_and(torch.gt(x, 0.7 * ones), torch.gt(0.8 * ones, x)),
            sinus,
            u0,
        )
        return u0
