from torch.utils.data import TensorDataset, DataLoader
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from copy import deepcopy
from dgm import DGMnet, DGMloss
from util import train


def train(
    model,
    criterion,
    num_epochs=10,
    batch_size=128,
    num_batches=1,
    lr=0.01,
    num_lr_steps=0,
    gamma=None,
    u0_preset="square",
    print_every=10,
):
    # set up training tools
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if num_lr_steps > 0:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=int(num_epochs / (num_lr_steps + 1)), gamma=0.5
        )

    # generate training data
    n_data = batch_size * num_batches
    x = torch.rand(n_data, 1)  # box size set to 1
    t = torch.rand(n_data, 1)  # solving time set to 1
    x.requires_grad = True
    t.requires_grad = True
    dataset = TensorDataset(t, x)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # train the model
    training_loss = []
    for epoch in range(num_epochs):
        batch_loss = 0
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
        if num_lr_steps > 0:
            scheduler.step()

        # Print the loss every some epochs
        if print_every:
            if (epoch + 1) % print_every == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    return training_loss
