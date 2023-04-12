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
    batch_size=64,
    lr=0.01,
    num_lr_steps=0,
    gamma=None,
    u0_preset="square",
    print_every=10,
    save_path=None,
):
    # set up training tools
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if num_lr_steps > 0:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=int(num_epochs / (num_lr_steps + 1)), gamma=0.5
        )

    # train the model
    training_loss = []
    for epoch in range(num_epochs):
        # new random training data every epoch
        x_batch = torch.rand((batch_size, 1), requires_grad=True)  # box size set to 1
        t_batch = torch.rand(
            (batch_size, 1), requires_grad=True
        )  # solving time set to 1

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

    # save model
    if save_path:
        torch.save(model, save_path)
        print("saved model to " + save_path)
    return training_loss
