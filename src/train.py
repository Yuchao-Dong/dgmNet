import torch
import numpy as np


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

    # 32 x 32 grid for consistent loss reports
    t_grid = torch.Tensor(
        np.repeat(np.linspace(0, 1, 32), 32).reshape(-1, 1)
    ).requires_grad_(True)
    x_grid = torch.Tensor(
        np.tile(np.linspace(0, 1, 32), 32).reshape(-1, 1)
    ).requires_grad_(True)

    # train the model
    training_loss = []
    for epoch in range(num_epochs):
        # uniform random sample
        t_batch = torch.rand(
            (batch_size, 1), requires_grad=True
        )  # solving time set to 1
        x_batch = torch.rand((batch_size, 1), requires_grad=True)  # box size set to 1

        # Forward pass
        loss = criterion(f=model, t=t_batch, x=x_batch)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure uniform loss
        uniform_loss = criterion(f=model, t=t_grid, x=x_grid)

        # save loss
        training_loss.append(uniform_loss.item())
        if num_lr_steps > 0:
            scheduler.step()

        # Print the loss every some epochs
        if print_every:
            if (epoch + 1) % print_every == 0:
                criterion(f=model, t=t_batch, x=x_batch)
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Loss: {uniform_loss.item():.4f}"
                )

    # save model
    if save_path:
        torch.save(model, save_path)
        print("saved model to " + save_path)
    return training_loss
