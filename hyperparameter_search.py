from torch.utils.data import TensorDataset, DataLoader
import torch
import numpy as np
import random
from dgm import DGMnet, DGMloss
from train import train

# define parameter space
num_iterations = 5
hyperparameters = {
    "M": [10, 20, 30, 50],
    "num_highway_layers": [0, 1, 2, 3],
    "num_batches": [1, 2, 4, 6, 8],
    "lr": [0.02, 0.01, 0.001],
    "num_lr_steps": [0, 1, 2, 4],
    "gamma": [0.5, 0.25, 0.1],
}


def reward(loss_list):
    last_loss_reward = 1 / (1 + loss_list[-1])
    np_loss = np.array(training_loss)[100:]
    mean_loss_reward = 1 / (1 + np.mean(np_loss))
    smoothness_reward = 1 / (1 + np.mean(np.abs(np_loss[1:] - np_loss[:1])))
    score = last_loss_reward + mean_loss_reward
    return score


best_score = 0
best_hyperparameters = None
parameters = {}
for i in range(num_iterations):
    # select parameters
    for key, item in hyperparameters.items():
        parameters[key] = random.choice(item)
    print(f"({i + 1} / {num_iterations}) - - - - -")
    print("parameters:")
    print(parameters)

    # define model
    model = DGMnet(
        d=1, M=parameters["M"], num_highway_layers=parameters["num_highway_layers"]
    )
    criterion = DGMloss(u0_preset="square")

    # train model
    training_loss = train(
        model,
        criterion,
        print_every=False,
        num_epochs=200,
        batch_size=128,
        num_batches=parameters["num_batches"],
        lr=parameters["lr"],
        num_lr_steps=parameters["num_lr_steps"],
        gamma=parameters["gamma"],
    )

    # evaluate model
    score = reward(training_loss)
    print(f"score: {score:.3f}")
    if score > best_score:
        best_score = score
        best_hyperparameters = parameters

print("\n! ! ! ! !")
print("best hyper parameters:")
print(best_hyperparameters)
print(f"best score: {best_score:.3f}")
