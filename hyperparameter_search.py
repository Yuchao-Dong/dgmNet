from torch.utils.data import TensorDataset, DataLoader
import torch
import numpy as np
import random
from dgm import DGMnet, DGMloss
from train import train

# define parameter space
num_iterations = 10
hyperparameters = {
    "M": [10, 20, 30, 40, 50],
    "num_highway_layers": [0, 1, 2, 3],
    "num_epochs": [5000, 10000],
    "lr": [0.02, 0.01, 0.001],
    "num_lr_steps": [0, 1, 2, 3],
    "gamma": [0.5, 0.25, 0.1],
}


def find_score(loss_list):
    np_loss = np.array(training_loss)
    score = np_loss[-1] + np.mean(np_loss[-20:])
    return score


best_score = float("inf")
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
        batch_size=64,
        num_epochs=parameters["num_epochs"],
        lr=parameters["lr"],
        num_lr_steps=parameters["num_lr_steps"],
        gamma=parameters["gamma"],
    )

    # evaluate model
    score = find_score(training_loss)
    print(f"score: {score:.3f}")
    if score < best_score:
        best_score = score
        best_hyperparameters = parameters

print("\n! ! ! ! !")
print("best hyper parameters:")
print(best_hyperparameters)
print(f"best score: {best_score:.3f}")
