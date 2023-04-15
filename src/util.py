import torch
import numpy as np


def u0_presets(x, preset="square"):
    """
    args:
        x   tensor
        preset string
    returns:
        input function u0 evaluated at x
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


def importance_resample(t, x, importance=None, noise_std=None, shuffle_probability=0):
    """
    args:
        t, x        torch tensor data, x of dimension d
        importance  torch tensor importance values associated with each (t, x)
    returns:
        new torch tensor t, x sample based on the importance of the given values
        as well as added noise
    """
    # if no importance is provided, return a random sample
    shuffle = np.random.rand() <= shuffle_probability if shuffle_probability else False
    if importance is None or shuffle:
        t_sample = torch.rand(size=t.shape).requires_grad_(True)
        x_sample = torch.rand(size=x.shape).requires_grad_(True)
        return t_sample, x_sample
    # convert data to numpy
    n = len(t)
    d = x.shape[1]
    t_np, x_np = t.clone().detach().numpy(), x.clone().detach().numpy()
    importance_np = importance.clone().detach().numpy()
    # normalize
    total_importance = np.sum(importance_np)
    importance_np /= total_importance
    # add noise depending on the importance
    noise_std = 1 / np.sqrt(n) * np.exp(-((n * importance_np) ** 2))
    t_noise = np.random.normal(loc=0, scale=noise_std, size=(n, 1))
    x_noise = np.random.normal(loc=0, scale=noise_std, size=(n, d))
    t_np += t_noise
    x_np += x_noise
    # convert to torch tensor
    t_resample = torch.Tensor(t_np).requires_grad_(True)
    x_resample = torch.Tensor(x_np).requires_grad_(True)
    return t_resample, x_resample
