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
