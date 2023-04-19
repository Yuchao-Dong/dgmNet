import torch
import numpy as np


class u0Presets():
    def guass(self, x):
        return torch.exp(-100 * torch.square(x - 0.5))
    def sinus(self, x):
        return torch.cos(2 * np.pi * x)
    def square(self, x):
        return torch.where(x > 0.3, torch.where(x < 0.7, 1., 0.), 0.)
    def tanh(self, x):
        return 0.5 * torch.tanh(10 * (x - 0.5)) + 0.5
    def step(self, x, h):
        return torch.where(x > 0.3, 1.0, 0.0) - torch.where(x > 0.7, 1.0, 0.0)
    def composite(self, x):
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

def random_sample(d=1, T = 1, n = 64, subdomain = 'interior', bounds = None):
    """
    args:
        d       # dimensionality
        T       # solving time
        bounds  # bound pairs for each dimension (d x 2)
        n       # number of sample points, must be an integer multiple of d
        subdomain  # interior, boundary, initial
    returns:
        t, x    # random samples of the specified subdomain
    """
    # boundaries for each dimension are 0 or 1 by default
    bounds = torch.Tensor([[0,1]]).repeat(d,1) if bounds is None else bounds
    # time
    if subdomain == 'initial':
        t = torch.zeros(n, 1)
    else:
        t = T * torch.rand(n, 1)
    t.requires_grad_(True)
    # space
    x = (bounds[..., 1] - bounds[..., 0]) * torch.rand(n, d) + bounds[..., 0]
    if subdomain == 'boundary':
        # project each point in x onto a random boundary
        x_mirror = x.clone() # project onto opposite boundary
        rows = torch.arange(n) # each point in x
        cols = torch.randint(0,d,(n,)) # random boundary
        upper_or_lower = torch.randint(0,2,(n,)) # binary choice in each dim
        x[rows, cols] = bounds[cols, upper_or_lower]
        x_mirror[rows, cols] = bounds[cols, 1 - upper_or_lower]
        x.requires_grad_(True)
        x_mirror.requires_grad_(True)
        return t, x, x_mirror
    x.requires_grad_(True)
    return t, x


def regular_sample(n = 64, d=1, T = 1, bounds = None, requires_grad = True):
    """
    args:
        d       # dimensionality
        T       # solving time
        bounds  # bound pairs for each dimension (d x 2)
        n       # number of sample points in each dimension
        requires_grad
    returns:
        t, x    # linearly spaced samples of the entire domain
    """
    # boundaries for each dimension are 0 or 1 by default
    bounds = torch.Tensor([[0,1]]).repeat(d,1) if bounds is None else bounds
    # time, length n^(d + 1)
    t = T * torch.linspace(0,1,n).repeat_interleave(n ** d).reshape(-1,1)
    # space, combinations of every point
    X = torch.linspace(0,1,n).reshape(-1,1).repeat(1,d)
    # apply boundaries
    X = (bounds[..., 1] - bounds[..., 0]) * X + bounds[..., 0]
    # generate combinations of the unique values of X
    combinations = torch.cartesian_prod(*([X[..., i] for i in range(d)])) if d > 1 else X
    # repeat for each value in time
    x = combinations.repeat(n, 1)
    # finish
    if requires_grad:
        t.requires_grad_(True)
        x.requires_grad_(True)
    return t, x
