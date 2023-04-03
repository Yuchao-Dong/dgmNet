import torch.nn as nn
import torch
import numpy as np


class DualLinear(nn.Module):
    def __init__(self, input_size, output_size, bias=True):
        """
        linear layer with two inputs x and S of the same size
        """
        super(DualLinear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.Wx = nn.Parameter(
            torch.Tensor(input_size, output_size), requires_grad=True
        )
        self.WS = nn.Parameter(
            torch.Tensor(output_size, output_size), requires_grad=True
        )
        self.bias = (
            nn.Parameter(torch.Tensor(output_size), requires_grad=True)
            if bias
            else None
        )
        nn.init.xavier_normal_(self.Wx)
        nn.init.xavier_normal_(self.WS)
        nn.init.uniform_(self.bias, -1 / np.sqrt(output_size), 1 / np.sqrt(output_size))

    def forward(self, x, S):
        return torch.matmul(x, self.Wx) + torch.matmul(S, self.WS) + self.bias


class Slayer(nn.Module):
    def __init__(self, input_size, output_size, bias=True):
        """
        given x, S at the previous layer (Sl), and S at the first layer
        (S1), evaluate Zl, Gl, Rl, H1 -> Sl+1
        """
        super(Slayer, self).__init__()
        self.sigma = nn.ReLU()  # non linearactivation function
        self.zl = DualLinear(input_size, output_size, bias=True)
        self.gl = DualLinear(input_size, output_size, bias=True)
        self.rl = DualLinear(input_size, output_size, bias=True)
        self.hl = DualLinear(input_size, output_size, bias=True)
        # no initialization required because CustomLinear already initializes

    def forward(self, x, Sl, S1=None):
        S1 = Sl if S1 is None else S1
        Zl = self.sigma(self.zl(x, Sl))
        Gl = self.sigma(self.gl(x, S1))
        Rl = self.sigma(self.rl(x, Sl))
        Hl = self.sigma(self.hl(x, Sl * Rl))
        Sl_plus_1 = (1 - Gl) * Hl + Zl * Sl
        return Sl_plus_1


class DGM(nn.Module):
    """
    PDE solving network following Sirignano and Spiliopoulos 2018
    """

    def __init__(self, d: int = 1, M: int = 50):
        super(DGM, self).__init__()
        self.sigma = nn.ReLU()  # non linearactivation function
        # 2 multistage layers preceeded and followed by a linear layer
        self.s1 = nn.Linear(d + 1, M, bias=True)
        self.s2 = Slayer(d + 1, M, bias=True)
        self.s3 = Slayer(d + 1, M, bias=True)
        self.sfinal = nn.Linear(M, 1, bias=True)
        # initialization
        nn.init.xavier_normal_(self.s1.weight)
        nn.init.xavier_normal_(self.sfinal.weight)

    def forward(self, t, x):
        X = torch.cat([t, x], dim=1)
        S1 = self.sigma(self.s1(X))
        S2 = self.s2(X, S1)
        S3 = self.s2(X, S2, S1)
        out = self.sfinal(S3)
        return out


class DGMloss(nn.Module):
    def __init__(self):
        super(DGMloss, self).__init__()

    def derivates(self, fx, x):
        return torch.autograd.grad(
            fx, x, grad_outputs=torch.ones_like(fx), create_graph=True
        )[0]

    def l2(self, x):
        return torch.sqrt(torch.sum(torch.square(x)))

    def burgers(self, fx, t, x):
        """
        burgers equation
        """
        dfdt = self.derivates(fx, t)
        dfdx = self.derivates(fx, x)
        d2fdx2 = self.derivates(dfdx, x)
        return dfdt - (1e-1) * d2fdx2 + (1) * fx * dfdx

    def advection(self, fx, t, x):
        """
        advection without diffusion with a speed of 0.5
        """
        dfdt = self.derivates(fx, t)
        dfdx = self.derivates(fx, x)
        return dfdt + (0.5) * dfdx

    def forward(self, f, t, x):
        """
        edit the differential operator term to solve a different equation
        """
        n = x.shape[0]
        f_ic = f(torch.zeros(n, 1), x)
        f_bc_left = f(t, torch.zeros(n, 1))
        f_bc_right = f(t, torch.ones(n, 1))
        f_bc = torch.cat([f_bc_left, f_bc_right], dim=1)
        fx = f(t, x)
        # ground truth
        g_ic = self.initial_condition(x)
        g_bc = self.boundary_condition(t)
        # differential operator
        L = self.advection(fx, t, x)
        return self.l2(L) + self.l2(f_ic - g_ic) + self.l2((f_bc - g_bc))

    def initial_condition(self, x):
        """
        select initial condition by modifying preset string
        """
        preset = "square"
        if preset == "guass":
            return torch.exp(-200 * torch.square(x - 0.25))
        if preset == "tanh":
            return 0.5 * torch.tanh(10 * (x - 0.5)) + 0.5
        if preset == "square":
            return torch.where(x > 0.1, 1, 0) - torch.where(x > 0.4, 1, 0)
        if preset == "composite":
            u0 = torch.zeros(x.shape)
            zeros = torch.zeros(x.shape)
            ones = torch.ones(x.shape)
            y = 2 * x
            guass = (
                1
                / 6
                * (
                    torch.exp(
                        -np.log(2) / 36 / 0.0025**2 * torch.square(y - 0.0025 - 0.15)
                    )
                    + torch.exp(
                        -np.log(2) / 36 / 0.0025**2 * torch.square(y + 0.0025 - 0.15)
                    )
                    + 4
                    * torch.exp(-np.log(2) / 36 / 0.0025**2 * torch.square(y - 0.15))
                )
            )
            square = 0.75
            cone = 1 - torch.abs(20 * (y - 0.55))
            sinus = (
                1
                / 6
                * (
                    torch.sqrt(
                        torch.maximum(1 - (20 * torch.square(y - 0.75 - 0.0025)), zeros)
                    )
                    + torch.sqrt(
                        torch.maximum(1 - torch.square(20 * (y - 0.75 + 0.0025)), zeros)
                    )
                    + 4
                    * torch.sqrt(
                        torch.maximum(1 - torch.square(20 * (y - 0.75)), zeros)
                    )
                )
            )
            u0 = torch.where(
                torch.logical_and(torch.gt(y, 0.1 * ones), torch.gt(0.2 * ones, y)),
                guass,
                u0,
            )
            u0 = torch.where(
                torch.logical_and(torch.gt(y, 0.3 * ones), torch.gt(0.4 * ones, y)),
                square,
                u0,
            )
            u0 = torch.where(
                torch.logical_and(torch.gt(y, 0.5 * ones), torch.gt(0.6 * ones, y)),
                cone,
                u0,
            )
            u0 = torch.where(
                torch.logical_and(torch.gt(y, 0.7 * ones), torch.gt(0.8 * ones, y)),
                sinus,
                u0,
            )
            return u0

    def boundary_condition(self, t):
        """
        set to the boundaries of the initial profile
        """
        n = t.shape[0]
        left_boundary = self.initial_condition(torch.Tensor([0]))
        right_boundary = self.initial_condition(torch.Tensor([1]))
        return torch.cat(
            [left_boundary * torch.ones(n, 1), right_boundary * torch.ones(n, 1)], dim=1
        )
