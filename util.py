import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt


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
        nn.init.uniform_(self.bias, -1 / np.sqrt(output_size) / np.sqrt(output_size))

    def forward(self, x, S):
        return torch.matmul(x, self.Wx) + torch.matmul(S, self.WS) + self.bias


class Slayer(nn.Module):
    def __init__(self, input_size, output_size, bias=True):
        """
        given x, S at the previous layer (Sl), and S at the first layer
        (S1), evaluate Zl, Gl, Rl, H1 -> Sl+1
        """
        super(Slayer, self).__init__()
        self.sigma = nn.Tanh()  # non linearactivation function
        self.zl = DualLinear(input_size, output_size, bias=True)
        self.gl = DualLinear(input_size, output_size, bias=True)
        self.rl = DualLinear(input_size, output_size, bias=True)
        self.hl = DualLinear(input_size, output_size, bias=True)
        # no initialization required because CustomLinear already initializes

    def forward(self, x, Sl, S1=None):
        S1 = Sl if S1 == None else S1
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
        self.sigma = nn.Tanh()  # non linearactivation function
        # layers go here
        self.s1 = nn.Linear(d + 1, M, bias=True)
        self.s2 = Slayer(d + 1, M, bias=True)
        self.s3 = Slayer(d + 1, M, bias=True)
        self.s4 = Slayer(d + 1, M, bias=True)
        self.sfinal = nn.Linear(M, 1, bias=True)
        # initialization
        nn.init.xavier_normal_(self.s1.weight)
        nn.init.xavier_normal_(self.sfinal.weight)

    def forward(self, t, x):
        X = torch.cat([t, x], dim=1)
        S1 = self.sigma(self.s1(X))
        S2 = self.s2(X, S1)
        S3 = self.s2(X, S2, S1)
        S4 = self.s2(X, S3, S1)
        out = self.sfinal(S2)
        return out


class DGMloss(nn.Module):
    def __init__(self):
        super(DGMloss, self).__init__()

    def derivates(self, fx, x):
        return torch.autograd.grad(fx, x, grad_outputs=torch.ones_like(fx), create_graph=True)[0]

    def l2(self, x):

        return torch.sqrt(torch.sum(torch.square(x)))

    def forward(self, f, t, x):
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
        L = self.dfdt_plus_Lf(fx, t, x)
        return self.l2(L) + self.l2(f_ic - g_ic) + self.l2((f_bc - g_bc))

    def dfdt_plus_Lf(self, fx, t, x):
        """
        edit me
        """
        dfdt = self.derivates(fx, t)
        dfdx = self.derivates(fx, x)
        d2fdx2 = self.derivates(dfdx, x)
        return dfdt - (1e-1) * d2fdx2 + (1e-1) * fx * dfdx
        # return dfdt - (1e-2) * d2fdx2 + (1) * a * dfdx


    def initial_condition(self, x):
        """
        edit me
        """
        return 0.5 * torch.tanh(10*(x - 0.5)) + 0.5

    def boundary_condition(self, t):
        """
        edit me
        """
        n = t.shape[0]
        return torch.cat([0 * torch.ones(n, 1), 1 * torch.ones(n, 1)], dim=1)

