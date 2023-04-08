import torch.nn as nn
import torch
from util import u0_presets


class HighwayLayer(nn.Module):
    def __init__(self, input_size, output_size, sigma=nn.ReLU, bias=True):
        super(HighwayLayer, self).__init__()
        # activation function
        self.sigma = sigma
        # four layers Z, G, R, H, each with two sets of weights
        # weight tensors for x (input size)
        self.U_z = nn.Parameter(
            nn.init.xavier_normal_(torch.empty(input_size, output_size)),
            requires_grad=True,
        )
        self.U_g = nn.Parameter(
            nn.init.xavier_normal_(torch.empty(input_size, output_size)),
            requires_grad=True,
        )
        self.U_r = nn.Parameter(
            nn.init.xavier_normal_(torch.empty(input_size, output_size)),
            requires_grad=True,
        )
        self.U_h = nn.Parameter(
            nn.init.xavier_normal_(torch.empty(input_size, output_size)),
            requires_grad=True,
        )
        # weight tensors for S (output size)
        self.W_z = nn.Parameter(
            nn.init.xavier_normal_(torch.empty(output_size, output_size)),
            requires_grad=True,
        )
        self.W_g = nn.Parameter(
            nn.init.xavier_normal_(torch.empty(output_size, output_size)),
            requires_grad=True,
        )
        self.W_r = nn.Parameter(
            nn.init.xavier_normal_(torch.empty(output_size, output_size)),
            requires_grad=True,
        )
        self.W_h = nn.Parameter(
            nn.init.xavier_normal_(torch.empty(output_size, output_size)),
            requires_grad=True,
        )
        # bias
        self.b_z = None
        self.b_g = None
        self.b_r = None
        self.b_h = None
        if bias:
            self.b_z = nn.Parameter(torch.zeros(output_size), requires_grad=True)
            self.b_g = nn.Parameter(torch.zeros(output_size), requires_grad=True)
            self.b_r = nn.Parameter(torch.zeros(output_size), requires_grad=True)
            self.b_h = nn.Parameter(torch.zeros(output_size), requires_grad=True)

    def forward(self, x, Sprev):
        Z = self.sigma(
            torch.matmul(x, self.U_z) + torch.matmul(Sprev, self.W_z) + self.b_z
        )
        G = self.sigma(
            torch.matmul(x, self.U_g) + torch.matmul(Sprev, self.W_g) + self.b_g
        )
        R = self.sigma(
            torch.matmul(x, self.U_r) + torch.matmul(Sprev, self.W_r) + self.b_r
        )
        H = self.sigma(
            torch.matmul(x, self.U_h) + torch.matmul(Sprev * R, self.W_h) + self.b_h
        )
        Snext = (1 - G) * H + Z * Sprev
        return Snext


class DGMnet(nn.Module):
    def __init__(
        self, d: int = 1, M: int = 50, num_highway_layers: int = 1, activation="relu"
    ):
        """
        args:
            d:   number of spatial input dimensions
            M:   nodes per layer
            number of layers
            activation function type
        returns:
            PDE solving model
        """
        super(DGMnet, self).__init__()
        if activation == "relu":
            self.sigma = nn.ReLU()
        if activation == "tanh":
            self.sigma = nn.Tanh()
        # first and last layers
        self.first_layer = nn.Linear(d + 1, M, bias=True)
        self.final_layer = nn.Linear(M, 1, bias=True)
        nn.init.xavier_normal_(self.first_layer.weight)
        nn.init.xavier_normal_(self.final_layer.weight)
        # highway layers in between
        self.highway_layers = nn.ModuleList()
        for _ in range(num_highway_layers):
            self.highway_layers.append(
                HighwayLayer(input_size=d + 1, output_size=M, sigma=self.sigma)
            )

    def forward(self, t, x):
        X = torch.cat([t, x], dim=1)
        S = self.sigma(self.first_layer(X))
        for layer in self.highway_layers:
            S = layer(x=X, Sprev=S)
        S = self.final_layer(S)
        return S


class DGMloss(nn.Module):
    def __init__(self, u0_preset="composite"):
        super(DGMloss, self).__init__()
        self.u0_preset = u0_preset

    def differentiate(self, fx, x):
        return torch.autograd.grad(
            fx, x, grad_outputs=torch.ones_like(fx), create_graph=True
        )[0]

    def norm(self, x):
        return torch.sqrt(torch.sum(torch.square(x)))

    def forward(self, f, t, x):
        # return self.initial_condition_loss(f, x) + self.boundary_condition_loss(f, t, x)
        return (
            self.boundary_condition_loss(f, t, x)
            + self.initial_condition_loss(f, x)
            + self.differential_loss(f, t, x)
        )

    def boundary_condition_loss(self, f, t, x):
        """
        edit me
        f(t, x_bc) - g(t, x)
        """
        n = len(t)
        f_left = f(t, torch.zeros(n, 1))
        f_right = f(t, torch.ones(n, 1))
        return self.norm(f_right - f_left)

    def initial_condition_loss(self, f, x):
        """
        edit me
        f(0, x) - u_0(x)
        """
        f0 = f(torch.zeros(len(x), 1), x)
        return self.norm(f0 - self.u0(x))

    def u0(self, x):
        return u0_presets(x, preset=self.u0_preset)

    def differential_loss(self, f, t, x):
        """
        edit me
        dfdx + L(f) = 0
        """
        fx = f(t, x)
        dfdt = self.differentiate(fx, t)
        dfdx = self.differentiate(fx, x)
        return self.norm(dfdt + 1 * dfdx)
