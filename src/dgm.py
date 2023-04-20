import torch.nn as nn
import torch


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
        self, d: int = 1, M: int = 50, L: int = 1, activation="relu"
    ):
        """
        args:
            d:  number of spatial input dimensions
            M:  nodes per layer
            L:  number of lstm layers 
            activation: relu or tanh
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
        for _ in range(L):
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
