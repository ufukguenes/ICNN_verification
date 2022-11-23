import torch
from torch import nn


class Flatten(nn.Module):

    def forward(self, x):
        x.size(0)
        return x.view(x.size(0), -1)


class SequentialNN(nn.Sequential):
    def __init__(self, layer_widths):
        super(SequentialNN, self).__init__()

        d_in = layer_widths[0]
        for lw in layer_widths[1:len(layer_widths)-1]:
            self.append(nn.Linear(d_in, lw, dtype=torch.float64))
            self.append(nn.ReLU())
            d_in = lw

        self.append(nn.Linear(d_in, layer_widths[-1], dtype=torch.float64))

        for tens in self.parameters():
            torch.nn.init.uniform_(tens, -1, 1)
            tens = tens


class ICNN(nn.Module):

    def __init__(self, layer_widths, force_positive_init=False):
        """
    layer_widths - ([int]) list of layer widths **including** input and output dim
    """
        super(ICNN, self).__init__()

        self.ws = nn.ParameterList([])  # positive weights for propagation
        self.us = nn.ParameterList([])  # weights tied to inputs

        self.ws.append(nn.Linear(layer_widths[0], layer_widths[1], bias=True, dtype=torch.float64))
        d_in = layer_widths[1]

        for lw in layer_widths[2:]:
            w = nn.Linear(d_in, lw, dtype=torch.float64)

            with torch.no_grad():
                if force_positive_init:
                    for p in w.parameters():
                        if len(p.size()) > 1:
                            p[:] = torch.maximum(torch.Tensor([0]), p)

            d_in = lw
            u = nn.Linear(layer_widths[0], lw, bias=False, dtype=torch.float64)

            self.ws.append(w)
            self.us.append(u)

    def forward(self, x):
        x = Flatten()(x)
        x1 = nn.ReLU()(self.ws[0](x))  # first layer is only W
        for w, u in zip(self.ws[1:-1], self.us[:-1]):
            a = w(x1)
            b = u(x)
            x1 = nn.ReLU()(a + b)

        x1 = self.ws[-1](x1) + self.us[-1](x)  # no ReLU in last layer"""

        return x1