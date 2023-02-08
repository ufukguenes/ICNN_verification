import torch
from torch import nn
from functorch import vmap

class Flatten(nn.Module):

    def forward(self, x):
        x.size(0)
        return x.view(x.size(0), -1)


class SequentialNN(nn.Sequential):
    def __init__(self, layer_widths):
        super(SequentialNN, self).__init__()
        self.layer_widths = layer_widths
        d_in = layer_widths[0]
        for lw in layer_widths[1:len(layer_widths) - 1]:
            self.append(nn.Linear(d_in, lw, dtype=torch.float64))
            self.append(nn.ReLU())
            d_in = lw

        self.append(nn.Linear(d_in, layer_widths[-1], dtype=torch.float64))

    def forward(self, input):
        x = Flatten()(input)
        return super().forward(x)


class ICNN(nn.Module):

    def __init__(self, layer_widths, force_positive_init=False):
        """
    layer_widths - ([int]) list of layer widths **including** input and output dim
    """
        super(ICNN, self).__init__()

        self.ws = nn.ParameterList([])  # positive weights for propagation
        self.us = nn.ParameterList([])  # weights tied to inputs
        self.layer_widths = layer_widths
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
        # x1 = nn.Softplus()(self.ws[0](x))  # first layer is only W
        # x1 = nn.Tanh()(self.ws[0](x))  # first layer is only W
        for w, u in zip(self.ws[1:-1], self.us[:-1]):
            a = w(x1)
            b = u(x)
            x1 = nn.ReLU()(a + b)
            # x1 = nn.Softplus()(a + b)
            # x1 = nn.Tanh()(a + b)

        x1 = self.ws[-1](x1) + self.us[-1](x)  # no ReLU in last layer"""
        return x1


class ICNN_Logical(nn.Module):


    def __init__(self, layer_widths, force_positive_init=False):
        """
        layer_widths - ([int]) list of layer widths **including** input and output dim
        """
        super(ICNN_Logical, self).__init__()

        self.training_setup = True

        self.ws = nn.ParameterList([])  # positive weights for propagation
        self.us = nn.ParameterList([])  # weights tied to inputs
        self.ls = []
        self.layer_widths = layer_widths
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

        self.ls.append(nn.Linear(layer_widths[0], 2 * layer_widths[0], bias=True, dtype=torch.float64))
        #self.ls.append(nn.Linear(2, 4, bias=False, dtype=torch.float64))
        #self.ls.append(nn.Linear(4, 3, bias=False, dtype=torch.float64))
        self.ls.append(nn.Linear(2, 3, bias=False, dtype=torch.float64))

        with torch.no_grad():
            #l1 = list(self.ls[1].parameters())
            #l2 = list(self.ls[2].parameters())

            # diese architektur ist für alle ICNNs gleich da alle genau 2 ausgaben haben eine vom eigentlichen ICNN
            # und eine von den Box Bounds
            #l1[0].data = torch.tensor([[1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=torch.float64)
            # todo das kann man auch in einem affine layer zusammen fassen, ist schneller aber auch unübersichtlicher
            #l2[0].data = torch.tensor([[1, 0, 0, -1], [0, 1, 0, -1], [0, 0, 1, -1]], dtype=torch.float64)
            # Vereinfachung von der zwei Layer Variante
            l1 = list(self.ls[1].parameters())
            l1[0].data = torch.tensor([[2, 2], [2, 0], [0, 2]], dtype=torch.float64)

    def forward(self, x):
        x = Flatten()(x)
        x1 = nn.ReLU()(self.ws[0](x))  # first layer is only W
        for w, u in zip(self.ws[1:-1], self.us[:-1]):
            a = w(x1)
            b = u(x)
            x1 = nn.ReLU()(a + b)

        icnn_out = self.ws[-1](x1) + self.us[-1](x)  # no ReLU in last layer"""

        # something like and
        box_out = self.ls[0](x)
        box_out = torch.max(box_out, dim=1, keepdim=True)[0]
        # box_out = torch.logsumexp(box_out, dim=1) # die sind alle unnötig, bei box bounds brauch ich nicht zu approximieren
        # box_out = boltzmann_op(box_out)
        # box_out = mellowmax(box_out)
        # box_out = smu_4(box_out)

        x_in = torch.cat([icnn_out, box_out], dim=1)


        if self.training_setup:
            #out = torch.max(x_in, dim=1)[0]
            #out = torch.logsumexp(x_in, dim=1)
            #out = boltzmann_op(x_in)
            #out = mellowmax(x_in)
            #out = smu_2(x_in)
            # out = smu_binary(icnn_out, box_out)

            x_in = self.ls[1](x_in)
            out = torch.max(x_in, dim=1)[0]

            """x_in = self.ls[1](x_in)
            # x_in = nn.ReLU()(x_in)
            x_in = self.ls[2](x_in)
            out = torch.max(x_in, dim=1)[0]"""
        else:
            out = torch.max(x_in, dim=1)[0]
        #out = icnn_out

        """x_in = self.ls[1](x_in)
        #x_in = nn.ReLU()(x_in)
        x_in = self.ls[2](x_in)
        out = torch.max(x_in, dim=1)[0]"""

        # Vereinfachung von der zwei Layer Variante
        """x_in = self.ls[1](x_in)
        out = torch.max(x_in, dim=1)[0]"""

        return out



def boltzmann_op(x):
    scale = torch.mul(x, 10)
    exp = torch.exp(scale)
    summed = torch.mul(x, exp).sum(dim=1, keepdim=True)
    summed_exp = exp.sum(dim=1, keepdim=True)
    return torch.div(summed, summed_exp)

def mellowmax(x):
    a = 10
    scale = torch.mul(x, a)
    exp = torch.exp(scale)
    size = x.size(1)
    summed = torch.div(exp.sum(dim=1, keepdim=True), size)
    out = torch.div(torch.log(summed), a)
    return out

def smu_4(x):
    out = vmap(lambda a: smu_binary(smu_binary(a[0], a[1]), smu_binary(a[2], a[3])))(x)
    out = torch.unsqueeze(out, dim=1)
    return out

def smu_2(x):
    out = vmap(lambda a: smu_binary(a[0], a[1]))(x)
    return out

def smu_binary(a, b):
    e = 0.1
    out = torch.div(a.add(b).add(torch.pow(torch.pow(torch.sub(a, b), 2).add(e), 0.5)), 2)
    return out