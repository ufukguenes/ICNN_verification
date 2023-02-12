import math

import torch
from torch import nn
from functorch import vmap
import script.Verification.VerificationBasics as verbas
from abc import ABC, abstractmethod
from script.settings import device, data_type

class Flatten(nn.Module):

    def forward(self, x):
        x.size(0)
        return x.view(x.size(0), -1)


class VerifiableNet(ABC):
    def __init__(self):
        self.ws = None

    @abstractmethod
    def calculate_box_bounds(self, input_bounds, with_relu=True):
        pass

    @abstractmethod
    def add_constraints(self, model, input_vars, bounds):
        pass

    def add_max_output_constraints(self, model, input_vars, bounds):
        output = self.add_constraints(model, input_vars, bounds)
        model.addConstr(output[0] <= 0)
        return output

    def apply_enlargement(self, value):
        with torch.no_grad():
            last_layer = list(self.ws[-1].parameters())
            b = last_layer[1]
            b.data = b - value

    @abstractmethod
    def apply_normalisation(self, mean, std):
        pass

    @abstractmethod
    def init(self, lower_bounds, upper_bounds):
        pass


class SequentialNN(nn.Sequential, VerifiableNet):
    def __init__(self, layer_widths):
        super(SequentialNN, self).__init__()
        self.layer_widths = layer_widths
        d_in = layer_widths[0]
        for lw in layer_widths[1:len(layer_widths) - 1]:
            self.append(nn.Linear(d_in, lw, dtype=data_type).to(device))
            self.append(nn.ReLU())
            d_in = lw

        self.append(nn.Linear(d_in, layer_widths[-1], dtype=data_type).to(device))

    def forward(self, x):
        x = Flatten()(x)
        return super().forward(x)

    def calculate_box_bounds(self, input_bounds, with_relu=True):
        parameter_list = list(self.parameters())

        if input_bounds is None:
            bounds_per_layer = [([torch.tensor([-5000 for k in range(len(parameter_list[i]))]),
                                  torch.tensor([5000 for k in range(len(parameter_list[i]))])]) for i in
                                range(0, len(parameter_list), 2)]
            return bounds_per_layer  # todo None entfernen aus aufrufen und durch sinnvolle eingabe ersetzen

        next_lower_bounds = input_bounds[0]
        next_upper_bounds = input_bounds[1]
        bounds_per_layer = []
        for i in range(0, len(parameter_list), 2):
            W, b = parameter_list[i], parameter_list[i + 1]
            w_plus = torch.maximum(W, torch.tensor(0, dtype=data_type).to(device))
            w_minus = torch.minimum(W, torch.tensor(0, dtype=data_type).to(device))
            lb = torch.matmul(w_plus, next_lower_bounds).add(torch.matmul(w_minus, next_upper_bounds)).add(b)
            ub = torch.matmul(w_plus, next_upper_bounds).add(torch.matmul(w_minus, next_lower_bounds)).add(b)

            if with_relu and i < len(parameter_list) - 2:
                next_upper_bounds = torch.maximum(torch.tensor(0, dtype=data_type).to(device), ub)
                next_lower_bounds = torch.maximum(torch.tensor(0, dtype=data_type).to(device), lb)
            else:
                next_upper_bounds = ub
                next_lower_bounds = lb

            bounds_per_layer.append([next_lower_bounds, next_upper_bounds])

        return bounds_per_layer

    def add_constraints(self, model, input_vars, bounds):
        parameter_list = list(self.parameters())
        output_vars = model.addMVar(self.layer_widths[-1], lb=-float('inf'))  # todo bounds anpassen

        in_var = input_vars
        for i in range(0, len(parameter_list), 2):
            lb = bounds[int(i / 2)][0].detach().cpu().numpy()
            ub = bounds[int(i / 2)][1].detach().cpu().numpy()
            W, b = parameter_list[i].detach().cpu().numpy(), parameter_list[i + 1].detach().cpu().numpy()

            out_fet = len(b)
            out_vars = model.addMVar(out_fet, lb=lb, ub=ub, name="affine_var" + str(i))
            const = model.addConstrs((W[i] @ in_var + b[i] == out_vars[i] for i in range(len(W))))
            in_var = out_vars

            if i < len(parameter_list) - 2:
                relu_vars = verbas.add_relu_constr(model, in_var, out_fet, [-10000 for i in range(len(W))], ub, i)
                in_var = relu_vars
                out_vars = relu_vars

        const = model.addConstrs(out_vars[i] == output_vars[i] for i in range(out_fet))
        return output_vars

    def apply_normalisation(self, mean, std):
        with torch.no_grad():
            parameter_list = list(self.parameters())
            parameter_list[0].data = torch.div(parameter_list[0], std)
            parameter_list[1].data = torch.add(- torch.matmul(parameter_list[0], mean), parameter_list[1])

    def init(self, lower_bounds, upper_bounds):
        if self.init_all_with_zeros:
            for i in range(0, len(self.ws)):
                ws = list(self.ws[i].parameters())
                ws[1].data = torch.zeros_like(ws[1], dtype=data_type).to(device)
                ws[0].data = torch.zeros_like(ws[0], dtype=data_type).to(device)

            # init non box-bound layer with zeros
            for elem in self.us:
                w = list(elem.parameters())
                w[0].data = torch.zeros_like(w[0], dtype=data_type).to(device)


class ICNN(nn.Module, VerifiableNet):

    def __init__(self, layer_widths, force_positive_init=False, activation_function="ReLU",
                 init_scaling=10, init_all_with_zeros=False):

        """
    layer_widths - ([int]) list of layer widths **including** input and output dim
    """
        super(ICNN, self).__init__()

        valid_activation_functions = ["ReLU", "Softplus", "Tanh"]

        if activation_function not in valid_activation_functions:
            raise AttributeError(
                "Expected activation function to be, one of: {}, actual: {}".format(valid_activation_functions,
                                                                                    activation_function))

        if activation_function == "ReLU":
            self.activation_function = nn.ReLU()
        elif activation_function == "Softplus":
            self.activation_function = nn.Softplus()
        elif activation_function == "Tanh":
            self.activation_function = nn.Tanh()

        self.init_scaling = init_scaling
        self.init_all_with_zeros = init_all_with_zeros

        self.ws = nn.ParameterList([])  # positive weights for propagation
        self.us = nn.ParameterList([])  # weights tied to inputs
        self.layer_widths = layer_widths
        self.ws.append(nn.Linear(layer_widths[0], layer_widths[1], bias=True, dtype=data_type).to(device))

        d_in = layer_widths[1]

        for lw in layer_widths[2:]:
            w = nn.Linear(d_in, lw, dtype=data_type).to(device)

            with torch.no_grad():
                if force_positive_init:
                    for p in w.parameters():
                        if len(p.size()) > 1:
                            p[:] = torch.maximum(torch.Tensor([0]), p)

            d_in = lw
            u = nn.Linear(layer_widths[0], lw, bias=False, dtype=data_type).to(device)

            self.ws.append(w)
            self.us.append(u)

    def forward(self, x):
        x = Flatten()(x)
        x1 = self.activation_function(self.ws[0](x))  # first layer is only W
        for w, u in zip(self.ws[1:-1], self.us[:-1]):
            a = w(x1)
            b = u(x)
            x1 = self.activation_function(a + b)

        x1 = self.ws[-1](x1) + self.us[-1](x)  # no ReLU in last layer"""
        return x1

    def init(self, lower_bounds, upper_bounds):
        with torch.no_grad():
            num_of_layers = len(self.layer_widths)
            last_layer_index = num_of_layers - 2
            penultimate_layer_index = num_of_layers - 3
            size_of_last_layer = self.layer_widths[num_of_layers - 1]
            size_of_penultimate_layer = self.layer_widths[num_of_layers - 2]
            size_of_input = self.layer_widths[0]

            if size_of_last_layer != 1:
                raise AttributeError("To use box-bound initialization, the output of the last layer should be a scalar")
            elif size_of_penultimate_layer != 2 * size_of_input:
                raise AttributeError("To use box-bound initialization, the size of the penultimate layer needs to be "
                                     "2*input size")

            if self.init_all_with_zeros:
                for i in range(0, len(self.ws)):
                    ws = list(self.ws[i].parameters())
                    ws[1].data = torch.zeros_like(ws[1], dtype=data_type).to(device)
                    ws[0].data = torch.zeros_like(ws[0], dtype=data_type).to(device)

                # init non box-bound layer with zeros
                for elem in self.us:
                    w = list(elem.parameters())
                    w[0].data = torch.zeros_like(w[0], dtype=data_type).to(device)
            else:
                third_last_layer = list(self.ws[penultimate_layer_index - 1].parameters())
                third_last_w = third_last_layer[0]
                third_last_b = third_last_layer[1]
                third_last_w.data = torch.zeros_like(third_last_w, dtype=data_type).to(device)
                third_last_b.data = torch.zeros_like(third_last_b, dtype=data_type).to(device)

                last_us = list(self.us[last_layer_index - 1].parameters())[0]
                last_us.data = torch.zeros_like(last_us, dtype=data_type).to(device)

            # bias ist set in W, because U does not have bias
            ws = list(self.ws[penultimate_layer_index].parameters())

            # us is used to represent Box-Bounds, because values in W are set to 0 when negative
            us = list(self.us[penultimate_layer_index - 1].parameters())

            b = torch.zeros_like(ws[1], dtype=data_type).to(device)
            u = torch.zeros_like(us[0], dtype=data_type).to(device)
            num_of_bounds = len(lower_bounds)
            for i in range(num_of_bounds):
                u[2 * i][i] = 1
                u[2 * i + 1][i] = -1
                b[2 * i] = - upper_bounds[i]  # upper bound
                b[2 * i + 1] = lower_bounds[i]  # lower bound
            ws[1].data = b
            us[0].data = u

            # last layer needs to sum the output of the box-bounds after ReLU activation
            last = list(self.ws[last_layer_index].parameters())
            last[0].data = torch.mul(torch.ones_like(last[0], dtype=data_type).to(device), self.init_scaling)
            last[1].data = torch.zeros_like(last[1], dtype=data_type).to(device)

    def calculate_box_bounds(self, input_bounds, with_relu=True):  # todo für andere Netze die Architektur anpassen
        parameter_list = list(self.parameters())
        # todo for now this only works for sequential nets

        if input_bounds is None:
            bounds_per_layer = [([torch.tensor([-5000 for k in range(len(parameter_list[i]))]),
                                  torch.tensor([5000 for k in range(len(parameter_list[i]))])]) for i in
                                range(0, len(parameter_list), 2)]
            return bounds_per_layer  # todo None entfernen aus aufrufen und durch sinnvolle eingabe ersetzen

        next_lower_bounds = input_bounds[0]
        next_upper_bounds = input_bounds[1]
        bounds_per_layer = []
        for i in range(0, len(parameter_list), 2):
            W, b = parameter_list[i], parameter_list[i + 1]
            w_plus = torch.maximum(W, torch.tensor(0, dtype=data_type).to(device))
            w_minus = torch.minimum(W, torch.tensor(0, dtype=data_type).to(device))
            lb = torch.matmul(w_plus, next_lower_bounds).add(torch.matmul(w_minus, next_upper_bounds)).add(b)
            ub = torch.matmul(w_plus, next_upper_bounds).add(torch.matmul(w_minus, next_lower_bounds)).add(b)
            if i != 0:
                affine_u = self.us[i // 2 - 1]
                u_plus = torch.maximum(affine_u, torch.tensor(0, dtype=data_type).to(device))
                u_minus = torch.minimum(affine_u, torch.tensor(0, dtype=data_type).to(device))
                lb = lb.add(torch.matmul(u_plus, next_lower_bounds).add(torch.matmul(u_minus, next_upper_bounds)))
                ub = ub.add(torch.matmul(u_plus, next_upper_bounds).add(torch.matmul(u_minus, next_lower_bounds)))

            if with_relu and i < len(parameter_list) - 2:
                next_upper_bounds = torch.maximum(torch.tensor(0, dtype=data_type).to(device), ub)
                next_lower_bounds = torch.maximum(torch.tensor(0, dtype=data_type).to(device), lb)
            else:
                next_upper_bounds = ub
                next_lower_bounds = lb

            bounds_per_layer.append([next_lower_bounds, next_upper_bounds])

        return bounds_per_layer

    def add_constraints(self, model, input_vars, bounds):
        ws = list(self.ws.parameters())
        us = list(self.us.parameters())
        output_vars = model.addMVar(self.layer_widths[-1], lb=-float('inf'))  # todo bounds anpassen

        in_var = input_vars
        for i in range(0, len(ws), 2):
            lb = bounds[int(i / 2)][0].detach().cpu().numpy()
            ub = bounds[int(i / 2)][1].detach().cpu().numpy()
            affine_w, affine_b = ws[i].detach().cpu().numpy(), ws[i + 1].detach().cpu().numpy()

            out_fet = len(affine_b)
            affine_var = model.addMVar(out_fet, lb=lb, ub=ub, name="affine_var" + str(i))
            out_vars = model.addMVar(out_fet, lb=lb, ub=ub, name="affine_skip_var" + str(i))

            affine_const = model.addConstrs(
                affine_w[i] @ in_var + affine_b[i] == affine_var[i] for i in range(len(affine_w)))
            if i != 0:
                k = math.floor(i / 2) - 1
                skip_W = torch.clone(us[k]).detach().cpu().numpy()  # has no bias
                skip_var = model.addMVar(out_fet, lb=lb, ub=ub, name="skip_var" + str(k))
                skip_const = model.addConstrs(skip_W[i] @ input_vars == skip_var[i] for i in range(len(affine_w)))
                affine_skip_cons = model.addConstrs(
                    affine_var[i] + skip_var[i] == out_vars[i] for i in range(len(affine_w)))
            else:
                affine_no_skip_cons = model.addConstrs(affine_var[i] == out_vars[i] for i in range(len(affine_w)))

            in_var = out_vars

            if i < len(ws) - 2:
                relu_vars = verbas.add_relu_constr(model, in_var, out_fet, lb, ub, i)
                in_var = relu_vars
                out_vars = relu_vars

        const = model.addConstrs(out_vars[i] == output_vars[i] for i in range(out_fet))
        return output_vars

    def apply_normalisation(self, mean, std):
        with torch.no_grad():
            parameter_list = list(self.ws[0].parameters())
            parameter_list[0].data = torch.div(parameter_list[0], std)
            parameter_list[1].data = torch.add(- torch.matmul(parameter_list[0], mean), parameter_list[1])

            for i in range(len(self.us)):
                parameter_list = list(self.us[i].parameters())
                parameter_list[0].data = torch.div(parameter_list[0], std)

                internal_parameter_list = list(self.ws[i + 1].parameters())
                internal_parameter_list[1].data = torch.add(- torch.matmul(parameter_list[0], mean),
                                                            internal_parameter_list[1])


class ICNNLogical(ICNN):

    def __init__(self, *args, with_two_layers=False, **kwargs):
        """
        layer_widths - ([int]) list of layer widths **including** input and output dim
        """
        super(ICNNLogical, self).__init__(*args, **kwargs)

        self.with_two_layers = with_two_layers
        self.ls = []

        self.ls.append(nn.Linear(self.layer_widths[0], 2 * self.layer_widths[0], bias=True, dtype=data_type).to(device))
        if self.with_two_layers:
            self.ls.append(nn.Linear(2, 4, bias=False, dtype=data_type).to(device))
            self.ls.append(nn.Linear(4, 3, bias=False, dtype=data_type).to(device))
        else:
            self.ls.append(nn.Linear(2, 3, bias=False, dtype=data_type).to(device))

        with torch.no_grad():
            if self.with_two_layers:
                l1 = list(self.ls[1].parameters())
                l2 = list(self.ls[2].parameters())

                # diese architektur ist für alle ICNNs gleich da alle genau 2 ausgaben haben eine vom eigentlichen ICNN
                # und eine von den Box Bounds
                l1[0].data = torch.tensor([[1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=data_type).to(device)
                l2[0].data = torch.tensor([[1, 0, 0, -1], [0, 1, 0, -1], [0, 0, 1, -1]], dtype=data_type).to(device)
            else:
                # Vereinfachung von der zwei Layer Variante
                l1 = list(self.ls[1].parameters())
                l1[0].data = torch.tensor([[2, 2], [2, 0], [0, 2]], dtype=data_type).to(device)

    def forward(self, x):
        icnn_out = super().forward(x)
        box_out = self.ls[0](x)
        box_out = torch.max(box_out, dim=1, keepdim=True)[0]
        x_in = torch.cat([icnn_out, box_out], dim=1)

        if self.with_two_layers:
            x_in = self.ls[1](x_in)
            # x_in = nn.ReLU()(x_in)
            x_in = self.ls[2](x_in)
            out = torch.max(x_in, dim=1)[0]
        else:
            x_in = self.ls[1](x_in)
            out = torch.max(x_in, dim=1)[0]
        return out

    def init(self, lower_bounds, upper_bounds):
        with torch.no_grad():
            if self.init_all_with_zeros:
                for i in range(len(self.ws)):
                    ws = list(self.ws[i].parameters())
                    ws[1].data = torch.zeros_like(ws[1], dtype=data_type).to(device)
                    ws[0].data = torch.zeros_like(ws[0], dtype=data_type).to(device)
                last_ws = list(self.ws[-1].parameters())
                last_ws[0].data = torch.ones_like(last_ws[0])
                last_ws[1].data = torch.zeros_like(last_ws[1])

                for i in range(len(self.us)):
                    us = list(self.us[i].parameters())
                    us[0].data = torch.zeros_like(us[0], dtype=data_type).to(device)

            numer_of_bounds = len(lower_bounds)
            bb = list(self.ls[0].parameters())  # us is used because values in ws are set to 0 when negative
            u = torch.zeros_like(bb[0], dtype=data_type).to(device)
            b = torch.zeros_like(bb[1], dtype=data_type).to(device)
            for i in range(numer_of_bounds):
                u[2 * i][i] = 1
                u[2 * i + 1][i] = -1
                b[2 * i] = - upper_bounds[i]  # upper bound
                b[2 * i + 1] = lower_bounds[i]  # lower bound
            bb[0].data = u
            bb[1].data = b

    def add_max_output_constraints(self, model, input_vars, bounds):
        icnn_output_var = super().add_constraints(model, input_vars, bounds)
        output_of_and = model.addMVar(1, lb=-float('inf'))
        lb = bounds[-1][0].detach().cpu().numpy()
        ub = bounds[-1][1].detach().cpu().numpy()  # todo richtige box bounds anwenden (die vom zu approximierenden Layer)
        ls = self.ls

        bb_w, bb_b = ls[0].weight.data.detach().cpu().numpy(), ls[0].bias.data.detach().cpu().numpy()
        bb_var = model.addMVar(4, lb=lb, ub=ub, name="skip_var")
        skip_const = model.addConstrs(bb_w[i] @ input_vars + bb_b[i] == bb_var[i] for i in range(len(bb_w)))
        max_var = model.addVar(lb=-float("inf"))
        model.addGenConstrMax(max_var, bb_var.tolist())

        new_in = model.addMVar(2, lb=-float('inf'))
        model.addConstr(new_in[0] == icnn_output_var[0])
        model.addConstr(new_in[1] == max_var)

        if self.with_two_layers:
            affine_w = ls[1].weight.data.detach().cpu().numpy()
            affine_var1 = model.addMVar(4, lb=-float("inf"))
            affine_const = model.addConstrs(
                affine_w[i] @ new_in == affine_var1[i] for i in range(len(affine_w)))

            affine_W2 = ls[2].weight.data.detach().cpu().numpy()
            affine_var2 = model.addMVar(3, lb=-float("inf"))
            affine_const = model.addConstrs(
                affine_W2[i] @ affine_var1 == affine_var2[i] for i in range(len(affine_W2)))
            max_var2 = model.addVar(lb=-float("inf"))
            model.addGenConstrMax(max_var2, affine_var2.tolist())
        else:
            affine_w = ls[1].weight.data.detach().cpu().numpy()
            affine_var1 = model.addMVar(3, lb=-float("inf"))
            affine_const = model.addConstrs(
                affine_w[i] @ new_in == affine_var1[i] for i in range(len(affine_w)))
            max_var2 = model.addVar(lb=-float("inf"))
            model.addGenConstrMax(max_var2, affine_var1.tolist())

        max_var2 = model.addVar(lb=-float("inf"))
        model.addGenConstrMax(max_var2, affine_var1.tolist())

        const = model.addConstr(max_var2 == output_of_and.tolist()[0])

        model.addConstr(output_of_and.tolist()[0] <= 0)

        return output_of_and

    def apply_normalisation(self, mean, std):
        super().apply_normalisation(mean, std)

        parameter_list = list(self.ls[0].parameters())
        parameter_list[0].data = torch.div(parameter_list[0], std)
        parameter_list[1].data = torch.add(- torch.matmul(parameter_list[0], mean), parameter_list[1])


class ICNNApproxMax(ICNN):

    def __init__(self, *args, maximum_function="max", function_parameter=1, use_training_setup=True, **kwargs):
        super().__init__(*args, **kwargs)

        self.valid_maximum_functions = ["max", "Boltzmann", "LogSumExp", "Mellowmax", "SMU"]
        if maximum_function not in self.valid_maximum_functions:
            raise AttributeError(
                "Expected activation function to be, one of: {}, actual: {}".format(self.valid_maximum_functions,
                                                                                    maximum_function))
        self.maximum_function = maximum_function
        self.function_parameter = function_parameter
        self.use_training_setup = use_training_setup
        self.ls = []

        self.ls.append(nn.Linear(self.layer_widths[0], 2 * self.layer_widths[0], bias=True, dtype=data_type).to(device))

    def forward(self, x):
        icnn_out = super().forward(x)
        box_out = self.ls[0](x)
        box_out = torch.max(box_out, dim=1, keepdim=True)[0]
        x_in = torch.cat([icnn_out, box_out], dim=1)

        if self.use_training_setup:
            if self.maximum_function == "max":
                out = torch.max(x_in, dim=1)[0]
            elif self.maximum_function == "Boltzmann":
                out = boltzmann_op(x_in, self.function_parameter)
            elif self.maximum_function == "LogSumExp":
                out = torch.logsumexp(x_in, dim=1)
            elif self.maximum_function == "Mellowmax":
                out = mellowmax(x_in, self.function_parameter)
            elif self.maximum_function == "SMU":
                out = smu_2(x_in, self.function_parameter)
            else:
                raise AttributeError(
                    "Expected activation function to be, one of: {}, actual: {}".format(self.valid_maximum_functions,
                                                                                        self.maximum_function))
        else:
            out = torch.max(x_in, dim=1)[0]

        return out

    def init(self, lower_bounds, upper_bounds):
        with torch.no_grad():
            if self.init_all_with_zeros:
                for i in range(len(self.ws)):
                    ws = list(self.ws[i].parameters())
                    ws[1].data = torch.zeros_like(ws[1], dtype=data_type).to(device)
                    ws[0].data = torch.zeros_like(ws[0], dtype=data_type).to(device)
                last_ws = list(self.ws[-1].parameters())
                last_ws[0].data = torch.ones_like(last_ws[0])
                last_ws[1].data = torch.zeros_like(last_ws[1])

                for i in range(len(self.us)):
                    us = list(self.us[i].parameters())
                    us[0].data = torch.zeros_like(us[0], dtype=data_type).to(device)

            numer_of_bounds = len(lower_bounds)
            bb = list(self.ls[0].parameters())  # us is used because values in ws are set to 0 when negative
            u = torch.zeros_like(bb[0], dtype=data_type).to(device)
            b = torch.zeros_like(bb[1], dtype=data_type).to(device)
            for i in range(numer_of_bounds):
                u[2 * i][i] = 1
                u[2 * i + 1][i] = -1
                b[2 * i] = - upper_bounds[i]  # upper bound
                b[2 * i + 1] = lower_bounds[i]  # lower bound
            bb[0].data = u
            bb[1].data = b

    def add_max_output_constraints(self, model, input_vars, bounds):
        icnn_output_var = super().add_constraints(model, input_vars, bounds)
        output_of_and = model.addMVar(1, lb=-float('inf'))
        lb = bounds[-1][0].detach().cpu().numpy()
        ub = bounds[-1][1].detach().cpu().numpy() # todo richtige box bounds anwenden (die vom zu approximierenden Layer)
        ls = self.ls

        bb_w, bb_b = ls[0].weight.data.detach().cpu().numpy(), ls[0].bias.data.detach().cpu().numpy()
        bb_var = model.addMVar(4, lb=lb, ub=ub, name="skip_var")
        skip_const = model.addConstrs(bb_w[i] @ input_vars + bb_b[i] == bb_var[i] for i in range(len(bb_w)))
        max_var = model.addVar(lb=-float("inf"))
        model.addGenConstrMax(max_var, bb_var.tolist())

        new_in = model.addMVar(2, lb=-float('inf'))
        model.addConstr(new_in[0] == icnn_output_var[0])
        model.addConstr(new_in[1] == max_var)

        max_var2 = model.addVar(lb=-float("inf"))

        if False and self.use_training_setup: #todo only works with exact max constraints
            if self.maximum_function == "max":
                model.addGenConstrMax(max_var2, new_in.tolist())
            elif self.maximum_function == "Boltzmann":
                boltzmann_constraints(model, new_in, max_var2, self.function_parameter)
            elif self.maximum_function == "LogSumExp":
                logsummax_constraints(model, new_in, max_var2)
            elif self.maximum_function == "Mellowmax":
                mellowmax_constraints(model, new_in, max_var2, self.function_parameter)
            elif self.maximum_function == "SMU":
                smu_constraints(model, new_in, max_var2, self.function_parameter)
            else:
                raise AttributeError(
                    "Expected activation function to be, one of: {}, actual: {}".format(self.valid_maximum_functions,
                                                                                        self.maximum_function))
        else:
            model.addGenConstrMax(max_var2, new_in.tolist())

        model.addConstr(max_var2 == output_of_and.tolist()[0])

        model.addConstr(output_of_and.tolist()[0] <= 0)

        return output_of_and

    def apply_normalisation(self, mean, std):
        super().apply_normalisation(mean, std)

        parameter_list = list(self.ls[0].parameters())
        parameter_list[0].data = torch.div(parameter_list[0], std)
        parameter_list[1].data = torch.add(- torch.matmul(parameter_list[0], mean), parameter_list[1])


def boltzmann_op(x, parameter):
    scale = torch.mul(x, parameter)
    exp = torch.exp(scale)
    summed = torch.mul(x, exp).sum(dim=1, keepdim=True)
    summed_exp = exp.sum(dim=1, keepdim=True)
    return torch.div(summed, summed_exp)


def boltzmann_constraints(model, input_var, output_var, parameter):
    scale_var = model.addMVar(input_var.size, lb=-float('inf'))
    model.addConstrs(scale_var[i] == parameter * input_var[i] for i in range(input_var.size))

    exp_var = model.addMVar(input_var.size, lb=-float('inf'))
    for i in range(input_var.size):
        model.addGenConstrExp(scale_var[i], exp_var[i])

    mul_var = model.addMVar(input_var.size, lb=-float('inf'))
    model.addConstrs(mul_var[i] == exp_var[i] * input_var[i] for i in range(input_var.size))

    divide_var = model.addMVar(1, lb=-float('inf'))
    model.addConstr(exp_var.sum() * divide_var == 1)

    model.addConstr(output_var == mul_var.sum() * divide_var)


def mellowmax(x, parameter):
    scale = torch.mul(x, parameter)
    exp = torch.exp(scale)
    size = x.size(1)
    summed = torch.div(exp.sum(dim=1, keepdim=True), size)
    out = torch.div(torch.log(summed), parameter)
    return out


def mellowmax_constraints(model, input_var, output_var, parameter):
    scale_var = model.addMVar(input_var.size, lb=-float('inf'))
    model.addConstrs(scale_var[i] == parameter * input_var[i] for i in range(input_var.size))

    exp_var = model.addMVar(input_var.size, lb=-float('inf'))
    for i in range(input_var.size):
        model.addGenConstrExp(scale_var[i], exp_var[i])

    summed_divided = model.addMVar(1, lb=0)
    divide_var = model.addMVar(1, lb=-float('inf'))
    model.addConstr(input_var.size * divide_var == 1)
    model.addConstr(summed_divided[0] == exp_var.sum() * divide_var)

    log_var = model.addMVar(1, lb=-float('inf'))
    model.addGenConstrLog(summed_divided, log_var)

    divide_var2 = model.addMVar(1, lb=-float('inf'))
    model.addConstr(parameter * divide_var2 == 1)
    model.addConstr(output_var == log_var * divide_var2)


def logsummax_constraints(model, input_var, output_var):
    exp_var = model.addMVar(input_var.size, lb=-float('inf'))
    for i in range(input_var.size):
        model.addGenConstrExp(input_var[i], exp_var[i])

    summed = model.addMVar(1, lb=0)
    model.addConstr(summed[0] == exp_var.sum())

    log_var = model.addMVar(1, lb=-float('inf'))
    model.addGenConstrLog(summed, log_var)

    model.addConstr(output_var == log_var)


def smu_2(x, parameter):
    out = vmap(lambda a: smu_binary(a[0], a[1], parameter))(x)
    return out


def smu_binary(a, b, parameter):
    e = parameter
    out = torch.div(a.add(b).add(torch.pow(torch.pow(torch.sub(a, b), 2).add(e), 0.5)), 2)
    return out


def smu_constraints(model, input_var, output_var, parameter):
    subtract_var = model.addMVar(1, lb=-float('inf'))
    power_var = model.addMVar(1, lb=0)
    root_var = model.addMVar(1, lb=0)
    power_eps_var = model.addMVar(1, lb=-float('inf'))

    model.addConstr(subtract_var[0] == input_var[0] - input_var[1])
    model.addGenConstrPow(power_var, subtract_var, 2)
    model.addConstr(power_eps_var[0] == power_var[0] + parameter)
    model.addGenConstrPow(root_var, power_eps_var, 0.5)
    model.addConstr(output_var == (input_var[0] + input_var[1] + root_var[0]) * 0.5)
