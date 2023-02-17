import math
import time
from abc import ABC, abstractmethod
import gurobipy as grp
import torch
from gurobipy import abs_

import script.Verification.VerificationBasics as verbas


class Verifier(ABC):
    def __init__(self, net, input_x, eps, time_limit=None, solver_bound=None, print_log=False):
        self.time_limit=time_limit
        self.solver_bound = solver_bound
        self.net = net
        self.input_x = input_x
        self.eps = eps
        self.print_log = print_log
        self.model = None
        self.output_vars = None
        self.input_vars = None

    @abstractmethod
    def generate_constraints_for_net(self):
        self.model = None
        self.output_vars = None
        self.input_vars = None


    def test_feasibility(self, input_sample):
        #t = time.time()

        self.model.update()
        model = self.model.copy()
        model.setParam("BestObjStop", 0.01)

        input_size = len(input_sample)
        in_vars = []
        for i in range(input_size):
            in_vars.append(model.getVarByName("in_var[{}]".format(i)))

        """
        output_size = len(self.output_vars.tolist())

        difference = model.addVars(output_size, lb=-float("inf"))
        model.addConstrs(difference[i] == self.output_vars[i] - output_sample[i] for i in range(output_size))

        abs_diff = model.addVars(output_size)
        model.addConstrs(abs_diff[i] == abs_(difference[i]) for i in range(output_size))

        max_var = model.addVar()
        model.addConstr(max_var == grp.max_(abs_diff))"""

        model.addConstrs(in_vars[i] == input_sample[i] for i in range(input_size))

        model.update()

        #print("constr generateion {}".format(time.time() - t))
        #t = time.time()
        model.optimize()

        if model.Status == grp.GRB.OPTIMAL:
            """out_val = []
            for i in range(len(self.output_vars.tolist())):
                out_val.append(self.output_vars.tolist()[i].getAttr("x"))

            difference_val = []
            for i in range(len(difference)):
                difference_val.append(difference[i].getAttr("x"))

            abs_value = []
            for i in range(len(abs_diff)):
                abs_value.append(abs_diff[i].getAttr("x"))"""

            #print("solving {}".format(time.time() - t))
            #max_var_value = max_var.getAttr("x")
            return True #max_var_value <= 0

        return False


class SingleNeuronVerifier(Verifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate_constraints_for_net(self):
        m = grp.Model()

        if self.time_limit is not None:
            m.setParam("TimeLimit", self.time_limit)

        if self.solver_bound is not None:
            m.setParam("BestObjStop", self.solver_bound)

        if not self.print_log:
            m.Params.LogToConsole = 0

        input_flattened = torch.flatten(self.input_x)
        input_size = input_flattened.size(0)
        bounds_affine_out, bounds_layer_out = self.net.calculate_box_bounds([input_flattened.add(-self.eps), input_flattened.add(self.eps)])

        input_flattened = input_flattened.cpu().numpy()

        input_var = m.addMVar(input_size, lb=[elem - self.eps for elem in input_flattened],
                              ub=[elem + self.eps for elem in input_flattened], name="in_var")
        m.addConstrs((input_var[i] <= input_flattened[i] + self.eps for i in range(input_size)), name="in_const0")
        m.addConstrs((input_var[i] >= input_flattened[i] - self.eps for i in range(input_size)), name="in_const1")


        parameter_list = list(self.net.parameters())
        in_var = input_var
        for i in range(0, len(parameter_list) - 2, 2):
            in_lb = bounds_affine_out[int(i / 2)][0].detach().cpu().numpy()
            in_ub = bounds_affine_out[int(i / 2)][1].detach().cpu().numpy()
            W, b = parameter_list[i].detach().cpu().numpy(), parameter_list[i + 1].detach().cpu().numpy()

            out_fet = len(b)
            out_vars = m.addMVar(out_fet, lb=in_lb, ub=in_ub, name="affine_var" + str(i))
            const = m.addConstrs((W[i] @ in_var + b[i] == out_vars[i] for i in range(len(W))), name="affine_const" + str(i))

            relu_in_var = out_vars
            out_lb = bounds_layer_out[int(i / 2)][0].detach().cpu().numpy()
            out_ub = bounds_layer_out[int(i / 2)][1].detach().cpu().numpy()
            relu_vars = verbas.add_single_neuron_constr(m, relu_in_var, out_fet, in_lb, in_ub, out_lb, out_ub, i=i)
            in_var = relu_vars

        lb = bounds_affine_out[-1][0].detach().cpu().numpy()
        ub = bounds_affine_out[-1][1].detach().cpu().numpy()
        W, b = parameter_list[len(parameter_list) - 2].detach().cpu().numpy(), parameter_list[-1].detach().cpu().numpy()

        out_fet = len(b)
        out_vars = m.addMVar(out_fet, lb=lb, ub=ub, name="last_affine_var")
        const = m.addConstrs((W[i] @ in_var + b[i] == out_vars[i] for i in range(len(W))), name="out_const")
        m.update()
        self.model = m
        self.output_vars = out_vars
        self.input_vars = input_var


class MILPVerifier(Verifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate_constraints_for_net(self):
        m = grp.Model()

        if self.time_limit is not None:
            m.setParam("TimeLimit", self.time_limit)

        if self.solver_bound is not None:
            m.setParam("BestObjStop", self.solver_bound)

        if not self.print_log:
            m.Params.LogToConsole = 0

        input_flattened = torch.flatten(self.input_x)
        input_size = input_flattened.size(0)
        # todo hier darf with_relu nicht wahr sein, weil man sonst ggf bestimmte abhängigkeiten
        #  der Neruonen unterbindet, aber möglicherweise kann ich das bei SNV oder DHOV verwenden?
        bounds_affine_out, bounds_layer_out = self.net.calculate_box_bounds([input_flattened.add(-self.eps), input_flattened.add(self.eps)])

        input_flattened = input_flattened.cpu().numpy()

        input_var = m.addMVar(input_size, lb=[elem - self.eps for elem in input_flattened],
                              ub=[elem + self.eps for elem in input_flattened], name="in_var")
        m.addConstrs(input_var[i] <= input_flattened[i] + self.eps for i in range(input_size))
        m.addConstrs(input_var[i] >= input_flattened[i] - self.eps for i in range(input_size))


        parameter_list = list(self.net.parameters())
        in_var = input_var
        for i in range(0, len(parameter_list) - 2, 2):
            in_lb = bounds_affine_out[int(i / 2)][0].detach().numpy()
            in_ub = bounds_affine_out[int(i / 2)][1].detach().numpy()
            W, b = parameter_list[i].detach().numpy(), parameter_list[i + 1].detach().numpy()

            out_fet = len(b)
            out_vars = m.addMVar(out_fet, lb=in_lb, ub=in_ub, name="affine_var" + str(i))
            const = m.addConstrs((W[i] @ in_var + b[i] == out_vars[i] for i in range(len(W))))

            relu_in_var = out_vars
            out_lb = bounds_layer_out[int(i / 2)][0].detach().cpu().numpy()
            out_ub = bounds_layer_out[int(i / 2)][1].detach().cpu().numpy()
            relu_vars = verbas.add_relu_constr(m, relu_in_var, out_fet, in_lb, in_ub, out_lb, out_ub)
            in_var = relu_vars

        lb = bounds_affine_out[-1][0].detach().numpy()
        ub = bounds_affine_out[-1][1].detach().numpy()
        W, b = parameter_list[len(parameter_list) - 2].detach().numpy(), parameter_list[-1].detach().numpy()

        out_fet = len(b)
        out_vars = m.addMVar(out_fet, lb=lb, ub=ub, name="last_affine_var")
        const = m.addConstrs((W[i] @ in_var + b[i] == out_vars[i] for i in range(len(W))), name="out_const")
        m.update()
        self.model = m
        self.output_vars = out_vars
        self.input_vars = input_var


class DHOVVerifier(Verifier):
    def __init__(self, icnns, group_size, *args, with_affine=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.icnns = icnns
        self.with_affine = with_affine
        self.group_size = group_size

    def generate_constraints_for_net(self):
        m = grp.Model()

        if self.time_limit is not None:
            m.setParam("TimeLimit", self.time_limit)

        if self.solver_bound is not None:
            m.setParam("BestObjStop", self.solver_bound)

        if not self.print_log:
            m.Params.LogToConsole = 0

        input_flattened = torch.flatten(self.input_x)
        input_size = input_flattened.size(0)
        bounds_affine_out, bounds_layer_out = self.net.calculate_box_bounds([input_flattened.add(-self.eps), input_flattened.add(self.eps)])

        input_flattened = input_flattened.cpu().numpy()
        in_var_bounds = [[elem - self.eps for elem in input_flattened], [elem + self.eps for elem in input_flattened]]
        input_var = m.addMVar(input_size, lb=in_var_bounds[0],
                              ub=in_var_bounds[1], name="in_var")
        m.addConstrs(input_var[i] <= input_flattened[i] + self.eps for i in range(input_size))
        m.addConstrs(input_var[i] >= input_flattened[i] - self.eps for i in range(input_size))

        parameter_list = list(self.net.parameters())
        in_var = input_var

        for i in range(0, len(parameter_list) - 2, 2):
            lb = bounds_affine_out[int(i / 2)][0].detach().cpu().numpy()
            ub = bounds_affine_out[int(i / 2)][1].detach().cpu().numpy()
            W, b = parameter_list[i].detach().cpu().numpy(), parameter_list[i + 1].detach().cpu().numpy()

            out_fet = len(b)
            out_vars = m.addMVar(out_fet, lb=lb, ub=ub, name="affine_var" + str(i))
            if self.with_affine:
                const = m.addConstrs((W[i] @ in_var + b[i] == out_vars[i] for i in range(len(W))))

            if i == len(parameter_list) - 2 - 2 or (not self.with_affine):
                num_groups = math.ceil(len(b) / self.group_size)
                for group_i in range(num_groups):
                    if group_i == num_groups - 1 and len(b) % group_i > 0:
                        from_neuron = self.group_size * group_i
                        to_neuron = self.group_size * group_i + self.group_size % group_i  # upper bound is exclusive
                    else:
                        from_neuron = self.group_size * group_i
                        to_neuron = self.group_size * group_i + self.group_size  # upper bound is exclusive

                    constraint_bounds_affine_out, constraint_bounds_layer_out = self.icnns[i // 2][group_i].calculate_box_bounds(in_var_bounds)
                    self.icnns[i // 2][group_i].add_max_output_constraints(m, in_var[from_neuron:to_neuron], constraint_bounds_affine_out, constraint_bounds_layer_out)

            #m.update()
            in_var = out_vars
            in_var_bounds = [lb, ub]

        lb = bounds_affine_out[-1][0].detach().cpu().numpy()
        ub = bounds_affine_out[-1][1].detach().cpu().numpy()
        W, b = parameter_list[len(parameter_list) - 2].detach().cpu().numpy(), parameter_list[-1].detach().cpu().numpy()

        out_fet = len(b)
        out_vars = m.addMVar(out_fet, lb=lb, ub=ub, name="last_affine_var")
        const = m.addConstrs((W[i] @ in_var + b[i] == out_vars[i] for i in range(len(W))), name="out_const")

        m.update()

        self.model = m
        self.output_vars = out_vars
        self.input_vars = input_var
