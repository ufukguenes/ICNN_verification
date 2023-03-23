import math
import time
from abc import ABC, abstractmethod
import gurobipy as grp
import torch
from gurobipy import abs_

import script.Verification.VerificationBasics as verbas
from script.settings import device, data_type


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


    def test_feasibility(self, output_sample):
        #t = time.time()

        self.model.update()
        model = self.model.copy()
        model.setParam("BestObjStop", 0.01)

        output_size = len(output_sample)
        out_vars = []
        for i in range(output_size):
            out_vars.append(model.getVarByName("last_affine_var[{}]".format(i)))

        """
        output_size = len(self.output_vars.tolist())

        difference = model.addVars(output_size, lb=-float("inf"))
        model.addConstrs(difference[i] == self.output_vars[i] - output_sample[i] for i in range(output_size))

        abs_diff = model.addVars(output_size)
        model.addConstrs(abs_diff[i] == abs_(difference[i]) for i in range(output_size))

        max_var = model.addVar()
        model.addConstr(max_var == grp.max_(abs_diff))"""

        model.addConstrs(out_vars[i] == output_sample[i] for i in range(len(output_sample)))

        model.setObjective(out_vars[0], grp.GRB.MINIMIZE)

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
    def __init__(self, *args, optimize_bounds=True, print_new_bounds=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.print_new_bounds = print_new_bounds
        self.optimize_bounds = optimize_bounds

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

            #print("================ layer {} ===============".format(i // 2))
            if self.optimize_bounds:
                m.update()
                # todo code duplicat
                for neuron_to_optimize in range(len(out_vars.tolist())):
                    m.setObjective(out_vars[neuron_to_optimize], grp.GRB.MINIMIZE)
                    m.optimize()
                    if m.Status == grp.GRB.OPTIMAL:
                        value = out_vars.getAttr("x")
                        if self.print_new_bounds and abs(value[neuron_to_optimize] - bounds_affine_out[i // 2][0][neuron_to_optimize]) > 0.00001:
                            print("        {}, lower: new {}, old {}".format(neuron_to_optimize, value[neuron_to_optimize],
                                                                         bounds_affine_out[i // 2][0][
                                                                             neuron_to_optimize]))
                        bounds_affine_out[i // 2][0][neuron_to_optimize] = value[neuron_to_optimize]

                    m.setObjective(out_vars[neuron_to_optimize], grp.GRB.MAXIMIZE)
                    m.optimize()
                    if m.Status == grp.GRB.OPTIMAL:
                        value = out_vars.getAttr("x")
                        if self.print_new_bounds and abs(value[neuron_to_optimize] - bounds_affine_out[i // 2][1][neuron_to_optimize]) > 0.00001:
                            print("        {}, upper: new {}, old {}".format(neuron_to_optimize, value[neuron_to_optimize],
                                                                         bounds_affine_out[i // 2][1][
                                                                             neuron_to_optimize]))
                        bounds_affine_out[i // 2][1][neuron_to_optimize] = value[neuron_to_optimize]

                relu_out_lb, relu_out_ub = verbas.calc_relu_out_bound(bounds_affine_out[i // 2][0],
                                                                      bounds_affine_out[i // 2][1])
                bounds_layer_out[i // 2][0] = relu_out_lb
                bounds_layer_out[i // 2][1] = relu_out_ub


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

    def generate_constraints_for_net(self, until_layer_neuron=None):
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
        m.addConstrs(input_var[i] <= input_flattened[i] + self.eps for i in range(input_size))
        m.addConstrs(input_var[i] >= input_flattened[i] - self.eps for i in range(input_size))


        parameter_list = list(self.net.parameters())
        in_var = input_var
        break_early = False
        for i in range(0, len(parameter_list) - 2, 2):
            if until_layer_neuron != None and until_layer_neuron[0] == i // 2:
                neuron_index = until_layer_neuron[1]
                in_lb = [bounds_affine_out[int(i / 2)][0].detach().numpy()[neuron_index]]
                in_ub = [bounds_affine_out[int(i / 2)][1].detach().numpy()[neuron_index]]
                W, b = parameter_list[i].detach().numpy()[neuron_index], parameter_list[i + 1].detach().numpy()[neuron_index]

                out_fet = 1
                out_vars = m.addMVar(out_fet, lb=in_lb, ub=in_ub)
                const = m.addConstr(W @ in_var + b == out_vars)

                relu_in_var = out_vars
                out_lb = [bounds_layer_out[int(i / 2)][0].detach().cpu().numpy()[neuron_index]]
                out_ub = [bounds_layer_out[int(i / 2)][1].detach().cpu().numpy()[neuron_index]]
                relu_vars = verbas.add_relu_constr(m, relu_in_var, out_fet, in_lb, in_ub, out_lb, out_ub, i=i)
                in_var = relu_vars
                break_early = True
                break

            in_lb = bounds_affine_out[int(i / 2)][0].detach().numpy()
            in_ub = bounds_affine_out[int(i / 2)][1].detach().numpy()
            W, b = parameter_list[i].detach().numpy(), parameter_list[i + 1].detach().numpy()

            out_fet = len(b)
            out_vars = m.addMVar(out_fet, lb=in_lb, ub=in_ub, name="affine_var" + str(i))
            const = m.addConstrs((W[i] @ in_var + b[i] == out_vars[i] for i in range(len(W))))

            relu_in_var = out_vars
            out_lb = bounds_layer_out[int(i / 2)][0].detach().cpu().numpy()
            out_ub = bounds_layer_out[int(i / 2)][1].detach().cpu().numpy()
            relu_vars = verbas.add_relu_constr(m, relu_in_var, out_fet, in_lb, in_ub, out_lb, out_ub, i=i)
            in_var = relu_vars

        if not break_early:
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
    def __init__(self, icnns, group_size, last_layer_group_indices, fixed_neuron_last_layer_lower, fixed_neuron_last_layer_upper, bounds_affine_out, bounds_layer_out, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.icnns = icnns
        self.group_size = group_size
        self.fixed_neuron_last_layer_lower = fixed_neuron_last_layer_lower
        self.fixed_neuron_last_layer_upper = fixed_neuron_last_layer_upper
        self.last_layer_group_indices = last_layer_group_indices
        self.bounds_affine_out = bounds_affine_out
        self.bounds_layer_out = bounds_layer_out

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

        parameter_list = list(self.net.parameters())
        bounds_affine_out_output_layer, bounds_layer_out_output_layer = self.net.calculate_box_bounds(
            [input_flattened.add(-self.eps), input_flattened.add(self.eps)])

        bounds_affine_out_output_layer, bounds_layer_out_output_layer = bounds_affine_out_output_layer[-1], bounds_layer_out_output_layer[-1]
        # only use the icnn for the last layer
        last_icnn = self.icnns[-1]
        input_size = len(parameter_list[-4])
        lb = self.bounds_layer_out[-1][0].detach().cpu().numpy()
        ub = self.bounds_layer_out[-1][1].detach().cpu().numpy()
        in_var = m.addMVar(input_size, lb=lb, ub=ub, name="icnn_var")

        for group_i, index_to_select in enumerate(self.last_layer_group_indices):
            current_var = [in_var[x] for x in index_to_select]
            index_to_select = torch.tensor(index_to_select).to(device)
            low = torch.index_select(self.bounds_layer_out[-1][0], 0, index_to_select)
            up = torch.index_select(self.bounds_layer_out[-1][1], 0, index_to_select)
            constraint_bounds_affine_out, constraint_bounds_layer_out = last_icnn[group_i].calculate_box_bounds([low, up])
            last_icnn[group_i].add_max_output_constraints(m, current_var, constraint_bounds_affine_out, constraint_bounds_layer_out)

        for neuron_index in self.fixed_neuron_last_layer_upper:
            m.addConstr(in_var[neuron_index] == 0)

        for neuron_index in self.fixed_neuron_last_layer_lower:
            m.addConstr(lb[neuron_index] <= in_var[neuron_index])
            m.addConstr(lb[neuron_index] <= in_var[neuron_index])
            m.addConstr(in_var[neuron_index] <= ub[neuron_index])

        lb = bounds_affine_out_output_layer[0].detach().cpu().numpy()
        ub = bounds_affine_out_output_layer[1].detach().cpu().numpy()
        W, b = parameter_list[len(parameter_list) - 2].detach().cpu().numpy(), parameter_list[-1].detach().cpu().numpy()

        out_fet = len(b)
        out_vars = m.addMVar(out_fet, lb=lb, ub=ub, name="last_affine_var")
        const = m.addConstrs((W[i] @ in_var + b[i] == out_vars[i] for i in range(len(W))), name="out_const")

        m.update()

        self.model = m
        self.output_vars = out_vars

