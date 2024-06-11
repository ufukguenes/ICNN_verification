import math
import time
from abc import ABC, abstractmethod
import gurobipy as grp
import torch
from gurobipy import abs_

import script.Verification.VerificationBasics as verbas
from script.settings import device, data_type


class Verifier(ABC):
    def __init__(self, net, input_x, input_bounds, time_limit=None, solver_bound=None, print_log=False):
        self.time_limit = time_limit
        self.solver_bound = solver_bound
        self.net = net
        self.input_x = input_x
        self.input_bounds = input_bounds
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
        output_layer_index = (len(list(self.net.parameters())) - 2) // 2
        for i in range(output_size):
            out_vars.append(model.getVarByName("affine_var{}[{}]".format(output_layer_index, i)))

        model.addConstrs(out_vars[i] == output_sample[i] for i in range(len(output_sample)))

        model.setObjective(out_vars[0], grp.GRB.MINIMIZE)
        model.update()
        model.optimize()

        if model.Status == grp.GRB.OPTIMAL:
            return True

        return False


class SingleNeuronVerifier(Verifier):
    def __init__(self, *args, optimize_bounds=True, print_new_bounds=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.print_new_bounds = print_new_bounds
        self.optimize_bounds = optimize_bounds
        self.bounds_affine_out = None
        self.bounds_layer_out = None

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
        bounds_affine_out, bounds_layer_out = self.net.calculate_box_bounds(self.input_bounds)

        self.bounds_affine_out, self.bounds_layer_out = bounds_affine_out, bounds_layer_out

        lower_bound = self.input_bounds[0].cpu().numpy()
        upper_bound = self.input_bounds[1].cpu().numpy()
        input_var = m.addMVar(input_size, lb=lower_bound, ub=upper_bound, name="in_var")


        parameter_list = list(self.net.parameters())
        in_var = input_var
        i = -2
        for i in range(0, len(parameter_list) - 2, 2):

            in_lb = bounds_affine_out[int(i / 2)][0].detach().cpu().numpy()
            in_ub = bounds_affine_out[int(i / 2)][1].detach().cpu().numpy()
            W, b = parameter_list[i].detach().cpu().numpy(), parameter_list[i + 1].detach().cpu().numpy()

            out_vars = verbas.add_affine_constr(m, W, b, in_var, in_lb, in_ub, i)

            if self.print_new_bounds:
                print("================ layer {} ===============".format(i // 2))
            if self.optimize_bounds and i != 0:
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

            for var_index, var in enumerate(out_vars):
                var.setAttr("LB", bounds_affine_out[i//2][0][var_index])
                var.setAttr("UB", bounds_affine_out[i//2][1][var_index])
            relu_in_var = out_vars
            out_lb = bounds_layer_out[int(i / 2)][0].detach().cpu().numpy()
            out_ub = bounds_layer_out[int(i / 2)][1].detach().cpu().numpy()
            out_vars = verbas.add_single_neuron_constr(m, relu_in_var, len(b), in_lb, in_ub, out_lb, out_ub, i=i)
            in_var = out_vars


        lb = bounds_affine_out[-1][0].detach().cpu().numpy()
        ub = bounds_affine_out[-1][1].detach().cpu().numpy()
        W, b = parameter_list[len(parameter_list) - 2].detach().cpu().numpy(), parameter_list[-1].detach().cpu().numpy()

        out_vars = verbas.add_affine_constr(m, W, b, in_var, lb, ub, i+2)

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
        bounds_affine_out, bounds_layer_out = self.net.calculate_box_bounds(self.input_bounds)

        lower_bound = self.input_bounds[0].cpu().numpy()
        upper_bound = self.input_bounds[1].cpu().numpy()
        input_var = m.addMVar(input_size, lb=lower_bound,
                              ub=upper_bound, name="in_var")

        parameter_list = list(self.net.parameters())
        in_var = input_var
        break_early = False
        i = -2
        for i in range(0, len(parameter_list) - 2, 2):
            if until_layer_neuron is not None and until_layer_neuron[0] == i // 2:
                neuron_index = until_layer_neuron[1]
                in_lb = [bounds_affine_out[int(i / 2)][0].detach().cpu().numpy()[neuron_index]]
                in_ub = [bounds_affine_out[int(i / 2)][1].detach().cpu().numpy()[neuron_index]]
                W, b = parameter_list[i].detach().cpu().numpy()[neuron_index], parameter_list[i + 1].detach().cpu().numpy()[neuron_index]

                out_vars = verbas.add_affine_constr(m, W, b, in_var, in_lb, in_ub, i)

                relu_in_var = out_vars
                out_lb = [bounds_layer_out[int(i / 2)][0].detach().cpu().numpy()[neuron_index]]
                out_ub = [bounds_layer_out[int(i / 2)][1].detach().cpu().numpy()[neuron_index]]
                relu_vars = verbas.add_relu_constr(m, relu_in_var, len(b), in_lb, in_ub, out_lb, out_ub, i=i)
                in_var = relu_vars
                break_early = True
                break

            in_lb = bounds_affine_out[int(i / 2)][0].detach().cpu().numpy()
            in_ub = bounds_affine_out[int(i / 2)][1].detach().cpu().numpy()
            W, b = parameter_list[i].detach().cpu().numpy(), parameter_list[i + 1].detach().cpu().numpy()

            out_vars = verbas.add_affine_constr(m, W, b, in_var, in_lb, in_ub, i)

            relu_in_var = out_vars
            out_lb = bounds_layer_out[int(i / 2)][0].detach().cpu().numpy()
            out_ub = bounds_layer_out[int(i / 2)][1].detach().cpu().numpy()
            relu_vars = verbas.add_relu_constr(m, relu_in_var, len(b), in_lb, in_ub, out_lb, out_ub, i=i)
            in_var = relu_vars

        if not break_early:
            lb = bounds_affine_out[-1][0].detach().cpu().numpy()
            ub = bounds_affine_out[-1][1].detach().cpu().numpy()
            W, b = parameter_list[len(parameter_list) - 2].detach().cpu().numpy(), parameter_list[-1].detach().cpu().numpy()

            out_vars = verbas.add_affine_constr(m, W, b, in_var, lb, ub, i+2)

        m.update()
        self.model = m
        self.output_vars = out_vars
        self.input_vars = input_var

