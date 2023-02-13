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


    def test_feasibility(self, output_sample):
        #t = time.time()

        model = self.model.copy()
        model.setParam("BestObjStop", -0.01)

        num_of_out_fet = len(output_sample)
        out_vars = []
        for i in range(num_of_out_fet):
            out_vars.append(model.getVarByName("last_affine_var[{}]".format(i)))

        """
        output_size = len(self.output_vars.tolist())

        difference = model.addVars(output_size, lb=-float("inf"))
        model.addConstrs(difference[i] == self.output_vars[i] - output_sample[i] for i in range(output_size))

        abs_diff = model.addVars(output_size)
        model.addConstrs(abs_diff[i] == abs_(difference[i]) for i in range(output_size))

        max_var = model.addVar()
        model.addConstr(max_var == grp.max_(abs_diff))"""

        model.addConstrs(out_vars[i] == output_sample[i] for i in range(num_of_out_fet))

        model.update()
        model.setObjective(out_vars[0], grp.GRB.MINIMIZE)

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
        bounds = self.net.calculate_box_bounds([input_flattened.add(-self.eps), input_flattened.add(self.eps)], with_relu=True)

        input_flattened = input_flattened.cpu().numpy()

        input_var = m.addMVar(input_size, lb=-float("inf"), name="in_var")
        m.addConstrs((input_var[i] <= input_flattened[i] + self.eps for i in range(input_size)), name="in_const1")
        m.addConstrs((input_var[i] >= input_flattened[i] - self.eps for i in range(input_size)), name="in_const1")


        parameter_list = list(self.net.parameters())
        in_var = input_var
        for i in range(0, len(parameter_list) - 2, 2):
            lb = bounds[int(i / 2)][0].detach().cpu().numpy()
            ub = bounds[int(i / 2)][1].detach().cpu().numpy()
            W, b = parameter_list[i].detach().cpu().numpy(), parameter_list[i + 1].detach().cpu().numpy()

            out_fet = len(b)
            out_vars = m.addMVar(out_fet, lb=-float("inf"), name="affine_var" + str(i))
            const = m.addConstrs((W[i] @ in_var + b[i] == out_vars[i] for i in range(len(W))), name="affine_const" + str(i))

            relu_in_var = out_vars
            relu_vars = verbas.add_singel_neuron_constr(m, relu_in_var, out_fet, lb, ub, i=i)  # verbas.add_relu_constr(m, relu_in_var, out_fet, lb, ub)
            in_var = relu_vars

        lb = bounds[-1][0].detach().cpu().numpy()
        ub = bounds[-1][1].detach().cpu().numpy()
        W, b = parameter_list[len(parameter_list) - 2].detach().cpu().numpy(), parameter_list[-1].detach().cpu().numpy()

        out_fet = len(b)
        out_vars = m.addMVar(out_fet, lb=-float("inf"), name="last_affine_var")
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
        bounds = self.net.calculate_box_bounds([input_flattened.add(-self.eps), input_flattened.add(self.eps)], with_relu=True)

        input_flattened = input_flattened.numpy()

        input_var = m.addMVar(input_size, lb=-float("inf"), name="in_var")
        m.addConstrs(input_var[i] <= input_flattened[i] + self.eps for i in range(input_size))
        m.addConstrs(input_var[i] >= input_flattened[i] - self.eps for i in range(input_size))

        parameter_list = list(self.net.parameters())
        in_var = input_var
        for i in range(0, len(parameter_list) - 2, 2):
            lb = bounds[int(i / 2)][0].detach().numpy()
            ub = bounds[int(i / 2)][1].detach().numpy()
            W, b = parameter_list[i].detach().numpy(), parameter_list[i + 1].detach().numpy()

            out_fet = len(b)
            out_vars = m.addMVar(out_fet, lb=-float("inf"), name="affine_var" + str(i))
            const = m.addConstrs((W[i] @ in_var + b[i] == out_vars[i] for i in range(len(W))))

            relu_in_var = out_vars
            relu_vars = verbas.add_relu_constr(m, relu_in_var, out_fet, lb, ub)
            in_var = relu_vars

        lb = bounds[-1][0].detach().numpy()
        ub = bounds[-1][1].detach().numpy()
        W, b = parameter_list[len(parameter_list) - 2].detach().numpy(), parameter_list[-1].detach().numpy()

        out_fet = len(b)
        out_vars = m.addMVar(out_fet, lb=-float("inf"), name="last_affine_var")
        const = m.addConstrs((W[i] @ in_var + b[i] == out_vars[i] for i in range(len(W))), name="out_const")
        m.update()
        self.model = m
        self.output_vars = out_vars
        self.input_vars = input_var


class DHOVVerifier(Verifier):
    def __init__(self, icnns, *args, with_affine=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.icnns = icnns
        self.with_affine = with_affine

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
        bounds = self.net.calculate_box_bounds([input_flattened.add(-self.eps), input_flattened.add(self.eps)], with_relu=True)

        input_flattened = input_flattened.cpu().numpy()

        input_var = m.addMVar(input_size, lb=[elem - self.eps for elem in input_flattened],
                              ub=[elem + self.eps for elem in input_flattened], name="in_var")
        m.addConstrs(input_var[i] <= input_flattened[i] + self.eps for i in range(input_size))
        m.addConstrs(input_var[i] >= input_flattened[i] - self.eps for i in range(input_size))

        parameter_list = list(self.net.parameters())
        in_var = input_var
        for i in range(0, len(parameter_list) - 2, 2):
            lb = bounds[int(i / 2)][0].detach().cpu().numpy()
            ub = bounds[int(i / 2)][1].detach().cpu().numpy()
            W, b = parameter_list[i].detach().cpu().numpy(), parameter_list[i + 1].detach().cpu().numpy()

            out_fet = len(b)
            out_vars = m.addMVar(out_fet, lb=lb, ub=ub, name="affine_var" + str(i))
            if self.with_affine:
                const = m.addConstrs((W[i] @ in_var + b[i] == out_vars[i] for i in range(len(W))))

            if i == len(parameter_list) - 2 - 2 or (not self.with_affine):
                self.icnns[i // 2].add_max_output_constraints(m, in_var, self.icnns[i // 2].calculate_box_bounds(None))

            #m.update()
            in_var = out_vars

        lb = bounds[-1][0].detach().cpu().numpy()
        ub = bounds[-1][1].detach().cpu().numpy()
        W, b = parameter_list[len(parameter_list) - 2].detach().cpu().numpy(), parameter_list[-1].detach().cpu().numpy()

        out_fet = len(b)
        out_vars = m.addMVar(out_fet, lb=lb, ub=ub, name="last_affine_var")
        const = m.addConstrs((W[i] @ in_var + b[i] == out_vars[i] for i in range(len(W))), name="out_const")

        m.update()

        self.model = m
        self.output_vars = out_vars
        self.input_vars = input_var
