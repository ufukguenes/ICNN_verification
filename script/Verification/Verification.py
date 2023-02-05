import math

import torch
from gurobipy import Model, GRB, max_
import script.Verification.VerificationBasics as verbas

from script.Verification.VerificationBasics import add_affine_constr, add_relu_constr


def load(icnn):
    icnn.load_state_dict(torch.load("../convexHullModel.pth"), strict=False)


def verification(icnn, center_eps_W_b=None, A_b=None, icnn_W_b_c=None, has_ReLU=False, sequential=False, use_logical_bound=False):
    m = Model()

    #m.Params.LogToConsole = 0

    input_size = icnn.layer_widths[0]
    output_size = icnn.layer_widths[-1]
    output_var = m.addMVar(output_size, lb=-float('inf'), name="output_var")


    # A, b = Rhombus().get_A(), Rhombus().get_b()
    # todo hier wird das <= wahrscheinlich nicht als <= sondern nur als < erkannt, Gurobi meckert nämlich auch nicht, wenn man "<k" dahin schreibt
    # m.addMConstr(A, input_var, "<=", b)

    if center_eps_W_b is not None:
        center = center_eps_W_b[0]
        eps = center_eps_W_b[1]
        W = center_eps_W_b[2]
        b = center_eps_W_b[3]

        input_to_previous_layer_size = W.shape[1]
        input_to_previous_layer = m.addMVar(input_to_previous_layer_size, lb=-float('inf'))

        lb = [-10000 for i in range(input_to_previous_layer_size)]
        ub = [10000 for i in range(input_to_previous_layer_size)] #todo mit boxbounds anpassen

        max_vars = m.addVars(input_to_previous_layer_size, lb=-float('inf'))
        min_vars = m.addVars(input_to_previous_layer_size, lb=-float('inf'))

        m.addConstrs(max_vars[i] == center[i] + eps for i in range(input_to_previous_layer_size))
        m.addConstrs(min_vars[i] == center[i] - eps for i in range(input_to_previous_layer_size))

        m.addConstrs(input_to_previous_layer[i] <= max_vars[i] for i in range(input_to_previous_layer_size))
        m.addConstrs(input_to_previous_layer[i] >= min_vars[i] for i in range(input_to_previous_layer_size))

        affine_out = add_affine_constr(m, W, b, input_to_previous_layer, lb, ub)

        if has_ReLU:
            relu_out = add_relu_constr(m, affine_out, input_size, lb, ub)
            input_var = relu_out
        else:
            input_var = affine_out


    elif A_b is not None:
        input_var = m.addMVar(input_size, lb=-float('inf'), name="in_var")
        A = A_b[0]
        b = A_b[1]
        m.addMConstr(A, input_var, "<=", b)

    elif icnn_W_b_c is not None:
        constraint_icnn = icnn_W_b_c[0]
        input_var = m.addMVar(input_size, lb=-float('inf'), name="in_var")
        # W und b bilden die affine Transformation des Layers, dass gerade verifiziert/ vergrößert werden soll
        W = icnn_W_b_c[1]
        b = icnn_W_b_c[2]
        c = icnn_W_b_c[3]

        # todo wie kann ich hier wiederverwenden, dass ich das constraint_icnn schon mal verifiziert habe?
        constraint_icnn_input_size = constraint_icnn.layer_widths[0]
        input_to_previous_layer = m.addMVar(constraint_icnn_input_size, lb=-float('inf'))
        output_of_layer_approx = m.addMVar(1, lb=-float('inf'))

        bounds = verbas.calculate_box_bounds(constraint_icnn, None, is_sequential=False)
        verbas.add_constr_for_non_sequential_icnn(m, constraint_icnn, input_to_previous_layer, output_of_layer_approx, bounds)

        if use_logical_bound:
            output_of_and = m.addMVar(1, lb=-float('inf'))
            verbas.add_constr_and_logic(m, constraint_icnn, input_to_previous_layer, output_of_layer_approx, output_of_and, bounds)
            output_of_layer_approx = output_of_and

        m.addConstr(output_of_layer_approx[0] <= 0) # todo 0 ersetzen mit c?

        """affine_out = add_affine_constr(m, W, b, input_to_previous_layer, lb, ub)
        if has_ReLU:
            relu_out = add_relu_constr(m, affine_out, constraint_icnn_input_size, lb, ub)
            input_var = relu_out
        else:
            input_var = affine_out"""

        if has_ReLU:
            relu_var = m.addMVar(input_size, lb=-float('inf'), name="in_var")
            m.addConstrs(W[i] @ input_to_previous_layer + b[i] == relu_var[i] for i in range(len(W)))
            m.addConstrs(input_var[i] == max_(0, relu_var[i]) for i in range(input_size))
        else:
            m.addConstrs(W[i] @ input_to_previous_layer + b[i] == input_var[i] for i in range(len(W)))

    else:
        return

    if sequential:
        bounds = verbas.calculate_box_bounds(icnn, None)
        verbas.add_constr_for_sequential_icnn(m, icnn, input_var, output_var, bounds)
        # model_constr = gml.add_sequential_constr(m, icnn, input_var, output_var) # Gurobi api for sequential NN
    else:
        bounds = verbas.calculate_box_bounds(icnn, None, is_sequential=False)
        verbas.add_constr_for_non_sequential_icnn(m, icnn, input_var, output_var, bounds)

    m.update()
    m.setObjective(output_var[0], GRB.MAXIMIZE)
    m.optimize()

    if m.Status == GRB.OPTIMAL:
        inp = input_var.getAttr("x")
        #inp = torch.tensor([[inp[0], inp[1]]], dtype=torch.float64)
        #true_out = icnn(inp)
        print("optimum solution at: {}, with value {}, true output: {}".format(inp, output_var.getAttr("x"), 0)) #

        """if center_eps_W_b is not None:
            input = input_to_previous_layer.getAttr("x")
            affine_verification_out = affine_out.getAttr("x")
            relu_inp = relu_out.getAttr("x")
            print("input for constraint icnn: {}, affine transform: {}, relu transform: {}".format(input, affine_verification_out, relu_inp))
            input = torch.tensor([[input[0], input[1]]], dtype=torch.float64)
            center = center_eps_W_b[0]
            eps = center_eps_W_b[1]
            W =  torch.tensor(center_eps_W_b[2], dtype=torch.float64)
            b =  torch.tensor(center_eps_W_b[3], dtype=torch.float64)
            affine_out = torch.matmul(W, input[0]) + b
            print("affine output = {}".format(affine_out))

        if icnn_W_b_c is not None:
            constraint_icnn_input = input_to_previous_layer.getAttr("x")
            #affine_inp = affine_out.getAttr("x")
            #relu_inp = relu_out.getAttr("x")
            #print("input for constraint icnn: {}, affine transform: {}, relu transform: {}".format(constraint_icnn_input, affine_inp, relu_inp))
            print("input for constraint icnn: {}".format(constraint_icnn_input))
            constraint_icnn_input = torch.tensor([[constraint_icnn_input[0], constraint_icnn_input[1]]], dtype=torch.float64)
            constraint_icnn = icnn_W_b_c[0]
            W = torch.tensor(icnn_W_b_c[1], dtype=torch.float64)
            b = torch.tensor(icnn_W_b_c[2], dtype=torch.float64)
            c = icnn_W_b_c[3]
            cons_out = constraint_icnn(constraint_icnn_input)
            #affine_out = torch.matmul(W, constraint_icnn_input[0]) + b
            print("output of constraint icnn: {}".format(cons_out))"""

        """for i in range(1, m.getAttr("SolCount")):
            m.setParam("SolutionNumber", i)
            inp = m.getAttr("Xn")
            inp = [inp[0], inp[1]]
            print("sub-optimal solution at: {}, with value {}".format(inp, m.getAttr("PoolObjVal")))"""
        return inp, output_var.X[0]


def find_minima(icnn, sequential=False, input_bounds=None):
    m = Model()

    # m.Params.LogToConsole = 0

    input_size = icnn.layer_widths[0]
    output_size = icnn.layer_widths[-1]
    input_var = m.addMVar(input_size, lb=-float('inf'), name="in_var")
    output_var = m.addMVar(output_size, lb=-float('inf'), name="output_var")

    if sequential:
        bounds = verbas.calculate_box_bounds(icnn, input_bounds)
        verbas.add_constr_for_sequential_icnn(m, icnn, input_var, output_var, bounds)
    else:
        bounds = verbas.calculate_box_bounds(icnn, input_bounds, is_sequential=False)
        verbas.add_constr_for_non_sequential_icnn(m, icnn, input_var, output_var, bounds)

    m.update()
    m.setObjective(output_var[0], GRB.MINIMIZE)
    m.optimize()

    if m.Status == GRB.OPTIMAL:
        inp = input_var.getAttr("x")
        print("optimum solution at: {}, with value {}, true output: {}".format(inp, output_var.getAttr("x"), 0))  #
        return inp, output_var.X[0]