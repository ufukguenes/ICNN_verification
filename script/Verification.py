import math

import numpy as np
import torch
from gurobipy import Model, GRB, max_, MVar

from script.Networks import ICNN
from script.dataInit import Rhombus


def load(icnn):
    icnn.load_state_dict(torch.load("../convexHullModel.pth"), strict=False)


def verification(icnn, center_eps_W_b=None, A_b=None, icnn_W_b_c=None, has_ReLU=False, sequential=False):
    m = Model()
    input_size = icnn.layer_widths[0]
    output_size = icnn.layer_widths[-1]
    output_var = m.addMVar(output_size, lb=-float('inf'), name="output_var")

    lb = -10000
    ub = 10000

    # A, b = Rhombus().get_A(), Rhombus().get_b()
    # todo hier wird das <= wahrscheinlich nicht als <= sondern nur als < erkannt, Gurobi meckert nämlich auch nicht, wenn man "<k" dahin schreibt
    # m.addMConstr(A, input_var, "<=", b)

    if center_eps_W_b is not None:
        center = center_eps_W_b[0]
        eps = center_eps_W_b[1]
        W = center_eps_W_b[2]
        b = center_eps_W_b[3]

        input_to_previous_layer_size = b.shape
        input_to_previous_layer = m.addMVar(input_to_previous_layer_size, lb=-float('inf'))

        max_vars = m.addVars(input_size, lb=-float('inf'))
        min_vars = m.addVars(input_size, lb=-float('inf'))

        m.addConstrs(max_vars[i] == center[i] + eps for i in range(input_size))
        m.addConstrs(min_vars[i] == center[i] - eps for i in range(input_size))

        m.addConstrs(input_to_previous_layer[i] <= max_vars[i] for i in range(input_size))
        m.addConstrs(input_to_previous_layer[i] >= min_vars[i] for i in range(input_size))

        affine_out = add_affine_constr(m, W, b, input_to_previous_layer, lb, ub)

        if has_ReLU:
            relu_out = add_relu_constr(m, affine_out, input_to_previous_layer_size, lb, ub)
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
        output_of_constraint_icnn = m.addMVar(1, lb=-float('inf'))

        add_non_sequential_constr(m, constraint_icnn, input_to_previous_layer, output_of_constraint_icnn)
        m.addConstr(output_of_constraint_icnn[0] <= c)

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
        add_sequential_constr(m, icnn, input_var, output_var)
        # model_constr = gml.add_sequential_constr(m, icnn, input_var, output_var) # Gurobi api for sequential NN
    else:
        add_non_sequential_constr(m, icnn, input_var, output_var)

    m.update()
    m.setObjective(output_var[0], GRB.MAXIMIZE)
    m.optimize()

    if m.Status == GRB.OPTIMAL:
        inp = input_var.getAttr("x")
        inp = torch.tensor([[inp[0], inp[1]]], dtype=torch.float64)
        true_out = icnn(inp)
        print("optimum solution at: {}, with value {}, true output: {}".format(inp, output_var.getAttr("x"), true_out))

        if center_eps_W_b is not None:
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
            print("output of constraint icnn: {}".format(cons_out))

        """for i in range(1, m.getAttr("SolCount")):
            m.setParam("SolutionNumber", i)
            inp = m.getAttr("Xn")
            inp = [inp[0], inp[1]]
            print("sub-optimal solution at: {}, with value {}".format(inp, m.getAttr("PoolObjVal")))"""
        return inp, output_var.X[0]


def verification_of_normal(model, input, label, eps=0.01, time_limit=None, bound=None):
    m = Model()
    input_size = 32 * 32 * 3
    input_flattened = torch.flatten(input).numpy()

    if time_limit is not None:
        m.setParam("TimeLimit", time_limit)

    if bound is not None:
        m.setParam("BestObjStop", bound)

    input_var = m.addMVar(input_size, lb=-1, ub=1, name="in_var")
    output_var = m.addMVar(10, lb=-10, ub=10, name="output_var")

    m.addConstrs(input_var[i] <= input_flattened[i] + eps for i in range(input_size))

    m.addConstrs(input_var[i] >= input_flattened[i] - eps for i in range(input_size))

    add_sequential_constr(m, model, input_var, output_var)

    difference = m.addVars(9, lb=-float('inf'))
    m.addConstrs(difference[i] == output_var.tolist()[i] - output_var.tolist()[label] for i in range(0, label))
    m.addConstrs(difference[i - 1] == output_var.tolist()[i] - output_var.tolist()[label] for i in range(label + 1, 10))

    # m.addConstrs(difference[i] == output_var.tolist()[i] - output_var.tolist()[label] for i in range(10))
    max_var = m.addVar(lb=-float('inf'), ub=10)
    m.addConstr(max_var == max_(difference))

    m.update()
    m.setObjective(max_var, GRB.MAXIMIZE)
    m.optimize()

    if m.Status == GRB.OPTIMAL or m.Status == GRB.TIME_LIMIT or m.Status == GRB.USER_OBJ_LIMIT:
        inp = input_var.getAttr("x")
        for o in difference.select():
            print(o.getAttr("x"))
        print("optimum solution with value \n {}".format(output_var.getAttr("x")))
        print("max_var {}".format(max_var.getAttr("x")))
        test_inp = torch.tensor([inp], dtype=torch.float64)
        return test_inp, output_var


def add_non_sequential_constr(model, predictor: ICNN, input_vars, output_vars):
    ws = list(predictor.ws.parameters())
    us = list(predictor.us.parameters())

    in_var = input_vars
    # todo lower und upper bounds für variablen und relu berechnen
    lb = -100000
    ub = 100000
    for i in range(0, len(ws), 2):
        affine_W, affine_b = ws[i].detach().numpy(), ws[i + 1].detach().numpy()

        out_fet = len(affine_b)
        affine_var = model.addMVar(out_fet, lb=lb, ub=ub, name="affine_var" + str(i))
        out_vars = model.addMVar(out_fet, lb=lb, ub=ub, name="affine_skip_var" + str(i))

        affine_const = model.addConstrs(
            affine_W[i] @ in_var + affine_b[i] == affine_var[i] for i in range(len(affine_W)))
        if i != 0:
            k = math.floor(i / 2) - 1
            skip_W = torch.clone(us[k]).detach().numpy()  # has no bias
            skip_var = model.addMVar(out_fet, lb=lb, ub=ub, name="skip_var" + str(k))
            skip_const = model.addConstrs(skip_W[i] @ input_vars == skip_var[i] for i in range(len(affine_W)))
            affine_skip_cons = model.addConstrs(
                affine_var[i] + skip_var[i] == out_vars[i] for i in range(len(affine_W)))
        else:
            affine_no_skip_cons = model.addConstrs(affine_var[i] == out_vars[i] for i in range(len(affine_W)))

        in_var = out_vars

        if i < len(ws) - 2:
            relu_vars = add_relu_constr(model, in_var, out_fet, lb, ub, i)
            in_var = relu_vars
            out_vars = relu_vars

    const = model.addConstrs(out_vars[i] == output_vars[i] for i in range(out_fet))


def add_sequential_constr(model, predictor: Model, input_vars, output_vars):
    parameter_list = list(predictor.parameters())

    in_var = input_vars
    # todo lower und upper bounds für variablen und relu berechnen
    lb = -100
    ub = 100
    for i in range(0, len(parameter_list), 2):
        W, b = parameter_list[i].detach().numpy(), parameter_list[i + 1].detach().numpy()

        out_fet = len(b)
        out_vars = model.addMVar(out_fet, lb=lb, ub=ub, name="affine_var" + str(i))
        const = model.addConstrs((W[i] @ in_var + b[i] == out_vars[i] for i in range(len(W))))
        in_var = out_vars

        if i < len(parameter_list) - 2:
            relu_vars = add_relu_constr(model, in_var, out_fet, lb, ub, i)
            in_var = relu_vars
            out_vars = relu_vars

    const = model.addConstrs(out_vars[i] == output_vars[i] for i in range(out_fet))


def add_affine_constr(model, W, b, input_vars, lb, ub, i=0):
    out_fet = len(b)
    out_vars = model.addMVar(out_fet, lb=lb, ub=ub, name="affine_var" + str(i))
    const = model.addConstrs((W[i] @ input_vars + b[i] == out_vars[i] for i in range(len(W))))
    return out_vars


def add_relu_constr(model, input_vars, number_of_out_features, lb, ub, i=0):
    a = model.addMVar(number_of_out_features, vtype=GRB.BINARY, name="a" + str(i))
    relu_vars = model.addMVar(number_of_out_features, lb=lb, ub=ub, name="relu_var" + str(i))
    model.update()
    relu_const_0 = model.addConstrs((var >= 0) for var in relu_vars.tolist())
    relu_const_1 = model.addConstrs((var >= x) for var, x in zip(relu_vars.tolist(), input_vars.tolist()))
    relu_const_if1 = model.addConstrs((var <= a_i * ub) for var, a_i in zip(relu_vars.tolist(), a.tolist()))
    relu_const_if0 = model.addConstrs(
        (var <= x - lb * (1 - a_i)) for var, x, a_i in zip(relu_vars.tolist(), input_vars.tolist(), a.tolist()))

    """
    # Variante 2: Füge Constraints für jede Komponente im Vektor einzeln hinzu
    mvars = []
    for k, relu_in in enumerate(in_var.tolist()):
        a = model.addVar(vtype=GRB.BINARY, name="a" + str(i) + str(k))
        relu_var = model.addVar(lb=lb, ub=ub, name="relu_var" + str(i) + str(k))
        r1 = model.addConstr(relu_var >= 0)
        r2 = model.addConstr(relu_var >= relu_in)
        r3 = model.addConstr(relu_var <= relu_in - lb * (1 - a))
        r4 = model.addConstr(relu_var <= ub * a)
        mvars.append(relu_var)
    in_var = MVar(mvars)
    out_vars = MVar(mvars)    
    """

    """
    # Variante 3: Nutze Gurobi max Constraints
    relu_vars = model.addMVar(out_fet, lb=lb, ub=ub, name="relu_var"+str(i))
    for var, x in zip(relu_vars.tolist(), in_var.tolist()):
        const = model.addConstr(var == max_(0, x))
    """

    return relu_vars
