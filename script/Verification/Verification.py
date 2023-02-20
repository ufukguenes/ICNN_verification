import math

import torch
from gurobipy import Model, GRB, max_
import script.Verification.VerificationBasics as verbas

from script.Verification.VerificationBasics import add_affine_constr, add_relu_constr
from script.settings import device, data_type

def load(icnn):
    icnn.load_state_dict(torch.load("../convexHullModel.pth"), strict=False)


def verification(icnn, from_to_neuron, current_layer_index, bounds_affine_out, bounds_layer_out, center_eps_w_b=None, a_b=None, icnn_w_b_c=None, has_relu=False):
    m = Model()

    if center_eps_w_b is not None:
        center = center_eps_w_b[0]
        eps = center_eps_w_b[1]
        affine_w = center_eps_w_b[2]
        b = center_eps_w_b[3]
        input_size = len(b)

        input_to_previous_layer_size = affine_w.shape[1]
        input_to_previous_layer = m.addMVar(input_to_previous_layer_size, lb=[elem - eps for elem in center],
                              ub=[elem + eps for elem in center])

        m.addConstrs(input_to_previous_layer[i] <= center[i] + eps for i in range(input_to_previous_layer_size))
        m.addConstrs(input_to_previous_layer[i] >= center[i] - eps for i in range(input_to_previous_layer_size))

        in_lb = bounds_affine_out[current_layer_index][0].detach().cpu().numpy()
        in_ub = bounds_affine_out[current_layer_index][1].detach().cpu().numpy()
        out_lb = bounds_layer_out[current_layer_index][0].detach().cpu().numpy()
        out_ub = bounds_layer_out[current_layer_index][1].detach().cpu().numpy()

        affine_out = add_affine_constr(m, affine_w, b, input_to_previous_layer, in_lb, in_ub)

        if has_relu:
            relu_out = add_relu_constr(m, affine_out, input_size, in_lb, in_ub, out_lb, out_ub)
            input_var = relu_out
        else:
            input_var = affine_out

    elif a_b is not None:
        A = a_b[0]
        b = a_b[1]
        input_size = len(b)
        input_var = m.addMVar(input_size, lb=-float('inf'), name="in_var") # todo mit boxbounds anpassen
        m.addMConstr(A, input_var, "<=", b)

    elif icnn_w_b_c is not None:
        constraint_icnn = icnn_w_b_c[0]
        # affine_w und b bilden die affine Transformation des Layers, dass gerade verifiziert/ vergrößert werden soll
        affine_w = icnn_w_b_c[1]
        b = icnn_w_b_c[2]
        input_size = len(b)
        input_var = m.addMVar(input_size, lb=-float('inf'), name="in_var")

        # todo wie kann ich hier wiederverwenden, dass ich das constraint_icnn schon mal verifiziert habe?
        constraint_icnn_input_size = len(affine_w)
        input_to_previous_layer = m.addMVar(constraint_icnn_input_size, lb=-float('inf'))

        for k in range(len(constraint_icnn)):
            low = bounds_layer_out[current_layer_index - 1][0][from_to_neuron[0]: from_to_neuron[1]]
            up = bounds_layer_out[current_layer_index - 1][1][from_to_neuron[0]: from_to_neuron[1]]
            constraint_icnn_bounds_affine_out, constraint_icnn_bounds_layer_out = constraint_icnn[k].calculate_box_bounds([low, up])
            constraint_icnn[k].add_max_output_constraints(m, input_to_previous_layer[from_to_neuron[0]: from_to_neuron[1]], constraint_icnn_bounds_affine_out, constraint_icnn_bounds_layer_out)

        if has_relu:
            relu_var = m.addMVar(input_size, lb=-float('inf'), name="in_var")
            m.addConstrs(affine_w[i] @ input_to_previous_layer + b[i] == relu_var[i] for i in range(len(affine_w)))
            m.addConstrs(input_var[i] == max_(0, relu_var[i]) for i in range(input_size))
        else:
            m.addConstrs(affine_w[i] @ input_to_previous_layer + b[i] == input_var[i] for i in range(len(affine_w)))

    else:
        return

    # todo das kann man optimieren in dem man die constraints, nur für die relevanten neuronen erstellt und nicht für alle
    input_var = input_var[from_to_neuron[0]: from_to_neuron[1]]

    low = bounds_layer_out[current_layer_index][0][from_to_neuron[0]: from_to_neuron[1]]
    up = bounds_layer_out[current_layer_index][1][from_to_neuron[0]: from_to_neuron[1]]
    icnn_bounds_affine_out, icnn_bounds_layer_out = icnn.calculate_box_bounds([low, up])
    output_var = icnn.add_constraints(m, input_var, icnn_bounds_affine_out, icnn_bounds_layer_out)

    m.update()
    m.setObjective(output_var[0], GRB.MAXIMIZE)
    m.optimize()

    if m.Status == GRB.OPTIMAL:
        inp = input_var.getAttr("x")
        # inp = torch.tensor([[inp[0], inp[1]]], dtype=data_type).to(device)
        # true_out = icnn(inp)
        print("optimum solution at: {}, with value {}, true output: {}".format(inp, output_var.getAttr("x"), 0))  #

        """if center_eps_W_b is not None:
            input = input_to_previous_layer.getAttr("x")
            affine_verification_out = affine_out.getAttr("x")
            relu_inp = relu_out.getAttr("x")
            print("input for constraint icnn: {}, affine transform: {}, relu transform: {}".format(input, affine_verification_out, relu_inp))
            input = torch.tensor([[input[0], input[1]]], dtype=data_type).to(device)
            center = center_eps_W_b[0]
            eps = center_eps_W_b[1]
            affine_w =  torch.tensor(center_eps_W_b[2], dtype=data_type).to(device)
            b =  torch.tensor(center_eps_W_b[3], dtype=data_type).to(device)
            affine_out = torch.matmul(affine_w, input[0]) + b
            print("affine output = {}".format(affine_out))

        if icnn_W_b_c is not None:
            constraint_icnn_input = input_to_previous_layer.getAttr("x")
            #affine_inp = affine_out.getAttr("x")
            #relu_inp = relu_out.getAttr("x")
            #print("input for constraint icnn: {}, affine transform: {}, relu transform: {}".format(constraint_icnn_input, affine_inp, relu_inp))
            print("input for constraint icnn: {}".format(constraint_icnn_input))
            constraint_icnn_input = torch.tensor([[constraint_icnn_input[0], constraint_icnn_input[1]]], dtype=data_type).to(device)
            constraint_icnn = icnn_W_b_c[0]
            affine_w = torch.tensor(icnn_W_b_c[1], dtype=data_type).to(device)
            b = torch.tensor(icnn_W_b_c[2], dtype=data_type).to(device)
            c = icnn_W_b_c[3]
            cons_out = constraint_icnn(constraint_icnn_input)
            #affine_out = torch.matmul(affine_w, constraint_icnn_input[0]) + b
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
        bounds = verbas.calculate_box_bounds(icnn, input_bounds, with_relu=True)
        verbas.add_constr_for_sequential_icnn(m, icnn, input_var, output_var, bounds)
    else:
        bounds = verbas.calculate_box_bounds(icnn, input_bounds, is_sequential=False, with_relu=True)
        verbas.add_constr_for_non_sequential_icnn(m, icnn, input_var, output_var, bounds)

    m.update()
    m.setObjective(output_var[0], GRB.MINIMIZE)
    m.optimize()

    if m.Status == GRB.OPTIMAL:
        inp = input_var.getAttr("x")
        print("optimum solution at: {}, with value {}, true output: {}".format(inp, output_var.getAttr("x"), 0))  #
        return inp, output_var.X[0]
