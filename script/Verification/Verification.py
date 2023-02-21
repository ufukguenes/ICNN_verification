import math
import time

import numpy as np
import torch
from gurobipy import Model, GRB, max_
import gurobipy as grp
import script.Verification.VerificationBasics as verbas

from script.settings import device, data_type


def load(icnn):
    icnn.load_state_dict(torch.load("../convexHullModel.pth"), strict=False)


def generate_model_center_eps(center, eps):
    m = Model()
    m.Params.LogToConsole = 0

    input_to_previous_layer_size = len(center)
    input_approx_layer = m.addMVar(input_to_previous_layer_size, lb=[elem - eps for elem in center],
                                        ub=[elem + eps for elem in center])
    m.addConstrs(input_approx_layer[i] <= center[i] + eps for i in range(input_to_previous_layer_size))
    m.addConstrs(input_approx_layer[i] >= center[i] - eps for i in range(input_to_previous_layer_size))

    for i, var in enumerate(input_approx_layer.tolist()):
        var.setAttr("varname", "input_approx_layer"+str(i))
    return m


def generate_model_icnns(constraint_icnns, all_from_to_neurons, current_layer_index, bounds_layer_out):
    m = Model()
    m.Params.LogToConsole = 0

    # todo wie kann ich hier wiederverwenden, dass ich das constraint_icnn schon mal verifiziert habe?
    input_size = len(bounds_layer_out[current_layer_index][0])
    lb = bounds_layer_out[current_layer_index - 1][0].detach().cpu().numpy()
    ub = bounds_layer_out[current_layer_index - 1][1].detach().cpu().numpy()
    input_approx_layer = m.addMVar(input_size, lb=lb, ub=ub)

    for k in range(len(constraint_icnns)):
        low = bounds_layer_out[current_layer_index - 1][0][all_from_to_neurons[k][0]: all_from_to_neurons[k][1]]
        up = bounds_layer_out[current_layer_index - 1][1][all_from_to_neurons[k][0]: all_from_to_neurons[k][1]]
        constraint_icnn_bounds_affine_out, constraint_icnn_bounds_layer_out = constraint_icnns[k].calculate_box_bounds(
            [low, up])
        constraint_icnns[k].add_max_output_constraints(m,
                                                      input_approx_layer[all_from_to_neurons[k][0]: all_from_to_neurons[k][1]],
                                                      constraint_icnn_bounds_affine_out,
                                                      constraint_icnn_bounds_layer_out)

    for i, var in enumerate(input_approx_layer.tolist()):
        var.setAttr("varname", "input_approx_layer" + str(i))
    return m


def generate_model_A_b(a_matrix, b_vector):
    m = Model()
    m.Params.LogToConsole = 0
    input_size = len(b_vector)
    input_var = m.addMVar(input_size, lb=-float('inf'), name="in_var")  # todo mit boxbounds anpassen
    m.addMConstr(a_matrix, input_var, "<=", b_vector)

    for i, var in enumerate(input_var.tolist()):
        var.setAttr("varname", "input_approx_layer" + str(i))
    return m


def verification(icnn, model, affine_w, b, from_to_neuron, current_layer_index, bounds_affine_out, bounds_layer_out, has_relu=False):

    input_approx_layer = []
    for i in range(affine_w.shape[1]):
        input_approx_layer.append(model.getVarByName("input_approx_layer"+str(i)))
    input_approx_layer = grp.MVar.fromlist(input_approx_layer)

    rows_to_delete = list(range(affine_w.shape[0]))
    rows_to_delete = rows_to_delete[:from_to_neuron[0]] + rows_to_delete[from_to_neuron[1]:]
    affine_w = np.delete(affine_w, rows_to_delete, axis=0)
    b = np.delete(b, rows_to_delete, axis=0)

    in_lb = bounds_affine_out[current_layer_index][0].detach().cpu().numpy()[from_to_neuron[0]: from_to_neuron[1]]
    in_ub = bounds_affine_out[current_layer_index][1].detach().cpu().numpy()[from_to_neuron[0]: from_to_neuron[1]]
    out_lb = bounds_layer_out[current_layer_index][0].detach().cpu().numpy()[from_to_neuron[0]: from_to_neuron[1]]
    out_ub = bounds_layer_out[current_layer_index][1].detach().cpu().numpy()[from_to_neuron[0]: from_to_neuron[1]]
    affine_out = verbas.add_affine_constr(model, affine_w, b, input_approx_layer, in_lb, in_ub)
    if has_relu:
        input_size = len(b)
        relu_out = verbas.add_relu_constr(model, affine_out, input_size, in_lb, in_ub, out_lb, out_ub)
        # relu_out = verbas.add_single_neuron_constr(m, affine_out, input_size, in_lb, in_ub, out_lb, out_ub)
        input_var = relu_out
    else:
        input_var = affine_out


    low = bounds_layer_out[current_layer_index][0][from_to_neuron[0]: from_to_neuron[1]]
    up = bounds_layer_out[current_layer_index][1][from_to_neuron[0]: from_to_neuron[1]]
    icnn_bounds_affine_out, icnn_bounds_layer_out = icnn.calculate_box_bounds([low, up])
    output_var = icnn.add_constraints(model, input_var, icnn_bounds_affine_out, icnn_bounds_layer_out)

    model.update()
    model.setObjective(output_var[0], GRB.MAXIMIZE)

    t = time.time()
    model.optimize()

    if model.Status == GRB.OPTIMAL:
        print("        actual verification time {}".format(time.time() - t))
        inp = input_var.getAttr("x")
        # inp = torch.tensor([[inp[0], inp[1]]], dtype=data_type).to(device)
        # true_out = icnn(inp)
        # print("optimum solution at: {}, with value {}, true output: {}".format(inp, output_var.getAttr("x"), 0))  #

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
