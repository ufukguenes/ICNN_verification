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
                                        ub=[elem + eps for elem in center], name="input_approx_layer")
    m.addConstrs(input_approx_layer[i] <= center[i] + eps for i in range(input_to_previous_layer_size))
    m.addConstrs(input_approx_layer[i] >= center[i] - eps for i in range(input_to_previous_layer_size))

    return m


def generate_model_icnns(constraint_icnns, group_indices, last_bounds_layer_out, fixed_neuron_lower, fixed_neuron_upper):
    m = Model()
    m.Params.LogToConsole = 0

    # todo wie kann ich hier wiederverwenden, dass ich das constraint_icnn schon mal verifiziert habe?
    all_indices = [index for group in group_indices for index in group]
    input_size = len(last_bounds_layer_out[0])
    lb = last_bounds_layer_out[0].detach().cpu().numpy()
    ub = last_bounds_layer_out[1].detach().cpu().numpy()
    input_approx_layer = m.addMVar(input_size, lb=lb, ub=ub, name="input_approx_layer")

    for k in range(len(constraint_icnns)):
        index_to_select = torch.tensor(group_indices[k]).to(device)
        low = torch.index_select(last_bounds_layer_out[0], 0, index_to_select)
        up = torch.index_select(last_bounds_layer_out[1], 0, index_to_select)
        constraint_icnn_bounds_affine_out, constraint_icnn_bounds_layer_out = constraint_icnns[k].calculate_box_bounds(
            [low, up])
        current_in_vars = [input_approx_layer[x] for x in group_indices[k]]
        constraint_icnns[k].add_max_output_constraints(m, current_in_vars, constraint_icnn_bounds_affine_out,
                                                      constraint_icnn_bounds_layer_out)

    for neuron_index in fixed_neuron_upper:
        m.addConstr(input_approx_layer[neuron_index] == 0)

    for neuron_index in fixed_neuron_lower:
        m.addConstr(last_bounds_layer_out[0][neuron_index] <= input_approx_layer[neuron_index])
        m.addConstr(input_approx_layer[neuron_index] <= last_bounds_layer_out[1][neuron_index])

    for i, var in enumerate(input_approx_layer.tolist()):
        var.setAttr("varname", "input_approx_layer[{}]".format(i))
    return m


def generate_complete_model_icnn(center, eps, affine_w_list, affine_b_list, constraint_icnns_per_layer, group_indices_per_layer, bounds_affine_out, bounds_layer_out, fixed_neuron_lower_per_layer, fixed_neuron_upper_per_layer):
    m = Model()
    m.Params.LogToConsole = 0

    input_to_previous_layer_size = len(center)
    input_var = m.addMVar(input_to_previous_layer_size, lb=[elem - eps for elem in center],
                                   ub=[elem + eps for elem in center], name="inp_eps")
    m.addConstrs((input_var[i] <= center[i] + eps for i in range(input_to_previous_layer_size)), name="in_const_ub")
    m.addConstrs((input_var[i] >= center[i] - eps for i in range(input_to_previous_layer_size)), name="in_const_lb")

    in_var = input_var
    for i in range(0, len(affine_w_list)):
        in_lb = bounds_affine_out[i][0].detach().numpy()
        in_ub = bounds_affine_out[i][1].detach().numpy()
        W, b = affine_w_list[i].detach().numpy(), affine_b_list[i].detach().numpy()

        out_fet = len(b)
        affine_var = m.addMVar(out_fet, lb=in_lb, ub=in_ub, name="affine_var" + str(i))
        const = m.addConstrs((W[i] @ in_var + b[i] == affine_var[i] for i in range(len(W))), name="affine_const"+str(i))

        out_lb = bounds_layer_out[i][0].detach().numpy()
        out_ub = bounds_layer_out[i][1].detach().numpy()
        out_vars = m.addMVar(out_fet, lb=out_lb, ub=out_ub, name="out_var" + str(i))

        for neuron_index in fixed_neuron_upper_per_layer[i]:
            m.addConstr(out_vars[neuron_index] == 0)

        for neuron_index in fixed_neuron_lower_per_layer[i]:
            m.addConstr(out_vars[neuron_index] == affine_var[neuron_index])


        for k in range(len(constraint_icnns_per_layer[i])):
            constraint_icnns = constraint_icnns_per_layer[i]
            index_to_select = torch.tensor(group_indices_per_layer[i][k]).to(device)
            low = torch.index_select(bounds_layer_out[i][0], 0, index_to_select)
            up = torch.index_select(bounds_layer_out[i][1], 0, index_to_select)
            constraint_icnn_bounds_affine_out, constraint_icnn_bounds_layer_out = constraint_icnns[
                k].calculate_box_bounds(
                [low, up])
            current_in_vars = [out_vars[x] for x in group_indices_per_layer[i][k]]
            constraint_icnns[k].add_max_output_constraints(m, current_in_vars, constraint_icnn_bounds_affine_out,
                                                           constraint_icnn_bounds_layer_out)

        in_var = out_vars

    for i, var in enumerate(in_var.tolist()):
        var.setAttr("varname", "input_approx_layer[{}]".format(i))
    return m


def generate_model_A_b(a_matrix, b_vector):
    m = Model()
    m.Params.LogToConsole = 0
    input_size = len(b_vector)
    input_var = m.addMVar(input_size, lb=-float('inf'), name="input_approx_layer")  # todo mit boxbounds anpassen
    m.addMConstr(a_matrix, input_var, "<=", b_vector)

    return m


def verification(icnn, model, affine_w, b, index_to_select, curr_bounds_affine_out, curr_bounds_layer_out, has_relu=False):

    input_approx_layer = []
    for i in range(affine_w.shape[1]):
        input_approx_layer.append(model.getVarByName("input_approx_layer[{}]".format(i)))
    input_approx_layer = grp.MVar.fromlist(input_approx_layer)

    rows_to_delete = list(range(affine_w.shape[0]))
    rows_to_delete = [x for x in rows_to_delete if x not in index_to_select]
    new_w = np.delete(affine_w, rows_to_delete, axis=0)
    new_b = np.delete(b, rows_to_delete, axis=0)

    index_to_select = torch.tensor(index_to_select).to(device)
    in_lb = torch.index_select(curr_bounds_affine_out[0], 0, index_to_select).detach().cpu().numpy()
    in_ub = torch.index_select(curr_bounds_affine_out[1], 0, index_to_select).detach().cpu().numpy()
    out_lb = torch.index_select(curr_bounds_layer_out[0], 0, index_to_select).detach().cpu().numpy()
    out_ub = torch.index_select(curr_bounds_layer_out[1], 0, index_to_select).detach().cpu().numpy()
    affine_out = verbas.add_affine_constr(model, new_w, new_b, input_approx_layer, in_lb, in_ub)
    if has_relu:
        input_size = len(new_b)
        relu_out = verbas.add_relu_constr(model, affine_out, input_size, in_lb, in_ub, out_lb, out_ub)
        # relu_out = verbas.add_single_neuron_constr(m, affine_out, input_size, in_lb, in_ub, out_lb, out_ub)
        input_var = relu_out
    else:
        input_var = affine_out


    low = torch.index_select(curr_bounds_layer_out[0], 0, index_to_select)
    up = torch.index_select(curr_bounds_layer_out[1], 0, index_to_select)
    icnn_bounds_affine_out, icnn_bounds_layer_out = icnn.calculate_box_bounds([low, up])
    output_var = icnn.add_constraints(model, input_var, icnn_bounds_affine_out, icnn_bounds_layer_out)

    model.update()
    model.setObjective(output_var[0], GRB.MAXIMIZE)

    t = time.time()
    model.optimize()

    if model.Status == GRB.OPTIMAL:
        print("        actual verification time {}".format(time.time() - t))
        inp = input_approx_layer.getAttr("x")
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
        return inp, output_var[0].X


def min_max_of_icnns(model, bounds_affine_out, bounds_layer_out, current_layer_index, affine_w, affine_b):

    input_to_current_layer = []
    for i in range(affine_w.shape[1]):
        input_to_current_layer.append(model.getVarByName("input_approx_layer[{}]".format(i)))
    input_to_current_layer = grp.MVar.fromlist(input_to_current_layer)

    lb = bounds_affine_out[current_layer_index][0].detach()
    ub = bounds_affine_out[current_layer_index][1].detach()
    affine_var = verbas.add_affine_constr(model, affine_w, affine_b, input_to_current_layer, lb, ub)

    model.update()

    for neuron_to_optimize in range(len(affine_var.tolist())):
        model.setObjective(affine_var[neuron_to_optimize], GRB.MINIMIZE)
        model.optimize()
        if model.Status == GRB.OPTIMAL:
            value = affine_var.getAttr("x")
            print("        lower: new {}, old {}".format(value[neuron_to_optimize], bounds_affine_out[current_layer_index][0][neuron_to_optimize]))
            bounds_affine_out[current_layer_index][0][neuron_to_optimize] = value[neuron_to_optimize]

        model.setObjective(affine_var[neuron_to_optimize], GRB.MAXIMIZE)
        model.optimize()
        if model.Status == GRB.OPTIMAL:
            value = affine_var.getAttr("x")
            print("        upper: new {}, old {}".format(value[neuron_to_optimize], bounds_affine_out[current_layer_index][1][neuron_to_optimize]))
            bounds_affine_out[current_layer_index][1][neuron_to_optimize] = value[neuron_to_optimize]

    relu_out_lb, relu_out_ub = verbas.calc_relu_out_bound(bounds_affine_out[current_layer_index][0], bounds_affine_out[current_layer_index][1])
    bounds_layer_out[current_layer_index][0] = relu_out_lb
    bounds_layer_out[current_layer_index][1] = relu_out_ub
    return bounds_affine_out, bounds_affine_out