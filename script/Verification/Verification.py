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


def generate_model_center_eps(model, center, eps, layer_index):
    input_to_previous_layer_size = len(center)
    input_approx_layer = model.addMVar(input_to_previous_layer_size, lb=[elem - eps for elem in center],
                                        ub=[elem + eps for elem in center], name="output_layer_[{}]_".format(layer_index))
    return input_approx_layer


def add_layer_to_model(model, affine_w, affine_b, curr_constraint_icnns, curr_group_indices, curr_bounds_affine_out, curr_bounds_layer_out, curr_fixed_neuron_lower, curr_fixed_neuron_upper, current_layer_index):

    prev_layer_index = current_layer_index - 1
    output_prev_layer = []
    for i in range(affine_w.shape[1]):
        output_prev_layer.append(model.getVarByName("output_layer_[{}]_[{}]".format(prev_layer_index, i)))
    output_prev_layer = grp.MVar.fromlist(output_prev_layer)

    in_lb = curr_bounds_affine_out[0].detach().numpy()
    in_ub = curr_bounds_affine_out[1].detach().numpy()

    out_fet = len(affine_b)
    affine_var = verbas.add_affine_constr(model, affine_w, affine_b, output_prev_layer, in_lb, in_ub, i=current_layer_index)

    out_lb = curr_bounds_layer_out[0].detach().numpy()
    out_ub = curr_bounds_layer_out[1].detach().numpy()
    out_vars = model.addMVar(out_fet, lb=out_lb, ub=out_ub, name="out_var_[{}]".format(current_layer_index))

    for neuron_index in curr_fixed_neuron_upper:
        model.addConstr(out_vars[neuron_index] == 0, name="fixed_upper_{}_{}".format(current_layer_index, neuron_index))

    for neuron_index in curr_fixed_neuron_lower:
        model.addConstr(out_vars[neuron_index] == affine_var[neuron_index], name="fixed_lower_{}_{}".format(current_layer_index, neuron_index))

    not_fixed_neurons = [x for x in range(out_fet) if x not in (curr_fixed_neuron_lower + curr_fixed_neuron_upper)]
    for neuron_index in not_fixed_neurons:
        in_lb = curr_bounds_affine_out[0][neuron_index].item()
        in_ub = curr_bounds_affine_out[1][neuron_index].item()
        model.addConstr(out_vars[neuron_index] >= 0, name="snr_gt0" + str(current_layer_index) + "k" + str(neuron_index))
        model.addConstr(out_vars[neuron_index] >= affine_var[neuron_index],
                        name="snr_gtX" + str(current_layer_index) + "k" + str(neuron_index))
        model.addConstr(out_vars[neuron_index] <= (in_ub * (affine_var[neuron_index] - in_lb)) / (in_ub - in_lb),
                        name="snr_lt" + str(current_layer_index) + "k" + str(neuron_index))


    for k, constraint_icnn in enumerate(curr_constraint_icnns):
        index_to_select = torch.tensor(curr_group_indices[k]).to(device)
        low = torch.index_select(curr_bounds_layer_out[0], 0, index_to_select)
        up = torch.index_select(curr_bounds_layer_out[1], 0, index_to_select)
        constraint_icnn_bounds_affine_out, constraint_icnn_bounds_layer_out = constraint_icnn.calculate_box_bounds([low, up])
        current_in_vars = model.addMVar(len(curr_group_indices[k]), lb=low.detach().numpy(), ub=up.detach().numpy(), name="icnn_var_group_{}_{}".format(current_layer_index, curr_group_indices[k]))


        constraint_icnn.add_max_output_constraints(model, current_in_vars, constraint_icnn_bounds_affine_out,
                                                       constraint_icnn_bounds_layer_out)

        model.addConstrs((out_vars[neuron_index] == current_in_vars[i] for i, neuron_index in enumerate(curr_group_indices[k])), name="group_out_icnn_{}_{}".format(current_layer_index, curr_group_indices[k]))

    in_var = out_vars

    for i, var in enumerate(in_var.tolist()):
        var.setAttr("varname", "output_layer_[{}]_[{}]".format(current_layer_index, i))


def generate_model_A_b(model, a_matrix, b_vector, layer_index):
    input_size = len(b_vector)
    input_var = model.addMVar(input_size, lb=-float('inf'), name="output_layer_[{}]_".format(layer_index))
    model.addMConstr(a_matrix, input_var, "<=", b_vector)


def verification(icnn, model, affine_w, affine_b, index_to_select, curr_bounds_affine_out, curr_bounds_layer_out, prev_layer_index, has_relu=False):

    output_prev_layer = []
    for i in range(affine_w.shape[1]):
        output_prev_layer.append(model.getVarByName("output_layer_[{}]_[{}]".format(prev_layer_index, i)))
    output_prev_layer = grp.MVar.fromlist(output_prev_layer)

    rows_to_delete = list(range(affine_w.shape[0]))
    rows_to_delete = [x for x in rows_to_delete if x not in index_to_select]
    new_w = np.delete(affine_w, rows_to_delete, axis=0)
    new_b = np.delete(affine_b, rows_to_delete, axis=0)

    index_to_select = torch.tensor(index_to_select).to(device)
    in_lb = torch.index_select(curr_bounds_affine_out[0], 0, index_to_select).detach().cpu().numpy()
    in_ub = torch.index_select(curr_bounds_affine_out[1], 0, index_to_select).detach().cpu().numpy()
    out_lb = torch.index_select(curr_bounds_layer_out[0], 0, index_to_select).detach().cpu().numpy()
    out_ub = torch.index_select(curr_bounds_layer_out[1], 0, index_to_select).detach().cpu().numpy()
    affine_out = verbas.add_affine_constr(model, new_w, new_b, output_prev_layer, in_lb, in_ub)
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
        inp = output_prev_layer.getAttr("x")
        return inp, output_var[0].X


def update_bounds_with_icnns(model, bounds_affine_out, bounds_layer_out, current_layer_index, affine_w, affine_b, print_new_bounds=False):

    output_prev_layer = []
    prev_layer_index = current_layer_index - 1
    for i in range(affine_w.shape[1]):
        output_prev_layer.append(model.getVarByName("output_layer_[{}]_[{}]".format(prev_layer_index, i)))
    output_prev_layer = grp.MVar.fromlist(output_prev_layer)

    lb = bounds_affine_out[current_layer_index][0].detach()
    ub = bounds_affine_out[current_layer_index][1].detach()
    affine_var = verbas.add_affine_constr(model, affine_w, affine_b, output_prev_layer, lb, ub)

    model.update()

    for neuron_to_optimize in range(len(affine_var.tolist())):
        model.setObjective(affine_var[neuron_to_optimize], GRB.MINIMIZE)
        model.optimize()
        if model.Status == GRB.OPTIMAL:
            value = affine_var.getAttr("x")
            if print_new_bounds and abs(value[neuron_to_optimize] - bounds_affine_out[current_layer_index][0][neuron_to_optimize]) > 0.00001:
                print("        {}, lower: new {}, old {}".format(neuron_to_optimize, value[neuron_to_optimize], bounds_affine_out[current_layer_index][0][neuron_to_optimize]))
            bounds_affine_out[current_layer_index][0][neuron_to_optimize] = value[neuron_to_optimize]

        model.setObjective(affine_var[neuron_to_optimize], GRB.MAXIMIZE)
        model.optimize()
        if model.Status == GRB.OPTIMAL:
            value = affine_var.getAttr("x")
            if print_new_bounds and abs(value[neuron_to_optimize] - bounds_affine_out[current_layer_index][1][neuron_to_optimize]) > 0.00001:
                print("        {}, upper: new {}, old {}".format(neuron_to_optimize, value[neuron_to_optimize], bounds_affine_out[current_layer_index][1][neuron_to_optimize]))
            bounds_affine_out[current_layer_index][1][neuron_to_optimize] = value[neuron_to_optimize]

    relu_out_lb, relu_out_ub = verbas.calc_relu_out_bound(bounds_affine_out[current_layer_index][0], bounds_affine_out[current_layer_index][1])
    bounds_layer_out[current_layer_index][0] = relu_out_lb
    bounds_layer_out[current_layer_index][1] = relu_out_ub
