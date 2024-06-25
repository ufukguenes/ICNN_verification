import math
import time
import warnings

import numpy as np
import torch
from gurobipy import GRB
import gurobipy as grp
import script.Verification.VerificationBasics as verbas

from script.settings import device, data_type
import torch.multiprocessing


def generate_model_center_bounds(model, center, input_bounds, layer_index):
    input_to_previous_layer_size = len(center)
    lower_bound = input_bounds[0].cpu().numpy()
    upper_bound = input_bounds[1].cpu().numpy()
    input_approx_layer = model.addMVar(input_to_previous_layer_size, lb=lower_bound,
                                        ub=upper_bound, name="output_layer_[{}]_".format(layer_index))
    return input_approx_layer


def add_layer_to_model(model, affine_w, affine_b, curr_constraint_icnns, curr_group_indices, curr_bounds_affine_out, curr_bounds_layer_out, curr_fixed_neuron_lower, curr_fixed_neuron_upper, current_layer_index):

    prev_layer_index = current_layer_index - 1
    output_prev_layer = []
    for i in range(affine_w.shape[1]):
        output_prev_layer.append(model.getVarByName("output_layer_[{}]_[{}]".format(prev_layer_index, i)))
    output_prev_layer = grp.MVar.fromlist(output_prev_layer)

    in_lb = curr_bounds_affine_out[0].detach().cpu().numpy()
    in_ub = curr_bounds_affine_out[1].detach().cpu().numpy()

    out_fet = len(affine_b)
    affine_var = verbas.add_affine_constr(model, affine_w, affine_b, output_prev_layer, in_lb, in_ub, i=current_layer_index)

    out_lb = curr_bounds_layer_out[0].detach().cpu().numpy()
    out_ub = curr_bounds_layer_out[1].detach().cpu().numpy()
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
        current_in_vars = model.addMVar(len(curr_group_indices[k]), lb=low.detach().cpu().numpy(), ub=up.detach().cpu().numpy(), name="icnn_var_group_{}_{}".format(current_layer_index, curr_group_indices[k]))

        constraint_icnn.add_max_output_constraints(model, current_in_vars, constraint_icnn_bounds_affine_out,
                                                   constraint_icnn_bounds_layer_out)

        model.addConstrs((out_vars[neuron_index] == current_in_vars[i] for i, neuron_index in enumerate(curr_group_indices[k])), name="group_out_icnn_{}_{}".format(current_layer_index, curr_group_indices[k]))

    in_var = out_vars

    for i, var in enumerate(in_var.tolist()):
        var.setAttr("varname", "output_layer_[{}]_[{}]".format(current_layer_index, i))


def verification(icnn, model, affine_w, affine_b, index_to_select, curr_bounds_affine_out, curr_bounds_layer_out, prev_layer_index, has_relu=False, relu_as_lp=False, icnn_as_lp=False):
    back_up_model = model.copy()
    output_prev_layer = []
    for i in range(affine_w.shape[1]):
        output_prev_layer.append(model.getVarByName("output_layer_[{}]_[{}]".format(prev_layer_index, i)))
    output_prev_layer = grp.MVar.fromlist(output_prev_layer)

    rows_to_delete = list(range(affine_w.shape[0]))
    rows_to_delete = [x for x in rows_to_delete if x not in index_to_select]
    new_w = np.delete(affine_w, rows_to_delete, axis=0)
    new_b = np.delete(affine_b, rows_to_delete, axis=0)

    index_to_select_torch = torch.tensor(index_to_select).to(device)
    in_lb = torch.index_select(curr_bounds_affine_out[0], 0, index_to_select_torch).detach().cpu().numpy()
    in_ub = torch.index_select(curr_bounds_affine_out[1], 0, index_to_select_torch).detach().cpu().numpy()
    out_lb = torch.index_select(curr_bounds_layer_out[0], 0, index_to_select_torch).detach().cpu().numpy()
    out_ub = torch.index_select(curr_bounds_layer_out[1], 0, index_to_select_torch).detach().cpu().numpy()
    affine_out = verbas.add_affine_constr(model, new_w, new_b, output_prev_layer, in_lb, in_ub)
    if has_relu:
        input_size = len(new_b)
        if relu_as_lp:
            relu_out = verbas.add_single_neuron_constr(model, affine_out, input_size, in_lb, in_ub, out_lb, out_ub)
        else:
            relu_out = verbas.add_relu_constr(model, affine_out, input_size, in_lb, in_ub, out_lb, out_ub)
        input_var = relu_out
    else:
        input_var = affine_out


    low = torch.index_select(curr_bounds_layer_out[0], 0, index_to_select_torch)
    up = torch.index_select(curr_bounds_layer_out[1], 0, index_to_select_torch)
    icnn_bounds_affine_out, icnn_bounds_layer_out = icnn.calculate_box_bounds([low, up])
    output_var = icnn.add_constraints(model, input_var, icnn_bounds_affine_out, icnn_bounds_layer_out, as_lp=icnn_as_lp)

    model.update()
    model.setObjective(output_var[0], GRB.MAXIMIZE)

    t = time.time()
    model.optimize()

    if model.Status == GRB.OPTIMAL:
        # print("        actual verification time {}".format(time.time() - t))
        inp = output_prev_layer.getAttr("x")
        return inp, output_var[0].X
    elif model.Status == GRB.TIME_LIMIT:
        # print("        actual verification time (time limit) {}".format(time.time() - t))
        inp = None

        if model.SolCount == 0:
            _, c = verification(icnn, back_up_model, affine_w, affine_b, index_to_select, curr_bounds_affine_out, curr_bounds_layer_out, prev_layer_index, has_relu=has_relu, relu_as_lp=True, icnn_as_lp=True)
            # print("            fall back to lp solving of icnn enlargement")
        else:
            c = model.ObjBound

        # print("            enlarge with {}".format(c))
        return inp, c
    elif model.Status == GRB.INFEASIBLE:
        raise RuntimeError("Model is infeasible")


def update_bounds_with_icnns(model, bounds_affine_out, bounds_layer_out, current_layer_index, affine_w, affine_b):

    output_prev_layer = []
    prev_layer_index = current_layer_index - 1
    for i in range(affine_w.shape[1]):
        output_prev_layer.append(model.getVarByName("output_layer_[{}]_[{}]".format(prev_layer_index, i)))
    output_prev_layer = grp.MVar.fromlist(output_prev_layer)

    lb = bounds_affine_out[current_layer_index][0].detach()
    ub = bounds_affine_out[current_layer_index][1].detach()
    affine_var = verbas.add_affine_constr(model, affine_w, affine_b, output_prev_layer, lb.cpu(), ub.cpu(), i="myVarWithNewName")

    model.setParam(GRB.Param.Threads, 2)
    model.update()

    num_processors = torch.multiprocessing.cpu_count()
    Cache.model = model
    Cache.affine_var_name = list(affine_var.VarName)
    neuron_indices = list(range(len(affine_var.tolist())))
    with torch.multiprocessing.Pool(num_processors) as pool:
        result = pool.map(parallel_call, neuron_indices)

    for neuron_to_optimize in range(len(affine_var.tolist())):
        bounds_affine_out[current_layer_index][0][neuron_to_optimize] = result[neuron_to_optimize][0]
        bounds_affine_out[current_layer_index][1][neuron_to_optimize] = result[neuron_to_optimize][1]

    relu_out_lb, relu_out_ub = verbas.calc_relu_out_bound(bounds_affine_out[current_layer_index][0],
                                                          bounds_affine_out[current_layer_index][1])

    bounds_layer_out[current_layer_index][0] = relu_out_lb
    bounds_layer_out[current_layer_index][1] = relu_out_ub
    return True



class Cache:
    model = None
    affine_var_name = None


def parallel_call(neuron_index):

    model = Cache.model.copy()
    affine_var = model.getVarByName(Cache.affine_var_name[neuron_index])

    lb_ub = []
    model.setObjective(affine_var, GRB.MINIMIZE)
    model.optimize()
    if model.Status == GRB.OPTIMAL:
        value = affine_var.getAttr("x")
    elif model.Status == GRB.TIME_LIMIT:
        value = affine_var.getAttr("lb") if model.SolCount == 0 else model.objbound
    elif model.Status == GRB.INFEASIBLE:
        raise RuntimeError("Model is infeasible lb")

    lb_ub.append(value)

    model.setObjective(affine_var, GRB.MAXIMIZE)
    model.optimize()
    if model.Status == GRB.OPTIMAL:
        value = affine_var.getAttr("x")
    elif model.Status == GRB.TIME_LIMIT:
        value = affine_var.getAttr("ub") if model.SolCount == 0 else model.objbound
    elif model.Status == GRB.INFEASIBLE:
        raise RuntimeError("Model is infeasible ub")
    lb_ub.append(value)

    return lb_ub
