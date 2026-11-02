import math
import time
import warnings

import numpy as np
import torch
from gurobipy import GRB
import gurobipy as grp
import script.Verification.VerificationBasics as verbas

from script.settings import device, data_type, get_gurobi_params
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
    out_vars = model.addMVar(out_fet, lb=out_lb, ub=out_ub)

    for neuron_index in curr_fixed_neuron_upper:
        model.addConstr(out_vars[neuron_index] == 0)

    for neuron_index in curr_fixed_neuron_lower:
        model.addConstr(out_vars[neuron_index] == affine_var[neuron_index])

    not_fixed_neurons = [x for x in range(out_fet) if x not in (curr_fixed_neuron_lower + curr_fixed_neuron_upper)]
    for neuron_index in not_fixed_neurons:
        in_lb = curr_bounds_affine_out[0][neuron_index].item()
        in_ub = curr_bounds_affine_out[1][neuron_index].item()
        model.addConstr(out_vars[neuron_index] >= 0)
        model.addConstr(out_vars[neuron_index] >= affine_var[neuron_index])
        model.addConstr(out_vars[neuron_index] <= (in_ub * (affine_var[neuron_index] - in_lb)) / (in_ub - in_lb))


    for k, constraint_icnn in enumerate(curr_constraint_icnns):
        index_to_select = torch.tensor(curr_group_indices[k]).to(device)
        low = torch.index_select(curr_bounds_layer_out[0], 0, index_to_select)
        up = torch.index_select(curr_bounds_layer_out[1], 0, index_to_select)
        constraint_icnn_bounds_affine_out, constraint_icnn_bounds_layer_out = constraint_icnn.calculate_box_bounds([low, up])
        current_in_vars = model.addMVar(len(curr_group_indices[k]), lb=low.detach().cpu().numpy(), ub=up.detach().cpu().numpy())

        constraint_icnn.add_max_output_constraints(model, current_in_vars, constraint_icnn_bounds_affine_out,
                                                   constraint_icnn_bounds_layer_out)

        model.addConstrs((out_vars[neuron_index] == current_in_vars[i] for i, neuron_index in enumerate(curr_group_indices[k])))    

    in_var = out_vars

    for i, var in enumerate(in_var.tolist()):
        var.setAttr("varname", "output_layer_[{}]_[{}]".format(current_layer_index, i))


def verification(icnn, model, affine_w, affine_b, index_to_select, curr_bounds_affine_out, curr_bounds_layer_out, prev_layer_index, has_relu=False, relu_as_lp=False, icnn_as_lp=False, report=None):
    """
    Enlargement problem of one group: maximises the ICNN over the group's neurons on top of the relaxation in model.

    Returns (inp, c): inp is the incumbent's input of the group's layer (None if there is none), c a proven upper
    bound on the maximum, widened by the solver tolerance, so that b_K -= c makes {ICNN <= 0} contain the group's
    part of the relaxation. c is the smallest of
    * the dual bound of the problem (ObjBound, also after a time limit and without an incumbent; never the
      incumbent value, which may lie below the maximum by up to MIPGap),
    * without a dual bound: c of the same problem with the group's ReLUs and the ICNN relaxed to LPs,
    * the ICNN's IBP upper bound over the group's box (always valid).
    The old version returned the incumbent value when OPTIMAL and raised a RuntimeError after a time limit without
    incumbent (legacy-notes #5). If report is a dict, it gets status, source, primal (incumbent value or None) and
    shift (= c).
    """
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
    print("        actual verification time {} (status {})".format(time.time() - t, model.Status))

    if model.Status == GRB.INFEASIBLE:
        raise RuntimeError("Model is infeasible")

    inp = output_prev_layer.getAttr("x") if model.SolCount > 0 else None
    primal = model.ObjVal if model.SolCount > 0 else None

    # (value, source) of every proven upper bound, each widened by the solver tolerance
    candidates = [(verbas.widen_up(icnn_bounds_layer_out[-1][1].item()), "ibp")]
    dual = verbas.proven_bound(model)
    if dual is not None:
        candidates.append((verbas.widen_up(dual), "milp" if model.IsMIP else "lp"))
    elif not (relu_as_lp and icnn_as_lp):
        print("            no dual bound, fall back to lp solving of icnn enlargement")
        back_up_model.Params.TimeLimit = model.Params.TimeLimit
        back_up_model.Params.Threads = model.Params.Threads
        _, c_lp = verification(icnn, back_up_model, affine_w, affine_b, index_to_select, curr_bounds_affine_out,
                               curr_bounds_layer_out, prev_layer_index, has_relu=has_relu, relu_as_lp=True,
                               icnn_as_lp=True)
        candidates.append((c_lp, "lp"))
    c, source = min(candidates, key=lambda candidate: candidate[0])

    print("            enlarge with {} ({})".format(c, source))
    if report is not None:
        status_names = {GRB.OPTIMAL: "optimal", GRB.TIME_LIMIT: "time_limit"}
        report.update(status=status_names.get(model.Status, str(model.Status)), source=source, primal=primal,
                      shift=c)
    return inp, c


def update_bounds_with_icnns(model, bounds_affine_out, bounds_layer_out, current_layer_index, affine_w, affine_b, time_limit=None, report=None):
    """
    LP bound tightening of layer current_layer_index over the relaxation in model: min and max of every
    pre-activation, one solve each, in parallel (one process and one Gurobi thread per solve).

    Each new bound is a proven bound (LP optimum or MIP dual bound), widened by the solver tolerance, rounded outward
    to data_type and intersected with the old bound; a solve that proves nothing keeps the old bound. The old
    version stored the LP value rounded to nearest and gave every process 2 Gurobi threads (cpu_count processes:
    oversubscribed, legacy-notes #11). time_limit caps each solve. If report is a dict, it gets the counts
    solved / without_bound. Returns True.
    """

    output_prev_layer = []
    prev_layer_index = current_layer_index - 1
    for i in range(affine_w.shape[1]):
        output_prev_layer.append(model.getVarByName("output_layer_[{}]_[{}]".format(prev_layer_index, i)))
    output_prev_layer = grp.MVar.fromlist(output_prev_layer)

    lb = bounds_affine_out[current_layer_index][0].detach()
    ub = bounds_affine_out[current_layer_index][1].detach()
    affine_var = verbas.add_affine_constr(model, affine_w, affine_b, output_prev_layer, lb.cpu(), ub.cpu(), i="myVarWithNewName")

    model.update()

    num_processors = torch.multiprocessing.cpu_count()
    Cache.model = model
    Cache.affine_var_name = list(affine_var.VarName)
    Cache.time_limit = time_limit
    neuron_indices = list(range(len(affine_var.tolist())))
    with torch.multiprocessing.Pool(num_processors) as pool:
        result = pool.map(parallel_call, neuron_indices)

    old_lb = bounds_affine_out[current_layer_index][0].detach().to(torch.float64)
    old_ub = bounds_affine_out[current_layer_index][1].detach().to(torch.float64)
    new_lb = old_lb.clone()
    new_ub = old_ub.clone()
    without_bound = 0
    for neuron_to_optimize, (lb_value, ub_value) in enumerate(result):
        if lb_value is None:
            without_bound += 1
        else:
            new_lb[neuron_to_optimize] = max(old_lb[neuron_to_optimize].item(), verbas.widen_down(lb_value))
        if ub_value is None:
            without_bound += 1
        else:
            new_ub[neuron_to_optimize] = min(old_ub[neuron_to_optimize].item(), verbas.widen_up(ub_value))

    bounds_affine_out[current_layer_index][0] = verbas.to_data_type_down(new_lb).to(device)
    bounds_affine_out[current_layer_index][1] = verbas.to_data_type_up(new_ub).to(device)

    relu_out_lb, relu_out_ub = verbas.calc_relu_out_bound(bounds_affine_out[current_layer_index][0],
                                                          bounds_affine_out[current_layer_index][1])

    bounds_layer_out[current_layer_index][0] = relu_out_lb
    bounds_layer_out[current_layer_index][1] = relu_out_ub
    if report is not None:
        report.update(solved=2 * len(result) - without_bound, without_bound=without_bound)
    return True



class Cache:
    model = None
    affine_var_name = None
    time_limit = None


def parallel_call(neuron_index):
    """Returns [lower, upper]: the proven min / max of one pre-activation (None where a solve proved nothing)."""
    gurobi_params = get_gurobi_params()

    with grp.Env(params=gurobi_params) as env, Cache.model.copy(env) as model:
        model.Params.Threads = 1
        if Cache.time_limit is not None:
            model.Params.TimeLimit = Cache.time_limit
        affine_var = model.getVarByName(Cache.affine_var_name[neuron_index])

        lb_ub = []
        for sense in (GRB.MINIMIZE, GRB.MAXIMIZE):
            model.setObjective(affine_var, sense)
            model.optimize()
            if model.Status == GRB.INFEASIBLE:
                raise RuntimeError("Model is infeasible ({})".format("lb" if sense == GRB.MINIMIZE else "ub"))
            lb_ub.append(verbas.proven_bound(model))

        return lb_ub
