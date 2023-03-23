import math

import numpy as np
import torch
import gurobipy as grp
from script.settings import data_type, device

def add_relu_constr(model, input_vars, number_of_out_features, in_lb, in_ub, out_lb, out_ub, i=0):
    relu_vars = model.addMVar(number_of_out_features, lb=out_lb, ub=out_ub, name="relu_var" + str(i))
    for k in range(number_of_out_features):
        if in_ub[k] <= 0:
            model.addConstr(relu_vars[k] == 0, name="relu=0" + str(i))
        elif in_lb[k] >= 0:
            model.addConstr(relu_vars[k] == input_vars[k], name="relu=x" + str(i))
        else:
            # Variante 1: Dreiecksungleichung
            a = model.addVar(vtype=grp.GRB.BINARY, name="a" + str(i) + str(k))
            r1 = model.addConstr(relu_vars[k] >= 0, name="relu_0_lt"+str(i)+"k"+str(k))
            r2 = model.addConstr(relu_vars[k] >= input_vars[k], name="relu_var_lt"+str(i)+"k"+str(k))
            r3 = model.addConstr(relu_vars[k] <= input_vars[k] - in_lb[k] * (1 - a), name="relu_var_lt_1-a"+str(i)+"k"+str(k))
            r4 = model.addConstr(relu_vars[k] <= in_ub[k] * a, name="relu_var_lt_a"+str(i)+"k"+str(k))

            # Variante 3: Nutze Gurobi max Constraints
            # const = model.addConstr(relu_vars[k] == grp.max_(0, input_vars[k]))

    return relu_vars


def add_affine_constr(model, affine_w, b, input_vars, lb, ub, i=0):
    out_fet = len(b)
    out_vars = model.addMVar(out_fet, lb=lb, ub=ub, name="affine_var" + str(i))
    const = model.addConstr(affine_w @ input_vars + b == out_vars, name="affine_const_[{}]".format(i))
    return out_vars


def add_single_neuron_constr(model, input_vars, number_of_out_features, in_lb, in_ub, out_lb, out_ub, i=0):
    relu_vars = model.addMVar(number_of_out_features, lb=out_lb, ub=out_ub, name="relu_var"+str(i))
    for k in range(number_of_out_features):
        if in_ub[k] <= 0:
            model.addConstr(relu_vars[k] == 0, name="relu=0" + str(i)+"k"+str(k))
        elif in_lb[k] >= 0:
            model.addConstr(relu_vars[k] == input_vars[k], name="relu>0" + str(i)+"k"+str(k))
        else:
            model.addConstr(relu_vars[k] >= 0, name="relu_0_lt"+str(i)+"k"+str(k))
            model.addConstr(relu_vars[k] >= input_vars[k], name="relu_var_lt"+str(i)+"k"+str(k))
            model.addConstr(relu_vars[k] <= (in_ub[k] * (input_vars[k] - in_lb[k])) / (in_ub[k] - in_lb[k]), name="ub_const" + str(i)+"k"+str(k))
    return relu_vars


def add_relu_as_lp(model, input_vars, number_of_out_features, out_lb, out_ub, i=0):
    relu_vars = model.addMVar(number_of_out_features, lb=out_lb, ub=out_ub, name="relu_var" + str(i))
    for k in range(number_of_out_features):
        r1 = model.addConstr(relu_vars[k] >= 0, name="relu_0_lt" + str(i) + "k" + str(k))
        r2 = model.addConstr(relu_vars[k] >= input_vars[k], name="relu_var_lt" + str(i) + "k" + str(k))

    return relu_vars


def calc_affine_out_bound(affine_w, affine_b, neuron_min_value, neuron_max_value):
    w_plus = torch.maximum(affine_w, torch.tensor(0, dtype=data_type).to(device))
    w_minus = torch.minimum(affine_w, torch.tensor(0, dtype=data_type).to(device))
    affine_out_lb = torch.matmul(w_plus, neuron_min_value).add(torch.matmul(w_minus, neuron_max_value)).add(affine_b)
    affine_out_ub = torch.matmul(w_plus, neuron_max_value).add(torch.matmul(w_minus, neuron_min_value)).add(affine_b)
    return affine_out_lb, affine_out_ub


def calc_relu_out_bound(neuron_min_value, neuron_max_value):
    relu_out_lb = torch.maximum(torch.tensor(0, dtype=data_type).to(device), neuron_min_value)
    relu_out_ub = torch.maximum(torch.tensor(0, dtype=data_type).to(device), neuron_max_value)
    return relu_out_lb, relu_out_ub
