import math

import numpy as np
import torch
import gurobipy as grp


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
    const = model.addConstrs((affine_w[i] @ input_vars + b[i] == out_vars[i] for i in range(len(affine_w))), name="affine_const"+str(i))
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