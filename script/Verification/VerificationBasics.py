import math

import numpy as np
import torch
import gurobipy as grp


def add_relu_constr(model, input_vars, number_of_out_features, lb, ub, i=0):
    a = model.addMVar(number_of_out_features, vtype=grp.GRB.BINARY, name="a" + str(i))
    relu_vars = model.addMVar(number_of_out_features, lb=lb, name="relu_var" + str(i))
    model.update()
    relu_const_0 = model.addConstrs((var >= 0) for var in relu_vars.tolist())
    relu_const_1 = model.addConstrs((var >= x) for var, x in zip(relu_vars.tolist(), input_vars.tolist()))
    # relu_const_if1 = model.addConstrs((var <= a_i * ub) for var, a_i in zip(relu_vars.tolist(), a.tolist()))
    relu_const_if1 = model.addConstrs(
        (relu_vars.tolist()[k] <= a.tolist()[k] * ub[k]) for k in range(number_of_out_features))
    # relu_const_if0 = model.addConstrs((var <= x - lb * (1 - a_i)) for var, x, a_i in zip(relu_vars.tolist(), input_vars.tolist(), a.tolist()))
    relu_const_if0 = model.addConstrs(
        (relu_vars.tolist()[k] <= input_vars.tolist()[k] - lb[k] * (1 - a.tolist()[k])) for k in
        range(number_of_out_features))

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


def add_affine_constr(model, affine_w, b, input_vars, lb, ub, i=0):
    out_fet = len(b)
    out_vars = model.addMVar(out_fet, lb=lb, ub=ub, name="affine_var" + str(i))
    const = model.addConstrs((affine_w[i] @ input_vars + b[i] == out_vars[i] for i in range(len(affine_w))))
    return out_vars

def add_singel_neuron_constr(model, input_vars, number_of_out_features, lb, ub, i=0):
    relu_vars = model.addMVar(number_of_out_features, lb=lb, name="in_relu"+str(i))
    model.addConstrs((relu_vars[k] >= 0) for k in range(number_of_out_features))
    model.addConstrs((relu_vars[k] >= input_vars[k]) for k in range(number_of_out_features))
    div_var = model.addMVar(number_of_out_features, lb=-float("inf"), name="div_var" + str(i))
    model.addConstrs((div_var[k] * (ub[k] - lb[k]) == 1 for k in range(number_of_out_features)))
    model.addConstrs(((relu_vars[k] <= (ub[k] ) * (input_vars[k] - lb[k])) for k in range(number_of_out_features)), name="lb_const" + str(i))
    return relu_vars