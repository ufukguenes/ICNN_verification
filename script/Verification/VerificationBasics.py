import math

import torch
import gurobipy as grp


def add_relu_constr(model, input_vars, number_of_out_features, lb, ub, i=0):
    a = model.addMVar(number_of_out_features, vtype=grp.GRB.BINARY, name="a" + str(i))
    relu_vars = model.addMVar(number_of_out_features, lb=lb, name="relu_var" + str(i))
    model.update()
    relu_const_0 = model.addConstrs((var >= 0) for var in relu_vars.tolist())
    relu_const_1 = model.addConstrs((var >= x) for var, x in zip(relu_vars.tolist(), input_vars.tolist()))
    #relu_const_if1 = model.addConstrs((var <= a_i * ub) for var, a_i in zip(relu_vars.tolist(), a.tolist()))
    relu_const_if1 = model.addConstrs((relu_vars.tolist()[k] <= a.tolist()[k] * ub[k]) for k in range(number_of_out_features))
    #relu_const_if0 = model.addConstrs((var <= x - lb * (1 - a_i)) for var, x, a_i in zip(relu_vars.tolist(), input_vars.tolist(), a.tolist()))
    relu_const_if0 = model.addConstrs((relu_vars.tolist()[k] <= input_vars.tolist()[k] - lb[k] * (1 - a.tolist()[k])) for k in range(number_of_out_features))

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


def add_affine_constr(model, W, b, input_vars, lb, ub, i=0):
    out_fet = len(b)
    out_vars = model.addMVar(out_fet, lb=lb, ub=ub, name="affine_var" + str(i))
    const = model.addConstrs((W[i] @ input_vars + b[i] == out_vars[i] for i in range(len(W))))
    return out_vars


def calculate_box_bounds(nn, input_bounds, is_sequential=True, with_ReLU=True):
    parameter_list = list(nn.parameters())
    # todo for now this only works for sequential nets

    if input_bounds is None:
        bounds_per_layer = [([torch.tensor([-5000 for k in range(len(parameter_list[i]))]),
                              torch.tensor([5000 for k in range(len(parameter_list[i]))])]) for i in
                            range(0, len(parameter_list), 2)]
        return bounds_per_layer #todo None entfernen aus aufrufen und durch sinnvolle eingabe ersetzen

    next_lower_bounds = input_bounds[0]
    next_upper_bounds = input_bounds[1]
    bounds_per_layer = []
    for i in range(0, len(parameter_list), 2):
        W, b = parameter_list[i], parameter_list[i + 1]
        W_plus = torch.maximum(W, torch.tensor(0, dtype=torch.float64))
        W_minus = torch.minimum(W, torch.tensor(0, dtype=torch.float64))
        lb = torch.matmul(W_plus, next_lower_bounds).add(torch.matmul(W_minus, next_upper_bounds)).add(b)
        ub = torch.matmul(W_plus, next_upper_bounds).add(torch.matmul(W_minus, next_lower_bounds)).add(b)
        if not is_sequential and i != 0:
            U = nn.us[i / 2 - 1]
            U_plus = torch.maximum(U, torch.tensor(0, dtype=torch.float64))
            U_minus = torch.minimum(U, torch.tensor(0, dtype=torch.float64))
            lb = lb.add(torch.matmul(U_plus, next_lower_bounds).add(torch.matmul(U_minus, next_upper_bounds)))
            ub = ub.add(torch.matmul(U_plus, next_upper_bounds).add(torch.matmul(U_minus, next_lower_bounds)))
        if with_ReLU:
            next_upper_bounds = torch.maximum(torch.tensor(0, dtype=torch.float64), ub)
            next_lower_bounds = torch.maximum(torch.tensor(0, dtype=torch.float64), lb)
        else:
            next_upper_bounds = ub
            next_lower_bounds = lb

        bounds_per_layer.append([next_lower_bounds, next_upper_bounds])

    return bounds_per_layer


def add_constr_for_non_sequential_icnn(model, predictor, input_vars, output_vars, bounds):
    ws = list(predictor.ws.parameters())
    us = list(predictor.us.parameters())

    in_var = input_vars
    for i in range(0, len(ws), 2):
        lb = bounds[int(i / 2)][0]
        ub = bounds[int(i / 2)][1]
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


def add_constr_for_sequential_icnn(model, predictor, input_vars, output_vars, bounds):
    parameter_list = list(predictor.parameters())

    in_var = input_vars
    for i in range(0, len(parameter_list), 2):
        lb = bounds[int(i / 2)][0]
        ub = bounds[int(i / 2)][1]
        W, b = parameter_list[i].detach().numpy(), parameter_list[i + 1].detach().numpy()

        out_fet = len(b)
        out_vars = model.addMVar(out_fet, lb=lb, ub=ub, name="affine_var" + str(i))
        const = model.addConstrs((W[i] @ in_var + b[i] == out_vars[i] for i in range(len(W))))
        in_var = out_vars

        if i < len(parameter_list) - 2:
            #relu_vars = add_relu_constr(model, in_var, out_fet, [-10000 for i in range(len(W))], [10000 for i in range(len(W))], i)
            relu_vars = add_relu_constr(model, in_var, out_fet, [-10000 for i in range(len(W))], ub, i)
            #relu_vars = add_relu_constr(model, in_var, out_fet, lb, ub, i)
            in_var = relu_vars
            out_vars = relu_vars

    const = model.addConstrs(out_vars[i] == output_vars[i] for i in range(out_fet))
