import numpy as np
import torch
import gurobipy as grp
import script.Verification.VerificationBasics as verbas


def sequential(predictor, input_x, eps, feasible_input=None, time_limit=None, bound=None):
    m = grp.Model()
    input_flattened = torch.flatten(input_x)
    input_size = input_flattened.size(0)
    bounds = predictor.calculate_box_bounds([input_flattened.add(-eps), input_flattened.add(eps)], with_relu=False)

    input_flattened = input_flattened.numpy()

    if time_limit is not None:
        m.setParam("TimeLimit", time_limit)

    if bound is not None:
        m.setParam("BestObjStop", bound)

    if feasible_input is None:
        input_var = m.addMVar(input_size, lb=[elem - eps for elem in input_flattened], ub=[elem + eps for elem in input_flattened], name="in_var")
        m.addConstrs(input_var[i] <= input_flattened[i] + eps for i in range(input_size))
        m.addConstrs(input_var[i] >= input_flattened[i] - eps for i in range(input_size))
    else:
        input_var = m.addMVar(input_size, lb=-float("inf"), name="in_var")
        m.addConstrs(input_var[i] <= feasible_input[i] for i in range(input_size))
        m.addConstrs(input_var[i] >= feasible_input[i] for i in range(input_size))


    parameter_list = list(predictor.parameters())
    in_var = input_var
    for i in range(0, len(parameter_list) - 2, 2):
        lb = bounds[int(i / 2)][0].detach().numpy()
        ub = bounds[int(i / 2)][1].detach().numpy()
        W, b = parameter_list[i].detach().numpy(), parameter_list[i + 1].detach().numpy()

        out_fet = len(b)
        out_vars = m.addMVar(out_fet, lb=lb, ub=ub, name="affine_var" + str(i))
        const = m.addConstrs((W[i] @ in_var + b[i] == out_vars[i] for i in range(len(W))))

        relu_in_var = out_vars
        relu_vars = verbas.add_singel_neuron_constr(m, relu_in_var, out_fet, lb, ub, i=i) #verbas.add_relu_constr(m, relu_in_var, out_fet, lb, ub)
        m.update()
        in_var = relu_vars
        

    lb = bounds[-1][0].detach().numpy()
    ub = bounds[-1][1].detach().numpy()
    W, b = parameter_list[len(parameter_list)-2].detach().numpy(), parameter_list[-1].detach().numpy()

    out_fet = len(b)
    out_vars = m.addMVar(out_fet, lb=lb, ub=ub, name="last_affine_var")
    const = m.addConstrs((W[i] @ in_var + b[i] == out_vars[i] for i in range(len(W))), name="out_const")

    m.update()
    m.setObjective(out_vars[1], grp.GRB.MAXIMIZE)
    m.optimize()

    if m.Status == grp.GRB.OPTIMAL:
        print(input_var.getAttr("x"))
        print(out_vars.getAttr("x"))


    return m
