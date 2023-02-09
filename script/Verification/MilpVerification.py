import torch
import gurobipy as grp


def sequential(predictor, input_x, output_size, label, eps=0.01, time_limit=None, bound=None):
    m = grp.Model()
    input_flattened = torch.flatten(input_x)
    input_size = input_flattened.size(0)
    bounds = predictor.calculate_box_bounds(m, [input_flattened.add(-eps), input_flattened.add(eps)], with_ReLU=False)

    bounds = [[elem[0].detach().tolist(), elem[1].detach().tolist()] for elem in bounds]

    input_flattened = input_flattened.numpy()

    if time_limit is not None:
        m.setParam("TimeLimit", time_limit)

    if bound is not None:
        m.setParam("BestObjStop", bound)

    input_var = m.addMVar(input_size, lb=[elem - eps for elem in input_flattened], ub=[elem + eps for elem in input_flattened], name="in_var")
    output_var = m.addMVar(output_size, lb=bounds[-1][0], ub=bounds[-1][1], name="output_var")

    m.addConstrs(input_var[i] <= input_flattened[i] + eps for i in range(input_size))
    m.addConstrs(input_var[i] >= input_flattened[i] - eps for i in range(input_size))

    predictor.add_constraints(m, input_var, bounds)

    lower_diff_bound = [lb - bounds[-1][1][label] for lb in bounds[-1][0]]
    upper_diff_bound = [ub - bounds[-1][0][label] for ub in bounds[-1][1]]

    lower_diff_bound.pop(label)
    upper_diff_bound.pop(label)

    difference = m.addVars(output_size - 1, lb=lower_diff_bound, ub=upper_diff_bound)
    m.addConstrs(difference[i] == output_var.tolist()[i] - output_var.tolist()[label] for i in range(0, label))
    m.addConstrs(
        difference[i - 1] == output_var.tolist()[i] - output_var.tolist()[label] for i in range(label + 1, output_size))

    min_diff_bound = min(lower_diff_bound)
    max_diff_bound = min(upper_diff_bound)
    max_var = m.addVar(lb=min_diff_bound, ub=bound)
    m.addConstr(max_var == grp.max_(difference))

    m.update()
    m.setObjective(max_var, grp.GRB.MAXIMIZE)
    m.optimize()

    if m.Status == grp.GRB.OPTIMAL or m.Status == grp.GRB.TIME_LIMIT or m.Status == grp.GRB.USER_OBJ_LIMIT:
        inp = input_var.getAttr("x")
        for o in difference.select():
            print(o.getAttr("x"))
        print("optimum solution with value \n {}".format(output_var.getAttr("x")))
        print("max_var {}".format(max_var.getAttr("x")))
        test_inp = torch.tensor([inp], dtype=torch.float64)
        return test_inp, output_var
