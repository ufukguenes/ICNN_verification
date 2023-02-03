import random

import numpy as np
import torch
import gurobipy as grp

import script.Verification.VerificationBasics as verbas
import script.Verification.MilpVerification as milp


def sample_max_radius(icnn, sample_size, c=0, box_bounds=None):
    m = grp.Model()
    m.Params.LogToConsole = 0

    icnn_input_size = icnn.layer_widths[0]
    input_to_icnn_one = m.addMVar(icnn_input_size, lb=-float('inf'))
    input_to_icnn_two = m.addMVar(icnn_input_size, lb=-float('inf'))
    output_of_icnn_one = m.addMVar(1, lb=-float('inf'))
    output_of_icnn_two = m.addMVar(1, lb=-float('inf'))
    bounds = verbas.calculate_box_bounds(icnn, None, is_sequential=False)
    verbas.add_constr_for_non_sequential_icnn(m, icnn, input_to_icnn_one, output_of_icnn_one, bounds)
    m.addConstr(output_of_icnn_one <= c)
    verbas.add_constr_for_non_sequential_icnn(m, icnn, input_to_icnn_two, output_of_icnn_two, bounds)
    m.addConstr(output_of_icnn_two <= c)

    difference = m.addVar(lb=-float('inf'))

    center_values = []
    eps_values = []
    for i in range(icnn_input_size):
        diff_const = m.addConstr(difference == input_to_icnn_one[i] - input_to_icnn_two[i])
        m.setObjective(difference, grp.GRB.MAXIMIZE)
        m.optimize()
        if m.Status == grp.GRB.OPTIMAL:
            point_one = input_to_icnn_one.getAttr("x")
            point_two = input_to_icnn_two.getAttr("x")
            max_dist = difference.getAttr("x")
            center_point = (point_one + point_two) / 2
            eps = max_dist / 2
            center_values.append(center_point[i])
            eps_values.append(eps)
        m.remove(diff_const)

    input_size = len(center_values)
    included_space = torch.empty(0, dtype=torch.float64)
    ambient_space = torch.empty(0, dtype=torch.float64)

    best_upper_bound = []
    best_lower_bound = []

    for k, val in enumerate(eps_values):
        choice_for_upper_bound = [center_values[k] + eps_values[k], center_values[k] + eps_values[k]]
        choice_for_lower_bound = [center_values[k] - eps_values[k], center_values[k] - eps_values[k]]

        if box_bounds is not None:
            choice_for_upper_bound.append(box_bounds[1][k].item())
            choice_for_lower_bound.append(box_bounds[0][k].item())

        distance_to_center_upper = [abs(center_values[k] - choice_for_upper_bound[i]) for i in
                                    range(len(choice_for_upper_bound))]
        distance_to_center_lower = [abs(center_values[k] - choice_for_lower_bound[i]) for i in
                                    range(len(choice_for_lower_bound))]
        arg_min_upper_bound = np.argmin(distance_to_center_upper)
        arg_min_lower_bound = np.argmin(distance_to_center_lower)
        best_upper_bound.append(choice_for_upper_bound[arg_min_upper_bound])
        best_lower_bound.append(choice_for_lower_bound[arg_min_lower_bound])

    best_upper_bound = torch.tensor(best_upper_bound, dtype=torch.float64)
    best_lower_bound = torch.tensor(best_lower_bound, dtype=torch.float64)

    samples = (best_upper_bound - best_lower_bound) * torch.rand((sample_size, input_size),
                                                                 dtype=torch.float64) + best_lower_bound

    for samp in samples:
        samp = torch.unsqueeze(samp, 0)
        out = icnn(samp)
        if out <= c:
            included_space = torch.cat([included_space, samp], dim=0)
        else:
            ambient_space = torch.cat([ambient_space, samp], dim=0)

    print("included space num samples {}, ambient space num samples {}".format(len(included_space), len(ambient_space)))
    return included_space, ambient_space



def regroup_samples(icnn, included_space, ambient_space, c=0):
    moved = 0
    for i, elem in enumerate(ambient_space):
        elem = torch.unsqueeze(elem, 0)
        output = icnn(elem)
        if output <= c:
            included_space = torch.cat([included_space, elem], dim=0)
            ambient_space = torch.cat([ambient_space[:i - moved], ambient_space[i + 1 - moved:]])
            moved += 1

    return included_space, ambient_space


def samples_uniform_over(data_samples, amount, bounds, keep_samples=True, padding=0):
    lb = bounds[0] - padding
    ub = bounds[1] + padding
    shape = data_samples.size(1)
    random_samples = (ub - lb) * torch.rand((amount, shape), dtype=torch.float64) + lb
    if keep_samples:
        data_samples = torch.cat([data_samples, random_samples], dim=0)
    else:
        data_samples = random_samples


    return data_samples


def sample_uniform_excluding(data_samples, amount, including_bound, excluding_bound=None, icnn=None, keep_samples=True, padding=0):
    input_size = data_samples.size(dim=1)

    lower = including_bound[0] - padding
    upper = including_bound[1] + padding
    new_samples = (upper - lower) * torch.rand((amount, input_size), dtype=torch.float64) + lower

    for i, samp in enumerate(new_samples):
        max_samp = torch.max(samp)
        min_samp = torch.min(samp)
        shift = False

        if excluding_bound is not None:
            lower_excluding_bound = excluding_bound[0]
            upper_excluding_bound = excluding_bound[1]

            max_greater_then = max_samp.gt(upper_excluding_bound)
            min_less_then = min_samp.lt(lower_excluding_bound)
            if True not in max_greater_then and True not in min_less_then:
                shift = True

        if icnn is not None and not shift:
            inp = torch.unsqueeze(samp, 0)
            out = icnn(inp)
            if out <= 0:
                shift = True

        if shift:
            rand_index = random.randint(0, samp.size(0) - 1)
            rand_bound = random.random()
            if rand_bound < 0.5:
                samp[rand_index] = upper[rand_index]
            else:
                samp[rand_index] = lower[rand_index]

    if keep_samples:
        data_samples = torch.cat([data_samples, new_samples], dim=0)
    else:
        data_samples = new_samples

    return data_samples


def apply_affine_transform(W, b, data_samples):
    transformed_samples = torch.empty((data_samples.size(0), b.size(0)), dtype=torch.float64)
    for i in range(data_samples.shape[0]):
        transformed_samples[i] = torch.matmul(W, data_samples[i]).add(b)


    return transformed_samples


def apply_ReLU_transform(data_samples):
    relu = torch.nn.ReLU()
    transformed_samples = relu(data_samples)

    return transformed_samples
