import math
import random

import numpy as np
import torch
import gurobipy as grp
from script.settings import device, data_type

def sample_max_radius(icnns, sample_size, group_indices, curr_bounds_layer_out, fixed_neuron_lower, fixed_neuron_upper):
    center_values = torch.zeros(len(curr_bounds_layer_out[0]), dtype=data_type).to(device)
    eps_values = torch.zeros(len(curr_bounds_layer_out[0]), dtype=data_type).to(device)
    for group_i, icnn in enumerate(icnns):
        icnn_input_size = icnn.layer_widths[0]
        index_to_select = group_indices[group_i]

        m = grp.Model()
        m.Params.LogToConsole = 0

        input_to_icnn_one = m.addMVar(icnn_input_size, lb=-float('inf'))
        input_to_icnn_two = m.addMVar(icnn_input_size, lb=-float('inf'))

        index_to_select = torch.tensor(index_to_select).to(device)
        low = torch.index_select(curr_bounds_layer_out[0], 0, index_to_select)
        up = torch.index_select(curr_bounds_layer_out[1], 0, index_to_select)
        icnn_bounds_affine_out, icnn_bounds_layer_out = icnn.calculate_box_bounds([low, up])
        icnn.add_max_output_constraints(m, input_to_icnn_one, icnn_bounds_affine_out, icnn_bounds_layer_out)
        icnn.add_max_output_constraints(m, input_to_icnn_two, icnn_bounds_affine_out, icnn_bounds_layer_out)

        difference = m.addVar(lb=-float('inf'))


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
                center_values[group_indices[group_i][i]] = center_point[i]
                eps_values[group_indices[group_i][i]] = eps
            m.remove(diff_const)

    for neuron_index in fixed_neuron_upper:
        center_values[neuron_index] = 0
        eps_values[neuron_index] = 0

    for neuron_index in fixed_neuron_lower:
        point_one = curr_bounds_layer_out[0][neuron_index]
        point_two = curr_bounds_layer_out[1][neuron_index]
        max_dist = point_one - point_two
        center_point = (point_one + point_two) / 2
        eps = max_dist / 2
        center_values[neuron_index] = center_point
        eps_values[neuron_index] = eps

    input_size = len(center_values)
    included_space = torch.empty(0, dtype=data_type).to(device)
    ambient_space = torch.empty(0, dtype=data_type).to(device)

    best_upper_bound = []
    best_lower_bound = []

    for k, val in enumerate(eps_values):
        choice_for_upper_bound = [center_values[k] + eps_values[k], center_values[k] + eps_values[k]]
        choice_for_lower_bound = [center_values[k] - eps_values[k], center_values[k] - eps_values[k]]

        distance_to_center_upper = [abs(center_values[k] - choice_for_upper_bound[i]) for i in
                                    range(len(choice_for_upper_bound))]
        distance_to_center_lower = [abs(center_values[k] - choice_for_lower_bound[i]) for i in
                                    range(len(choice_for_lower_bound))]
        distance_to_center_lower = [x.detach().cpu().numpy() for x in distance_to_center_lower]
        distance_to_center_upper = [x.detach().cpu().numpy() for x in distance_to_center_upper]
        arg_min_upper_bound = np.argmin(distance_to_center_upper)
        arg_min_lower_bound = np.argmin(distance_to_center_lower)
        best_upper_bound.append(choice_for_upper_bound[arg_min_upper_bound])
        best_lower_bound.append(choice_for_lower_bound[arg_min_lower_bound])

    best_upper_bound = torch.tensor(best_upper_bound, dtype=data_type).to(device)
    best_lower_bound = torch.tensor(best_lower_bound, dtype=data_type).to(device)

    samples = (best_upper_bound - best_lower_bound) * torch.rand((sample_size, input_size),
                                                                 dtype=data_type).to(device) + best_lower_bound

    for samp in samples:
        is_included = True
        """
        index_start = 0
        index_end = 0
        for group_i, icnn in enumerate(icnns):
            index_end += len(group_indices[group_i])
            index_to_select = torch.tensor(range(index_start, index_end)).to(device)
            index_start = index_end
        """
        samp = torch.unsqueeze(samp, 0)
        for group_i, icnn in enumerate(icnns):
            index_to_select = group_indices[group_i]
            index_to_select = torch.tensor(index_to_select).to(device)
            reduced_elem = torch.index_select(samp, 1, index_to_select)
            output = icnn(reduced_elem)
            if output > 0:
                is_included = False
                break


        if is_included:
            included_space = torch.cat([included_space, samp], dim=0)
        else:
            ambient_space = torch.cat([ambient_space, samp], dim=0)

    print("included space num samples {}, ambient space num samples {}".format(len(included_space), len(ambient_space)))
    return included_space, ambient_space


def regroup_samples(icnns, included_space, ambient_space, group_indices, c=0):
    moved = 0
    for i, elem in enumerate(ambient_space):
        elem = torch.unsqueeze(elem, 0)
        is_included = True
        for group_i, icnn in enumerate(icnns):
            index_to_select = group_indices[group_i]
            index_to_select = torch.tensor(index_to_select).to(device)
            reduced_elem = torch.index_select(elem, 1, index_to_select)
            output = icnn(reduced_elem)
            if output > c:
                is_included = False
                break

        if is_included:
            included_space = torch.cat([included_space, elem], dim=0)
            ambient_space = torch.cat([ambient_space[:i - moved], ambient_space[i + 1 - moved:]])
            moved += 1

    return included_space, ambient_space


def samples_uniform_over(data_samples, amount, bounds, keep_samples=True, padding=0):
    lb = bounds[0] - padding
    ub = bounds[1] + padding
    shape = data_samples.size(1)
    random_samples = (ub - lb) * torch.rand((amount, shape), dtype=data_type).to(device) + lb
    if keep_samples and data_samples.size(0) > 0:
        data_samples = torch.cat([data_samples, random_samples], dim=0)
    else:
        data_samples = random_samples

    return data_samples

def sample_linspace(data_samples, amount, center, eps, keep_samples=True):
    step_number = 1 #math.floor(math.sqrt(amount))
    xs = torch.linspace(-eps, eps, steps=step_number)
    """#ys = torch.linspace(-eps, eps, steps=step_number)
    tens = [xs for i in range(step_number)]
    grid = torch.meshgrid(*tens)
    new_samples = torch.add(center, grid)"""
    mask = torch.combinations(xs, r=data_samples.size(1), with_replacement=True)
    new_samples = torch.add(center, mask)
    if keep_samples and data_samples.size(0) > 0:
        data_samples = torch.cat([data_samples, new_samples], dim=0)
    else:
        data_samples = new_samples
    return data_samples


def sample_uniform_excluding(data_samples, amount, including_bound, excluding_bound=None, icnns=None, layer_index=None,
                             group_size=None, keep_samples=True, padding=0):
    input_size = data_samples.size(dim=1)

    lower = including_bound[0] - padding
    upper = including_bound[1] + padding
    new_samples = (upper - lower) * torch.rand((amount, input_size), dtype=data_type).to(device) + lower
    lower = lower.detach()
    upper = upper.detach()
    new_samples = new_samples.detach()

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

        if icnns is not None and layer_index is not None and group_size is not None and not shift:
            number_of_groups = len(icnns)
            shift = True
            for group_i in range(number_of_groups):
                if group_i == number_of_groups - 1 and input_size % group_size > 0:
                    from_to_neurons = [group_size * group_i, group_size * group_i + (input_size % group_size)]
                else:
                    from_to_neurons = [group_size * group_i, group_size * group_i + group_size]  # upper bound is exclusive
                inp = torch.unsqueeze(samp.clone()[from_to_neurons[0]:from_to_neurons[1]], 0)
                out = icnns[group_i](inp)
                if out > 0:
                    shift = False
                    break

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


def apply_affine_transform(affine_w, affine_b, data_samples):
    transformed_samples = torch.empty((data_samples.size(0), affine_b.size(0)), dtype=data_type).to(device)
    for i in range(data_samples.shape[0]):
        transformed_samples[i] = torch.matmul(affine_w, data_samples[i]).add(affine_b)

    return transformed_samples


def apply_relu_transform(data_samples):
    relu = torch.nn.ReLU()
    transformed_samples = relu(data_samples)

    return transformed_samples
