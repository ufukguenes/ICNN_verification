import math
import random

import numpy as np
import torch
import gurobipy as grp
from script.settings import device, data_type
import script.Verification.VerificationBasics as verbas


def sample_max_radius(icnns, sample_size, group_indices, curr_bounds_layer_out, fixed_neuron_lower, fixed_neuron_upper):
    center_values = torch.zeros(len(curr_bounds_layer_out[0]), dtype=data_type).to(device)
    eps_values = torch.zeros(len(curr_bounds_layer_out[0]), dtype=data_type).to(device)
    for group_i, icnn in enumerate(icnns):
        icnn_input_size = icnn.layer_widths[0]
        index_to_select = group_indices[group_i]

        m = grp.Model()
        m.Params.LogToConsole = 0



        index_to_select = torch.tensor(index_to_select).to(device)
        low = torch.index_select(curr_bounds_layer_out[0], 0, index_to_select)
        up = torch.index_select(curr_bounds_layer_out[1], 0, index_to_select)
        icnn_bounds_affine_out, icnn_bounds_layer_out = icnn.calculate_box_bounds([low, up])

        lb = low.detach().numpy()
        ub = up.detach().numpy()
        input_to_icnn_one = m.addMVar(icnn_input_size, lb=lb, ub=ub)
        input_to_icnn_two = m.addMVar(icnn_input_size, lb=lb, ub=ub)
        icnn.add_max_output_constraints(m, input_to_icnn_one, icnn_bounds_affine_out, icnn_bounds_layer_out)
        icnn.add_max_output_constraints(m, input_to_icnn_two, icnn_bounds_affine_out, icnn_bounds_layer_out)

        difference = m.addVar(lb=min(lb - ub), ub=max(ub - lb))
        m.update()
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


def sample_uniform_over_icnn(data_samples, amount, icnns, group_indices, curr_bounds_layer_out, keep_samples=True):
    lb = curr_bounds_layer_out[0]
    ub = curr_bounds_layer_out[1]
    shape = data_samples.size(1)
    random_samples = (ub - lb) * torch.rand((amount, shape), dtype=data_type).to(device) + lb

    index_not_to_be_deleted = []
    for i, samp in enumerate(random_samples):
        is_included = True
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
            index_not_to_be_deleted.append(i)

    index_not_to_be_deleted = torch.tensor(index_not_to_be_deleted, dtype=torch.int64).to(device)
    random_samples = torch.index_select(random_samples, 0, index_not_to_be_deleted)
    if keep_samples and data_samples.size(0) > 0:
        data_samples = torch.cat([data_samples, random_samples], dim=0)
    else:
        data_samples = random_samples
    return data_samples

def sample_linspace(data_samples, amount, center, eps, keep_samples=True):
    step_number = 1  # math.floor(math.sqrt(amount))
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


def sample_boarder(data_samples, amount, center, eps, keep_samples=True):
    step_number = 10
    tensor_center = torch.zeros((step_number, data_samples.size(1)), dtype=data_type).to(device) + center
    tensor_mask = torch.zeros((data_samples.size(1), step_number, data_samples.size(1)), dtype=data_type).to(device)
    samples = torch.empty(step_number * data_samples.size(1), data_samples.size(1), dtype=data_type).to(device)
    xs = torch.linspace(-eps, eps, steps=step_number, dtype=data_type).to(device)

    for i in range(data_samples.size(1)):
        for k in range(step_number):
            tensor_mask[i][k][i] = xs[k]

    new_samples = torch.add(tensor_center, tensor_mask)
    new_samples = torch.flatten(new_samples, 0, 1)
    if keep_samples and data_samples.size(0) > 0:
        data_samples = torch.cat([data_samples, new_samples], dim=0)
    else:
        data_samples = new_samples
    return data_samples


def sample_random_sum_noise(data_samples, amount, center, eps, keep_samples=True):
    upper = data_samples.size(1) * eps
    lower = - upper
    random_sums = (upper - lower) * torch.rand((amount, 1), dtype=data_type).to(device) + lower
    random_divs = random_sums.div(data_samples.size(1))
    samples = torch.zeros((amount, data_samples.size(1)), dtype=data_type).to(device)
    samples = samples.add(random_divs)

    upper = torch.zeros((amount, data_samples.size(1)), dtype=data_type).to(device).add(eps).add(- samples)
    lower = torch.zeros((amount, data_samples.size(1)), dtype=data_type).to(device).add(-eps).add(- samples)

    noise_per_sample = (upper - lower) * torch.rand((amount, data_samples.size(1)), dtype=data_type).to(device) + lower
    sum_per_sample = torch.sum(noise_per_sample, dim=1)
    div_sum_per_sample = sum_per_sample.div(data_samples.size(1))
    for i in range(amount):
        noise_per_sample[i] = noise_per_sample[i].add(- div_sum_per_sample[i])
    samples = samples.add(noise_per_sample)

    eps_tensor = torch.zeros_like(samples, dtype=data_type).to(device) + eps
    samples = torch.where(samples <= eps, samples, eps_tensor)
    samples = torch.where(samples >= -eps, samples, -eps_tensor)

    new_samples = samples.add(center)
    if keep_samples and data_samples.size(0) > 0:
        data_samples = torch.cat([data_samples, new_samples], dim=0)
    else:
        data_samples = new_samples
    return data_samples


def sample_per_group_as_lp(data_samples, amount, affine_w, affine_b, index_to_select, model, curr_bounds_affine_out, prev_layer_index, rand_samples_percent=0, rand_sample_alternation_percent=0.2, keep_samples=True):
    upper = 1
    lower = - 1
    cs_temp = (upper - lower) * torch.rand((amount, len(index_to_select)),
                                           dtype=data_type).to(device) + lower

    cs = torch.zeros((amount, affine_w.size(0)), dtype=data_type).to(device)

    samples = torch.empty((amount, affine_w.size(1)), dtype=data_type).to(device)

    for i in range(amount):
        for k, index in enumerate(index_to_select):
            cs[i][index] = cs_temp[i][k]

    num_rand_samples = math.floor(amount * rand_samples_percent)
    alternations_per_sample = math.floor(affine_w.size(0) * rand_sample_alternation_percent)
    if num_rand_samples > 0 and alternations_per_sample > 0:
        rand_index = torch.randperm(affine_w.size(0))
        rand_index = rand_index[:alternations_per_sample]
        rand_samples = (upper - lower) * torch.rand((num_rand_samples, alternations_per_sample),
                                                    dtype=data_type).to(device) + lower
        for i in range(num_rand_samples):
            for k, index in enumerate(rand_index):
                cs[i][index] = rand_samples[i][k]

    output_prev_layer = []
    for i in range(affine_w.shape[1]):
        output_prev_layer.append(model.getVarByName("output_layer_[{}]_[{}]".format(prev_layer_index, i)))
    output_prev_layer = grp.MVar.fromlist(output_prev_layer)

    lb = curr_bounds_affine_out[0].detach().cpu().numpy()
    ub = curr_bounds_affine_out[1].detach().cpu().numpy()
    numpy_affine_w = affine_w.detach().cpu().numpy()
    numpy_affine_b = affine_b.detach().cpu().numpy()
    output_var = verbas.add_affine_constr(model, numpy_affine_w, numpy_affine_b, output_prev_layer, lb, ub, i=0)

    model.update()
    for index, c in enumerate(cs):
        c = c.detach().cpu().numpy()
        model.setObjective(c @ output_var, grp.GRB.MAXIMIZE)

        model.optimize()
        if model.Status == grp.GRB.OPTIMAL:
            samples[index] = torch.tensor(output_prev_layer.getAttr("X"), dtype=data_type).to(device)

    if keep_samples and data_samples.size(0) > 0:
        data_samples = torch.cat([data_samples, samples], dim=0)
    else:
        data_samples = samples
    return data_samples


def sample_per_group(data_samples, amount, affine_w, center, eps, index_to_select, keep_samples=True, rand_samples_percent=0, rand_sample_alternation_percent=0.2, with_noise=False, with_sign_swap=False):
    samples_per_bound = amount // 2
    eps_tensor = torch.tensor(eps, dtype=data_type).to(device)

    upper = 1
    lower = - 1
    cs_temp = (upper - lower) * torch.rand((samples_per_bound, len(index_to_select)),
                                      dtype=data_type).to(device) + lower

    cs = torch.zeros((samples_per_bound, affine_w.size(0)), dtype=data_type).to(device)

    for i in range(samples_per_bound):
        for k, index in enumerate(index_to_select):
            cs[i][index] = cs_temp[i][k]

    num_rand_samples = math.floor(amount * rand_samples_percent)
    alternations_per_sample = math.floor(affine_w.size(0) * rand_sample_alternation_percent)
    if num_rand_samples > 0 and alternations_per_sample > 0:
        rand_index = torch.randperm(affine_w.size(0))
        rand_index = rand_index[:alternations_per_sample]
        rand_samples = (upper - lower) * torch.rand((num_rand_samples, alternations_per_sample),
                                                    dtype=data_type).to(device) + lower
        for i in range(num_rand_samples):
            for k, index in enumerate(rand_index):
                cs[i][index] = rand_samples[i][k]

    affine_w_temp = torch.matmul(cs, affine_w)
    upper_samples = torch.where(affine_w_temp > 0, eps_tensor, - eps_tensor)
    lower_samples = torch.where(affine_w_temp < 0, eps_tensor, - eps_tensor)

    if with_noise:
        upper = eps
        lower = - eps
        noise_per_sample = (upper - lower) * torch.rand((samples_per_bound, data_samples.size(1)),
                                                        dtype=data_type).to(device) + lower

        upper_samples.add_(noise_per_sample)
        lower_samples.add_(noise_per_sample)

        upper_samples = torch.where(upper_samples <= eps, upper_samples, eps_tensor)
        upper_samples = torch.where(upper_samples >= -eps, upper_samples, -eps_tensor)

        lower_samples = torch.where(lower_samples <= eps, lower_samples, eps_tensor)
        lower_samples = torch.where(lower_samples >= -eps, lower_samples, -eps_tensor)

    if with_sign_swap:
        # changing sign
        swap_probability = 0.2
        if swap_probability == 0:
            swaps = torch.ones((samples_per_bound, data_samples.size(1)))
        elif swap_probability == 1:
            swaps = -1 * torch.ones((samples_per_bound, data_samples.size(1)))
        elif swap_probability <= 0.5:
            swap_probability = int(1 / swap_probability)
            swaps = torch.randint(-1, swap_probability, (samples_per_bound, data_samples.size(1)),
                                  dtype=torch.int8)
            swaps = torch.where(swaps >= 0, 1, swaps)
        else:
            swap_probability = 1 - swap_probability
            swap_probability = - int(1 / swap_probability)
            swaps = torch.randint(swap_probability, 1, (samples_per_bound, data_samples.size(1)),
                                  dtype=torch.int8)
            swaps = torch.where(swaps <= 0, 1, swaps)
        upper_samples = torch.mul(upper_samples, swaps)
        lower_samples = torch.mul(lower_samples, swaps)


    all_samples = torch.cat([upper_samples, lower_samples], dim=0)
    all_samples.add_(center)

    if keep_samples and data_samples.size(0) > 0:
        data_samples = torch.cat([data_samples, all_samples], dim=0)
    else:
        data_samples = all_samples
    return data_samples


def sample_alternate_min_max(data_samples, amount, affine_w, center, eps, keep_samples=True):
    samples_per_neuron = math.ceil(amount / affine_w.size(0))
    samples_per_bound = samples_per_neuron // 2
    eps_tensor = torch.zeros((affine_w.size(0), data_samples.size(1)), dtype=data_type).to(device) + eps

    upper = 1
    lower = - 1
    cs = (upper - lower) * torch.rand((samples_per_bound, 1, affine_w.size(0)),
                                                    dtype=data_type).to(device) + lower

    cs[1][0] = torch.zeros(affine_w.size(0), dtype=data_type).to(device)
    cs[1][0][1] = 1
    cs[1][0][23] = 1


    affine_w_temp = torch.matmul(cs, affine_w)
    upper_samples = torch.where(affine_w_temp > 0, eps_tensor, - eps_tensor)
    lower_samples = torch.where(affine_w_temp < 0, eps_tensor, - eps_tensor)

    upper = eps
    lower = - eps
    noise_per_sample = (upper - lower) * torch.rand((samples_per_bound, affine_w.size(0), data_samples.size(1)),
                                                    dtype=data_type).to(device) + lower

    upper_samples.add_(noise_per_sample)
    lower_samples.add_(noise_per_sample)


    upper_samples = torch.where(upper_samples <= eps, upper_samples, eps_tensor)
    upper_samples = torch.where(upper_samples >= -eps, upper_samples, -eps_tensor)

    lower_samples = torch.where(lower_samples <= eps, lower_samples, eps_tensor)
    lower_samples = torch.where(lower_samples >= -eps, lower_samples, -eps_tensor)

    eps_tensor = torch.zeros((affine_w.size(0), data_samples.size(1)), dtype=data_type).to(device) + eps
    upper_input = torch.where(affine_w > 0, eps_tensor, - eps_tensor)
    lower_input = torch.where(affine_w < 0, eps_tensor, - eps_tensor)
    upper_samples[0] = upper_input
    lower_samples[0] = lower_input
    upper_samples = torch.flatten(upper_samples, 0, 1)
    lower_samples = torch.flatten(lower_samples, 0, 1)
    all_samples = torch.cat([upper_samples, lower_samples], dim=0)
    all_samples.add_(center)

    return all_samples

def sample_min_max_perturbation(data_samples, amount, affine_w, center, eps, keep_samples=True, swap_probability=0.2):
    samples_per_neuron = math.ceil(amount / affine_w.size(0))
    samples_per_bound = samples_per_neuron // 2
    eps_tensor = torch.zeros((affine_w.size(0), data_samples.size(1)), dtype=data_type).to(device) + eps
    upper_input = torch.where(affine_w > 0, eps_tensor, - eps_tensor)
    lower_input = torch.where(affine_w < 0, eps_tensor, - eps_tensor)

    upper_samples = torch.zeros((samples_per_bound, affine_w.size(0), data_samples.size(1)), dtype=data_type).to(
        device) + upper_input
    lower_samples = torch.zeros((samples_per_bound, affine_w.size(0), data_samples.size(1)), dtype=data_type).to(
        device) + lower_input

    upper = eps
    lower = - eps
    noise_per_sample = (upper - lower) * torch.rand((samples_per_bound, affine_w.size(0), data_samples.size(1)),
                                                    dtype=data_type).to(device) + lower

    upper_samples.add_(noise_per_sample)
    lower_samples.add_(noise_per_sample)

    eps_tensor = torch.zeros_like(upper_samples, dtype=data_type).to(device) + eps
    upper_samples = torch.where(upper_samples <= eps, upper_samples, eps_tensor)
    upper_samples = torch.where(upper_samples >= -eps, upper_samples, -eps_tensor)

    lower_samples = torch.where(lower_samples <= eps, lower_samples, eps_tensor)
    lower_samples = torch.where(lower_samples >= -eps, lower_samples, -eps_tensor)

    # changing sign
    if swap_probability == 0:
        swaps = torch.ones((samples_per_bound, affine_w.size(0), data_samples.size(1)))
    elif swap_probability == 1:
        swaps = -1 * torch.ones((samples_per_bound, affine_w.size(0), data_samples.size(1)))
    elif swap_probability <= 0.5:
        swap_probability = int(1 / swap_probability)
        swaps = torch.randint(-1, swap_probability, (samples_per_bound, affine_w.size(0), data_samples.size(1)), dtype=torch.int8)
        swaps = torch.where(swaps >= 0, 1, swaps)
    else:
        swap_probability = 1 - swap_probability
        swap_probability = - int(1 / swap_probability)
        swaps = torch.randint(swap_probability, 1, (samples_per_bound, affine_w.size(0), data_samples.size(1)), dtype=torch.int8)
        swaps = torch.where(swaps <= 0, 1, swaps)
    upper_samples = torch.mul(upper_samples, swaps)
    lower_samples = torch.mul(lower_samples, swaps)

    upper_samples[0] = upper_input
    lower_samples[0] = lower_input

    new_samples = torch.cat([upper_samples, lower_samples], dim=0)
    new_samples.add_(center)
    new_samples = torch.flatten(new_samples, 0, 1)

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
                    from_to_neurons = [group_size * group_i,
                                       group_size * group_i + group_size]  # upper bound is exclusive
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
