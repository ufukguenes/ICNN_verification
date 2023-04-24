import math
import random
import time

import gurobipy
import numpy as np
import torch
import gurobipy as grp
from script.settings import device, data_type
import script.Verification.VerificationBasics as verbas
from multiprocessing.dummy import Pool

def sample_max_radius(icnns, sample_size, group_indices, curr_bounds_layer_out, keep_ambient_space=False):
    input_size = len(curr_bounds_layer_out[0])
    samples = (curr_bounds_layer_out[1] - curr_bounds_layer_out[0]) * torch.rand((sample_size, input_size),
                                                                 dtype=data_type).to(device) + curr_bounds_layer_out[0]
    included_space = torch.empty(0, dtype=data_type).to(device)
    ambient_space = torch.empty(0, dtype=data_type).to(device)

    i = 0
    for samp in samples:
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
            included_space = torch.cat([included_space, samp], dim=0)
            print(i)
            i+=1
            if len(included_space) == sample_size:
                break
        elif keep_ambient_space:
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
        else:
            print("Model unfeasible?")

    if keep_samples and data_samples.size(0) > 0:
        data_samples = torch.cat([data_samples, samples], dim=0)
    else:
        data_samples = samples
    return data_samples

def sample_feasible(included_space, amount, affine_w, affine_b, index_to_select, model, curr_bounds_affine_out, curr_bounds_layer_out, prev_layer_index, keep_samples=True):

    out_samples_1 = torch.empty((0, len(index_to_select)))
    bounds_of_group_lb = torch.index_select(curr_bounds_layer_out[0], 0, index_to_select)
    bounds_of_group_ub = torch.index_select(curr_bounds_layer_out[1], 0, index_to_select)
    out_samples_1 = samples_uniform_over(out_samples_1, amount // 2, [bounds_of_group_lb, bounds_of_group_ub])

    out_samples_2 = torch.empty((0, len(index_to_select)))
    bounds_of_group_lb = torch.index_select(curr_bounds_affine_out[0], 0, index_to_select)
    bounds_of_group_ub = torch.index_select(curr_bounds_affine_out[1], 0, index_to_select)
    out_samples_2 = samples_uniform_over(out_samples_2, amount // 2, [bounds_of_group_lb, bounds_of_group_ub])

    out_samples = torch.cat([out_samples_1, out_samples_2], dim=0)

    output_prev_layer = []
    for i in range(affine_w.shape[1]):
        output_prev_layer.append(model.getVarByName("output_layer_[{}]_[{}]".format(prev_layer_index, i)))
    output_prev_layer = grp.MVar.fromlist(output_prev_layer)

    lb = curr_bounds_affine_out[0].detach().cpu().numpy()
    ub = curr_bounds_affine_out[1].detach().cpu().numpy()
    numpy_affine_w = affine_w.detach().cpu().numpy()
    numpy_affine_b = affine_b.detach().cpu().numpy()
    output_var = verbas.add_affine_constr(model, numpy_affine_w, numpy_affine_b, output_prev_layer, lb, ub, i=0)
    """out_lb = curr_bounds_layer_out[0].detach().cpu().numpy()
    out_ub = curr_bounds_layer_out[1].detach().cpu().numpy()
    output_var = verbas.add_relu_as_lp(model, output_var, len(affine_b), out_lb, out_ub)"""
    model.update()

    in_samples = torch.empty((0, affine_w.size(1)), dtype=data_type).to(device)
    ambient_samples = torch.empty((0, len(index_to_select)), dtype=data_type).to(device)
    overall_time = 0
    only_feasible_time = 0
    for progress, sample in enumerate(out_samples):
        sample_np = sample.detach().cpu().numpy()

        for count_i, actual_index in enumerate(index_to_select):
            output_var[actual_index].setAttr("LB", sample_np[count_i])
            output_var[actual_index].setAttr("UB", sample_np[count_i])

        model.setObjective(output_var[index_to_select[0]], grp.GRB.MAXIMIZE)

        t = time.time()
        model.optimize()
        overall_time += time.time() - t

        if model.Status == grp.GRB.OPTIMAL:
            only_feasible_time += time.time() - t
            in_sample = torch.tensor(output_prev_layer.getAttr("X"), dtype=data_type).to(device)
            in_samples = torch.cat([in_samples, torch.unsqueeze(in_sample, 0)], dim=0)
            #print("worked: {}".format(progress))
        else:
            ambient_samples = torch.cat([ambient_samples, torch.unsqueeze(sample, 0)], dim=0)
            pass
            #print("Model unfeasible?")

    print("        overall time for feasible test: {}".format(overall_time))
    print("        time for only feasible points: {}".format(only_feasible_time))
    if keep_samples and included_space.size(0) > 0:
        included_space = torch.cat([included_space, in_samples], dim=0)
    else:
        included_space = in_samples

    return included_space, ambient_samples

def sample_at_0(data_samples, amount, affine_w, affine_b, index_to_select, model, curr_bounds_affine_out, prev_layer_index):

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

    max_x_value_by_index = []
    for i, index in enumerate(index_to_select):
        to_be_removed = []
        indices_leq_0 = index_to_select.tolist()
        indices_leq_0.pop(i)
        for const in indices_leq_0:
            to_be_removed.append(model.addConstr(output_var[const] <= 0))

        model.setObjective(output_var[index], grp.GRB.MAXIMIZE)
        model.update()

        model.optimize()
        if model.Status == grp.GRB.OPTIMAL:
            max_x_value_by_index.append(torch.tensor(output_var[index].getAttr("X"), dtype=data_type).to(device))
        else:
            print("Model unfeasible?")
        model.remove(to_be_removed)

    amount_per_max = amount // len(max_x_value_by_index)
    for index, max_val in enumerate(max_x_value_by_index):
        new_samples = torch.zeros((amount_per_max, affine_b.size(0)), dtype=data_type).to(device)
        line_space = torch.linspace(0, max_val, amount_per_max, dtype=data_type)
        max_val_index = index_to_select[index]
        new_samples[:, max_val_index] = line_space
        data_samples = torch.cat([data_samples, new_samples], dim=0)

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

def parallel_per_group_as_lp(inc_space_sample_count, affine_w, affine_b, index_to_select, curr_bounds_affine_out, prev_layer_index, rand_samples_percent=0, rand_sample_alternation_percent=0.2):
    num_processes = 1
    pool = Pool(num_processes)

    args = []
    for i in range(num_processes):
        detached_bounds = [[x.detach().numpy() for x in curr_bounds_affine_out[0]],
                           [x.detach().numpy() for x in curr_bounds_affine_out[1]]]
        args.append([inc_space_sample_count // num_processes, affine_w.detach().clone(), affine_b.detach().clone(), index_to_select,
             detached_bounds, prev_layer_index, rand_samples_percent, rand_sample_alternation_percent])

    results = pool.map(parallel_helper, args)
    return torch.cat(results, dim=0)

def parallel_helper(args):
    keep_samples = False
    amount = args[0]
    affine_w = args[1]
    data_samples = torch.empty((0, affine_w.size(1)), dtype=data_type).to(device).detach()
    affine_b = args[2]
    index_to_select= args[3]
    curr_bounds_affine_out= args[4]
    prev_layer_index= args[5]
    rand_samples_percent = args[6]
    rand_sample_alternation_percent = args[7]
    env = gurobipy.Env("")
    model = gurobipy.read("temp_model.lp", env)
    model.Params.LogToConsole = 0
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

    lb = curr_bounds_affine_out[0]
    ub = curr_bounds_affine_out[1]
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
        else:
            model.computeIIS()
            print("constraint")
            all_constr = model.getConstrs()

            for const in all_constr:
                if const.IISConstr:
                    print(const)

            print("lower bound")
            all_var = model.getVars()
            for var in all_var:
                if var.IISLB:
                    print(var)

            print("upper bound")
            all_var = model.getVars()
            for var in all_var:
                if var.IISUB:
                    print(var)
            return

    if keep_samples and data_samples.size(0) > 0:
        data_samples = torch.cat([data_samples, samples], dim=0)
    else:
        data_samples = samples
    return data_samples



