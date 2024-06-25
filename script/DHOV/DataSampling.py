import math
import random
import time

import gurobipy
import numpy as np
import torch
import gurobipy as grp
from script.settings import device, data_type
import script.Verification.VerificationBasics as verbas

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

    cs = torch.nn.functional.normalize(cs, dim=1)

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


def sample_per_group(data_samples, amount, affine_w, input_bounds, index_to_select, keep_samples=True, rand_samples_percent=0, rand_sample_alternation_percent=0.2, with_noise=False, with_sign_swap=False):
    samples_per_bound = amount // 2
    lower_bounds = input_bounds[0]
    upper_bounds = input_bounds[1]

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
    upper_samples = torch.where(affine_w_temp > 0, upper_bounds, lower_bounds)
    lower_samples = torch.where(affine_w_temp < 0, upper_bounds, lower_bounds)

    if with_noise:
        noise_per_sample = (upper_bounds - lower_bounds) * torch.rand((samples_per_bound, data_samples.size(1)),
                                                        dtype=data_type).to(device) + lower_bounds

        upper_samples = upper_samples.add(noise_per_sample)
        lower_samples = lower_samples.add(noise_per_sample)

        upper_samples = torch.where(upper_samples <= upper_bounds, upper_samples, upper_bounds)
        upper_samples = torch.where(upper_samples >= lower_bounds, upper_samples, lower_bounds)

        lower_samples = torch.where(lower_samples <= upper_bounds, lower_samples, upper_bounds)
        lower_samples = torch.where(lower_samples >= lower_bounds, lower_samples, lower_bounds)

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

    if keep_samples and data_samples.size(0) > 0:
        data_samples = torch.cat([data_samples, all_samples], dim=0)
    else:
        data_samples = all_samples
    return data_samples
