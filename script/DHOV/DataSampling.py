import numpy as np
import torch
import gurobipy as grp

import script.Verification.VerificationBasics as verbas
import script.Verification.MilpVerification as milp


def sample_max_radius(icnn, c, sample_size, box_bounds=None):
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


def add_samples_uniform_over(data_samples, amount, bounds, keep_samples=True, padding=0):
    lb = bounds[0] - padding
    ub = bounds[1] + padding
    shape = data_samples.size(1)
    random_samples = (ub - lb) * torch.rand((amount, shape), dtype=torch.float64) + lb
    if keep_samples:
        data_samples = torch.cat([data_samples, random_samples], dim=0)
    else:
        data_samples = random_samples


    return data_samples


def regroup_samples(icnn, c, included_space, ambient_space):
    moved = 0
    size_ambient_space = ambient_space.size(0)
    for i, elem in enumerate(size_ambient_space):
        elem = torch.unsqueeze(elem, 0)
        output = icnn(elem)
        if output <= c:
            included_space = torch.cat([included_space, elem], dim=0)
            ambient_space = torch.cat([ambient_space[:i - moved], ambient_space[i + 1 - moved:]])
            moved += 1

    return included_space, ambient_space


def sample_uniform_from(input_flattened, eps, sample_size, icnn_c=None, lower_bound=None, upper_bound=None, ):
    input_size = input_flattened.size(dim=0)

    lb = - eps - 0.001
    ub = eps + 0.001

    if lower_bound is not None:
        lb = - eps - lower_bound
    if upper_bound is not None:
        ub = eps + upper_bound

    lower_bound = lb
    upper_bound = ub


    if type(eps) == list and icnn_c is not None and len(eps) == input_size:

        icnn = icnn_c[0]
        c = icnn_c[1]
        included_space = torch.empty(0, dtype=torch.float64)
        ambient_space = torch.empty(0, dtype=torch.float64)
        displacements = torch.rand((sample_size, input_size), dtype=torch.float64)
        displacements = displacements.detach()
        for i, disp in enumerate(displacements):
            for k, val in enumerate(disp):
                upper_bound = min(eps[k] * 1.3, eps[k] + 0.004)
                lower_bound = max(- eps[k] * 1.3, - eps[k] - 0.004)
                val = (upper_bound - lower_bound) * val + lower_bound
                displacements[i][k] = val


        for i in range(sample_size):
            disp = input_flattened + displacements[i]
            disp = torch.unsqueeze(disp, 0)
            out = icnn(disp)
            if out <= c:
                included_space = torch.cat([included_space, disp], dim=0)
            else:
                ambient_space = torch.cat([ambient_space, disp], dim=0)

        print("included space num samples {}, ambient space num samples {}".format(len(included_space), len(ambient_space)))
        return included_space, ambient_space

    if icnn_c is None:
        sample_size = int(sample_size / 2)
        included_space = torch.empty((sample_size, input_size), dtype=torch.float64)
        ambient_space = torch.empty((sample_size, input_size), dtype=torch.float64)

        lower = - eps
        upper = eps
        displacements_included_space = (upper - lower) * torch.rand((sample_size, input_size),
                                                                    dtype=torch.float64) + lower
        lower = lower_bound
        upper = upper_bound
        displacements_ambient_space = (upper - lower) * torch.rand((sample_size, input_size),
                                                                   dtype=torch.float64) + lower

        # making sure that at least one value is outside the ball with radius eps
        for i, displacement in enumerate(displacements_ambient_space):
            argmax_displacement = torch.argmax(displacement)
            argmin_displacement = torch.argmin(displacement)
            max_displacement = displacement[argmax_displacement]
            min_displacement = displacement[argmin_displacement]

            if max_displacement >= eps or min_displacement < -eps:
                continue
            if max_displacement < eps:
                displacement[argmax_displacement] = upper
                continue
            if min_displacement >= -eps:
                displacement[argmin_displacement] = lower

        for i in range(sample_size):
            included_space[i] = input_flattened + displacements_included_space[i]
            ambient_space[i] = input_flattened + displacements_ambient_space[i]

    else:
        icnn = icnn_c[0]
        c = icnn_c[1]
        included_space = torch.empty(0, dtype=torch.float64)
        ambient_space = torch.empty(0, dtype=torch.float64)
        displacements = (upper_bound - lower_bound) * torch.rand((sample_size, input_size),
                                                                 dtype=torch.float64) + lower_bound
        for i in range(sample_size):
            disp = input_flattened + displacements[i]
            disp = torch.unsqueeze(disp, 0)
            out = icnn(disp)
            if out <= c:
                included_space = torch.cat([included_space, disp], dim=0)
            else:
                ambient_space = torch.cat([ambient_space, disp], dim=0)

    return included_space, ambient_space


def apply_affine_transform(W, b, data_samples):
    transformed_samples = torch.empty_like(data_samples, dtype=torch.float64)
    for i in range(data_samples.shape[0]):
        transformed_samples[i] = torch.matmul(W, data_samples[i]).add(b)

    return transformed_samples


def apply_ReLU_transform(data_samples):
    relu = torch.nn.ReLU()
    transformed_samples = relu(data_samples)

    return transformed_samples
