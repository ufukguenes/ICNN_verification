# DeepHull Over approximated Verification
import time
from functools import reduce

import matplotlib
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision as tv
from torchvision.transforms import Compose, ToTensor, Normalize

from script import dataInit
from script.Networks import SequentialNN, ICNN
from gurobipy import Model, GRB, max_, norm, abs_, quicksum
import torch
import numpy as np

from script.Normalisation import get_std, get_mean, normalize_nn, normalize_data
from script.Verification import verification, add_non_sequential_constr, add_affine_constr
from script.dataInit import ConvexDataset, Rhombus
from script.eval import Plots_for
from script.trainFunction import train_icnn, train_sequential, train_sequential_2


def start_verification(nn: SequentialNN, input, eps=0.001, solver_time_limit=None, solver_bound=None,
                       icnn_batch_size=10,
                       icnn_epochs=10, sample_count=10000, sample_new=True):
    # todo Achtung ich muss schauen, ob gurobi upper bound inklusive ist, da ich aktuell die upper bound mit eps nicht inklusive habe
    input_flattened = torch.flatten(input)
    eps_bounds = [input_flattened.add(-eps), input_flattened.add(eps)]
    box_bounds = calculate_box_bounds(nn, eps_bounds)  # todo abbrechen, wenn die box bounds schon die eigenschaft erfüllen

    included_space, ambient_space = sample_uniform_from(input_flattened, eps, sample_count, lower_bound=0.5,
                                                        upper_bound=0.5) #todo test for when lower/upper bound is smaller then eps

    """imshow_flattened(input_flattened, (3, 32, 32))
    imshow_flattened(included_space[0], (3, 32, 32))
    imshow_flattened(included_space[100], (3, 32, 32))
    imshow_flattened(ambient_space[0], (3, 32, 32))
    imshow_flattened(ambient_space[100], (3, 32, 32))"""

    def plt_inc_amb(caption, inc, amb):
        plt.figure(figsize=(20, 10))
        plt.scatter(list(map(lambda x: x[0], inc)), list(map(lambda x: x[1], inc)))
        plt.scatter(list(map(lambda x: x[0], amb)), list(map(lambda x: x[1], amb)))
        plt.title(caption)
        plt.show()

    if should_plot:
        plt_inc_amb("start", included_space, ambient_space)

    parameter_list = list(nn.parameters())

    icnns = []
    c_values = []
    center = input_flattened
    for i in range(0, len(parameter_list) - 2, 2):  # -2 because last layer has no ReLu activation
        current_layer_index = int(i / 2)
        icnn_input_size = nn.layer_widths[current_layer_index + 1]
        icnns.append(ICNN([icnn_input_size, 10, 10, icnn_input_size, 2*icnn_input_size, 1]))
        current_icnn = icnns[current_layer_index]

        W, b = parameter_list[i], parameter_list[i + 1]

        included_space, ambient_space = apply_affine_transform(W, b, included_space, ambient_space)
        if should_plot:
            plt_inc_amb("affin e" + str(i), included_space.tolist(), ambient_space.tolist())
        included_space, ambient_space = apply_ReLU_transform(included_space, ambient_space)
        if should_plot:
            plt_inc_amb("relu " + str(i), included_space.tolist(), ambient_space.tolist())

        ambient_space = add_to_ambient_space(ambient_space, int(sample_count / 2), box_bounds[current_layer_index])

        if should_plot:
            plt_inc_amb("enhanced ambient space " + str(i), included_space.tolist(), ambient_space.tolist())

        mean = get_mean(included_space, ambient_space)
        std = get_std(included_space, ambient_space)

        normalized_included_space, normalized_ambient_space = normalize_data(included_space, ambient_space, mean, std)
        dataset = ConvexDataset(data=normalized_included_space)
        train_loader = DataLoader(dataset, batch_size=icnn_batch_size, shuffle=True)
        dataset = ConvexDataset(data=normalized_ambient_space)
        ambient_loader = DataLoader(dataset, batch_size=icnn_batch_size, shuffle=True)

        # torch.save(included_space, "../included_space_after_relu.pt")
        # torch.save(ambient_space, "../ambient_space_after_relu.pt")

        ws = list(current_icnn.ws.parameters())
        us = list(current_icnn.us.parameters())
        low = box_bounds[current_layer_index][0]
        up = box_bounds[current_layer_index][1]
        low = torch.div(torch.add(low, -mean), std)
        up = torch.div(torch.add(up, -mean), std)
        init_icnn(current_icnn, [low, up])
        # train icnn1
        train_icnn(current_icnn, train_loader, ambient_loader, epochs=icnn_epochs, hyper_lambda=1)

        normalize_nn(current_icnn, mean, std, isICNN=True)
        """        with torch.no_grad():
            ws = list(current_icnn.ws[3].parameters())
            #us = list(current_icnn.us[2].parameters())
            m2 = - torch.tensor([mean[0], mean[0], mean[1], mean[1]])
            std2 = torch.tensor([std[0], std[0], std[1], std[1]])
            #ws[1].data = torch.div(torch.add(ws[1], m2), std2)"""
        #matplotlib.use("TkAgg")
        if should_plot:
            plots = Plots_for(0, current_icnn, included_space.detach(), ambient_space.detach(), true_extremal_points,
                              [-1, 3], [-1, 3])
            plots.plt_dotted()
            plots.plt_mesh()

        # verify and enlarge convex approximation
        if i == 0:
            adversarial_input, c = verification(current_icnn, center_eps_W_b=[center.detach().numpy(), eps, W.detach().numpy(), b.detach().numpy()], has_ReLU=True)
        else:
            prev_icnn = icnns[current_layer_index - 1]
            #prev_W, prev_b = parameter_list[i-2].detach().numpy(), parameter_list[i - 1].detach().numpy()
            prev_c = c_values[current_layer_index - 1]
            adversarial_input, c = verification(current_icnn, icnn_W_b_c=[prev_icnn, W.detach().numpy(), b.detach().numpy(), prev_c], has_ReLU=True)
        c_values.append(c)
        if should_plot:
            plots.c = c
            plots.plt_dotted()

        c_values.append(0)

        # entweder oder:
        # nutze die samples weiter (dafür muss man dann das ReLU layer anwenden), und man muss schauen ob die
        # samples jetzt von ambient_space zu included_space gewechselt haben (wegen überapproximation)
        # damit könnte man gut den Fehler messen, da man ganz genau weiß wie viele elemente aus Ambient space in Icluded space übergegenagen sind

        # oder sample neue punkte

        if sample_new:
            included_space, ambient_space = sample_max_radius(current_icnn, c, sample_count, box_bounds=box_bounds[current_layer_index])
            if should_plot:
                plt_inc_amb("sampled new", included_space, ambient_space)

        else:
            included_space, ambient_space = split_new(current_icnn, included_space, ambient_space, c)

    index = len(parameter_list) - 2
    W, b = parameter_list[index], parameter_list[index + 1]
    # last_layer_picture(icnns[-1], c_values[-1], W, b, 6, solver_time_limit, solver_bound)  # todo nicht hardcoden
    A_out, b_out = Rhombus().get_A(), Rhombus().get_b()


    included_space, ambient_space = apply_affine_transform(W, b, included_space, ambient_space)
    plt_inc_amb("affin e" + str(i), included_space.tolist(), ambient_space.tolist())


    last_layer_identity(icnns[-1], c_values[-1], W, b, A_out, b_out, solver_time_limit, solver_bound)


def last_layer_identity(last_icnn: ICNN, last_c, W, b, A_out, b_out, solver_time_limit, solver_bound):
    m = Model()
    m.Params.LogToConsole = 0
    if solver_time_limit is not None:
        m.setParam("TimeLimit", solver_time_limit)

    if solver_bound is not None:
        m.setParam("BestObjStop", solver_bound)

    output_var = m.addMVar(2, lb=-float('inf'))

    icnn_input_size = last_icnn.layer_widths[0]
    output_of_penultimate_layer = m.addMVar(icnn_input_size, lb=-float('inf'))
    output_of_icnn = m.addMVar(1, lb=-float('inf'))
    add_non_sequential_constr(m, last_icnn, output_of_penultimate_layer, output_of_icnn)
    m.addConstr(output_of_icnn[0] <= last_c)

    W = W.detach().numpy()
    b = b.detach().numpy()
    #output_var = add_affine_constr(m, W, b, output_of_penultimate_layer, -1000, 1000)

    #m.addConstrs((W[i] @ output_var >= b[i] for i in range(len(W))))

    m.addConstrs(W[i] @ output_of_penultimate_layer + b[i] == output_var[i] for i in range(len(W)))

    #constr = m.addMConstr(A_out, output_var.tolist(), ">", b_out)

    max_var = m.addMVar(2, lb=-float('inf'), ub=1000)

    m.addConstr(max_var[0] == abs_(output_var[0]))  # maximize distance to origin without norm
    m.addConstr(max_var[1] == abs_(output_var[1])) # todo diese nachbedinung garantiert nicht, dass etwas außerhalb maximal ist!

    m.update()
    m.setObjective(max_var[0] + max_var[1], GRB.MAXIMIZE)
    m.optimize()

    solution = 0
    if m.Status == GRB.OPTIMAL or m.Status == GRB.TIME_LIMIT:
        # print("optimum solution with value \n {}".format(output_var.getAttr("x")))
        print("icnn_in_var {}".format(output_of_penultimate_layer.getAttr("x")))
        print("max_var {}".format(output_of_icnn.getAttr("x")))
        output = output_var.getAttr("x")
        print("output {}".format(output_var.getAttr("x")))
        sol = output_of_penultimate_layer.getAttr("x")
        sol = torch.tensor(sol, dtype=torch.float64)
        sol = torch.unsqueeze(sol, 0)
        out = last_icnn(sol)
        print(out)

        print("======================= \n")
        if Rhombus().f(output):
            print("Verification was successful")
        else:
            print("Verification failed")
        print("\n =======================")

def init_icnn(icnn: ICNN, box_bounds):
    # todo check for the layer to have the right dimensions
    with torch.no_grad():
        k = len(icnn.ws)
        for i in range(0, len(icnn.ws)):
            ws = list(icnn.ws[i].parameters())
            ws[0].data = torch.zeros_like(ws[0], dtype=torch.float64)
            ws[1].data = torch.zeros_like(ws[1], dtype=torch.float64)

        for elem in icnn.us:
            w = list(elem.parameters())
            w[0].data = torch.zeros_like(w[0], dtype=torch.float64)

        ws = list(icnn.ws[3].parameters()) # bias for first relu activation with weights from us (in second layer)
        us = list(icnn.us[2].parameters()) # us is used because values in ws are set to 0 when negative
        b = torch.zeros_like(ws[1], dtype=torch.float64)
        w = torch.zeros_like(us[0], dtype=torch.float64)
        for i in range(len(box_bounds)):
            w[2*i][i] = 1
            w[2*i+1][i] = -1
            b[2*i] = - box_bounds[1][i] # upper bound
            b[2*i+1] = box_bounds[0][i] # lower bound
        ws[0].data = torch.zeros_like(ws[0], dtype=torch.float64)
        ws[1].data = b
        us[0].data = w

        last = list(icnn.ws[4].parameters())
        last[0].data = torch.ones_like(last[0], dtype=torch.float64)
        last[1].data = torch.zeros_like(last[1], dtype=torch.float64)


def last_layer_picture(last_icnn: ICNN, last_c, W, b, label, solver_time_limit, solver_bound):
    m = Model()

    if solver_time_limit is not None:
        m.setParam("TimeLimit", solver_time_limit)

    if solver_bound is not None:
        m.setParam("BestObjStop", solver_bound)

    icnn_input_size = last_icnn.layer_widths[0]
    output_of_penultimate_layer = m.addMVar(icnn_input_size, lb=-float('inf'))
    output_of_icnn = m.addMVar(1, lb=-float('inf'))
    add_non_sequential_constr(m, last_icnn, output_of_penultimate_layer, output_of_icnn)

    W = W.detach().numpy()
    b = b.detach().numpy()
    output_var = add_affine_constr(m, W, b, output_of_penultimate_layer, -1000, 1000)
    m.addConstr(output_of_icnn <= last_c)

    difference = m.addVars(9, lb=-float('inf'))
    m.addConstrs(difference[i] == output_var.tolist()[i] - output_var.tolist()[label] for i in range(0, label))
    m.addConstrs(difference[i - 1] == output_var.tolist()[i] - output_var.tolist()[label] for i in range(label + 1, 10))

    # m.addConstrs(difference[i] == output_var.tolist()[i] - output_var.tolist()[label] for i in range(10))
    max_var = m.addVar(lb=-float('inf'), ub=1000)
    m.addConstr(max_var == max_(difference))

    m.update()
    m.setObjective(max_var, GRB.MAXIMIZE)
    m.optimize()

    solution = 0
    if m.Status == GRB.OPTIMAL or m.Status == GRB.TIME_LIMIT or m.Status == GRB.USER_OBJ_LIMIT:
        for o in difference.select():
            print(o.getAttr("x"))
        print("optimum solution with value \n {}".format(output_var.getAttr("x")))
        print("max_var {}".format(max_var.getAttr("x")))
        sol = output_of_penultimate_layer.getAttr("x")

        sol = torch.tensor(sol, dtype=torch.float64)
        sol = torch.unsqueeze(sol, 0)
        out = last_icnn(sol)
        print(out)


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

def sample_max_radius(icnn, c, sample_size, box_bounds=None):
    m = Model()
    m.Params.LogToConsole = 0

    icnn_input_size = icnn.layer_widths[0]
    input_to_icnn_one = m.addMVar(icnn_input_size, lb=-float('inf'))
    input_to_icnn_two = m.addMVar(icnn_input_size, lb=-float('inf'))
    output_of_icnn_one = m.addMVar(1, lb=-float('inf'))
    output_of_icnn_two = m.addMVar(1, lb=-float('inf'))
    add_non_sequential_constr(m, icnn, input_to_icnn_one, output_of_icnn_one)
    m.addConstr(output_of_icnn_one <= c)
    add_non_sequential_constr(m, icnn, input_to_icnn_two, output_of_icnn_two)
    m.addConstr(output_of_icnn_two <= c)


    difference = m.addVar(lb=-float('inf'))

    center_values = []
    eps_values = []
    for i in range(icnn_input_size):
        diff_const = m.addConstr(difference == input_to_icnn_one[i] - input_to_icnn_two[i])
        m.setObjective(difference, GRB.MAXIMIZE)
        m.optimize()
        if m.Status == GRB.OPTIMAL:
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

        distance_to_center_upper = [abs(center_values[k] - choice_for_upper_bound[i]) for i in range(len(choice_for_upper_bound))]
        distance_to_center_lower = [abs(center_values[k] - choice_for_lower_bound[i]) for i in range(len(choice_for_lower_bound))]
        arg_min_upper_bound = np.argmin(distance_to_center_upper)
        arg_min_lower_bound = np.argmin(distance_to_center_lower)
        best_upper_bound.append(choice_for_upper_bound[arg_min_upper_bound])
        best_lower_bound.append(choice_for_lower_bound[arg_min_lower_bound])

    best_upper_bound = torch.tensor(best_upper_bound, dtype=torch.float64)
    best_lower_bound = torch.tensor(best_lower_bound, dtype=torch.float64)

    samples = (best_upper_bound - best_lower_bound) * torch.rand((sample_size, input_size),dtype=torch.float64) + best_lower_bound

    for samp in samples:
        samp = torch.unsqueeze(samp, 0)
        out = icnn(samp)
        if out <= c:
            included_space = torch.cat([included_space, samp], dim=0)
        else:
            ambient_space = torch.cat([ambient_space, samp], dim=0)

    print("included space num samples {}, ambient space num samples {}".format(len(included_space), len(ambient_space)))
    return included_space, ambient_space

    # i cant use norm, because its not convex
    """
    max_var = m.addVar(lb=-float('inf'))
    difference = m.addVars(icnn_input_size, lb=-float('inf'))
    
    m.addConstrs(difference[i] == input_to_icnn_one[i] - input_to_icnn_two[i] for i in range(icnn_input_size))
    m.addConstr(max_var == max_(difference))
    m.setObjective(max_var, GRB.MAXIMIZE)
    m.optimize()

    if m.Status == GRB.OPTIMAL:
        point_one = input_to_icnn_one.getAttr("x")
        point_two = input_to_icnn_two.getAttr("x")
        max_dist = max_var.getAttr("x")
        center_point = (point_one + point_two) / 2 # das ist nicht der wahre mittelpunkt, aber da p1 und p2 am rand liegen müssen und in einer variable maximal entfernt liegen ist das eine gute? annährung für einen mittelpunkt

        eps = max_dist / 2

        return sample_uniform_from(torch.tensor(center_point, dtype=torch.float64), eps, sample_size, icnn_c=[icnn, c])

"""

def add_to_ambient_space(ambient_space, count, box_bounds):
    lb = box_bounds[0] - 0.5
    ub = box_bounds[1] + 0.5
    shape = ambient_space.size()[1]
    samples = (ub - lb) * torch.rand((count, shape), dtype=torch.float64) + lb
    ambient_space = torch.cat([ambient_space, samples], dim=0)
    return ambient_space

def calculate_box_bounds(nn, input_bounds):
    # todo for now this only works for sequential nets
    parameter_list = list(nn.parameters())

    next_lower_bounds = input_bounds[0]
    next_upper_bounds = input_bounds[1]

    bounds_per_layer = []
    for i in range(0, len(parameter_list), 2):
        W, b = parameter_list[i], parameter_list[i + 1]

        new_upper_bound = []
        new_lower_bound = []
        for k, row in enumerate(W):
            upper_multiply_value = []
            lower_multiply_value = []
            for l in range(row.size(dim=0)):
                if row[l] < 0:
                    upper_multiply_value.append(next_lower_bounds[l])
                    lower_multiply_value.append(next_upper_bounds[l])
                else:
                    upper_multiply_value.append(next_upper_bounds[l])
                    lower_multiply_value.append(next_lower_bounds[l])

            upper_multiply_value = torch.tensor(upper_multiply_value, dtype=torch.float64)
            lower_multiply_value = torch.tensor(lower_multiply_value, dtype=torch.float64)
            affine_bound_upper = torch.sum(torch.mul(W[k], upper_multiply_value)).add(b[k]).item()
            affine_bound_lower = torch.sum(torch.mul(W[k], lower_multiply_value)).add(b[k]).item()
            new_upper_bound.append(max(0, affine_bound_upper)) # ReLU bound
            new_lower_bound.append(max(0, affine_bound_lower))

        next_lower_bounds = torch.tensor(new_lower_bound, dtype=torch.float64)
        next_upper_bounds = torch.tensor(new_upper_bound, dtype=torch.float64)
        bounds_per_layer.append([next_lower_bounds, next_upper_bounds])

    return bounds_per_layer

def split_new(icnn, included_space, ambient_space, c):
    moved = 0

    in_outs = []
    for elem in included_space:
        elem = torch.unsqueeze(elem, 0)
        out = icnn(elem)
        in_outs.append(out.item())

    outputs = []
    for i, elem in enumerate(ambient_space):
        elem = torch.unsqueeze(elem, 0)
        output = icnn(elem)
        outputs.append(output.item())
        if output <= c:
            included_space = torch.cat([included_space, elem], dim=0)
            ambient_space = torch.cat([ambient_space[:i - moved], ambient_space[i + 1 - moved:]])
            moved += 1
    mini = min(outputs)
    maxi = max(in_outs)

    return included_space, ambient_space


def apply_affine_transform(W, b, included_space, ambient_space):
    t = time.time()
    affine_inc2 = torch.empty((included_space.shape[0], b.shape[0]), dtype=torch.float64)
    affine_amb2 = torch.empty((ambient_space.shape[0], b.shape[0]), dtype=torch.float64)

    for i in range(included_space.shape[0]):
        affine_inc2[i] = torch.matmul(W, included_space[i]).add(b)
    for i in range(ambient_space.shape[0]):
        affine_amb2[i] = torch.matmul(W, ambient_space[i]).add(b)
    t = time.time() - t
    print("Time: {}".format(t))

    return affine_inc2, affine_amb2


def apply_ReLU_transform(included_space, ambient_space):
    relu = torch.nn.ReLU()
    included_space = relu(included_space)
    ambient_space = relu(ambient_space)
    return included_space, ambient_space


def imshow_flattened(img_flattened, shape):
    img = np.reshape(img_flattened, shape)
    img = img / 2 + .05  # revert normalization for viewing
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()


def plot_2d(nn, included, ambient):
    plt.figure(figsize=(20, 10))
    included_out_x = []
    included_out_y = []
    for x in included:
        x = torch.unsqueeze(x, dim=0)
        out = nn(x)
        included_out_x.append(out[0][0].item())
        included_out_y.append(out[0][1].item())
    ambient_out_x = []
    ambient_out_y = []
    for x in ambient:
        x = torch.unsqueeze(x, dim=0)
        out = nn(x)
        ambient_out_x.append(out[0][0].item())
        ambient_out_y.append(out[0][1].item())

    plt.scatter(included_out_x, included_out_y)
    plt.scatter(ambient_out_x, ambient_out_y)
    plt.show()


"""transform = Compose([ToTensor(),
                     Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
                    )

training_data = CIFAR10(root="../cifar",
                        train=True,
                        download=True,
                        transform=transform)
images, labels = training_data.__getitem__(0)
testimage, testlabel = torch.unsqueeze(images, 0), torch.unsqueeze(torch.tensor(labels), 0)

nn = SequentialNN([32 * 32 * 3, 1024, 512, 10])
nn.load_state_dict(torch.load("../cifar_fc.pth", map_location=torch.device('cpu')), strict=False)

torch.set_default_dtype(torch.float64)
start_verification(nn, images)"""

batch_size = 10
epochs = 30
number_of_train_samples = 10000
hyper_lambda = 1
x_range = [-1.5, 1.5]
y_range = [-1.5, 1.5]

included_space, ambient_space = Rhombus().get_uniform_samples(number_of_train_samples, x_range,
                                                              y_range)  # samples will be split in inside and outside the rhombus


true_extremal_points = Rhombus().get_extremal_points()
dataset_in = ConvexDataset(data=included_space)
train_loader = DataLoader(dataset_in, batch_size=batch_size, shuffle=True)
dataset = ConvexDataset(data=ambient_space)
ambient_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)



"""W1 = [1. 1.; 1. -1.]
b1 = [0., 0.]
W2 = [1. 1.; 1. -1.]
b2 = [-0.5, 0.]
W3 = [-1. 1.; 1. 1.]
b3 = [3., 0.] """

nn = SequentialNN([2, 2, 2, 2])
should_plot = True

with torch.no_grad():
    parameter_list = list(nn.parameters())
    parameter_list[0].data = torch.tensor([[1, 1], [1, -1]], dtype=torch.float64)
    parameter_list[1].data = torch.tensor([0, 0], dtype=torch.float64)
    parameter_list[2].data = torch.tensor([[1, 1], [1, -1]], dtype=torch.float64)
    parameter_list[3].data = torch.tensor([-0.5, 0], dtype=torch.float64)
    parameter_list[4].data = torch.tensor([[-1, 1], [1, 1]], dtype=torch.float64)
    parameter_list[5].data = torch.tensor([3, 0], dtype=torch.float64)

#nn.load_state_dict(torch.load("nn_2x2.pt"), strict=False)
#train_sequential_2(nn, train_loader, ambient_loader, epochs=epochs)


#matplotlib.use('TkAgg')

#plot_2d(nn, included_space, ambient_space)
#torch.save(nn.state_dict(), "nn_2x2.pt")


test_image = torch.tensor([[0, 0]], dtype=torch.float64)
start_verification(nn, test_image, eps=1, sample_new=True)
