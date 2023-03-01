# DeepHull Over approximated Verification
import math
import random
import time

from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from script.NeuralNets.Networks import SequentialNN
import torch
import numpy as np

import script.DHOV.Normalisation as norm
import script.Verification.Verification as ver
import script.Verification.VerificationBasics as verbas
import script.DHOV.DataSampling as ds
import script.DHOV.DataOptimization as dop
from script.dataInit import ConvexDataset
from script.eval import Plots_for
from script.NeuralNets.trainFunction import train_icnn, train_icnn_outer
from script.settings import device, data_type

# todo zu Klasse umwandeln
"""
should_plot has values: none, simple, detailed, verification
optimizer has values: adam, LBFGS, SdLBFGS
adapt_lambda has values: none, high_low, included
"""


def start_verification(nn: SequentialNN, input, icnn_factory, group_size, eps=0.001, icnn_batch_size=1000,
                       icnn_epochs=100,
                       sample_count=1000, break_after=None, use_icnn_bounds=False, use_fixed_neurons=False,
                       keep_ambient_space=False, sample_new=True, use_over_approximation=True,
                       sample_over_input_space=False, sample_over_output_space=True, data_grad_descent_steps=0,
                       train_outer=False, preemptive_stop=True, even_gradient_training=False, force_inclusion_steps=0,
                       init_network=False, adapt_lambda="none", should_plot='none', optimizer="adam"):
    valid_adapt_lambda = ["none", "high_low", "included"]
    valid_should_plot = ["none", "simple", "detailed", "verification", "output"]
    valid_optimizer = ["adam", "LBFGS", "SdLBFGS"]

    parameter_list = list(nn.parameters())
    force_break = False

    if adapt_lambda not in valid_adapt_lambda:
        raise AttributeError(
            "Expected adaptive lambda mode one of: {}, actual: {}".format(valid_adapt_lambda, adapt_lambda))
    if should_plot not in valid_should_plot:
        raise AttributeError("Expected plotting mode one of: {}, actual: {}".format(valid_should_plot, should_plot))
    if optimizer not in valid_optimizer:
        raise AttributeError("Expected optimizer one of: {}, actual: {}".format(valid_optimizer, optimizer))
    if force_inclusion_steps < 0:
        raise AttributeError("Expected force_inclusion to be:  >= 0 , got: {}".format(force_inclusion_steps))
    if data_grad_descent_steps < 0 or data_grad_descent_steps > icnn_epochs:
        raise AttributeError(
            "Expected data_grad_descent_steps to be:  >= {}} , got: {}".format(icnn_epochs, data_grad_descent_steps))

    input_flattened = torch.flatten(input)
    eps_bounds = [input_flattened.add(-eps), input_flattened.add(eps)]

    if use_icnn_bounds:
        bounds_affine_out, bounds_layer_out = [], []
    else:
        bounds_affine_out, bounds_layer_out = nn.calculate_box_bounds(
            eps_bounds)  # todo abbrechen, wenn die box bounds schon die eigenschaft erfüllen

    included_space = torch.empty((0, input_flattened.size(0)), dtype=data_type).to(device)
    included_space = ds.samples_uniform_over(included_space, int(sample_count / 2), eps_bounds)
    ambient_space = torch.empty((0, input_flattened.size(0)), dtype=data_type).to(device)
    original_included_space = torch.empty((0, input_flattened.size(0)), dtype=data_type).to(device)
    original_ambient_space = torch.empty((0, input_flattened.size(0)), dtype=data_type).to(device)

    if should_plot in valid_should_plot and should_plot != "none":
        original_included_space, original_ambient_space = included_space, ambient_space

    center = input_flattened

    force_inclusion_steps += 1
    data_grad_descent_steps += 1

    fixed_neuron_per_layer_lower = []
    fixed_neuron_per_layer_upper = []
    num_fixed_neurons_layer = []
    number_of_fixed_neurons = 0
    all_group_indices = []
    list_of_icnns = []

    for i in range(0, len(parameter_list) - 2, 2):  # -2 because last layer has no ReLu activation
        current_layer_index = int(i / 2)
        print("")
        print("approximation of layer: {}".format(current_layer_index))

        affine_w, affine_b = parameter_list[i], parameter_list[i + 1]

        if use_icnn_bounds:
            if i == 0:
                affine_out_lb, affine_out_ub = verbas.calc_affine_out_bound(affine_w, affine_b, eps_bounds[0],
                                                                            eps_bounds[1])
            else:
                affine_out_lb, affine_out_ub = verbas.calc_affine_out_bound(affine_w, affine_b,
                                                                            bounds_layer_out[current_layer_index - 1][
                                                                                0],
                                                                            bounds_layer_out[current_layer_index - 1][
                                                                                1])
            relu_out_lb, relu_out_ub = verbas.calc_relu_out_bound(affine_out_lb, affine_out_ub)
            bounds_affine_out.append([affine_out_lb, affine_out_ub])
            bounds_layer_out.append([relu_out_lb, relu_out_ub])

        fix_upper = []
        fix_lower = []
        if use_fixed_neurons:
            for neuron_index, (lb, ub) in enumerate(
                    zip(bounds_affine_out[current_layer_index][0], bounds_affine_out[current_layer_index][1])):
                if ub <= 0:
                    fix_upper.append(neuron_index)
                    number_of_fixed_neurons += 1
                elif lb >= 0:
                    fix_lower.append(neuron_index)
                    number_of_fixed_neurons += 1
        fixed_neuron_per_layer_lower.append(fix_lower)
        fixed_neuron_per_layer_upper.append(fix_upper)
        num_fixed_neurons_layer.append(len(fix_lower) + len(fix_upper))
        print("    number of fixed neurons for current layer: {}".format(len(fix_lower) + len(fix_upper)))

        if not keep_ambient_space:
            ambient_space = torch.empty((0, nn.layer_widths[current_layer_index]), dtype=data_type).to(device)

        if sample_over_input_space:
            if i == 0:
                ambient_space = ds.sample_uniform_excluding(ambient_space, int(sample_count / 2), eps_bounds,
                                                            excluding_bound=eps_bounds, padding=0.5)
            else:
                # todo test for when lower/upper bound is smaller then eps
                ambient_space = ds.sample_uniform_excluding(ambient_space, int(sample_count / 2),
                                                            bounds_layer_out[current_layer_index - 1],
                                                            icnns=list_of_icnns[current_layer_index - 1],
                                                            layer_index=current_layer_index, group_size=group_size,
                                                            padding=0.5)

        if should_plot == "detailed":
            plt_inc_amb("start " + str(i), included_space.tolist(), ambient_space.tolist())

        included_space = ds.apply_affine_transform(affine_w, affine_b, included_space)
        ambient_space = ds.apply_affine_transform(affine_w, affine_b, ambient_space)

        if should_plot in valid_should_plot and should_plot != "none":
            original_included_space = ds.apply_affine_transform(affine_w, affine_b, original_included_space)
            original_ambient_space = ds.apply_affine_transform(affine_w, affine_b, original_ambient_space)
            if should_plot == "detailed":
                plt_inc_amb("affin e" + str(i), included_space.tolist(), ambient_space.tolist())
                plt_inc_amb("original affin e" + str(i), original_included_space.tolist(),
                            original_ambient_space.tolist())

        included_space = ds.apply_relu_transform(included_space)
        ambient_space = ds.apply_relu_transform(ambient_space)

        if should_plot in valid_should_plot and should_plot != "none":
            original_included_space = ds.apply_relu_transform(original_included_space)
            original_ambient_space = ds.apply_relu_transform(original_ambient_space)
            if should_plot == "detailed":
                plt_inc_amb("relu " + str(i), included_space.tolist(), ambient_space.tolist())
                plt_inc_amb("original relu e" + str(i), original_included_space.tolist(),
                            original_ambient_space.tolist())

        if sample_over_output_space:
            ambient_space = ds.samples_uniform_over(ambient_space, int(sample_count / 2),
                                                    bounds_layer_out[current_layer_index], padding=0.5)
            if should_plot in ["simple", "detailed"]:
                original_ambient_space = ds.samples_uniform_over(original_ambient_space, int(sample_count / 2),
                                                                 bounds_layer_out[current_layer_index], padding=0.5)

        if should_plot == "detailed":
            plt_inc_amb("enhanced ambient space " + str(i), included_space.tolist(), ambient_space.tolist())
            plt_inc_amb("original enhanced ambient space " + str(i), original_included_space.tolist(),
                        original_ambient_space.tolist())

        number_of_groups = get_num_of_groups(len(affine_b) - num_fixed_neurons_layer[current_layer_index], group_size)
        group_indices = get_current_group_indices(len(affine_b), group_size,
                                                  fixed_neuron_per_layer_lower[current_layer_index],
                                                  fixed_neuron_per_layer_upper[current_layer_index])

        # current_from_tos = get_from_tos(len(affine_b), group_size)

        if use_over_approximation:
            if i == 0:
                model = ver.generate_model_center_eps(center.detach().cpu().numpy(), eps)
            else:
                past_group_indices = all_group_indices[-1]
                prev_icnns = list_of_icnns[current_layer_index - 1]
                model = ver.generate_model_icnns(prev_icnns, past_group_indices,
                                                 bounds_layer_out[current_layer_index - 1],
                                                 fixed_neuron_per_layer_lower[current_layer_index - 1],
                                                 fixed_neuron_per_layer_upper[current_layer_index - 1])
            model.update()
            all_group_indices.append(group_indices)

        list_of_icnns.append([])
        for group_i in range(number_of_groups):
            if break_after is not None:
                break_after -= 1
            print("    layer progress, group {} of {} ".format(group_i + 1, number_of_groups))
            t = time.time()

            index_to_select = torch.tensor(group_indices[group_i]).to(device)
            group_inc_space = torch.index_select(included_space, 1, index_to_select)
            group_amb_space = torch.index_select(ambient_space, 1, index_to_select)

            size_of_icnn_input = len(index_to_select)
            current_icnn = icnn_factory.get_new_icnn(size_of_icnn_input)
            list_of_icnns[current_layer_index].append(current_icnn)

            mean = norm.get_mean(group_inc_space, group_amb_space)
            std = norm.get_std(group_inc_space, group_amb_space)
            group_norm_included_space, group_norm_ambient_space = norm.normalize_data(group_inc_space, group_amb_space,
                                                                                      mean,
                                                                                      std)

            dataset = ConvexDataset(data=group_norm_included_space)
            train_loader = DataLoader(dataset, batch_size=icnn_batch_size)
            dataset = ConvexDataset(data=group_norm_ambient_space)
            ambient_loader = DataLoader(dataset, batch_size=icnn_batch_size)

            if init_network:
                low = torch.index_select(bounds_layer_out[current_layer_index][0], 0, index_to_select)
                up = torch.index_select(bounds_layer_out[current_layer_index][1], 0, index_to_select)
                low = torch.div(torch.add(low, -mean), std)
                up = torch.div(torch.add(up, -mean), std)
                current_icnn.init_with_box_bounds(low, up)

            # train icnn
            epochs_per_inclusion = icnn_epochs // force_inclusion_steps
            epochs_in_last_inclusion = icnn_epochs % force_inclusion_steps
            for inclusion_round in range(force_inclusion_steps):
                if inclusion_round > 0:
                    if inclusion_round == force_inclusion_steps - 1 and epochs_in_last_inclusion > 0:
                        epochs_per_inclusion = epochs_in_last_inclusion

                    out = current_icnn(group_norm_included_space)
                    max_out = torch.max(out)
                    current_icnn.apply_enlargement(max_out)
                if data_grad_descent_steps > 1:
                    # todo sollte ich das noch dazu nehmen, dann wird ja ggf die anzahl samples
                    #  doppelt so groß und es dauert länger
                    untouched_group_norm_ambient_space = group_norm_ambient_space.detach().clone()
                    if should_plot in ["simple", "detailed"]:
                        plt_inc_amb("without gradient descent", group_norm_included_space.tolist(),
                                    group_norm_ambient_space.tolist())
                    for gd_round in range(data_grad_descent_steps):
                        optimization_steps = 1000

                        if gd_round == data_grad_descent_steps - 1:
                            epochs_in_run = epochs_per_inclusion % data_grad_descent_steps
                        else:
                            epochs_in_run = epochs_per_inclusion // data_grad_descent_steps

                        train_icnn(current_icnn, train_loader, ambient_loader, epochs=epochs_in_run, hyper_lambda=1,
                                   optimizer=optimizer, adapt_lambda=adapt_lambda, preemptive_stop=preemptive_stop)

                        for v in range(optimization_steps):
                            # normalized_ambient_space =
                            # dop.gradient_descent_data_optim(current_icnn, normalized_ambient_space.detach())
                            group_norm_ambient_space = dop.adam_data_optim(current_icnn,
                                                                           group_norm_ambient_space.detach())
                        dataset = ConvexDataset(data=torch.cat(
                            [group_norm_ambient_space.detach(), untouched_group_norm_ambient_space.detach()]))

                        # todo hier muss ich noch verwalten was passiert wenn ambient space in die nächste runde
                        #  übernommen wird
                        ambient_loader = DataLoader(dataset, batch_size=icnn_batch_size, shuffle=True)

                        if should_plot in ["simple", "detailed"]:
                            """plt_inc_amb("with gradient descent", normalized_included_space.tolist(),
                                        torch.cat([normalized_ambient_space.detach(),
                                                   untouched_normalized_ambient_space.detach()]).tolist())"""
                            plots = Plots_for(0, current_icnn, group_norm_included_space.detach(), torch.cat(
                                [group_norm_ambient_space.detach(),
                                 untouched_group_norm_ambient_space.detach()]).detach(),
                                              [-2, 3], [-2, 3])

                            plots.plt_mesh()

                else:
                    train_icnn(current_icnn, train_loader, ambient_loader, epochs=epochs_per_inclusion, hyper_lambda=1,
                               optimizer=optimizer, adapt_lambda=adapt_lambda, preemptive_stop=preemptive_stop)

            if train_outer:  # todo will ich train outer behalten oder einfach verwerfen?
                for k in range(icnn_epochs):
                    train_icnn_outer(current_icnn, train_loader, ambient_loader, epochs=1)
                    if k % 10 == 0:
                        plots = Plots_for(0, current_icnn, group_norm_ambient_space.detach(),
                                          group_norm_ambient_space.detach(),
                                          [-2, 3], [-2, 3])
                        plots.plt_mesh()

            current_icnn.apply_normalisation(mean, std)

            current_icnn.use_training_setup = False  # todo richtig integrieren in den code

            print("        time for training: {}".format(time.time() - t))

            t = time.time()
            # verify and enlarge convex approximation
            if use_over_approximation:
                copy_model = model.copy()
                adversarial_input, c = ver.verification(current_icnn, copy_model, affine_w.detach().numpy(),
                                                        affine_b.detach().numpy(), group_indices[group_i],
                                                        bounds_affine_out[current_layer_index],
                                                        bounds_layer_out[current_layer_index],
                                                        has_relu=True)

                current_icnn.apply_enlargement(c)
            print("        time for verification: {}".format(time.time() - t))

            """
            #visualisation for one single ReLu
            gt0 = []
            leq0 = []
            gt_x = []
            leq_x = []
            x_in = torch.tensor(x, dtype=data_type).to(device)
            for k, samp in enumerate(x_in):
                testsamp = torch.unsqueeze(samp, dim=0)
                testsamp = torch.unsqueeze(testsamp, dim=0)
                if current_icnn(testsamp) > 0:
                    gt0.append(samp)
                    gt_x.append(current_icnn(testsamp))
                else:
                    leq0.append(samp)
                    leq_x.append(current_icnn(testsamp))

            plt.scatter(list(map(lambda x: x.detach().numpy(), gt0)),
                        list(map(lambda x: x.detach().numpy(), gt_x)), c="#ff7f0e")
            plt.scatter(list(map(lambda x: x.detach().numpy(), leq0)),
                        list(map(lambda x: x.detach().numpy(), leq_x)), c="#1f77b4")
            plt.title("My ReLU post overapprox")
            plt.show()"""
            if break_after is not None and break_after == 0:
                force_break = True
                break

        if force_break:
            print("aborting because of force break")
            break

        if use_icnn_bounds:
            t = time.time()
            inp_bounds_icnn = bounds_layer_out[current_layer_index]
            new_bounds = ver.min_max_of_icnns(list_of_icnns[current_layer_index], inp_bounds_icnn,
                                              group_indices, print_log=False)
            bounds_layer_out[current_layer_index] = new_bounds
            print("    time for icnn_bound calculation: {}".format(time.time() - t))

        # entweder oder:
        # nutze die samples weiter (dafür muss man dann das ReLU layer anwenden), und man muss schauen ob die
        # samples jetzt von ambient_space zu included_space gewechselt haben (wegen überapproximation)
        # damit könnte man gut den Fehler messen, da man ganz genau weiß wie viele elemente aus Ambient space
        # in Icluded space übergegenagen sind

        # oder sample neue punkte
        t = time.time()
        if sample_new:
            # todo entweder über box bounds sampeln oder über maximum radius
            included_space, ambient_space = ds.sample_max_radius(list_of_icnns[current_layer_index], sample_count,
                                                                 group_indices, bounds_layer_out[current_layer_index])

        else:
            included_space, ambient_space = ds.regroup_samples(list_of_icnns[current_layer_index], included_space,
                                                               ambient_space, group_indices)
        print("    time for regrouping method: {}".format(time.time() - t))

    if should_plot in valid_should_plot and should_plot != "none":
        index = len(parameter_list) - 2
        affine_w, affine_b = parameter_list[index], parameter_list[index + 1]
        included_space = ds.apply_affine_transform(affine_w, affine_b, included_space)
        ambient_space = ds.apply_affine_transform(affine_w, affine_b, ambient_space)
        original_included_space = ds.apply_affine_transform(affine_w, affine_b, original_included_space)
        plots = Plots_for(0, current_icnn, included_space.detach().cpu(), ambient_space.detach().cpu(), [-1, 3],
                          [-1, 3],
                          extr=original_included_space.detach().cpu())
        plots.plt_initial()

    return list_of_icnns, all_group_indices[-1], fixed_neuron_per_layer_lower[-1], fixed_neuron_per_layer_upper[
        -1], bounds_affine_out, bounds_layer_out


def imshow_flattened(img_flattened, shape):
    img = np.reshape(img_flattened, shape)
    img = img / 2 + .05  # revert normalization for viewing
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()


def plt_inc_amb(caption, inc, amb):
    plt.figure(figsize=(20, 10))
    plt.scatter(list(map(lambda x: x[0], amb)), list(map(lambda x: x[1], amb)), c="#ff7f0e")
    plt.scatter(list(map(lambda x: x[0], inc)), list(map(lambda x: x[1], inc)), c="#1f77b4")
    plt.title(caption)
    plt.show()


def get_from_tos(layer_size, group_size):
    number_of_groups = get_num_of_groups(layer_size, group_size)
    all_from_to_neurons = []
    for group_i in range(number_of_groups):
        if group_i == number_of_groups - 1 and layer_size % group_size > 0:
            from_to_neurons = [group_size * group_i, group_size * group_i + (layer_size % group_size)]
        else:
            from_to_neurons = [group_size * group_i, group_size * group_i + group_size]  # upper bound is exclusive
        all_from_to_neurons.append(from_to_neurons)
    return all_from_to_neurons


def get_num_of_groups(layer_size, group_size):
    number_of_groups = layer_size / group_size
    if number_of_groups < 0:
        number_of_groups *= -1
    number_of_groups = math.ceil(number_of_groups)
    return number_of_groups


def get_current_group_indices(num_neurons, group_size, fixed_neurons_lower, fixed_neurons_upper):
    group_indices = []
    current_group = []
    fixed_neuron_index = (fixed_neurons_lower + fixed_neurons_upper)
    for index in range(num_neurons):
        if index in fixed_neuron_index:
            continue
        else:
            current_group.append(index)
            if len(current_group) == group_size:
                group_indices.append(current_group)
                current_group = []

    if len(current_group) > 0:
        group_indices.append(current_group)
    return group_indices
