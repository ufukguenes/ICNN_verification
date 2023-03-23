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
from script.NeuralNets.testFunction import test_icnn
from script.settings import device, data_type
import gurobipy as grp
import warnings


class MultiDHOV:
    def __init__(self):
        self.nn_encoding_model = None
        self.input_var = None
        self.output_var = None
        self.bounds_affine_out = None
        self.bounds_layer_out = None
        self.fixed_neuron_per_layer_lower = None
        self.fixed_neuron_per_layer_upper = None
        self.num_fixed_neurons_layer = None
        self.all_group_indices = None
        self.list_of_icnns = None
        self.list_of_included_samples = []
        self.list_of_ambient_samples = []

    def start_verification(self, nn: SequentialNN, input, icnn_factory, group_size, eps=0.001, icnn_batch_size=1000,
                           icnn_epochs=100, sample_count=1000, sampling_method="uniform",
                           break_after=None, use_icnn_bounds=False, use_fixed_neurons=False,
                           keep_ambient_space=False, sample_new=True, use_over_approximation=True, opt_steps_gd=100,
                           sample_over_input_space=False, sample_over_output_space=True, data_grad_descent_steps=0,
                           train_outer=False, preemptive_stop=True, even_gradient_training=False, store_samples=False,
                           force_inclusion_steps=0, grouping_method="consecutive", group_num_multiplier=None,
                           init_network=False, adapt_lambda="none", should_plot='none', optimizer="adam",
                           print_training_loss=False, print_optimization_steps=False, print_new_bounds=False):
        valid_adapt_lambda = ["none", "high_low", "included"]
        valid_should_plot = ["none", "simple", "detailed", "verification", "output"]
        valid_optimizer = ["adam", "LBFGS", "SdLBFGS"]
        valid_sampling_methods = ["uniform", "linespace", "boarder", "sum_noise", "min_max_perturbation",
                                  "alternate_min_max", "per_group_sampling"]
        valid_grouping_methods = ["consecutive", "random"]

        parameter_list = list(nn.parameters())
        force_break = False


        if adapt_lambda not in valid_adapt_lambda:
            raise AttributeError(
                "Expected adaptive lambda mode to be one of: {}, actual: {}".format(valid_adapt_lambda, adapt_lambda))
        if should_plot not in valid_should_plot:
            raise AttributeError("Expected plotting mode to be one of: {}, actual: {}".format(valid_should_plot, should_plot))
        if optimizer not in valid_optimizer:
            raise AttributeError("Expected optimizer to be one of: {}, actual: {}".format(valid_optimizer, optimizer))
        if force_inclusion_steps < 0:
            raise AttributeError("Expected force_inclusion to be:  >= 0 , got: {}".format(force_inclusion_steps))
        if data_grad_descent_steps < 0 or data_grad_descent_steps > icnn_epochs:
            raise AttributeError(
                "Expected data_grad_descent_steps to be:  >= {}} , got: {}".format(icnn_epochs,
                                                                                   data_grad_descent_steps))
        if grouping_method not in valid_grouping_methods:
            raise AttributeError("Expected grouping method to be one of: {}, actual: {}".format(valid_grouping_methods,
                                                                                   grouping_method))
        if grouping_method == "random" and group_num_multiplier is None and group_num_multiplier % 1 != 0:
            raise AttributeError(
                "Expected group_num_multiplier to be integer > 0 , got: {}".format(data_grad_descent_steps))
        if sampling_method not in valid_sampling_methods:
            raise AttributeError(
                "Expected sampling method to be one of: {} , got: {}".format(valid_sampling_methods, sampling_method))
        if keep_ambient_space and sampling_method == "per_group_sampling":
            warnings.warn("keep_ambient_space is True and sampling method is per_group_sampling. "
                          "Keeping previous samples is not supported when using per group sampling")
        if sample_over_input_space:
            sample_over_input_space = False
            sample_over_output_space = True
            warnings.warn("sample_over_input_space is True and sampling method is per_group_sampling. "
                          "Sampling over input space is not yet supported when using per group sampling. "
                          "Using sampling over output space instead...")
        if group_num_multiplier is not None and grouping_method == "consecutive":
            warnings.warn("value for group number multiplier is given with grouping method consecutive. "
                          "consecutive grouping does not use variable number of groups")

        input_flattened = torch.flatten(input)
        center = input_flattened
        eps_bounds = [input_flattened.add(-eps), input_flattened.add(eps)]

        bounds_affine_out, bounds_layer_out = nn.calculate_box_bounds(eps_bounds)

        nn_encoding_model = grp.Model()
        if print_optimization_steps:
            nn_encoding_model.Params.LogToConsole = 1
        else:
            nn_encoding_model.Params.LogToConsole = 0

        included_space = torch.empty((0, input_flattened.size(0)), dtype=data_type).to(device)

        inc_space_sample_count = sample_count // 2

        if sample_over_input_space and sample_over_output_space:
            amb_space_sample_count = sample_count // 4
        else:
            amb_space_sample_count = sample_count // 2

        if sampling_method != "per_group_sampling":
            if sampling_method == "uniform":
                included_space = ds.samples_uniform_over(included_space, inc_space_sample_count, eps_bounds)
            elif sampling_method == "linespace":
                included_space = ds.sample_linspace(included_space, inc_space_sample_count, center, eps)
            elif sampling_method == "boarder":
                included_space = ds.sample_boarder(included_space, inc_space_sample_count, center, eps)
            elif sampling_method == "sum_noise":
                included_space = ds.sample_random_sum_noise(included_space, inc_space_sample_count, center, eps)
            elif sampling_method == "min_max_perturbation":
                included_space = ds.sample_min_max_perturbation(included_space, inc_space_sample_count,
                                                                parameter_list[0],
                                                                center, eps)
            elif sampling_method == "alternate_min_max":
                included_space = ds.sample_alternate_min_max(included_space, inc_space_sample_count, parameter_list[0],
                                                             center, eps)

            # included_space = ds.samples_uniform_over(included_space, int(sample_count / 2), eps_bounds, keep_samples=True)

            ambient_space = torch.empty((0, input_flattened.size(0)), dtype=data_type).to(device)
            original_included_space = torch.empty((0, input_flattened.size(0)), dtype=data_type).to(device)
            original_ambient_space = torch.empty((0, input_flattened.size(0)), dtype=data_type).to(device)

            if should_plot in valid_should_plot and should_plot != "none":
                original_included_space, original_ambient_space = included_space, ambient_space

        force_inclusion_steps += 1
        data_grad_descent_steps += 1

        fixed_neuron_per_layer_lower = []
        fixed_neuron_per_layer_upper = []
        num_fixed_neurons_layer = []
        number_of_fixed_neurons = 0
        all_group_indices = []
        list_of_icnns = []
        current_layer_index = 0

        self.nn_encoding_model = nn_encoding_model
        self.input_var = ver.generate_model_center_eps(nn_encoding_model, center, eps, -1)
        nn_encoding_model.update()

        self.bounds_affine_out = bounds_affine_out
        self.bounds_layer_out = bounds_layer_out
        self.fixed_neuron_per_layer_lower = fixed_neuron_per_layer_lower
        self.fixed_neuron_per_layer_upper = fixed_neuron_per_layer_upper
        self.num_fixed_neurons_layer = num_fixed_neurons_layer
        self.all_group_indices = all_group_indices
        self.list_of_icnns = list_of_icnns

        for i in range(0, len(parameter_list) - 2, 2):  # -2 because last layer has no ReLu activation
            current_layer_index = i // 2
            prev_layer_index = current_layer_index - 1
            print("")
            print("approximation of layer: {}".format(current_layer_index))
            if store_samples:
                self.list_of_included_samples.append([])
                self.list_of_ambient_samples.append([])

            affine_w, affine_b = parameter_list[i], parameter_list[i + 1]

            if sampling_method != "per_group_sampling":
                if not keep_ambient_space:
                    ambient_space = torch.empty((0, nn.layer_widths[current_layer_index]), dtype=data_type).to(device)

                if sample_over_input_space:
                    if i == 0:
                        ambient_space = ds.sample_uniform_excluding(ambient_space, amb_space_sample_count, eps_bounds,
                                                                    excluding_bound=eps_bounds, padding=eps)
                    else:
                        ambient_space = ds.sample_uniform_excluding(ambient_space, amb_space_sample_count,
                                                                    bounds_layer_out[current_layer_index - 1],
                                                                    icnns=list_of_icnns[current_layer_index - 1],
                                                                    layer_index=current_layer_index,
                                                                    group_size=group_size,
                                                                    padding=eps)

                if should_plot == "detailed" and included_space.size(1) == 2:
                    plt_inc_amb("start " + str(i), included_space.tolist(), ambient_space.tolist())

                included_space = ds.apply_affine_transform(affine_w, affine_b, included_space)
                ambient_space = ds.apply_affine_transform(affine_w, affine_b, ambient_space)

                """plt_inc_amb("second layer output of neuron 2, 3 / number of samples {}".format(sample_count),
                            torch.index_select(included_space, 1, torch.tensor([1, 23])).tolist(),
                            torch.index_select(ambient_space, 1, torch.tensor([1, 23])).tolist())"""

                if should_plot in valid_should_plot and should_plot != "none" and included_space.size(1) == 2:
                    original_included_space = ds.apply_affine_transform(affine_w, affine_b, original_included_space)
                    original_ambient_space = ds.apply_affine_transform(affine_w, affine_b, original_ambient_space)
                    if should_plot == "detailed":
                        plt_inc_amb("affin e" + str(i), included_space.tolist(), ambient_space.tolist())
                        plt_inc_amb("original affin e" + str(i), original_included_space.tolist(),
                                    original_ambient_space.tolist())

                included_space = ds.apply_relu_transform(included_space)
                ambient_space = ds.apply_relu_transform(ambient_space)

                """plt_inc_amb("second layer output of neuron 2, 3", torch.index_select(included_space, 1, torch.tensor([2, 3])).tolist(),
                            torch.index_select(ambient_space, 1, torch.tensor([2, 3])).tolist())"""

                if should_plot in valid_should_plot and should_plot != "none" and included_space.size(1) == 2:
                    original_included_space = ds.apply_relu_transform(original_included_space)
                    original_ambient_space = ds.apply_relu_transform(original_ambient_space)
                    if should_plot == "detailed":
                        plt_inc_amb("relu " + str(i), included_space.tolist(), ambient_space.tolist())
                        plt_inc_amb("original relu e" + str(i), original_included_space.tolist(),
                                    original_ambient_space.tolist())

                if sample_over_output_space:
                    ambient_space = ds.samples_uniform_over(ambient_space, amb_space_sample_count,
                                                            bounds_layer_out[current_layer_index], padding=eps)
                    if should_plot in ["simple", "detailed"] and included_space.size(1) == 2:
                        original_ambient_space = ds.samples_uniform_over(original_ambient_space, amb_space_sample_count,
                                                                         bounds_layer_out[current_layer_index],
                                                                         padding=eps)

                if store_samples:
                    self.list_of_included_samples[current_layer_index].append(included_space)
                    self.list_of_ambient_samples[current_layer_index].append(ambient_space)

                if should_plot == "detailed" and included_space.size(1) == 2:
                    plt_inc_amb("enhanced ambient space " + str(i), included_space.tolist(), ambient_space.tolist())
                    plt_inc_amb("original enhanced ambient space " + str(i), original_included_space.tolist(),
                                original_ambient_space.tolist())

            """if use_over_approximation:
                if i == 0:
                    model = ver.generate_model_center_eps(center.detach().cpu().numpy(), eps)
                else:
                    affine_w_list = parameter_list[:i:2]
                    affine_b_list = parameter_list[1:i:2] #todo das model muss auch erstellt werden, wenn man per group sampling macht
                    model = ver.generate_complete_model_icnn(center, eps, affine_w_list, affine_b_list, list_of_icnns,
                                                             all_group_indices, bounds_affine_out, bounds_layer_out,
                                                             fixed_neuron_per_layer_lower, fixed_neuron_per_layer_upper)"""

            if use_icnn_bounds and i != 0:
                t = time.time()
                copy_model = nn_encoding_model.copy()
                ver.update_bounds_with_icnns(copy_model, bounds_affine_out, bounds_layer_out,
                                             current_layer_index, affine_w.detach().numpy(),
                                             affine_b.detach().numpy(), print_new_bounds=print_new_bounds)
                print("    time for icnn_bound calculation: {}".format(time.time() - t))

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


            if grouping_method == "consecutive":
                number_of_groups = get_num_of_groups(len(affine_b) - num_fixed_neurons_layer[current_layer_index],
                                                     group_size)
                group_indices = get_current_group_indices(len(affine_b), group_size,
                                                          fixed_neuron_per_layer_lower[current_layer_index],
                                                          fixed_neuron_per_layer_upper[current_layer_index])
            elif grouping_method == "random":
                number_of_groups = get_num_of_groups(len(affine_b) - num_fixed_neurons_layer[current_layer_index],
                                                     group_size)
                number_of_groups = group_num_multiplier * number_of_groups
                group_indices = get_random_groups(len(affine_b), number_of_groups, group_size,
                                                          fixed_neuron_per_layer_lower[current_layer_index],
                                                          fixed_neuron_per_layer_upper[current_layer_index])

            all_group_indices.append(group_indices)
            if force_break:
                print("aborting because of force break. Layer currently approximated is not ")
                return

            list_of_icnns.append([])
            for group_i in range(number_of_groups):
                if break_after is not None:
                    break_after -= 1
                print("    layer progress, group {} of {} ".format(group_i + 1, number_of_groups))

                index_to_select = torch.tensor(group_indices[group_i]).to(device)

                size_of_icnn_input = len(index_to_select)
                current_icnn = icnn_factory.get_new_icnn(size_of_icnn_input)
                list_of_icnns[current_layer_index].append(current_icnn)

                if sampling_method == "per_group_sampling":
                    included_space = torch.empty((0, affine_w.size(1)), dtype=data_type).to(device)
                    ambient_space = torch.empty((0, affine_w.size(1)), dtype=data_type).to(device)

                    if i == 0:
                        t_group = time.time()
                        """included_space = ds.sample_per_group(included_space, sample_count // 2, affine_w, center,
                                                             eps, index_to_select)"""
                        included_space = ds.sample_per_group(included_space, inc_space_sample_count // 2, affine_w,
                                                             center,
                                                             eps, index_to_select, rand_samples_percent=0.2,
                                                             rand_sample_alternation_percent=0.01)
                        included_space = ds.samples_uniform_over(included_space, inc_space_sample_count // 2,
                                                                 eps_bounds,
                                                                 keep_samples=True)
                        # included_space = ds.sample_min_max_perturbation(included_space, inc_space_sample_count // 2, affine_w, center, eps, keep_samples=True, swap_probability=0.2)
                        # included_space = ds.sample_per_group(included_space, inc_space_sample_count // 2, parameter_list[0], center, eps, index_to_select, with_noise=True, with_sign_swap=True)
                        # included_space = ds.sample_per_group(included_space, inc_space_sample_count // 2, parameter_list[0], center, eps, index_to_select, with_noise=False, with_sign_swap=True)
                        # included_space = ds.sample_per_group(included_space, inc_space_sample_count // 2, parameter_list[0], center, eps, index_to_select, with_noise=True, with_sign_swap=False)
                        print("        time for sampling for one group: {}".format(time.time() - t_group))
                    else:
                        copy_model = nn_encoding_model.copy()
                        t_group = time.time()
                        included_space = ds.sample_per_group_as_lp(included_space, inc_space_sample_count // 2,
                                                                   affine_w, affine_b,
                                                                   index_to_select, copy_model,
                                                                   bounds_affine_out[current_layer_index],
                                                                   prev_layer_index,
                                                                   rand_samples_percent=0.2,
                                                                   rand_sample_alternation_percent=0.2)
                        included_space = ds.sample_uniform_over_icnn(included_space, inc_space_sample_count // 2,
                                                                     list_of_icnns[current_layer_index - 1],
                                                                     all_group_indices[current_layer_index - 1],
                                                                     bounds_layer_out[current_layer_index - 1],
                                                                     keep_samples=True)
                        print("        time for sampling for one group: {}".format(time.time() - t_group))

                    if should_plot in valid_should_plot and should_plot not in ["none", "verification"] and len(
                            group_indices[group_i]) == 2:
                        plt_inc_amb("layer input ",
                                    torch.index_select(included_space, 1, index_to_select).tolist(),
                                    torch.index_select(ambient_space, 1, index_to_select).tolist())
                    elif should_plot in valid_should_plot and should_plot not in ["none", "verification"] and len(
                            group_indices[group_i]) == 3:
                        plt_inc_amb_3D("layer input ",
                                       torch.index_select(included_space, 1, index_to_select).tolist(),
                                       torch.index_select(ambient_space, 1, index_to_select).tolist())

                    included_space = ds.apply_affine_transform(affine_w, affine_b, included_space)
                    ambient_space = ds.apply_affine_transform(affine_w, affine_b, ambient_space)

                    if should_plot in valid_should_plot and should_plot not in ["none", "verification"] and len(
                            group_indices[group_i]) == 2:
                        plt_inc_amb("layer output ",
                                    torch.index_select(included_space, 1, index_to_select).tolist(),
                                    torch.index_select(ambient_space, 1, index_to_select).tolist())
                    elif should_plot in valid_should_plot and should_plot not in ["none", "verification"] and len(
                            group_indices[group_i]) == 3:
                        plt_inc_amb_3D("layer output ",
                                       torch.index_select(included_space, 1, index_to_select).tolist(),
                                       torch.index_select(ambient_space, 1, index_to_select).tolist())

                    included_space = ds.apply_relu_transform(included_space)
                    ambient_space = ds.apply_relu_transform(ambient_space)

                    if sample_over_output_space:
                        ambient_space = ds.samples_uniform_over(ambient_space, amb_space_sample_count,
                                                                bounds_layer_out[current_layer_index],
                                                                padding=eps)

                    if store_samples:
                        self.list_of_included_samples[current_layer_index].append(included_space)
                        self.list_of_ambient_samples[current_layer_index].append(ambient_space)

                t = time.time()
                group_inc_space = torch.index_select(included_space, 1, index_to_select)
                group_amb_space = torch.index_select(ambient_space, 1, index_to_select)

                mean = norm.get_mean(group_inc_space, group_amb_space)
                std = norm.get_std(group_inc_space, group_amb_space)
                group_norm_included_space, group_norm_ambient_space = norm.normalize_data(group_inc_space,
                                                                                          group_amb_space,
                                                                                          mean, std)
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
                current_icnn.use_training_setup = True  # is only relevant for ApproxMaxICNNs
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
                        untouched_amb_space = group_norm_ambient_space.detach().clone()
                        if should_plot in ["simple", "detailed"] and len(group_indices[group_i]) == 2:
                            plt_inc_amb("without gradient descent", group_norm_included_space.tolist(),
                                        group_norm_ambient_space.tolist())
                        for gd_round in range(data_grad_descent_steps):
                            optimization_steps = opt_steps_gd
                            if data_grad_descent_steps > epochs_per_inclusion:
                                epochs_in_run = 1
                            elif gd_round == data_grad_descent_steps - 1 and epochs_per_inclusion % data_grad_descent_steps != 0:
                                epochs_in_run = epochs_per_inclusion % data_grad_descent_steps
                            else:
                                epochs_in_run = epochs_per_inclusion // data_grad_descent_steps
                            print("===== grad descent =====")
                            train_icnn(current_icnn, train_loader, ambient_loader, epochs=epochs_in_run, hyper_lambda=1,
                                       optimizer=optimizer, adapt_lambda=adapt_lambda, preemptive_stop=preemptive_stop,
                                       verbose=print_training_loss)

                            for v in range(optimization_steps):
                                # normalized_ambient_space =
                                # dop.gradient_descent_data_optim(current_icnn, normalized_ambient_space.detach())
                                group_norm_ambient_space = dop.adam_data_optim(current_icnn,
                                                                               group_norm_ambient_space.detach())
                            dataset = ConvexDataset(group_norm_ambient_space.detach())

                            # todo hier muss ich noch verwalten was passiert wenn ambient space in die nächste runde
                            #  übernommen wird
                            ambient_loader = DataLoader(dataset, batch_size=icnn_batch_size, shuffle=True)

                            if should_plot in ["simple", "detailed"] and len(group_indices[group_i]) == 2:
                                """plt_inc_amb("with gradient descent", normalized_included_space.tolist(),
                                            torch.cat([normalized_ambient_space.detach(),
                                                       untouched_normalized_ambient_space.detach()]).tolist())"""
                                min_x, max_x, min_y, max_y = get_min_max_x_y(torch.cat(
                                    [group_norm_included_space.detach(), group_norm_ambient_space.detach()]))

                                plots = Plots_for(0, current_icnn, group_norm_included_space.detach(),
                                                  group_norm_ambient_space.detach(),
                                                  [min_x, max_x], [min_y, max_y])

                                plots.plt_mesh()

                        if print_training_loss:
                            dataset = ConvexDataset(data=group_norm_included_space)
                            train_loader = DataLoader(dataset, batch_size=icnn_batch_size)
                            dataset = ConvexDataset(data=untouched_amb_space)
                            ambient_loader = DataLoader(dataset, batch_size=icnn_batch_size)
                            test_icnn(current_icnn, train_loader, ambient_loader)


                    else:
                        train_icnn(current_icnn, train_loader, ambient_loader, epochs=epochs_per_inclusion,
                                   hyper_lambda=1,
                                   optimizer=optimizer, adapt_lambda=adapt_lambda, preemptive_stop=preemptive_stop,
                                   verbose=print_training_loss)

                if train_outer:  # todo will ich train outer behalten oder einfach verwerfen?
                    for k in range(icnn_epochs):
                        train_icnn_outer(current_icnn, train_loader, ambient_loader, epochs=1)
                        if k % 10 == 0:
                            min_x, max_x, min_y, max_y = get_min_max_x_y(torch.cat(
                                [group_norm_included_space.detach(), group_norm_ambient_space.detach()]))
                            plots = Plots_for(0, current_icnn, group_norm_included_space.detach(),
                                              group_norm_ambient_space.detach(),
                                              [min_x, max_x], [min_y, max_y])
                            plots.plt_mesh()

                current_icnn.apply_normalisation(mean, std)

                current_icnn.use_training_setup = False

                print("        time for training: {}".format(time.time() - t))

                t = time.time()
                # verify and enlarge convex approximation

                if should_plot in ["detailed", "verification"] and should_plot != "none":
                    if len(group_indices[group_i]) == 2:
                        min_x, max_x, min_y, max_y = \
                            get_min_max_x_y(torch.cat([group_inc_space.detach(), group_amb_space.detach()]))
                        plots = Plots_for(0, current_icnn, group_inc_space.detach(), group_amb_space.detach(),
                                          [min_x, max_x], [min_y, max_y])
                        plots.plt_mesh()

                    elif len(group_indices[group_i]) == 1:
                        visualize_single_neuron(current_icnn, group_indices[group_i], current_layer_index, bounds_affine_out)

                if use_over_approximation:
                    copy_model = nn_encoding_model.copy()
                    adversarial_input, c = ver.verification(current_icnn, copy_model, affine_w.detach().numpy(),
                                                            affine_b.detach().numpy(), group_indices[group_i],
                                                            bounds_affine_out[current_layer_index],
                                                            bounds_layer_out[current_layer_index], prev_layer_index,
                                                            has_relu=True)

                    current_icnn.apply_enlargement(c)

                print("        time for verification: {}".format(time.time() - t))

                if should_plot in ["detailed", "verification"] and should_plot != "none":
                    if len(group_indices[group_i]) == 2:
                        min_x, max_x, min_y, max_y = \
                            get_min_max_x_y(torch.cat([group_inc_space.detach(), group_amb_space.detach()]))
                        plots = Plots_for(0, current_icnn, group_inc_space.detach(), group_amb_space.detach(),
                                          [min_x, max_x], [min_y, max_y])
                        plots.plt_mesh()

                    elif len(group_indices[group_i]) == 1:
                        visualize_single_neuron(current_icnn, group_indices[group_i], current_layer_index,
                                                bounds_affine_out)

                # inp_bounds_icnn = bounds_layer_out[current_layer_index]
                # ver.min_max_of_icnns([current_icnn], inp_bounds_icnn, [group_indices[group_i]], print_log=False)
                # ver.min_max_of_icnns([current_icnn], inp_bounds_icnn, [group_indices[group_i]], print_log1, 23=False)


                if break_after is not None and break_after == 0:
                    force_break = True

            # add current layer to model
            curr_constraint_icnns = list_of_icnns[current_layer_index]
            curr_group_indices = all_group_indices[current_layer_index]
            curr_bounds_affine_out = bounds_affine_out[current_layer_index]
            curr_bounds_layer_out = bounds_layer_out[current_layer_index]
            curr_fixed_neuron_lower = fixed_neuron_per_layer_lower[current_layer_index]
            curr_fixed_neuron_upper = fixed_neuron_per_layer_upper[current_layer_index]
            ver.add_layer_to_model(nn_encoding_model, affine_w.detach().numpy(), affine_b.detach().numpy(),
                                   curr_constraint_icnns, curr_group_indices,
                                   curr_bounds_affine_out, curr_bounds_layer_out,
                                   curr_fixed_neuron_lower, curr_fixed_neuron_upper,
                                   current_layer_index)
            nn_encoding_model.update()

            # entweder oder:
            # nutze die samples weiter (dafür muss man dann das ReLU layer anwenden), und man muss schauen ob die
            # samples jetzt von ambient_space zu included_space gewechselt haben (wegen überapproximation)
            # damit könnte man gut den Fehler messen, da man ganz genau weiß wie viele elemente aus Ambient space
            # in Icluded space übergegenagen sind

            # oder sample neue punkte
            t = time.time()
            if sample_new and sampling_method != "per_group_sampling":
                included_space, ambient_space = ds.sample_max_radius(list_of_icnns[current_layer_index], sample_count,
                                                                     group_indices,
                                                                     bounds_layer_out[current_layer_index],
                                                                     fixed_neuron_per_layer_lower[current_layer_index],
                                                                     fixed_neuron_per_layer_upper[current_layer_index])

            elif sampling_method != "per_group_sampling":
                # re-grouping
                included_space, ambient_space = ds.regroup_samples(list_of_icnns[current_layer_index], included_space,
                                                                   ambient_space, group_indices)
            print("    time for regrouping method: {}".format(time.time() - t))


        if should_plot in valid_should_plot and should_plot != "none" and sampling_method != "per_group_sampling" and included_space.size(
                1) == 2:
            index = len(parameter_list) - 2
            affine_w, affine_b = parameter_list[index], parameter_list[index + 1]
            included_space = ds.apply_affine_transform(affine_w, affine_b, included_space)
            ambient_space = ds.apply_affine_transform(affine_w, affine_b, ambient_space)
            original_included_space = ds.apply_affine_transform(affine_w, affine_b, original_included_space)

            min_x, max_x, min_y, max_y = \
                get_min_max_x_y(torch.cat([included_space.detach(), ambient_space.detach()]))
            plots = Plots_for(0, current_icnn, included_space.detach().cpu(), ambient_space.detach().cpu(),
                              [min_x, max_x], [min_y, max_y],
                              extr=original_included_space.detach().cpu())
            plots.plt_initial()

        affine_w, affine_b = parameter_list[-2].detach().numpy(), parameter_list[-1].detach().numpy()
        last_layer_index = current_layer_index + 1
        output_second_last_layer = []
        for m in range(affine_w.shape[1]):
            output_second_last_layer.append(
                nn_encoding_model.getVarByName("output_layer_[{}]_[{}]".format(last_layer_index - 1, m)))
        output_second_last_layer = grp.MVar.fromlist(output_second_last_layer)
        in_lb = bounds_affine_out[last_layer_index][0].detach().numpy()
        in_ub = bounds_affine_out[last_layer_index][1].detach().numpy()

        out_fet = len(affine_b)
        output_nn = nn_encoding_model.addMVar(out_fet, lb=in_lb, ub=in_ub,
                                              name="output_layer_[{}]_".format(last_layer_index))
        const = nn_encoding_model.addConstrs(
            (affine_w[i] @ output_second_last_layer + affine_b[i] == output_nn[i] for i in range(len(affine_w))),
            name="affine_const_[{}]".format(last_layer_index))

        nn_encoding_model.update()

        if use_icnn_bounds:
            #todo this is a code duplicat
            for neuron_to_optimize in range(len(output_nn.tolist())):
                nn_encoding_model.setObjective(output_nn[neuron_to_optimize], grp.GRB.MINIMIZE)
                nn_encoding_model.optimize()
                if nn_encoding_model.Status == grp.GRB.OPTIMAL:
                    value = output_nn.getAttr("x")
                    if print_new_bounds and abs(value[neuron_to_optimize] - bounds_affine_out[last_layer_index][0][neuron_to_optimize]) > 0.00001:
                        print("        {}, lower: new {}, old {}".format(neuron_to_optimize, value[neuron_to_optimize], bounds_affine_out[last_layer_index][0][neuron_to_optimize]))
                    bounds_affine_out[last_layer_index][0][neuron_to_optimize] = value[neuron_to_optimize]

                nn_encoding_model.setObjective(output_nn[neuron_to_optimize], grp.GRB.MAXIMIZE)
                nn_encoding_model.optimize()
                if nn_encoding_model.Status == grp.GRB.OPTIMAL:
                    value = output_nn.getAttr("x")
                    if print_new_bounds and abs(value[neuron_to_optimize] - bounds_affine_out[last_layer_index][1][neuron_to_optimize]) > 0.00001:
                        print("        {}, upper: new {}, old {}".format(neuron_to_optimize, value[neuron_to_optimize], bounds_affine_out[last_layer_index][1][neuron_to_optimize]))
                    bounds_affine_out[last_layer_index][1][neuron_to_optimize] = value[neuron_to_optimize]

            if i < len(parameter_list) - 2:
                relu_out_lb, relu_out_ub = verbas.calc_relu_out_bound(bounds_affine_out[last_layer_index][0],
                                                                      bounds_affine_out[last_layer_index][1])
            bounds_layer_out[last_layer_index][0] = relu_out_lb
            bounds_layer_out[last_layer_index][1] = relu_out_ub

        self.output_var = output_nn
        print("done...")
        return


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


def plt_inc_amb_3D(caption, inc, amb):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_ylim(-0.2, 0.2)
    ax.set_xlim(-0.1, 0.25)
    ax.set_zlim(-0.1, 0.3)
    ax.scatter(list(map(lambda x: x[0], amb)), list(map(lambda x: x[1], amb)), list(map(lambda x: x[2], amb)),
               c="#ff7f0e")
    ax.scatter(list(map(lambda x: x[0], inc)), list(map(lambda x: x[1], inc)), list(map(lambda x: x[2], inc)),
               c="#1f77b4")
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

def get_random_groups(num_neurons, num_groups, group_size, fixed_neurons_lower, fixed_neurons_upper):
    group_indices = []
    min_group_size = min(num_neurons, group_size)
    fixed_neuron_index = (fixed_neurons_lower + fixed_neurons_upper)
    neurons_to_group = [x for x in range(num_neurons) if x not in fixed_neuron_index]
    for index in range(num_groups):
        random.shuffle(neurons_to_group)
        current_group = neurons_to_group[:min_group_size]
        group_indices.append(current_group)

    return group_indices


def get_min_max_x_y(values):
    xs = torch.index_select(values, 1, torch.tensor(0))
    ys = torch.index_select(values, 1, torch.tensor(1))
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    return min_x, max_x, min_y, max_y


def visualize_single_neuron(icnn, neuron_index, layer_index, bounds_affine_out):
    gt0 = []
    leq0 = []
    gt_x = []
    leq_x = []
    x = np.linspace(bounds_affine_out[layer_index][0][neuron_index].item(), bounds_affine_out[layer_index][1][neuron_index].item(), 100)
    x_in = torch.tensor(x, dtype=data_type).to(device)
    for k, samp in enumerate(x_in):
        testsamp = torch.unsqueeze(samp, dim=0)
        testsamp = torch.unsqueeze(testsamp, dim=0)
        relu_out = torch.nn.ReLU()(testsamp)
        if icnn(testsamp) >= 0:
            gt0.append(samp)
            gt_x.append(relu_out)
        else:
            leq0.append(samp)
            leq_x.append(relu_out)

    plt.scatter(list(map(lambda x: x.detach().numpy(), gt0)),
                list(map(lambda x: x.detach().numpy(), gt_x)), c="#ff7f0e")
    plt.scatter(list(map(lambda x: x.detach().numpy(), leq0)),
                list(map(lambda x: x.detach().numpy(), leq_x)), c="#1f77b4")
    plt.title("ReLU")
    plt.show()
