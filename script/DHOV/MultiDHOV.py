# DeepHull Over approximated Verification
import math
import random
import time
import multiprocessing

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
from script.DHOV.Sampling import SamplingStrategy
from script.DHOV.Sampling.PropagateSamplingStrategy import UniformSamplingStrategy

#todo adapt everything to also work on gpu
class MultiDHOV:
    """
    Multiple groups - DeepHull-Over approximated-Verification
    This class provides a method which encodes a NN given a specific input as a Gurobi encoding.
    The encoding can be forced to be an over approximation.
    It can encode multiple groups within one layer together

    Attributes
        self.nn_encoding_model = None :  the final Gurobi model/ encoding of the NN
        self.input_var = None : the Gurobi variables describing the input neurons of the NN
        self.output_var = None : the Gurobi variables describing the output neurons of the NN
        self.bounds_affine_out = None : List of torch.Tensor describing bounds for each layer and each neuron
            before an activation function
        self.bounds_layer_out = None : List of torch.Tensor describing bounds for each layer and each neuron
            after an activation function. It is the same as self.bounds_affine_out for each layer where
            there is no activation function
        self.fixed_neuron_per_layer_lower = None : List of neurons which have been fixed to 0 and
            therefore don't need to be approximated by an ICNN (i.e. for Relu(x)=0 because x <= 0)
        self.fixed_neuron_per_layer_upper = None : List of neurons which have been fixed to x and
            therefore don't need to be approximated by an ICNN (i.e. for Relu(x)=x because x >= 0)
        self.num_fixed_neurons_layer = None : the total number of neurons which have been fixed in each layer
        self.all_group_indices = None : List of groups created for each layer by indices
            indicating the neuron of that layer
        self.list_of_icnns = None : List of all ICNNs trained for each group in each layer
        self.list_of_included_samples = [] : if wanted, the included space samples used for training each ICNN
            are stored here
        self.list_of_ambient_samples = [] : if wanted, the included ambient samples used for training each ICNN
            are stored here
    """
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
                           icnn_epochs=100, hyper_lambda=1, init_affine_bounds=None, init_layer_bounds=None,
                           break_after=None, tighten_bounds=False, use_fixed_neurons_in_grouping=False, layers_as_milp=[], layers_as_snr=[],
                           use_over_approximation=True, opt_steps_gd=100,
                           data_grad_descent_steps=0,
                           train_outer=False, preemptive_stop=True, store_samples=False,
                           force_inclusion_steps=0, grouping_method="consecutive", group_num_multiplier=None,
                           init_network=False, adapt_lambda="none", should_plot='none', optimizer="adam",
                           print_training_loss=False, print_last_loss=False, print_optimization_steps=False, print_new_bounds=False,
                           sampling_strategy: SamplingStrategy = UniformSamplingStrategy):
        """

        :param nn: the NN to encode as a Gurobi model given. This has to be a Sequential/ Feedforward NN
        :param input: the input to the NN for which the encoding should be generated
        :param icnn_factory: ICNNFactory that can generate new ICNNs given a needed input dimension
        :param group_size: the size of one group for the approximation with the ICNN. If not enough neurons are
            available the group size is smaller
        :param eps: the epsilon radius for around the input (l-infinity norm)
        :param icnn_batch_size: batch size for each ICNN. Usually we can fit all training data into one batch
        :param icnn_epochs: maximum number of epochs each ICNN is going to train if not interrupted
        :param sample_count: number of trainings data points generated for each ICNN.
            Depending on the parameters sampling_method, sample_new and keep_ambient_space the actual number of samples
            can be different from this number
        :param sampling_method: The method used for generating new training data points for the ICNN
        :param hyper_lambda: the hyperparameter controlling the balance between inclusion loss and ambient loss
        :param init_affine_bounds: List of torch.Tensor of each neuron in each layer giving a lower and upper bound
            of the output of that neuron before the activation function.
            If not provided and tighten_bounds is false these will be just box-bounds
        :param init_layer_bounds: List of torch.Tensor of each neuron in each layer giving a lower and upper bound
            of the output of that neuron after the activation function. It should be same as init_layer_bounds for
            each layer where there is no activation function.
            If not provided and tighten_bounds is false these will be just box-bounds
        :param break_after: int, parameter to preemptively interrupt the generation of the encoding for
            debugging purpose. breaks after n many groups of neurons have been approximated
        :param tighten_bounds: if true, the bounds for self.bounds_affine_out, self.bounds_layer_out are not just
            box-bounds/ the bound provided by init_affine_bounds, init_layer_bounds, for each neuron a new
            upper and lower bound will be determined by solving a maximization/ minimization problem
            given the current encoding of the NN
        :param use_fixed_neurons_in_grouping: boolean deciding whether neurons which can be fixed, hence be exactly
            encoded should instead be included in the grouping and therefore also in the
            approximation of the layer output
        :param layers_as_milp: list of indices indicating which layers should be encoded as a MILP instead of using DHOV
        :param layers_as_snr: list of indices indicating which layers should be encoded with Single-Neuron-Relaxation
            instead of using DHOV
        :param keep_ambient_space: if true, the ambient samples used for the previous layer will be used as ambient
            samples for the training of the approximation of the next layer # todo dependency on sampling_methods
        :param sample_new: if true, sample_count many new samples will be generated for the representation of the
            output of each layer, otherwise samples will only be generated in the input space and then propagated,
            regrouped and split into included and ambient samples for each layer # todo dependency on sampling_methods
        :param use_over_approximation: boolean, deciding whether each ICNN should be enlarged to guarantee the encoding
            being an over approximation. If true, it might also shrink an approximation to the minimal size needed
            to be an over approximation
        :param opt_steps_gd: number of steps the gradient descent should do for each optimization of a training data point #todo dependency on data_grad_descent_steps should be clear
        :param sample_over_input_space: if true, (only) ambient samples will be (uniformly) generated in the
            input space of each layer and through that layer propagated, for training the ICNNs.
            This is independent of the parameter sample_new
        :param sample_over_output_space: if true, (only) ambient samples will be (uniformly) generated over the
            output space of each layer, for training the ICNNs.
        :param data_grad_descent_steps: number of steps, that the training data points should be tried to optimized
            by pushing them in the direction of the current decission boundry with gradient descent
        :param train_outer: if true, (currently not fully working) there will be another training part which only
            focuses on training the boarder of the approximation
        :param preemptive_stop: decides if the training of each ICNN can be preemptively stopped if the moving average
            of the loss is smaller than a certain value
        :param store_samples: whether the generated training data points for each ICNN should be stored
        :param force_inclusion_steps: number of times the each ICNN should be enlarged during training to include
            all included space data samples
        :param grouping_method: method to use for grouping neurons
        :param group_num_multiplier: depending on the grouping method, this parameter is multiplied by the number of
            existing groups to determine a larger of groups (e.g. to have over lapping groups in random grouping) # todo dependency on each grouping method should be clear
        :param init_network: whether each ICNN should be initialized with the current
            upper and lower bounds for the neurons in the corresponding group
        :param adapt_lambda: the strategy to use for adapting the hyperparameter lambda based on the current state of
            balance between included and ambient space
        :param should_plot: plotting strategy which generates multiple plots for each group
            (samples, decision boundary... - depending on the strategy) only works if dimension of group is <= 3
        :param optimizer: optimizer to use for training each ICNN
        :param print_training_loss: if true, loss for each epoch during ICNN training will be printed
        :param print_last_loss: if true, the last loss for each ICNN after training will be printed
        :param print_optimization_steps: if true, optimization steps of Gurobi will be printed where ever
            the optimizer is called. That includes, calculating tighter bounds, the proof for over approximation
            and if applicable data sampling
        :param print_new_bounds: if true, tightened bounds are printed for each layer
        :param sampling_strategy: the sampling strategy to use for generating new training data points for each ICNN defaults to UniformSamplingStrategy
        :return:
        """
        valid_adapt_lambda = ["none", "high_low", "included"]
        valid_should_plot = ["none", "simple", "detailed", "verification", "output"]
        valid_optimizer = ["adam", "LBFGS", "SdLBFGS"]
        valid_sampling_methods = ["uniform", "linespace", "boarder", "sum_noise", "min_max_perturbation",
                                  "alternate_min_max", "per_group_sampling", "per_group_feasible"]
        valid_grouping_methods = ["consecutive", "random"]

        parameter_list = list(nn.parameters())

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
        if group_num_multiplier is not None and grouping_method == "consecutive":
            warnings.warn("value for group number multiplier is given with grouping method consecutive. "
                          "consecutive grouping does not use variable number of groups")

        input_flattened = torch.flatten(input)
        center = input_flattened
        eps_bounds = [input_flattened.add(-eps), input_flattened.add(eps)]

        bounds_affine_out, bounds_layer_out = nn.calculate_box_bounds(eps_bounds)

        if init_affine_bounds is not None:
            bounds_affine_out = init_affine_bounds

        if init_layer_bounds is not None:
            bounds_layer_out = init_layer_bounds

        nn_encoding_model = grp.Model()
        if print_optimization_steps:
            nn_encoding_model.Params.LogToConsole = 1
        else:
            nn_encoding_model.Params.LogToConsole = 0


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
        included_space, ambient_space = None, None

        for i in range(0, len(parameter_list) - 2, 2):  # -2 because last layer has no ReLu activation
            current_layer_index = i // 2
            prev_layer_index = current_layer_index - 1
            print("")
            print("approximation of layer: {}".format(current_layer_index))
            if store_samples:
                self.list_of_included_samples.append([])
                self.list_of_ambient_samples.append([])

            affine_w, affine_b = parameter_list[i], parameter_list[i + 1]

            if tighten_bounds and i != 0:
                t = time.time()
                copy_model = nn_encoding_model.copy()
                ver.update_bounds_with_icnns(copy_model, bounds_affine_out, bounds_layer_out,
                                             current_layer_index, affine_w.detach().cpu().numpy(),
                                             affine_b.detach().cpu().numpy(), print_new_bounds=print_new_bounds)
                print("    time for icnn_bound calculation: {}".format(time.time() - t))

            fix_upper = []
            fix_lower = []

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


            if current_layer_index in layers_as_milp or current_layer_index in layers_as_snr:
                all_group_indices.append([])
                list_of_icnns.append([])
                self.list_of_included_samples.append([])
                self.list_of_ambient_samples.append([])

                prev_layer_index = current_layer_index - 1
                output_prev_layer = []
                for k in range(affine_w.shape[1]):
                    output_prev_layer.append(nn_encoding_model.getVarByName("output_layer_[{}]_[{}]".format(prev_layer_index, k)))
                output_prev_layer = grp.MVar.fromlist(output_prev_layer)

                affine_out = verbas.add_affine_constr(nn_encoding_model, affine_w.detach().cpu().numpy(), affine_b.detach().cpu().numpy(), output_prev_layer,
                                                      bounds_affine_out[current_layer_index][0].detach().cpu().numpy(),
                                                      bounds_affine_out[current_layer_index][1].detach().cpu().numpy(),
                                                      i=current_layer_index)
                if current_layer_index in layers_as_milp:
                    print("    encode layer {} as MILP".format(current_layer_index))
                    relu_out = verbas.add_relu_constr(nn_encoding_model, affine_out, len(affine_b),
                                                      bounds_affine_out[current_layer_index][0].detach().cpu().numpy(),
                                                      bounds_affine_out[current_layer_index][1].detach().cpu().numpy(),
                                                      bounds_layer_out[current_layer_index][0].detach().cpu().numpy(),
                                                      bounds_layer_out[current_layer_index][1].detach().cpu().numpy(),
                                                      i=current_layer_index)

                elif current_layer_index in layers_as_snr:
                    print("    encode layer {} as SNR".format(current_layer_index))
                    relu_out = verbas.add_single_neuron_constr(nn_encoding_model, affine_out, len(affine_b),
                                                               bounds_affine_out[current_layer_index][0].detach().cpu().numpy(),
                                                               bounds_affine_out[current_layer_index][1].detach().cpu().numpy(),
                                                               bounds_layer_out[current_layer_index][0].detach().cpu().numpy(),
                                                               bounds_layer_out[current_layer_index][1].detach().cpu().numpy(),
                                                               i=current_layer_index)
                for k, var in enumerate(relu_out.tolist()):
                    var.setAttr("varname", "output_layer_[{}]_[{}]".format(current_layer_index, k))

                nn_encoding_model.update()
                continue

            if grouping_method == "consecutive":
                if use_fixed_neurons_in_grouping:
                    number_of_groups = get_num_of_groups(len(affine_b), group_size)
                    group_indices = get_current_group_indices(len(affine_b), group_size, [], [])
                else:
                    number_of_groups = get_num_of_groups(len(affine_b) - num_fixed_neurons_layer[current_layer_index],
                                                         group_size)
                    group_indices = get_current_group_indices(len(affine_b), group_size,
                                                              fixed_neuron_per_layer_lower[current_layer_index],
                                                              fixed_neuron_per_layer_upper[current_layer_index])

            elif grouping_method == "random":
                if use_fixed_neurons_in_grouping:
                    number_of_groups = get_num_of_groups(len(affine_b), group_size)
                    number_of_groups = group_num_multiplier * number_of_groups
                    group_indices = get_random_groups(len(affine_b), number_of_groups, group_size, [], [])
                else:
                    number_of_groups = get_num_of_groups(len(affine_b) - num_fixed_neurons_layer[current_layer_index],
                                                         group_size)
                    number_of_groups = group_num_multiplier * number_of_groups
                    group_indices = get_random_groups(len(affine_b), number_of_groups, group_size,
                                                      fixed_neuron_per_layer_lower[current_layer_index],
                                                      fixed_neuron_per_layer_upper[current_layer_index])

            all_group_indices.append(group_indices)

            gurobi_model = nn_encoding_model.copy()
            included_space, ambient_space = sampling_strategy.sampling_by_round(affine_w, affine_b, all_group_indices,
                                                                                gurobi_model, current_layer_index,
                                                                                bounds_affine_out, bounds_layer_out,
                                                                                list_of_icnns)



            list_of_icnns.append([])
            for group_i in range(number_of_groups):
                if break_after is not None:
                    break_after -= 1
                print("    layer progress, group {} of {} ".format(group_i + 1, number_of_groups))

                index_to_select = torch.tensor(group_indices[group_i]).to(device)

                size_of_icnn_input = len(index_to_select)
                current_icnn = icnn_factory.get_new_icnn(size_of_icnn_input)
                list_of_icnns[current_layer_index].append(current_icnn)



                t = time.time()
                group_inc_space = included_space[group_i]
                group_amb_space = ambient_space[group_i]


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
                            train_icnn(current_icnn, train_loader, ambient_loader, epochs=epochs_in_run, hyper_lambda=hyper_lambda,
                                       optimizer=optimizer, adapt_lambda=adapt_lambda, preemptive_stop=preemptive_stop,
                                       verbose=print_training_loss, print_last_loss=print_last_loss)

                            for v in range(optimization_steps):
                                # normalized_ambient_space =
                                # dop.gradient_descent_data_optim(current_icnn, normalized_ambient_space.detach())
                                group_norm_ambient_space = dop.adam_data_optim(current_icnn,
                                                                               group_norm_ambient_space.detach())
                            dataset = ConvexDataset(group_norm_ambient_space.detach())

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
                                   hyper_lambda=hyper_lambda,
                                   optimizer=optimizer, adapt_lambda=adapt_lambda, preemptive_stop=preemptive_stop,
                                   verbose=print_training_loss, print_last_loss=print_last_loss)

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
                    adversarial_input, c = ver.verification(current_icnn, copy_model, affine_w.cpu().detach().cpu().numpy(),
                                                            affine_b.detach().cpu().numpy(), group_indices[group_i],
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
                    return

            # add current layer to model
            curr_constraint_icnns = list_of_icnns[current_layer_index]
            curr_group_indices = all_group_indices[current_layer_index]
            curr_bounds_affine_out = bounds_affine_out[current_layer_index]
            curr_bounds_layer_out = bounds_layer_out[current_layer_index]
            curr_fixed_neuron_lower = fixed_neuron_per_layer_lower[current_layer_index]
            curr_fixed_neuron_upper = fixed_neuron_per_layer_upper[current_layer_index]
            ver.add_layer_to_model(nn_encoding_model, affine_w.detach().cpu().numpy(), affine_b.detach().cpu().numpy(),
                                   curr_constraint_icnns, curr_group_indices,
                                   curr_bounds_affine_out, curr_bounds_layer_out,
                                   curr_fixed_neuron_lower, curr_fixed_neuron_upper,
                                   current_layer_index)
            nn_encoding_model.update()


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
        t = time.time()
        affine_w, affine_b = parameter_list[-2].detach().cpu().numpy(), parameter_list[-1].detach().cpu().numpy()
        last_layer_index = current_layer_index + 1
        output_second_last_layer = []
        for m in range(affine_w.shape[1]):
            output_second_last_layer.append(
                nn_encoding_model.getVarByName("output_layer_[{}]_[{}]".format(last_layer_index - 1, m)))
        output_second_last_layer = grp.MVar.fromlist(output_second_last_layer)
        in_lb = bounds_affine_out[last_layer_index][0].detach().cpu().numpy()
        in_ub = bounds_affine_out[last_layer_index][1].detach().cpu().numpy()
        output_nn = verbas.add_affine_constr(nn_encoding_model, affine_w, affine_b, output_second_last_layer, in_lb, in_ub, i=last_layer_index)
        for m, var in enumerate(output_nn.tolist()):
            var.setAttr("varname", "output_layer_[{}]_[{}]".format(last_layer_index, m))

        nn_encoding_model.update()

        """if tighten_bounds:
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
            bounds_layer_out[last_layer_index][1] = relu_out_ub"""
        print("time for icnn_bound calculation (last layer):{}".format(time.time() - t))
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
        current_group.sort()
        #todo die reihnfolge der elemente in group ist wichtig, deswegen .sort(), das ist ein Bug, das sollte ich fixen,
        # über all wo index_select angewandt wird werden die bounds vertauscht weil torch bei index select so zurückgibt wie in der liste angegeben
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

    plt.scatter(list(map(lambda x: x.detach().cpu().numpy(), gt0)),
                list(map(lambda x: x.detach().cpu().numpy(), gt_x)), c="#ff7f0e")
    plt.scatter(list(map(lambda x: x.detach().cpu().numpy(), leq0)),
                list(map(lambda x: x.detach().cpu().numpy(), leq_x)), c="#1f77b4")
    plt.title("ReLU")
    plt.show()
