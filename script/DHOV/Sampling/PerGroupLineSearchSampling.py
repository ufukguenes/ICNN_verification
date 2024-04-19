import math
import warnings

import torch
import gurobipy as grp

import script.DHOV.DataSampling as ds
from script.settings import data_type, device
from script.DHOV.Sampling.SamplingStrategy import SamplingStrategy
from script.NeuralNets.Networks import SequentialNN
import matplotlib.pyplot as plt
import matplotlib


class PerGroupLineSearchSamplingStrategy(SamplingStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.keep_ambient_space:
            warnings.warn("keep_ambient_space is True and sampling method is per_group_sampling. "
                          "Keeping previous samples is not supported when using per group sampling")

        if self.sample_over_input_space:
            warnings.warn("sample_over_input_space is True and sampling method is per_group_sampling. "
                          "Sampling over input space is not yet supported when using per group sampling. "
                          "Using sampling over output space instead...")

    def sampling_by_round(self, affine_w, affine_b, all_group_indices, gurobi_model, current_layer_index,
                          bounds_affine_out,
                          bounds_layer_out, list_of_icnns):
        list_included_spaces = []
        list_ambient_spaces = []
        included_sample_count, ambient_sample_count = self.get_num_of_samples()
        group_indices = all_group_indices[current_layer_index]

        rand_samples_percent = 0.2
        rand_sample_alternation_percent = 0.01

        if current_layer_index == 0:
            sample_space = torch.empty((len(group_indices), 0, affine_w.size(1)), dtype=data_type).to(device)

            #  this is the same as for the PerGroupSamplingStrategy
            sample_space = ds.sample_per_group_all_groups(sample_space, included_sample_count, affine_w, self.center,
                                                          self.eps, group_indices,
                                                          rand_samples_percent=rand_samples_percent,
                                                          rand_sample_alternation_percent=rand_sample_alternation_percent)

            sample_space = (torch
                            .matmul(affine_w, sample_space.view(sample_space.shape[0], sample_space.shape[1], sample_space.shape[2], 1))
                            .view(sample_space.shape[0], sample_space.shape[1], affine_w.shape[0])
                            .add(affine_b)
                            )

        elif current_layer_index > 0:
            sample_space = torch.empty((0, affine_w.size(0)), dtype=data_type).to(device)

            current_bounds_affine_out = bounds_affine_out[current_layer_index]
            sample_space = self._sample_group_line_search_lp(sample_space, included_sample_count, self.nn_model,
                                                             index_to_select, affine_w, all_group_indices,
                                                             current_bounds_affine_out,
                                                             current_layer_index, list_of_icnns,
                                                             rand_samples_percent=rand_samples_percent,
                                                             rand_sample_alternation_percent=rand_sample_alternation_percent)
        else:
            raise ValueError("round_index must be a positive integer or zero. Got: ", current_layer_index)

        sample_space = ds.apply_relu_transform(sample_space)

        new_amb_space = ds.samples_uniform_over_all_groups((len(group_indices), ambient_sample_count, affine_w.size(0)),
                                                           bounds_layer_out[current_layer_index],
                                                           padding=self.eps)

        for index_to_select in group_indices:
            index_to_select = torch.tensor(index_to_select).to(device)
            list_included_spaces.append(torch.index_select(sample_space, 2, index_to_select))
            list_ambient_spaces.append(torch.index_select(new_amb_space, 2, index_to_select))

        return list_included_spaces, list_ambient_spaces

    # todo nn_model is in self, so I don't need to pass it as an argument
    def _sample_group_line_search_lp(self, data_samples, amount, nn_model, index_to_select, affine_w, all_group_indices,
                                     curr_bounds_affine_out, current_layer_index, list_of_icnns, rand_samples_percent=0,
                                     rand_sample_alternation_percent=0.2, keep_samples=True):
        # getting the random directions with controlled random noise
        upper = 1
        lower = - 1
        cs_temp = (upper - lower) * torch.rand((amount, len(index_to_select)),
                                               dtype=data_type).to(device) + lower

        cs = torch.zeros((amount, affine_w.size(0)), dtype=data_type).to(device)

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

        # sample a random point in the input space of the nn_model
        input_samples = torch.empty((0, self.center.size(0)), dtype=data_type).to(device)
        eps_bounds = [self.center.add(-self.eps), self.center.add(self.eps)]
        input_samples = ds.samples_uniform_over(input_samples, amount, eps_bounds)
        input_samples.requires_grad = True
        nn_model.requires_grad = False

        # calculate the gradient of the output in the direction of the output plus the direction
        # for W and b
        nn_until_current_layer = SequentialNN(nn_model.layer_widths[:current_layer_index + 2])
        parameter_list = list(nn_until_current_layer.parameters())

        for i in range((len(nn_until_current_layer.layer_widths) - 1) * 2):
            parameter_list[i].data = list(nn_model.parameters())[i].data

        output_samples = nn_until_current_layer(input_samples)

        optimizer = torch.optim.Adam([input_samples])

        # loss = torch.subtract(output_samples, torch.zeros_like(output_samples))
        loss = torch.sum(output_samples * cs)
        loss.backward()

        # gradients = input_samples.grad
        # gradients = torch.autograd.grad(output_samples, input_samples, grad_outputs=cs)[0]

        # do line search until any bound is violated
        for_plotting = []
        for x in [100]:  # [10, 20, 50, 80, 100]:
            max_iterations = x
            individual_step_size = torch.ones((amount, 1), dtype=data_type).to(device)
            for i in range(max_iterations):
                optimizer.zero_grad()
                loss = self._loss_for_boundary(nn_until_current_layer, input_samples, output_samples, cs, eps_bounds,
                                               list_of_icnns, all_group_indices)
                loss.backward()
                optimizer.step()
                adapted_input_samples = input_samples
                # todo precision rausnehmen wenn ich dabei bleibe nur outside_eps zu verwenden
                continue

                adapted_input_samples = input_samples + torch.mul(gradients, individual_step_size)

                # step in to direction of gradient until eps bound is violated (this will result in the input samples being
                # approximatly on the true boundry, however we want the over approximated boundry given by the icnns,
                # therefore continue gradient decent until the latest icnn is violated.
                # If there is no latest icnn for a neuron use the eps bounds

                # first solution with only check for eps bounds
                # because our first input is guaranteed to be in the eps bounds, we don't need the step size to be negative
                # todo 1. is_within_eps and is_outside_eps always False, why? It should only be false if it is on the boundry
                # todo 2. maybe make this more efficient (e.g. golden section search)
                # todo 3. need to implement check for icnn and ignoring eps bounds if icnn is NOT violated
                # attention: the step size needs to be a scalar, not a vector

                is_within_eps = self._within_eps(adapted_input_samples, eps_bounds)
                is_outside_eps = self._outside_eps(adapted_input_samples, eps_bounds)
                for j in range(amount):
                    if is_within_eps[j]:
                        individual_step_size[j] = individual_step_size[j] * 1.3

                for j in range(amount):
                    if is_outside_eps[j]:
                        individual_step_size[j] = individual_step_size[j] * 0.5

            # get intermediate output before current layer

            included_space = nn_until_current_layer(adapted_input_samples)
            """for_plotting.append(included_space.index_select(1, torch.IntTensor(index_to_select)).detach().cpu().numpy())


        for_plotting.append(output_samples.index_select(1, torch.IntTensor(index_to_select)).detach().cpu().numpy())
        matplotlib.use("TkAgg")
        self.plt_inc_amb_3D("test {}".format(max_iterations),
                            for_plotting)"""

        if keep_samples and included_space.size(0) > 0:
            included_space = torch.cat([data_samples, included_space], dim=0)
        else:
            included_space = included_space
        return included_space

    def _loss_for_boundary(self, nn_model, input_samples, output_samples, direction, eps_bounds, list_of_icnns,
                           all_group_indices):
        output_samples = nn_model(input_samples)
        loss = torch.sum(output_samples * direction)
        loss += 1000 * torch.logical_not(torch.logical_and(self._outside_eps(input_samples, eps_bounds),
                                                           self._check_icnns(nn_model, input_samples, list_of_icnns,
                                                                             all_group_indices))).any()  # self._outside_eps(input_samples, eps_bounds).any() #todo 1000 adaptable machen
        return loss

    def _check_icnns(self, nn_model, input_samples, list_of_icnns, all_group_indices, precision=1e-5):
        parameters = list(nn_model.parameters())
        current_input = input_samples
        intermediate_inputs = []
        relu = torch.nn.ReLU()
        for layer in range(0, len(parameters) - 2, 2):
            W = list(nn_model.parameters())[layer].data
            b = list(nn_model.parameters())[layer + 1].data
            current_input = torch.matmul(W, current_input.T).T.add(b)
            current_input = relu(current_input)
            intermediate_inputs.append(current_input)

        sample_is_outside_any_icnn = torch.zeros(len(input_samples)).bool()
        for layer_index in range(len(all_group_indices) - 1):
            for group_index in range(len(all_group_indices[layer_index])):
                current_icnn = list_of_icnns[layer_index][group_index]
                current_indices = all_group_indices[layer_index][group_index]
                current_intermediate = torch.index_select(intermediate_inputs[layer_index], 1,
                                                          torch.IntTensor(current_indices))
                current_is_outside_icnn = torch.greater(current_icnn(current_intermediate), 0 + precision)
                sample_is_outside_any_icnn = torch.logical_or(sample_is_outside_any_icnn, current_is_outside_icnn)

        return sample_is_outside_any_icnn

    def plt_inc_amb_3D(self, caption, myList: []):  # inc, amb
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.set_ylim(-0.2, 0.2)
        ax.set_xlim(-0.1, 0.25)
        ax.set_zlim(-0.1, 0.3)
        for x in myList:
            ax.scatter(list(map(lambda x: x[0], x)), list(map(lambda x: x[1], x)), list(map(lambda x: x[2], x)))

        """ax.scatter(list(map(lambda x: x[0], amb)), list(map(lambda x: x[1], amb)), list(map(lambda x: x[2], amb)),
                   c="#ff7f0e")
        ax.scatter(list(map(lambda x: x[0], inc)), list(map(lambda x: x[1], inc)), list(map(lambda x: x[2], inc)),
                   c="#1f77b4")"""
        plt.title(caption)
        plt.show()

    def _within_eps(self, input_samples, bounds, precision=1e-5):
        lower = bounds[0]
        upper = bounds[1]
        return torch.logical_and(torch.all(torch.greater_equal(input_samples, lower + precision), dim=1),
                                 torch.all(torch.less_equal(input_samples, upper - precision), dim=1))

    def _outside_eps(self, input_samples, bounds, precision=1e-5):
        lower = bounds[0]
        upper = bounds[1]
        return torch.logical_or(torch.any(torch.less_equal(input_samples, lower - precision), dim=1),
                                torch.any(torch.greater_equal(input_samples, upper + precision), dim=1))
