import math
import warnings

import torch
import gurobipy as grp

import script.DHOV.DataSampling as ds
from script.settings import data_type, device
from script.DHOV.Sampling.SamplingStrategy import SamplingStrategy
from script.NeuralNets.Networks import SequentialNN


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

    def sampling_by_round(self, affine_w, affine_b, group_indices, gurobi_model, current_layer_index, bounds_affine_out,
                          bounds_layer_out, list_of_icnns):
        list_included_spaces = []
        list_ambient_spaces = []
        included_sample_count, ambient_sample_count = self.get_num_of_samples()
        for i, index_to_select in enumerate(group_indices):

            rand_samples_percent = 0.2
            rand_sample_alternation_percent = 0.01

            if current_layer_index == 0:
                sample_space = torch.empty((0, affine_w.size(1)), dtype=data_type).to(device)
                ambient_space = torch.empty((0, affine_w.size(1)), dtype=data_type).to(device)

                #  this is the same as for the PerGroupSamplingStrategy
                sample_space = ds.sample_per_group(sample_space, included_sample_count, affine_w, self.center,
                                                   self.eps, index_to_select,
                                                   rand_samples_percent=rand_samples_percent,
                                                   rand_sample_alternation_percent=rand_sample_alternation_percent)

                sample_space = ds.apply_affine_transform(affine_w, affine_b, sample_space)
                ambient_space = ds.apply_affine_transform(affine_w, affine_b, ambient_space)

            elif current_layer_index > 0:
                sample_space = torch.empty((0, affine_w.size(0)), dtype=data_type).to(device)
                ambient_space = torch.empty((0, affine_w.size(0)), dtype=data_type).to(device)

                current_bounds_affine_out = bounds_affine_out[current_layer_index]
                sample_space = self._sample_group_line_search_lp(sample_space, included_sample_count, self.nn_model,
                                                                 index_to_select, affine_w,
                                                                 current_bounds_affine_out,
                                                                 current_layer_index, list_of_icnns,
                                                                 rand_samples_percent=rand_samples_percent,
                                                                 rand_sample_alternation_percent=rand_sample_alternation_percent)
            else:
                raise ValueError("round_index must be a positive integer or zero. Got: ", current_layer_index)

            sample_space = ds.apply_relu_transform(sample_space)
            ambient_space = ds.apply_relu_transform(ambient_space)

            index_to_select = torch.tensor(index_to_select).to(device)
            list_included_spaces.append(torch.index_select(sample_space, 1, index_to_select))

            list_ambient_spaces.append(ambient_space)

        if self.sample_over_output_space:
            for i in range(len(list_ambient_spaces)):
                new_amb_space = ds.samples_uniform_over(list_ambient_spaces[i], ambient_sample_count,
                                                        bounds_layer_out[current_layer_index],
                                                        padding=self.eps)
                new_amb_space = torch.index_select(new_amb_space, 1, torch.tensor(group_indices[i]).to(device))
                old_amb_space = torch.index_select(list_ambient_spaces[i], 1, torch.tensor(group_indices[i]).to(device))
                list_ambient_spaces[i] = torch.concat((old_amb_space, new_amb_space), dim=0)

        return list_included_spaces, list_ambient_spaces

    # todo nn_model is in self, so I don't need to pass it as an argument
    def _sample_group_line_search_lp(self, data_samples, amount, nn_model, index_to_select, affine_w,
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
        # output_samples_with_direction = torch.add(output_samples, cs)
        # loss = torch.subtract(output_samples_with_direction, output_samples) #.sum(dim=1)
        # loss.backward()
        # loss.backward(cs)
        output_samples.backward(cs)

        gradients = input_samples.grad

        # do line search until any bound is violated
        max_iterations = 100
        individual_step_size = torch.ones((amount, 1), dtype=data_type).to(device)
        for i in range(max_iterations):
            adapted_input_samples = input_samples + torch.mul(gradients, individual_step_size)

            # step in to direction of gradient until eps bound is violated (this will result in the input samples being
            # approximatly on the true boundry, however we want the over approximated boundry given by the icnns,
            # therefore continue gradient decent until the latest icnn is violated.
            # If there is no latest icnn for a neuron use the eps bounds

            # first solution with only check for eps bounds
            # because our first input is guaranteed to be in the eps bounds, we don't need the step size to be negative
            # todo maybe make this more efficient (e.g. golden section search)

            #todo is_within_eps and is_outside_eps always False, why? It should only be false if it is on the boundry
            is_within_eps = self._within_eps(adapted_input_samples, eps_bounds)
            for j in range(amount):
                if is_within_eps[j]:
                    individual_step_size[j] = individual_step_size[j] * 1.3

            is_outside_eps = self._outside_eps(adapted_input_samples, eps_bounds)
            for j in range(amount):
                if is_outside_eps[j]:
                    individual_step_size[j] = individual_step_size[j] * 0.5

        # get intermediate output before current layer
        included_space = nn_until_current_layer(adapted_input_samples)

        if keep_samples and included_space.size(0) > 0:
            included_space = torch.cat([data_samples, included_space], dim=0)
        else:
            included_space = included_space
        return included_space

    def _within_eps(self, input_samples, bounds, precision=1e-3):
        lower = bounds[0]
        upper = bounds[1]
        return torch.logical_and(torch.all(torch.greater_equal(input_samples, lower + precision), dim=1),
                                 torch.all(torch.less_equal(input_samples, upper - precision), dim=1))

    def _outside_eps(self, input_samples, bounds, precision=1e-3):
        lower = bounds[0]
        upper = bounds[1]
        return torch.logical_or(torch.all(torch.less_equal(input_samples, lower - precision), dim=1),
                                torch.all(torch.greater_equal(input_samples, upper + precision), dim=1))
