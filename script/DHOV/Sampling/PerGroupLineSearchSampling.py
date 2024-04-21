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
            sample_space = torch.empty((len(group_indices), 0, affine_w.size(0)), dtype=data_type).to(device)

            current_bounds_affine_out = bounds_affine_out[current_layer_index]
            sample_space = self._sample_group_line_search_lp(sample_space, included_sample_count, self.nn_model,
                                                             group_indices, affine_w, all_group_indices,
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

        for i, index_to_select in enumerate(group_indices):
            index_to_select = torch.tensor(index_to_select).to(device)
            list_included_spaces.append(torch.index_select(sample_space[i], 1, index_to_select))
            list_ambient_spaces.append(torch.index_select(new_amb_space[i], 1, index_to_select))

        return list_included_spaces, list_ambient_spaces

    # todo nn_model is in self, so I don't need to pass it as an argument
    def _sample_group_line_search_lp(self, data_samples, amount, nn_model, group_indices, affine_w, all_group_indices,
                                     curr_bounds_affine_out, current_layer_index, list_of_icnns, rand_samples_percent=0,
                                     rand_sample_alternation_percent=0.2, keep_samples=True):
        # getting the random directions with controlled random noise
        upper = 1
        lower = - 1
        num_rand_samples = math.floor(amount * rand_samples_percent)
        alternations_per_sample = math.floor(affine_w.size(0) * rand_sample_alternation_percent)

        cs = torch.zeros((len(group_indices), amount, affine_w.size(0)), dtype=data_type).to(device)
        for i, group in enumerate(group_indices):
            cs[i] = cs[i].index_fill(1, torch.LongTensor(group), -1)

            if num_rand_samples > 0 and alternations_per_sample > 0:
                rand_index = torch.randint(low=0, high=num_rand_samples * affine_w.size(0),
                                           size=(num_rand_samples * alternations_per_sample,), dtype=torch.int64).to(
                    device)
                cs[i][:num_rand_samples] = cs[i][:num_rand_samples].view(-1).index_fill(0, rand_index, -1).view(
                    num_rand_samples, -1)

            cs[i] = torch.nn.functional.normalize(torch.where(cs[i] == -1, (upper - lower) * torch.rand(affine_w.size(0)) + lower, cs[i]), dim=1)

        # sample a random point in the input space of the nn_model
        eps_bounds = [self.center.add(-self.eps), self.center.add(self.eps)]
        input_samples = ds.samples_uniform_over_all_groups((len(group_indices), amount, self.center.size(0)), eps_bounds)

        cs = cs.view(-1, cs.shape[-1])
        input_samples = input_samples.view(-1, input_samples.shape[-1]).detach()
        input_samples.requires_grad = True


        # calculate the gradient of the output in the direction of the output plus the direction
        # for W and b
        nn_model.requires_grad = False
        nn_until_current_layer = SequentialNN(nn_model.layer_widths[:current_layer_index + 2])
        parameter_list = list(nn_until_current_layer.parameters())

        for i in range((len(nn_until_current_layer.layer_widths) - 1) * 2):
            parameter_list[i].data = list(nn_model.parameters())[i].data


        output_samples = nn_until_current_layer(input_samples)

        optimizer = torch.optim.Adam([input_samples])

        # do line search until any bound is violated
        num_iterations = 100

        for i in range(num_iterations):
            optimizer.zero_grad()
            loss = self._loss_for_boundary(nn_until_current_layer, input_samples, cs, eps_bounds,
                                           list_of_icnns, all_group_indices)
            loss.backward()
            optimizer.step()
            adapted_input_samples = input_samples

        # get intermediate output before current layer
        included_space = nn_until_current_layer(adapted_input_samples)

        return included_space.view(len(group_indices), amount, output_samples.shape[-1])

    def _loss_for_boundary(self, nn_model, input_samples, direction, eps_bounds, list_of_icnns,
                           all_group_indices):
        output_samples = nn_model(input_samples)
        loss = torch.sum(output_samples * direction)
        loss += 1000 * torch.logical_not(torch.logical_and(self._outside_eps(input_samples, eps_bounds),
                                                           self._check_icnns(nn_model, input_samples, list_of_icnns,
                                                                             all_group_indices))).any() #todo 1000 adaptable machen
        return loss

    def _check_icnns(self, nn_model, input_samples, list_of_icnns, all_group_indices, precision=1e-5):
        parameters = list(nn_model.parameters())
        current_input = input_samples
        intermediate_inputs = []
        relu = torch.nn.ReLU()
        sample_is_outside_any_icnn = torch.zeros(len(input_samples)).bool()

        for layer in range(0, len(parameters) - 2, 2):
            W = list(nn_model.parameters())[layer].data
            b = list(nn_model.parameters())[layer + 1].data
            current_input = torch.matmul(W, current_input.T).T.add(b)
            current_input = relu(current_input)
            intermediate_inputs.append(current_input)

            layer_index = layer // 2

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
