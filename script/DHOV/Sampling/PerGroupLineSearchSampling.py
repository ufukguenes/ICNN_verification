import math
import warnings

import torch
import gurobipy as grp

import script.DHOV.DataSampling as ds
from script.settings import data_type, device
from script.DHOV.Sampling.SamplingStrategy import SamplingStrategy


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
                prev_layer_index = current_layer_index - 1
                sample_space = self._sample_group_line_search_lp(sample_space, included_sample_count, affine_w,
                                                                 affine_b,
                                                                 index_to_select, gurobi_model,
                                                                 current_bounds_affine_out,
                                                                 prev_layer_index, list_of_icnns[current_layer_index-1],
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

    def _sample_group_line_search_lp(self, data_samples, amount, affine_w, affine_b, index_to_select, model,
                                     curr_bounds_affine_out, prev_layer_index, current_icnns, rand_samples_percent=0,
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

        # find a point which is inside the current icnn, ideally it is somewhat in the center
        # (chebyshev center would be interesting maybe, but because of relu the problem is not convex anymore)

        uniform_points = ds.samples_uniform_over(data_samples, amount, curr_bounds_affine_out, keep_samples=False,
                                                 padding=0)

        included_space = torch.empty((amount, affine_w.size(0)), dtype=data_type).to(device)
        for i in range(amount):
            included_space[i] = self._line_search_by_direction(current_icnns, cs[i], uniform_points[i], index_to_select)

        if keep_samples and included_space.size(0) > 0:
            included_space = torch.cat([data_samples, included_space], dim=0)
        else:
            included_space = included_space
        return included_space

    def _line_search_by_direction(self, current_icnns, direction, sample, index_to_select, low=-1.0, up=1.0):
        # todo maybe make this more efficient (e.g. golden section search) and parallel execution with matrices
        print("low: {}, up: {}".format(low, up))

        scaled_direction = torch.mul(direction, up)
        new_x = torch.add(scaled_direction, sample)
        new_out = [icnn(torch.index_select(new_x.view(1, -1), 1, torch.tensor(index_to_select))) for icnn in current_icnns]
        #todo ich muss hier bei den icnns die indices auswählen, welche zu den icnns gehören und nicht die neuen indices
        #todo außerdem muss ich darauf achten, dass auch die dimensionen des samples korrekt sind für die keine icnns existieren, also die dimensionen, die nicht in index_to_select enthalten sind

        if max(new_out) < 0:
            return self._line_search_by_direction(current_icnns, direction, sample, index_to_select, low=low, up=2 * up)

        scaled_direction = torch.mul(direction, low)
        new_x = torch.add(scaled_direction, sample)
        new_out = [icnn(torch.index_select(new_x.view(1, -1), 1, torch.tensor(index_to_select))) for icnn in current_icnns]
        if min(new_out) < 0:
            return self._line_search_by_direction(current_icnns, direction, sample, index_to_select, low=low * 2, up=up)

        middle = (up + low) / 2
        scaled_direction = torch.mul(direction, middle)
        new_x = torch.add(scaled_direction, sample)
        new_out = [icnn(torch.index_select(new_x.view(1, -1), 1, torch.tensor(index_to_select))) for icnn in current_icnns]

        if max(new_out) < 0 and max(new_out) > - 0.2:
            return new_x
        if max(new_out) < 0:
            return self._line_search_by_direction(current_icnns, direction, sample, index_to_select, low=middle, up=up)
        if min(new_out) > 0:
            return self._line_search_by_direction(current_icnns, direction, sample, index_to_select, low=low, up=middle)
