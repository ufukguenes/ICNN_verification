from abc import ABC, abstractmethod

import torch

import script.DHOV.DataSampling as ds
from script.settings import data_type, device
from script.DHOV.Sampling.SamplingStrategy import SamplingStrategy
import warnings
class PropagateSamplingStrategy(SamplingStrategy, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        raise NotImplementedError("This sampling strategy still needs to be adapted to work without "
                                  "specification of eps and just use the input bounds")
        self.unaltered_included_space_prev_round = None
        self.unaltered_ambient_space_prev_round = None
        self.prev_group_indices = None

    @abstractmethod
    def _initial_sampling_method(self, affine_w, affine_b, current_layer_index, bounds_affine_out, bounds_layer_out):
        pass

    def _propagate_space(self, included_space, ambient_space, affine_w, affine_b):
        included_space = ds.apply_affine_transform(affine_w, affine_b, included_space)
        included_space = ds.apply_relu_transform(included_space)
        ambient_space = ds.apply_affine_transform(affine_w, affine_b, ambient_space)
        ambient_space = ds.apply_relu_transform(ambient_space)

        return included_space, ambient_space

    def _regroup(self, current_icnns, included_space, ambient_space, group_indices):
        included_space, ambient_space = ds.regroup_samples(current_icnns, included_space, ambient_space, group_indices)
        return included_space, ambient_space

    def _sample_new(self, current_icnns, sample_count, group_indices, bounds_layer_out):
        included_space, ambient_space = ds.sample_max_radius(current_icnns, sample_count, group_indices,
                                                             bounds_layer_out,
                                                             keep_ambient_space=self.keep_ambient_space)
        return included_space, ambient_space

    def _sample_over_input_space(self, ambient_space, current_icnns, sample_count, current_layer_index,
                                 bounds_layer_out):
        eps_bounds = [self.center.add(-self.eps), self.center.add(self.eps)]

        if current_layer_index == 0:
            ambient_space = ds.sample_uniform_excluding(ambient_space, sample_count, eps_bounds,
                                                        excluding_bound=eps_bounds, padding=self.eps)
        else:
            ambient_space = ds.sample_uniform_excluding(ambient_space, sample_count,
                                                        bounds_layer_out[current_layer_index - 1],
                                                        icnns=current_icnns[current_layer_index - 1],
                                                        layer_index=current_layer_index,
                                                        group_size=None,
                                                        padding=self.eps)
        return ambient_space

    def sampling_by_round(self, affine_w, affine_b, all_group_indices, gurobi_model, current_layer_index, bounds_affine_out,
                          bounds_layer_out, list_of_icnns):

        """

        Args:
            list_of_icnns:

        """

        group_indices = all_group_indices[current_layer_index]
        list_included_spaces = []
        list_ambient_spaces = []
        included_sample_count, ambient_sample_count = self.get_num_of_samples()

        if current_layer_index == 0:
            ambient_space = torch.empty((0, affine_w.size(1)), dtype=data_type).to(device)
            included_space = self._initial_sampling_method(affine_w, affine_b, current_layer_index, bounds_affine_out,
                                                           bounds_layer_out)

        else:
            if self.sample_new:
                included_space, ambient_space = self._sample_new(list_of_icnns[current_layer_index - 1],
                                                                 included_sample_count,
                                                                 self.prev_group_indices,
                                                                 bounds_layer_out[current_layer_index])
            else:
                included_space, ambient_space = self._regroup(list_of_icnns[current_layer_index - 1],
                                                              self.unaltered_included_space_prev_round,
                                                              self.unaltered_ambient_space_prev_round,
                                                              self.prev_group_indices)

        if self.sample_over_input_space:
            ambient_space_over_input = self._sample_over_input_space(ambient_space, list_of_icnns, ambient_sample_count,
                                                                     current_layer_index, bounds_layer_out)
            ambient_space = torch.concat((ambient_space, ambient_space_over_input), dim=0)

        included_space, ambient_space = self._propagate_space(included_space, ambient_space, affine_w, affine_b)

        if self.sample_over_output_space:
            new_ambient_space = torch.empty((0, affine_w.size(0)), dtype=data_type).to(device)
            new_ambient_space = ds.samples_uniform_over(new_ambient_space, ambient_sample_count,
                                                        bounds_layer_out[current_layer_index],
                                                        padding=self.eps)
            if self.keep_ambient_space:
                ambient_space = torch.concat((ambient_space, new_ambient_space), dim=0)
            else:
                ambient_space = new_ambient_space

        self.unaltered_included_space_prev_round = included_space
        self.unaltered_ambient_space_prev_round = ambient_space
        self.prev_group_indices = group_indices

        for index_to_select in group_indices:
            index_to_select = torch.tensor(index_to_select).to(device)
            list_included_spaces.append(torch.index_select(included_space, 1, index_to_select))
            list_ambient_spaces.append(torch.index_select(ambient_space, 1, index_to_select))

        return list_included_spaces, list_ambient_spaces


class UniformSamplingStrategy(PropagateSamplingStrategy, ABC):

    def _initial_sampling_method(self, affine_w, affine_b, current_layer_index, bounds_affine_out, bounds_layer_out):
        included_sample_count, ambient_sample_count = self.get_num_of_samples()

        included_space = torch.empty((0, self.center.size(0)), dtype=data_type).to(device)
        eps_bounds = [self.center.add(-self.eps), self.center.add(self.eps)]
        included_space = ds.samples_uniform_over(included_space, included_sample_count, eps_bounds)

        return included_space


class LinespaceSamplingStrategy(PropagateSamplingStrategy, ABC):
    def _initial_sampling_method(self, affine_w, affine_b, current_layer_index, bounds_affine_out, bounds_layer_out):
        included_sample_count, ambient_sample_count = self.get_num_of_samples()

        sample_space = torch.empty((0, self.center.size(0)), dtype=data_type).to(device)
        sample_space = ds.sample_linspace(sample_space, included_sample_count, self.center, self.eps)
        self.current_sample_space = sample_space
        return sample_space


class BoarderSamplingStrategy(PropagateSamplingStrategy, ABC):

    def _initial_sampling_method(self, affine_w, affine_b, current_layer_index, bounds_affine_out, bounds_layer_out):
        included_sample_count, ambient_sample_count = self.get_num_of_samples()

        sample_space = torch.empty((0, self.center.size(0)), dtype=data_type).to(device)
        sample_space = ds.sample_boarder(sample_space, included_sample_count, self.center, self.eps)
        self.current_sample_space = sample_space
        return sample_space


class SumNoiseSamplingStrategy(PropagateSamplingStrategy, ABC):

    def _initial_sampling_method(self, affine_w, affine_b, current_layer_index, bounds_affine_out, bounds_layer_out):
        included_sample_count, ambient_sample_count = self.get_num_of_samples()

        sample_space = torch.empty((0, self.center.size(0)), dtype=data_type).to(device)
        sample_space = ds.sample_random_sum_noise(included_sample_count, included_sample_count, self.center, self.eps)
        self.current_sample_space = sample_space
        return sample_space


class MinMaxPerturbationSamplingStrategy(PropagateSamplingStrategy, ABC):
    def _initial_sampling_method(self, affine_w, affine_b, current_layer_index, bounds_affine_out, bounds_layer_out):
        included_sample_count, ambient_sample_count = self.get_num_of_samples()

        sample_space = torch.empty((0, self.center.size(0)), dtype=data_type).to(device)
        sample_space = ds.sample_min_max_perturbation(sample_space, included_sample_count, affine_w, self.center,
                                                      self.eps)
        self.current_sample_space = sample_space
        return sample_space


class AlternateMinMaxSamplingStrategy(PropagateSamplingStrategy, ABC):
    def _initial_sampling_method(self, affine_w, affine_b, current_layer_index, bounds_affine_out, bounds_layer_out):
        included_sample_count, ambient_sample_count = self.get_num_of_samples()

        sample_space = torch.empty((0, self.center.size(0)), dtype=data_type).to(device)
        sample_space = ds.sample_alternate_min_max(sample_space, included_sample_count, affine_w, self.center, self.eps)
        self.current_sample_space = sample_space
        return sample_space