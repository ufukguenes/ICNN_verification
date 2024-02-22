import warnings
from abc import ABC, abstractmethod

import torch

import script.DHOV.DataSampling as ds
from script.settings import data_type, device


# todo i need to adapt each jupyter notebook to the new sampling strategy
# todo make each strategy return an iterator by group for before and after sampling
class SamplingStrategy(ABC):
    def __init__(self, center, eps, keep_ambient_space=False, sample_count=10, sample_over_output_space=True,
                 sample_over_input_space=False, sample_new=True):
        self.current_sample_space = None  # todo soll ich das behalten?, wenn ja, dann muss ich das updaten
        self.keep_ambient_space = keep_ambient_space
        self.sample_over_input_space = sample_over_input_space
        self.sample_over_output_space = sample_over_output_space
        self.sample_count = sample_count
        self.sample_new = sample_new
        self.center = center
        self.eps = eps

    def get_num_of_samples(self):
        inc_space_sample_count = self.sample_count // 2

        if self.sample_over_input_space and self.sample_over_output_space:
            amb_space_sample_count = self.sample_count // 4
        else:
            amb_space_sample_count = self.sample_count // 2

        return inc_space_sample_count, amb_space_sample_count

    @abstractmethod
    def sampling_by_round(self, affine_w, affine_b, group_indices, gurobi_model, current_layer_index, bounds_affine_out,
                          bounds_layer_out, list_of_icnns):
        """
        Use this method to sample data points for one layer for all groups.
        Args:
         list_of_icnns:
            affine_w: the weight matrix of the current layer
            affine_b: the bias vector of the current layer
            group_indices: the index of the neuron to be sampled, for all groups
            gurobi_model: the gurobi model to be used for sampling
            current_layer_index: the index of the current layer
            bounds_affine_out: the bounds of the affine output space of the all layers
            bounds_layer_out: the bounds of the activation output space of the all layers

        Returns: list of tuple of included_space, ambient_space

        """
        pass


class PerGroupSamplingStrategy(SamplingStrategy):
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
        for index_to_select in group_indices:
            sample_space = torch.empty((0, affine_w.size(1)), dtype=data_type).to(device)
            ambient_space = torch.empty((0, affine_w.size(1)), dtype=data_type).to(device)
            rand_samples_percent = 0.2
            rand_sample_alternation_percent = 0.01

            if current_layer_index == 0:
                sample_space = ds.sample_per_group(sample_space, included_sample_count, affine_w, self.center,
                                                   self.eps, index_to_select,
                                                   rand_samples_percent=rand_samples_percent,
                                                   rand_sample_alternation_percent=rand_sample_alternation_percent)
            elif current_layer_index > 0:
                current_bounds_affine_out = bounds_affine_out[current_layer_index]
                prev_layer_index = current_layer_index - 1
                sample_space = ds.sample_per_group_as_lp(sample_space, included_sample_count, affine_w, affine_b,
                                                         index_to_select, gurobi_model,
                                                         current_bounds_affine_out,
                                                         prev_layer_index,
                                                         rand_samples_percent=rand_samples_percent,
                                                         rand_sample_alternation_percent=rand_sample_alternation_percent)
            else:
                raise ValueError("round_index must be a positive integer or zero. Got: ", current_layer_index)

            sample_space = ds.apply_affine_transform(affine_w, affine_b, sample_space)
            ambient_space = ds.apply_affine_transform(affine_w, affine_b, ambient_space)

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
                    new_amb_space = torch.index_select(new_amb_space, 1, index_to_select)
                    old_amb_space = torch.index_select(list_ambient_spaces[i], 1, index_to_select)
                    list_ambient_spaces[i] = torch.concat((old_amb_space, new_amb_space), dim=0)

        return list_included_spaces, list_ambient_spaces


class PerGroupFeasibleSamplingStrategy(SamplingStrategy):
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
        for index_to_select in group_indices:
            included_space = torch.empty((0, affine_w.size(1)), dtype=data_type).to(device)
            prev_layer_index = current_layer_index - 1

            index_to_select = torch.tensor(index_to_select).to(device)
            included_space, ambient_space = ds.sample_feasible(included_space, included_sample_count,
                                                               affine_w, affine_b, index_to_select,
                                                               gurobi_model,
                                                               bounds_affine_out[current_layer_index],
                                                               bounds_layer_out[current_layer_index],
                                                               prev_layer_index)

            # test if list_included_spaces and list_ambient_spaces have more than zero elements
            if included_space.size(0) == 0:
                raise ValueError(
                    "No included space samples were generated. Possible solution might be to reduce group size.")

            included_space = ds.apply_affine_transform(affine_w, affine_b, included_space)

            list_included_spaces.append(torch.index_select(included_space, 1, index_to_select))
            list_ambient_spaces.append(ambient_space)

        if self.sample_over_output_space:
            for i in range(len(list_ambient_spaces)):
                new_ambient_space = torch.empty((0, affine_w.size(0)), dtype=data_type).to(device)
                new_amb_space = ds.samples_uniform_over(new_ambient_space, ambient_sample_count,
                                                        bounds_layer_out[current_layer_index],
                                                        padding=self.eps)
                new_amb_space = torch.index_select(new_amb_space, 1, index_to_select)
                list_ambient_spaces[i] = torch.concat((list_ambient_spaces[i], new_amb_space), dim=0)

        return list_included_spaces, list_ambient_spaces


class PropagateSamplingStrategy(SamplingStrategy, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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

    def sampling_by_round(self, affine_w, affine_b, group_indices, gurobi_model, current_layer_index, bounds_affine_out,
                          bounds_layer_out, list_of_icnns):

        """

        Args:
            list_of_icnns:

        """

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
