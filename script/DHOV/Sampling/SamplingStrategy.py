from abc import ABC, abstractmethod
import torch
import math
from script.settings import data_type, device


# todo i need to adapt each jupyter notebook to the new sampling strategy
# todo remove affine_w and affine_b from the sampling strategy, not neaded as nn_model includes that inforamtion
class SamplingStrategy(ABC):
    def __init__(self, center, input_bounds, nn_model, keep_ambient_space=False, sample_count=10,
                 sample_over_output_space=True,
                 sample_over_input_space=False, sample_new=True, rand_samples_percent=0.2,
                 rand_sample_alternation_percent=0.01):
        self.current_sample_space = None  # todo soll ich das behalten?, wenn ja, dann muss ich das updaten
        self.keep_ambient_space = keep_ambient_space
        self.sample_over_input_space = sample_over_input_space
        self.sample_over_output_space = sample_over_output_space
        self.sample_count = sample_count
        self.sample_new = sample_new
        self.center = center
        self.input_bounds = input_bounds
        self.nn_model = nn_model
        self.rand_samples_percent = rand_samples_percent
        self.rand_sample_alternation_percent = rand_sample_alternation_percent

    def get_num_of_samples(self):
        inc_space_sample_count = self.sample_count // 2

        if self.sample_over_input_space and self.sample_over_output_space:
            amb_space_sample_count = self.sample_count // 4
        else:
            amb_space_sample_count = self.sample_count // 2

        return inc_space_sample_count, amb_space_sample_count

    @abstractmethod
    def _sampling_strategy(self):
        pass

    def _sampling_strategy_for_first_layer(self):

        sample_space = self._sample_over_input_all_groups(self._current_included_sample_count, self._current_affine_w,
                                                          self.input_bounds, self._current_group_indices,
                                                          rand_samples_percent=self.rand_samples_percent,
                                                          rand_sample_alternation_percent=self.rand_sample_alternation_percent)

        self._current_sample_space = (torch
                                     .matmul(self._current_affine_w,
                                             sample_space.view(sample_space.shape[0], sample_space.shape[1],
                                                               sample_space.shape[2], 1))
                                     .view(sample_space.shape[0], sample_space.shape[1],
                                           self._current_affine_w.shape[0])
                                     .add(self._current_affine_b)
                                     )

    def sampling_by_round(self, affine_w, affine_b, all_group_indices, gurobi_model, current_layer_index,
                          bounds_affine_out,
                          bounds_layer_out, list_of_icnns):
        """
        Use this method to sample data points for one layer for all groups.
        Args:
         list_of_icnns:
            affine_w: the weight matrix of the current layer
            affine_b: the bias vector of the current layer
            all_group_indices: the index of the neuron to be sampled, for all groups of all layers until the current one
            gurobi_model: the gurobi model to be used for sampling
            current_layer_index: the index of the current layer
            bounds_affine_out: the bounds of the affine output space of the all layers
            bounds_layer_out: the bounds of the activation output space of the all layers

        Returns: list of tuple of included_space, ambient_space

        """
        self._current_list_included_spaces = []
        self._current_list_ambient_spaces = []
        self._current_included_sample_count, self._current_ambient_sample_count = self.get_num_of_samples()
        self._current_all_group_indices = all_group_indices
        self._current_group_indices = all_group_indices[current_layer_index]
        self._current_affine_w = affine_w
        self._current_affine_b = affine_b
        self._current_gurobi_model = gurobi_model
        self._current_layer_index = current_layer_index
        self._current_bounds_affine_out = bounds_affine_out
        self._current_bounds_layer_out = bounds_layer_out
        self._current_list_of_icnns = list_of_icnns
        self._current_sample_space = None

        # either provide a tensor in self._current_sample_space for all groups
        # or per group in self._current_list_included_spaces
        if current_layer_index == 0:
            self._sampling_strategy_for_first_layer()

        elif current_layer_index > 0:
            self._sampling_strategy()

        else:
            raise ValueError("round_index must be a positive integer or zero. Got: ", current_layer_index)

        new_amb_space = self._samples_uniform_over_all_groups(
            (len(self._current_group_indices), self._current_ambient_sample_count, affine_w.size(0)),
            bounds_layer_out[current_layer_index])

        if len(self._current_list_included_spaces) == 0 and self._current_sample_space is not None:

            expected_shape = [len(self._current_group_indices), self._current_included_sample_count, affine_w.size(0)]

            if list(self._current_sample_space.shape) != expected_shape:
                raise RuntimeError("Expected the tensor for all groups of the included space to be of shape {}, "
                                   "but got: {}".format(expected_shape, self._current_sample_space.shape))
            create_groups_for_included_space = True

        elif len(self._current_list_included_spaces) == len(
                self._current_group_indices) and self._current_sample_space is None:
            create_groups_for_included_space = False

        else:
            raise RuntimeError("The sampling strategy did not provide either a list of included spaces per group "
                               "or a tensor for the included spaces over all groups")

        for i, index_to_select in enumerate(self._current_group_indices):
            index_to_select = torch.tensor(index_to_select).to(device)
            self._current_list_ambient_spaces.append(torch.index_select(new_amb_space[i], 1, index_to_select))

            if create_groups_for_included_space:
                self._current_list_included_spaces.append(
                    torch.index_select(self._current_sample_space[i], 1, index_to_select))

        return self._current_list_included_spaces, self._current_list_ambient_spaces

    def _sample_over_input_all_groups(self, amount, affine_w, input_bounds, group_indices,
                                      rand_samples_percent=0, rand_sample_alternation_percent=0.2):
        samples_per_bound = amount // 2
        lower_bounds = input_bounds[0]
        upper_bounds = input_bounds[1]

        upper = 1
        lower = - 1
        num_rand_samples = math.floor(amount * rand_samples_percent)
        alternations_per_sample = math.floor(affine_w.size(0) * rand_sample_alternation_percent)

        cs = torch.zeros((len(group_indices), samples_per_bound, affine_w.size(0)), dtype=data_type).to(device)
        for i, group in enumerate(group_indices):
            cs[i] = cs[i].index_fill(1, torch.LongTensor(group).to(device), -1)

            if num_rand_samples > 0 and alternations_per_sample > 0:
                rand_index = torch.randint(low=0, high=num_rand_samples * affine_w.size(0),
                                           size=(num_rand_samples * alternations_per_sample,), dtype=torch.int64).to(
                    device)
                cs[i][:num_rand_samples] = cs[i][:num_rand_samples].view(-1).index_fill(0, rand_index, -1).view(
                    num_rand_samples, -1)

            cs[i] = torch.where(cs[i] == -1, (upper - lower) * torch.rand(affine_w.size(0)).to(device) + lower, cs[i])

        affine_w_temp = torch.matmul(cs, affine_w)
        upper_samples = torch.where(affine_w_temp > 0, upper_bounds, lower_bounds)
        lower_samples = torch.where(affine_w_temp < 0, upper_bounds, lower_bounds)

        all_samples = torch.cat([upper_samples, lower_samples], dim=1)

        return all_samples

    def _samples_uniform_over_all_groups(self, shape, bounds, padding=0):
        lb = bounds[0] - padding
        ub = bounds[1] + padding
        random_samples = (ub - lb) * torch.rand(shape, dtype=data_type).to(device) + lb
        data_samples = random_samples

        return data_samples

    def _apply_relu_transform(self, data_samples):
        relu = torch.nn.ReLU()
        transformed_samples = relu(data_samples)

        return transformed_samples

    def _apply_affine_transform(self, affine_w, affine_b, data_samples):
        transformed_samples = torch.empty((data_samples.size(0), affine_b.size(0)), dtype=data_type).to(device)
        for i in range(data_samples.shape[0]):
            transformed_samples[i] = torch.matmul(affine_w, data_samples[i]).add(affine_b)

        return transformed_samples
