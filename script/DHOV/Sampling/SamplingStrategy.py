from abc import ABC, abstractmethod
import torch
import math
from script.settings import data_type, device


# todo i need to adapt each jupyter notebook to the new sampling strategy
#todo remove affine_w and affine_b from the sampling strategy, not neaded as nn_model includes that inforamtion
class SamplingStrategy(ABC):
    def __init__(self, center, input_bounds, nn_model, keep_ambient_space=False, sample_count=10, sample_over_output_space=True,
                 sample_over_input_space=False, sample_new=True):
        self.current_sample_space = None  # todo soll ich das behalten?, wenn ja, dann muss ich das updaten
        self.keep_ambient_space = keep_ambient_space
        self.sample_over_input_space = sample_over_input_space
        self.sample_over_output_space = sample_over_output_space
        self.sample_count = sample_count
        self.sample_new = sample_new
        self.center = center
        self.input_bounds = input_bounds
        self.nn_model = nn_model

    def get_num_of_samples(self):
        inc_space_sample_count = self.sample_count // 2

        if self.sample_over_input_space and self.sample_over_output_space:
            amb_space_sample_count = self.sample_count // 4
        else:
            amb_space_sample_count = self.sample_count // 2

        return inc_space_sample_count, amb_space_sample_count

    @abstractmethod
    def sampling_by_round(self, affine_w, affine_b, all_group_indices, gurobi_model, current_layer_index, bounds_affine_out,
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
        pass

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
            cs[i] = cs[i].index_fill(1, torch.LongTensor(group), -1)

            if num_rand_samples > 0 and alternations_per_sample > 0:
                rand_index = torch.randint(low=0, high=num_rand_samples * affine_w.size(0),
                                           size=(num_rand_samples * alternations_per_sample,), dtype=torch.int64).to(
                    device)
                cs[i][:num_rand_samples] = cs[i][:num_rand_samples].view(-1).index_fill(0, rand_index, -1).view(
                    num_rand_samples, -1)

            cs[i] = torch.where(cs[i] == -1, (upper - lower) * torch.rand(affine_w.size(0)) + lower, cs[i])

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