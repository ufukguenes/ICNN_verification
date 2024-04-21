from abc import ABC, abstractmethod


# todo i need to adapt each jupyter notebook to the new sampling strategy
# todo make each strategy return an iterator by group for before and after sampling
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
