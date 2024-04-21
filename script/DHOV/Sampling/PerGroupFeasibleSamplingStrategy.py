import warnings
import torch

import script.DHOV.DataSampling as ds
from script.settings import data_type, device
from script.DHOV.Sampling.SamplingStrategy import SamplingStrategy


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

    def sampling_by_round(self, affine_w, affine_b, all_group_indices, gurobi_model, current_layer_index, bounds_affine_out,
                          bounds_layer_out, list_of_icnns):
        list_included_spaces = []
        list_ambient_spaces = []
        group_indices = all_group_indices[current_layer_index]
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
            included_space = ds.apply_relu_transform(included_space)

            list_included_spaces.append(torch.index_select(included_space, 1, index_to_select))
            list_ambient_spaces.append(ambient_space)

        if self.sample_over_output_space:
            for i in range(len(list_ambient_spaces)):
                new_amb_space = ds.samples_uniform_over(list_ambient_spaces[i], ambient_sample_count,
                                                        bounds_layer_out[current_layer_index])
                new_amb_space = torch.index_select(new_amb_space, 1, group_indices[i])
                old_amb_space = torch.index_select(list_ambient_spaces[i], 1, group_indices[i])
                list_ambient_spaces[i] = torch.concat((old_amb_space, new_amb_space), dim=0)

        return list_included_spaces, list_ambient_spaces
