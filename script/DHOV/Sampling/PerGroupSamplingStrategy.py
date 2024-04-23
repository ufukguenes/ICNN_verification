import warnings
import torch

import script.DHOV.DataSampling as ds
from script.settings import data_type, device
from script.DHOV.Sampling.SamplingStrategy import SamplingStrategy


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

    def _sampling_strategy(self):
        for index_to_select in self._current_group_indices:
            sample_space = torch.empty((0, self._current_affine_w.size(1)), dtype=data_type).to(device)
            current_bounds_affine_out = self._current_bounds_affine_out[self._current_layer_index]
            prev_layer_index = self._current_layer_index - 1
            sample_space = ds.sample_per_group_as_lp(sample_space, self._current_included_sample_count,
                                                     self._current_affine_w, self._current_affine_b,
                                                     index_to_select, self._current_gurobi_model,
                                                     current_bounds_affine_out,
                                                     prev_layer_index,
                                                     rand_samples_percent=self.rand_samples_percent,
                                                     rand_sample_alternation_percent=self.rand_sample_alternation_percent)
            sample_space = self._apply_affine_transform(self._current_affine_w, self._current_affine_b, sample_space)
            sample_space = self._apply_relu_transform(sample_space)

            index_to_select = torch.tensor(index_to_select).to(device)
            self._current_list_included_spaces.append(torch.index_select(sample_space, 1, index_to_select))
