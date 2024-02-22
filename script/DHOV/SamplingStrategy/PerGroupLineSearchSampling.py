import warnings

import torch

import script.DHOV.DataSampling as ds
from script.settings import data_type, device
from script.DHOV.SamplingStrategy import SamplingStrategy


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
        for index_to_select in group_indices:
            sample_space = torch.empty((0, affine_w.size(1)), dtype=data_type).to(device)
            ambient_space = torch.empty((0, affine_w.size(1)), dtype=data_type).to(device)
            rand_samples_percent = 0.2
            rand_sample_alternation_percent = 0.01

            if current_layer_index == 0:
                #  this is the same as for the PerGroupSamplingStrategy
                sample_space = ds.sample_per_group(sample_space, included_sample_count, affine_w, self.center,
                                                   self.eps, index_to_select,
                                                   rand_samples_percent=rand_samples_percent,
                                                   rand_sample_alternation_percent=rand_sample_alternation_percent)
            elif current_layer_index > 0:
                current_bounds_affine_out = bounds_affine_out[current_layer_index]
                prev_layer_index = current_layer_index - 1
                sample_space = self._sample_group_line_search_lp(sample_space, included_sample_count, affine_w, affine_b,
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

    def _sample_group_line_search_lp(self, data_samples, amount, affine_w, affine_b, index_to_select, model, curr_bounds_affine_out, prev_layer_index, rand_samples_percent=0, rand_sample_alternation_percent=0.2, keep_samples=True):
        upper = 1
        lower = - 1
        cs_temp = (upper - lower) * torch.rand((amount, len(index_to_select)),
                                               dtype=data_type).to(device) + lower

        cs = torch.zeros((amount, affine_w.size(0)), dtype=data_type).to(device)

        samples = torch.empty((amount, affine_w.size(1)), dtype=data_type).to(device)

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

        output_prev_layer = []
        for i in range(affine_w.shape[1]):
            output_prev_layer.append(model.getVarByName("output_layer_[{}]_[{}]".format(prev_layer_index, i)))
        output_prev_layer = grp.MVar.fromlist(output_prev_layer)

        lb = curr_bounds_affine_out[0].detach().cpu().numpy()
        ub = curr_bounds_affine_out[1].detach().cpu().numpy()
        numpy_affine_w = affine_w.detach().cpu().numpy()
        numpy_affine_b = affine_b.detach().cpu().numpy()
        output_var = verbas.add_affine_constr(model, numpy_affine_w, numpy_affine_b, output_prev_layer, lb, ub, i=0)

        model.update()
        for index, c in enumerate(cs):
            c = c.detach().cpu().numpy()
            model.setObjective(c @ output_var, grp.GRB.MAXIMIZE)

            model.optimize()
            if model.Status == grp.GRB.OPTIMAL:
                samples[index] = torch.tensor(output_prev_layer.getAttr("X"), dtype=data_type).to(device)
            else:
                print("Model unfeasible?")

        if keep_samples and data_samples.size(0) > 0:
            data_samples = torch.cat([data_samples, samples], dim=0)
        else:
            data_samples = samples
        return data_samples