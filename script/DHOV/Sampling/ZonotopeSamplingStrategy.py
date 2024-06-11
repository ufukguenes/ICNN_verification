import torch
import torch.nn as nn

import script.DHOV.DataSampling as ds
from script.settings import data_type, device
from script.DHOV.Sampling.SamplingStrategy import SamplingStrategy

from script.ZonotopePropagation.Zonotope import Zonotope
from script.ZonotopePropagation.ZonotopePropagation import ZonotopePropagator

class ZonotopeSamplingStrategy(SamplingStrategy):

    def __init__(self, *args, use_intermediate_bounds=True, **kwargs):
        super().__init__(*args, **kwargs)

        in_lbs = self.input_bounds[0]
        in_ubs = self.input_bounds[1]
        self.zono = Zonotope.from_bounds(in_lbs, in_ubs, shape=(len(in_lbs),), dtype=data_type)
        self.use_intermediate_bounds = use_intermediate_bounds

    def _sampling_strategy(self):
        pass

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
        z = ZonotopePropagator.propagate_linear(affine_w, affine_b, self.zono)

        
        current_bounds_affine_out = bounds_affine_out[current_layer_index]
        current_bounds_layer_out  = bounds_layer_out[current_layer_index]

        n_included, n_ambient = self.get_num_of_samples()
        included_samples = []
        ambient_samples = []
        for group in group_indices[current_layer_index]:
            z_group = z[group]
            P = z_group.sample_boundary(n_samples=n_included)
            #samples = torch.zeros((n_included, affine_b.shape[0]), dtype=data_type)
            #samples[:,group] = nn.functional.relu(P.T)
            samples = nn.functional.relu(P.T)
            included_samples.append(samples)

            new_amb_space = ds.samples_uniform_over(torch.zeros((n_ambient, affine_b.shape[0])), n_ambient,
                                                        current_bounds_layer_out, keep_samples=False)
            new_amb_space = torch.index_select(new_amb_space, 1, torch.tensor(group))

            ambient_samples.append(new_amb_space)

        lb = current_bounds_affine_out[0] if self.use_intermediate_bounds else None
        ub = current_bounds_affine_out[1] if self.use_intermediate_bounds else None
        self.zono = ZonotopePropagator.propagate_relu(z, lb=lb, ub=ub)

        return included_samples, ambient_samples
