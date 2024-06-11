
import torch
import torch.nn as nn
import torch.nn.functional as F

# why do I need these .. and . ???
from ..NeuralNets.Networks import VerifiableNet
from .Zonotope import Zonotope
    

class ZonotopePropagator:

    def __init__(self, nn : VerifiableNet) -> None:
        self.nn = nn

    @staticmethod
    def propagate_linear(W, b, z: Zonotope) -> Zonotope:
        return Zonotope(F.linear(z.center, W, b), F.linear(z.generators, W))

    @staticmethod
    def propagate_relu(z: Zonotope, lb=None, ub=None) -> Zonotope:
        """
        Propagates a zonotope through a ReLU layer.

        If no concrete lower and upper bounds are given, the bounds are calculated from the input zonotope.
        Otherwise, the tighter bound from the bounds calculated by the zonotope and the given concrete bounds are used.

        taken from https://github.com/fzuerkeraguilar/alpha-beta-deepZ/blob/main/abZono/trans_layers/ZonoReLU.py (with slight modification)
        """
        gen_abs_sum = z.generators.abs().sum(dim=0)
        l = z.center - gen_abs_sum
        u = z.center + gen_abs_sum

        if not lb is None:
            l = torch.maximum(lb, l)
            u = torch.minimum(ub, u)

        zero_tensor = torch.zeros_like(z.center)
        negative_mask = (u < 0)
        crossing_mask = (l < 0) & (u > 0)

        slope = u / (u - l)
        new_generator = -slope * l * 0.5 * crossing_mask.float()

        new_center = torch.where(crossing_mask, z.center * slope + new_generator, z.center)
        new_center = torch.where(negative_mask, zero_tensor, new_center)

        compressed_generator_indices = crossing_mask.nonzero(as_tuple=True)
        num_activations = compressed_generator_indices[0].size(0)
        do_not_repeat_other_dims = [1] * z.center.dim()
        stacked_generator_indices = (torch.arange(num_activations), *compressed_generator_indices)

        new_eps_terms = zero_tensor.unsqueeze(0).repeat(num_activations, *do_not_repeat_other_dims)
        new_eps_terms[stacked_generator_indices] = new_generator[compressed_generator_indices]

        old_new_generators = torch.where(crossing_mask, z.generators * slope, z.generators)
        old_new_generators = torch.where(negative_mask, torch.zeros_like(z.generators), old_new_generators)

        return Zonotope(new_center, torch.cat((old_new_generators, new_eps_terms), dim=0))


    def propagate(self, in_lbs, in_ubs, dtype=torch.float64, lbs=None, ubs=None, n_layers=torch.inf):
        """
        Propagates a zonotope through the network of the propagator given box bounds on the input set.

        args:
            in_lbs - concrete lower bounds on the input variables
            in_ubs - concrete upper bounds on the input variables
        
        kwargs:
            dtype - datatype of the pytorch tensors that make up the zonotope
            lbs - concrete lower bounds on the intermediate activation inputs
            ubs - concrete upper bounds on the intermediate activation inputs
            n_layers - number of layers to propagate (note that each layer in nn.children counts as a layer!)
        """
        z = Zonotope.from_bounds(in_lbs, in_ubs, shape=(len(in_lbs),), dtype=dtype)

        cnt = 0
        for i, layer in enumerate(self.nn.children()):
            if i > n_layers:
                break

            if isinstance(layer, nn.Linear):
                print("propagating linear layer")
                W, b = list(layer.parameters())
                z = self.propagate_linear(W, b, z)
            elif isinstance(layer, nn.ReLU):
                print("propagating ReLU layer")
                if not lbs is None:
                    z = self.propagate_relu(z, lb=lbs[cnt], ub=ubs[cnt])
                else:
                    z = self.propagate_relu(z)

                cnt += 1
            else:
                raise ValueError("Unknown type of layer: ", layer)

        return z