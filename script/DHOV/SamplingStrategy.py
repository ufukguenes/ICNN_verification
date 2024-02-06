from abc import ABC, abstractmethod

import torch

import script.DHOV.DataSampling as ds
from script.settings import data_type, device


# todo make each strategy return an iterator by group for before and after sampling
class SamplingStrategy(ABC):
	def __init__(self, center, eps, keep_ambient_space=False, sample_count=10):
		self.needs_initialization = False
		self.current_sample_space = None #todo soll ich das behalten?, wenn ja, dann muss ich das updaten
		self.keep_ambient_space = keep_ambient_space  # todo implement this for each strategy
		self.sample_over_input_space = True
		self.sample_over_output_space = True
		self.sample_count = sample_count
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
	def input_space_sampling_by_round(self, affine_w, affine_b, group_indices, gurobi_model,
																		 current_layer_index, bounds_affine_out, bounds_layer_out):
		"""
		Use this method to sample over the input space of a layer (before propagation).
		Args:
			affine_w: the weight matrix of the current layer
			affine_b: the bias vector of the current layer
			group_indices: the index of the neuron to be sampled, for all groups
			gurobi_model: the gurobi model to be used for sampling
			current_layer_index: the index of the current layer
			bounds_affine_out: the bounds of the affine output space of the all layers
			bounds_layer_out: the bounds of the activation output space of the all layers

		Returns: list of tuple of  included_space, ambient_space

		"""
		pass

	@abstractmethod
	def output_space_sampling_by_round(self, included_space, ambient_space, affine_w, affine_b, group_indices, gurobi_model,
																		 current_layer_index, bounds_affine_out, bounds_layer_out): #todo docstring
		"""
		Use this method to sample over the output space of a layer (after propagation).
		Args:
			affine_w: the weight matrix of the current layer
			affine_b: the bias vector of the current layer
			group_indices: the index of the neuron to be sampled, for all groups
			gurobi_model: the gurobi model to be used for sampling
			current_layer_index: the index of the current layer
			bounds_affine_out: the bounds of the affine output space of the all layers
			bounds_layer_out: the bounds of the activation output space of the all layers

		Returns: list of tuple of  included_space, ambient_space

		"""
		pass

	@abstractmethod
	def propagate_space(self, list_included_spaces, list_ambient_spaces, affine_w, affine_b): #todo docstring
		"""
		use this method if the strategy depends on affine transformations of the data points
		Args:
			affine_w: the weight matrix of the current layer
			affine_b: the bias vector of the current layer

		Returns: list of tuple of  included_space, ambient_space

		"""
		for i in range(len(list_included_spaces)):
			list_included_spaces[i] = ds.apply_affine_transform(affine_w, affine_b, list_included_spaces[i])
			list_ambient_spaces[i] = ds.apply_affine_transform(affine_w, affine_b, list_ambient_spaces[i])

			list_included_spaces[i] = ds.apply_relu_transform(list_included_spaces[i])
			list_ambient_spaces[i] = ds.apply_relu_transform(list_ambient_spaces[i])

		return list_included_spaces, list_ambient_spaces


class PerGroupSamplingStrategy(SamplingStrategy):
	def input_space_sampling_by_round(self, affine_w, affine_b, group_indices, gurobi_model,
																		 current_layer_index, bounds_affine_out, bounds_layer_out):
		all_included_spaces = []
		all_ambient_spaces = []
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

			all_included_spaces.append(sample_space)
			all_ambient_spaces.append(ambient_space)
		return all_included_spaces, all_ambient_spaces

	def output_space_sampling_by_round(self, list_included_space, list_ambient_space, affine_w, affine_b, group_indices, gurobi_model,
																		 current_layer_index, bounds_affine_out, bounds_layer_out):
		included_sample_count, ambient_sample_count = self.get_num_of_samples()
		if self.sample_over_output_space:
			for i in range(len(list_ambient_space)):
				list_ambient_space[i] = ds.samples_uniform_over(list_ambient_space[i], ambient_sample_count,
																												bounds_layer_out[current_layer_index],
																												padding=self.eps)

		return list_included_space, list_ambient_space #todo concat ambient space instead of overwriting it


class PerGroupFeasibleSamplingStrategy(SamplingStrategy):

	def input_space_sampling_by_round(self, included_space, ambient_space, affine_w, affine_b, group_indices, gurobi_model,
																		 current_layer_index, bounds_affine_out, bounds_layer_out):
		return included_space, ambient_space

	def output_space_sampling_by_round(self, included_space, ambient_space, affine_w, affine_b, group_indices, gurobi_model,
																		 current_layer_index, bounds_affine_out, bounds_layer_out):
		all_included_spaces = []
		all_ambient_spaces = []
		for index_to_select in group_indices:
			included_space = torch.empty((0, affine_w.size(1)), dtype=data_type).to(device)
			ambient_space = torch.empty((0, affine_w.size(1)), dtype=data_type).to(device)
			included_sample_count, ambient_sample_count = self.get_num_of_samples()
			prev_layer_index = current_layer_index - 1

			included_space, ambient_space = ds.sample_feasible(included_space, included_sample_count,
																																						affine_w, affine_b, index_to_select,
																																						gurobi_model,
																																						bounds_affine_out[current_layer_index],
																																						bounds_layer_out[current_layer_index],
																																						prev_layer_index)
		all_included_spaces.append(included_space)
		all_ambient_spaces.append(ambient_space)
		# todo check if amount of samples is correct
		return all_included_spaces, all_included_spaces


#todo hier muss ich für jede strat anpassen das der output in einer liste ist (da es genau eine gruppe gibt)
#todo sampling auf 0tes und darauf folgende layer anpassen
class PropagateSamplingStrategy(SamplingStrategy, ABC):
	"""
	# todo use this code in initalization and input_space_sampling_by_round. Attention: dont sample twice for the first layer
	                if sample_over_input_space:  # todo put this in sampling strategy (not used in per group strat)
                    if i == 0:
                        ambient_space = ds.sample_uniform_excluding(ambient_space, amb_space_sample_count, eps_bounds,
                                                                    excluding_bound=eps_bounds, padding=eps)
                    else:
                        ambient_space = ds.sample_uniform_excluding(ambient_space, amb_space_sample_count,
                                                                    bounds_layer_out[current_layer_index - 1],
                                                                    icnns=list_of_icnns[current_layer_index - 1],
                                                                    layer_index=current_layer_index,
                                                                    group_size=group_size,
                                                                    padding=eps)
	"""

	def propagate_space(self, affine_w, affine_b):
		if self.current_sample_space is None:
			raise RuntimeError("Initialization has not been done")

		self.current_sample_space = ds.apply_affine_transform(affine_w, affine_b, self.current_sample_space)
		self.current_sample_space = ds.apply_relu_transform(self.current_sample_space)
		return self.current_sample_space


class UniformSamplingStrategy(PropagateSamplingStrategy):
	def input_space_sampling_by_round(self, sample_count, center, eps):
		sample_space = torch.empty((0, center.size(0)), dtype=data_type).to(device)
		eps_bounds = [center.add(-eps), center.add(eps)]
		sample_space = ds.samples_uniform_over(sample_space, sample_count, eps_bounds)
		self.current_sample_space = sample_space
		return sample_space


class LinespaceSamplingStrategy(PropagateSamplingStrategy):
	def input_space_sampling_by_round(self, sample_count, center, eps):
		sample_space = torch.empty((0, center.size(0)), dtype=data_type).to(device)
		sample_space = ds.sample_linspace(sample_space, sample_count, center, eps)
		self.current_sample_space = sample_space
		return sample_space


class BoarderSamplingStrategy(PropagateSamplingStrategy):
	def input_space_sampling_by_round(self, sample_count, center, eps):
		sample_space = torch.empty((0, center.size(0)), dtype=data_type).to(device)
		sample_space = ds.sample_boarder(sample_space, sample_count, center, eps)
		self.current_sample_space = sample_space
		return sample_space


class SumNoiseSamplingStrategy(PropagateSamplingStrategy):
	def input_space_sampling_by_round(self, sample_count, center, eps):
		sample_space = torch.empty((0, center.size(0)), dtype=data_type).to(device)
		sample_space = ds.sample_random_sum_noise(sample_space, sample_count, center, eps)
		self.current_sample_space = sample_space
		return sample_space


class MinMaxPerturbationSamplingStrategy(PropagateSamplingStrategy):
	def input_space_sampling_by_round(self, sample_count, center, eps, affine_w):
		sample_space = torch.empty((0, center.size(0)), dtype=data_type).to(device)
		sample_space = ds.sample_min_max_perturbation(sample_space, sample_count, affine_w, center, eps)
		self.current_sample_space = sample_space
		return sample_space


class AlternateMinMaxSamplingStrategy(PropagateSamplingStrategy):
	def input_space_sampling_by_round(self, sample_count, center, eps, affine_w):
		sample_space = torch.empty((0, center.size(0)), dtype=data_type).to(device)
		sample_space = ds.sample_alternate_min_max(sample_space, sample_count, affine_w, center, eps)
		self.current_sample_space = sample_space
		return sample_space
