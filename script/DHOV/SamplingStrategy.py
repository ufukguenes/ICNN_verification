from abc import ABC, abstractmethod

import torch

import DataSampling as ds
from script.settings import data_type, device


class SamplingStrategy(ABC):
	def __init__(self):
		self.needs_initialization = False
		self.current_sample_space = None

	@abstractmethod
	def initialization(self, sample_count, center, eps):
		pass

	@abstractmethod
	def get_next_sampling_space(self):
		pass


class PerGroupSamplingStrategy(SamplingStrategy):
	def initialization(self, sample_count, center, eps):
		pass

	def get_next_sampling_space(self, sample_count, affine_w, center, eps, index_to_select, rand_samples_percent=0.2,
															rand_sample_alternation_percent=0.01):
		sample_space = torch.empty((0, affine_w.size(1)), dtype=data_type).to(device)
		self.current_sample_space = ds.sample_per_group(sample_space, sample_count, affine_w, center, eps, index_to_select,
																										rand_samples_percent=rand_samples_percent,
																										rand_sample_alternation_percent=rand_sample_alternation_percent)
		return self.current_sample_space


class LPSolverPerGroupSamplingStrategy(SamplingStrategy):
	def initialization(self, sample_count, center, eps):
		pass

	def get_next_sampling_space(self, sample_count, affine_w, affine_b, current_bounds_affine_out, prev_layer_index,
															gurobi_model, index_to_select, rand_samples_percent=0.2,
															rand_sample_alternation_percent=0.01):
		sample_space = torch.empty((0, affine_w.size(1)), dtype=data_type).to(device)
		self.current_sample_space = ds.sample_per_group_as_lp(sample_space, sample_count,
																													affine_w, affine_b,
																													index_to_select, gurobi_model,
																													current_bounds_affine_out,
																													prev_layer_index,
																													rand_samples_percent=0.2,
																													rand_sample_alternation_percent=0.2)

		return self.current_sample_space


class PerGroupFeasibleSamplingStrategy(SamplingStrategy):
	def initialization(self, sample_count, center, eps):
		pass

	def get_next_sampling_space(self):
		sample_space = torch.empty((0, affine_w.size(1)), dtype=data_type).to(device)
		return


class PropagateSamplingStrategy(SamplingStrategy, ABC):

	def get_next_sampling_space(self, affine_w, affine_b):
		if self.current_sample_space is None:
			raise RuntimeError("Initialization has not been done")

		self.current_sample_space = ds.apply_affine_transform(affine_w, affine_b, self.current_sample_space)
		self.current_sample_space = ds.apply_relu_transform(self.current_sample_space)
		return self.current_sample_space


class UniformSamplingStrategy(PropagateSamplingStrategy):
	def initialization(self, sample_count, center, eps):
		sample_space = torch.empty((0, center.size(0)), dtype=data_type).to(device)
		eps_bounds = [center.add(-eps), center.add(eps)]
		sample_space = ds.samples_uniform_over(sample_space, sample_count, eps_bounds)
		self.current_sample_space = sample_space
		return sample_space


class LinespaceSamplingStrategy(PropagateSamplingStrategy):
	def initialization(self, sample_count, center, eps):
		sample_space = torch.empty((0, center.size(0)), dtype=data_type).to(device)
		sample_space = ds.sample_linspace(sample_space, sample_count, center, eps)
		self.current_sample_space = sample_space
		return sample_space


class BoarderSamplingStrategy(PropagateSamplingStrategy):
	def initialization(self, sample_count, center, eps):
		sample_space = torch.empty((0, center.size(0)), dtype=data_type).to(device)
		sample_space = ds.sample_boarder(sample_space, sample_count, center, eps)
		self.current_sample_space = sample_space
		return sample_space


class SumNoiseSamplingStrategy(PropagateSamplingStrategy):
	def initialization(self, sample_count, center, eps):
		sample_space = torch.empty((0, center.size(0)), dtype=data_type).to(device)
		sample_space = ds.sample_random_sum_noise(sample_space, sample_count, center, eps)
		self.current_sample_space = sample_space
		return sample_space


class MinMaxPerturbationSamplingStrategy(PropagateSamplingStrategy):
	def initialization(self, sample_count, center, eps, affine_w):
		sample_space = torch.empty((0, center.size(0)), dtype=data_type).to(device)
		sample_space = ds.sample_min_max_perturbation(sample_space, sample_count, affine_w, center, eps)
		self.current_sample_space = sample_space
		return sample_space


class AlternateMinMaxSamplingStrategy(PropagateSamplingStrategy):
	def initialization(self, sample_count, center, eps, affine_w):
		sample_space = torch.empty((0, center.size(0)), dtype=data_type).to(device)
		sample_space = ds.sample_alternate_min_max(sample_space, sample_count, affine_w, center, eps)
		self.current_sample_space = sample_space
		return sample_space
