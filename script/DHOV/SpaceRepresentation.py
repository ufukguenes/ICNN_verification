from enum import Enum


class SpaceRepresentation:
	def __init__(self, sampling_method, center, eps, number_of_total_samples, store_all_values=False):
		self.sampling_method = self.check_if_sampling_method_exists(sampling_method)
		self.store_all_values = store_all_values
		self.all_included_space = []
		self.all_ambient_space = []
		self.included_space = []
		self.ambient_space = []
		self.number_of_total_samples = number_of_total_samples
		self.center = center
		self.eps = eps

	class ValidSamplingMethods(Enum):
		UNIFORM = "uniform"
		LINESPACE = "linespace"
		BOARDER = "boarder"
		SUM_NOISE = "sum_noise"
		MIN_MAX_PERTURBATION = "min_max_perturbation"
		ALTERNATE_MIN_MAX = "alternate_min_max"
		PER_GROUP_SAMPLING = "per_group_sampling"
		PER_GROUP_FEASIBLE = "per_group_feasible"

	def check_if_sampling_method_exists(self, sampling_method: str):
		for elem in self.ValidSamplingMethods:
			if elem.value == sampling_method:
				return elem
		raise AttributeError(
				"Expected sampling method to be one of: {} , got: {}".format(
						self.ValidSamplingMethods.value, sampling_method))
