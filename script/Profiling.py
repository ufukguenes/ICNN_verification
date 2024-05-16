
class Timings:
    def __init__(self):
        self.icnn_verification_time_per_layer_per_icnn = []
        self.icnn_verification_time_per_layer_total = []

        self.icnn_training_time_per_layer_per_icnn = []
        self.icnn_training_time_per_layer_total = []

        self.neuron_refinement_time_per_layer = []
        self.data_sampling_time_per_layer = []
        self.total_time_per_layer = []

    def get_all_results(self, do_round=False):
        return [
            [x if not do_round else round(x, 2) for x in self.total_time_per_layer],
            [x if not do_round else round(x, 2) for x in self.neuron_refinement_time_per_layer],
            [x if not do_round else round(x, 2) for x in self.data_sampling_time_per_layer],
            [x if not do_round else round(x, 2) for x in self.icnn_training_time_per_layer_total],
            [x if not do_round else [round(y, 2) for y in x] for x in self.icnn_training_time_per_layer_per_icnn],
            [x if not do_round else round(x, 2) for x in self.icnn_verification_time_per_layer_total],
            [x if not do_round else [round(y, 2) for y in x] for x in self.icnn_verification_time_per_layer_per_icnn]

        ]


    def get_ordering_as_list_of_strings(self):
        return [
            "total_time_per_layer",
            "neuron_refinement_time_per_layer",
            "data_sampling_time_per_layer",
            "icnn_training_time_per_layer_total",
            "icnn_training_time_per_layer_per_icnn",
            "icnn_verification_time_per_layer_total",
            "icnn_verification_time_per_layer_per_icnn"
        ]