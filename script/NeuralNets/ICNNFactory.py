from script.NeuralNets.Networks import ICNN, ICNNLogical, ICNNApproxMax


class ICNNFactory():
    def __init__(self, icnn_type, layer_width, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.icnn_type = icnn_type
        self.layer_width = layer_width
        valid_icnn_types = ["standard", "logical", "approx_max"]
        if icnn_type not in valid_icnn_types:
            raise AttributeError("Expected plotting mode one of: {}, actual: {}".format(valid_icnn_types, icnn_type))

    def get_new_icnn(self, in_layer_size_as_group_size):
        new_layer_width = self.layer_width.copy()
        new_layer_width.insert(0, in_layer_size_as_group_size)
        if self.icnn_type == "standard":
            return ICNN(new_layer_width, *self.args, **self.kwargs)
        elif self.icnn_type == "logical":
            return ICNNLogical(new_layer_width, *self.args, **self.kwargs)
        elif self.icnn_type == "approx_max":
            return ICNNApproxMax(new_layer_width, *self.args, **self.kwargs)
