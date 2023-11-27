from script.NeuralNets.Networks import ICNN, ICNNLogical, ICNNApproxMax


class ICNNFactory():
    """
    This is Class follows the Factory-Design-Pattern.
    It is used to generate ICNNs which are identical in their arguments except for the input size in to the ICNN,
    which has to be defined for each ICNN individually
    """
    def __init__(self, icnn_type, layer_width, adapt_layer_for_init=True, *args, **kwargs):
        """

        :param icnn_type: one of ["standard", "logical", "approx_max"], these are different ICNN architectures,
            which differ in the way they approximate the maximum function before the output
        :param layer_width: list of layer sizes except for the input size, which has to be determined for each ICNN
            individually
        :param adapt_layer_for_init: Boolean, deciding whether a new layer should be included which has the
            correct size to make an initialization of the ICNN based on known bounds
        :param args: further arguments for the init function of the specific ICNN architecture
        :param kwargs: further keyword arguments for the init function of the specific ICNN architecture
        """
        self.args = args
        self.kwargs = kwargs
        self.icnn_type = icnn_type
        self.layer_width = layer_width
        self.adapt_layer_for_init = adapt_layer_for_init
        valid_icnn_types = ["standard", "logical", "approx_max"]
        if icnn_type not in valid_icnn_types:
            raise AttributeError("Expected plotting mode one of: {}, actual: {}".format(valid_icnn_types, icnn_type))

    def get_new_icnn(self, in_layer_size_as_group_size):
        """
        Generator function to get a new ICNN
        :param in_layer_size_as_group_size: Input dimension of the ICNN
        :return: an ICNN
        """
        new_layer_width = self.layer_width.copy()
        new_layer_width.insert(0, in_layer_size_as_group_size)
        if self.icnn_type == "standard":
            if self.adapt_layer_for_init:
                new_layer_width.insert(len(new_layer_width) - 1, 2 * in_layer_size_as_group_size)
            return ICNN(new_layer_width, *self.args, **self.kwargs)
        elif self.icnn_type == "logical":
            return ICNNLogical(new_layer_width, *self.args, **self.kwargs)
        elif self.icnn_type == "approx_max":
            return ICNNApproxMax(new_layer_width, *self.args, **self.kwargs)
