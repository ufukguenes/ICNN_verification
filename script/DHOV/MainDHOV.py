
import numpy as np
from matplotlib import pyplot as plt
from torchvision.datasets import CIFAR10, MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
import torch

from script.NeuralNets.Networks import SequentialNN
from script.settings import device, data_type
import MultiDHOV as multidhov
from script.NeuralNets.ICNNFactory import ICNNFactory


def multi_net2D():
    """W1 = [1. 1.; 1. -1.]
    b1 = [0., 0.]
    W2 = [1. 1.; 1. -1.]
    b2 = [-0.5, 0.]
    W3 = [-1. 1.; 1. 1.]
    b3 = [3., 0.] """

    # 2D-NN
    """nn = SequentialNN([2, 2, 2, 2])

    with torch.no_grad():
        parameter_list = list(nn.parameters())
        parameter_list[0].data = torch.tensor([[1, 1], [1, -1]], dtype=data_type).to(device)
        parameter_list[1].data = torch.tensor([0, 0], dtype=data_type).to(device)
        parameter_list[2].data = torch.tensor([[1, 1], [1, -1]], dtype=data_type).to(device)
        parameter_list[3].data = torch.tensor([-0.5, 0], dtype=data_type).to(device)
        parameter_list[4].data = torch.tensor([[-1, 1], [1, 1]], dtype=data_type).to(device)
        parameter_list[5].data = torch.tensor([3, 0], dtype=data_type).to(device)

    test_image = torch.tensor([[0, 0]], dtype=data_type).to(device)"""

    # random-NN
    """nn = SequentialNN([500, 500, 50, 7])
    test_image = torch.zeros((1, 500), dtype=data_type).to(device)"""

    # CIFAR-NN
    """transform = Compose([ToTensor(),
                         Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
                        )

    training_data = CIFAR10(root="../../cifar", train=True, download=True, transform=transform)
    images, labels = training_data.__getitem__(0)
    test_image, test_label = torch.unsqueeze(images, 0).to(dtype=data_type).to(device), torch.unsqueeze(
        torch.tensor(labels), 0).to(dtype=data_type).to(device)

    nn = SequentialNN([32 * 32 * 3, 1024, 512, 10])
    nn.load_state_dict(torch.load("../../cifar_fc.pth", map_location=torch.device('cpu')), strict=False)"""

    # MNIST-NN
    transform = Compose([ToTensor(),
                         Normalize(0.5, 0.5)]
                        )

    training_data = MNIST(root="../../mnist",
                          train=True,
                          download=True,
                          transform=transform)
    images, labels = training_data.__getitem__(0)
    test_image, test_label = torch.unsqueeze(images, 0).to(dtype=data_type).to(device), torch.unsqueeze(
        torch.tensor(labels), 0).to(dtype=data_type).to(device)

    nn = SequentialNN([28*28*1, 100, 30, 10])
    nn.load_state_dict(torch.load("../../mnist_fc.pth", map_location=torch.device('cpu')), strict=False)
    pred = nn(test_image)

    # start of DHOV

    eps = 0.02
    #matplotlib.use('TkAgg')

    group_size = 2
    icnn_factory = ICNNFactory("logical", [10, 1], always_use_logical_layer=False)
    #icnn_factory = ICNNFactory("standard", [10, 1])
    # icnn_factory = ICNNFactory("approx_max", [5, 1], maximum_function="SMU", function_parameter=0.3)

    dhov_verifier = multidhov.MultiDHOV()
    dhov_verifier.start_verification(nn, test_image, icnn_factory, group_size, eps=eps, icnn_epochs=10,
                                     icnn_batch_size=1000, sample_count=1000, sample_new=True,
                                     use_over_approximation=True, break_after=None,
                                     sample_over_input_space=False, sample_over_output_space=True,
                                     use_icnn_bounds=True,
                                     use_fixed_neurons=True, sampling_method="per_group_sampling",
                                     force_inclusion_steps=0, preemptive_stop=False, even_gradient_training=False,
                                     keep_ambient_space=True, data_grad_descent_steps=0, opt_steps_gd=200,
                                     train_outer=False, print_training_loss=False, print_new_bounds=False,
                                     grouping_method="consecutive", group_num_multiplier=5, store_samples=False,
                                     print_optimization_steps=False,
                                     should_plot="detailed", optimizer="SdLBFGS", init_network=True,
                                     adapt_lambda="included")
    print(dhov_verifier.all_group_indices)


def to_A_b(model):
    # print(model.display())
    matrix_a = model.getA().toarray()
    constr = model.getConstrs()
    con_by_name = model.getConstrByName("lb_const0[0]")

    lhs = []
    rhs = []
    # print("==============================")
    for i, elem in enumerate(constr):
        # print(elem.ConstrName)
        if elem.Sense == "<":
            lhs.append(matrix_a[i])
            rhs.append(elem.RHS)
        elif elem.Sense == ">":
            lhs.append(-1 * matrix_a[i])
            rhs.append(-1 * elem.RHS)
        elif elem.Sense == "=":
            lhs.append(matrix_a[i])
            rhs.append(elem.RHS)
            lhs.append(-1 * matrix_a[i])
            rhs.append(-1 * elem.RHS)

    lhs = np.array(lhs)
    rhs = np.array(rhs)
    lhs[-0 == lhs] = 0
    rhs[-0 == rhs] = 0
    return lhs, rhs


def plt_inc_amb(caption, points, is_true):
    true_points = []
    false_points = []
    for i in range(len(points)):
        if is_true[i]:
            true_points.append(points[i])
        else:
            false_points.append(points[i])

    plt.figure(figsize=(20, 10))
    plt.scatter(list(map(lambda x: x[0], false_points)), list(map(lambda x: x[1], false_points)), c="#ff7f0e")
    plt.scatter(list(map(lambda x: x[0], true_points)), list(map(lambda x: x[1], true_points)), c="#1f77b4")
    plt.title(caption)
    plt.show()


multi_net2D()
