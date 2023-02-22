import numpy as np
import scipy.linalg
import torch
from gurobipy import Model
from matplotlib import pyplot as plt
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize
import gurobipy as grp
import torch

import DataSampling as ds
import DHOV as dhov
from torch.utils.data import DataLoader
from script.NeuralNets.Networks import SequentialNN, ICNN, ICNNApproxMax, ICNNLogical
from script.Verification.Verifier import SingleNeuronVerifier, MILPVerifier, DHOVVerifier
from script.dataInit import Rhombus, ConvexDataset
import polytope as pc
from script.settings import device, data_type
import MultiDHOV as multidhov


def last_layer_identity(last_icnn: ICNN, last_c, W, b, A_out, b_out, nn_bounds, solver_time_limit, solver_bound):
    m = Model()
    m.Params.LogToConsole = 0
    if solver_time_limit is not None:
        m.setParam("TimeLimit", solver_time_limit)

    if solver_bound is not None:
        m.setParam("BestObjStop", solver_bound)

    output_var = m.addMVar(2, lb=-float('inf'))

    icnn_input_size = last_icnn.layer_widths[0]
    output_of_penultimate_layer = m.addMVar(icnn_input_size, lb=-float('inf'))
    output_of_icnn = m.addMVar(1, lb=-float('inf'))
    bounds = verbas.calculate_box_bounds(last_icnn, nn_bounds[len(nn_bounds) - 1], is_sequential=False)
    verbas.add_constr_for_non_sequential_icnn(m, last_icnn, output_of_penultimate_layer, output_of_icnn, bounds)
    m.addConstr(output_of_icnn[0] <= last_c)

    W = W.detach().cpu().numpy()
    b = b.detach().cpu().numpy()
    # output_var = add_affine_constr(m, W, b, output_of_penultimate_layer, -1000, 1000)

    # m.addConstrs((W[i] @ output_var >= b[i] for i in range(len(W))))

    m.addConstrs(W[i] @ output_of_penultimate_layer + b[i] == output_var[i] for i in range(len(W)))

    # constr = m.addMConstr(A_out, output_var.tolist(), ">", b_out)

    max_var = m.addMVar(2, lb=-float('inf'), ub=1000)

    m.addConstr(max_var[0] == abs_(output_var[0]))  # maximize distance to origin without norm
    m.addConstr(max_var[1] == abs_(
        output_var[1]))  # todo diese nachbedinung garantiert nicht, dass etwas außerhalb maximal ist!

    m.update()
    m.setObjective(max_var[0] + max_var[1], GRB.MAXIMIZE)
    m.optimize()

    solution = 0
    if m.Status == GRB.OPTIMAL or m.Status == GRB.TIME_LIMIT:
        # print("optimum solution with value \n {}".format(output_var.getAttr("x")))
        print("icnn_in_var {}".format(output_of_penultimate_layer.getAttr("x")))
        print("max_var {}".format(output_of_icnn.getAttr("x")))
        output = output_var.getAttr("x")
        print("output {}".format(output_var.getAttr("x")))
        sol = output_of_penultimate_layer.getAttr("x")
        sol = torch.tensor(sol, dtype=data_type).to(device)
        sol = torch.unsqueeze(sol, 0)
        out = last_icnn(sol)
        print(out)

        print("======================= \n")
        if Rhombus().f(output):
            print("Verification was successful")
        else:
            print("Verification failed")
        print("\n =======================")


def last_layer_picture(last_icnn: ICNN, last_c, W, b, label, nn_bounds, solver_time_limit, solver_bound):
    m = Model()

    if solver_time_limit is not None:
        m.setParam("TimeLimit", solver_time_limit)

    if solver_bound is not None:
        m.setParam("BestObjStop", solver_bound)

    icnn_input_size = last_icnn.layer_widths[0]
    output_of_penultimate_layer = m.addMVar(icnn_input_size, lb=-float('inf'))
    output_of_icnn = m.addMVar(1, lb=-float('inf'))
    verbas.add_constr_for_non_sequential_icnn(m, last_icnn, output_of_penultimate_layer, output_of_icnn,
                                              nn_bounds[len(nn_bounds) - 2])

    W = W.detach().cpu().numpy()
    b = b.detach().cpu().numpy()
    output_var = verbas.add_affine_constr(m, W, b, output_of_penultimate_layer, -1000, 1000)
    m.addConstr(output_of_icnn <= last_c)

    difference = m.addVars(9, lb=-float('inf'))
    m.addConstrs(difference[i] == output_var.tolist()[i] - output_var.tolist()[label] for i in range(0, label))
    m.addConstrs(difference[i - 1] == output_var.tolist()[i] - output_var.tolist()[label] for i in range(label + 1, 10))

    # m.addConstrs(difference[i] == output_var.tolist()[i] - output_var.tolist()[label] for i in range(10))
    max_var = m.addVar(lb=-float('inf'), ub=1000)
    m.addConstr(max_var == max_(difference))

    m.update()
    m.setObjective(max_var, GRB.MAXIMIZE)
    m.optimize()

    solution = 0
    if m.Status == GRB.OPTIMAL or m.Status == GRB.TIME_LIMIT or m.Status == GRB.USER_OBJ_LIMIT:
        for o in difference.select():
            print(o.getAttr("x"))
        print("optimum solution with value \n {}".format(output_var.getAttr("x")))
        print("max_var {}".format(max_var.getAttr("x")))
        sol = output_of_penultimate_layer.getAttr("x")

        sol = torch.tensor(sol, dtype=data_type).to(device)
        sol = torch.unsqueeze(sol, 0)
        out = last_icnn(sol)
        print(out)


def net_2d():
    batch_size = 10
    epochs = 30
    number_of_train_samples = 10000
    hyper_lambda = 1
    x_range = [-1.5, 1.5]
    y_range = [-1.5, 1.5]

    included_space, ambient_space = Rhombus().get_uniform_samples(number_of_train_samples, x_range,
                                                                  y_range)  # samples will be split in inside and outside the rhombus

    dataset_in = ConvexDataset(data=included_space)
    train_loader = DataLoader(dataset_in, batch_size=batch_size, shuffle=True)
    dataset = ConvexDataset(data=ambient_space)
    ambient_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    """W1 = [1. 1.; 1. -1.]
    b1 = [0., 0.]
    W2 = [1. 1.; 1. -1.]
    b2 = [-0.5, 0.]
    W3 = [-1. 1.; 1. 1.]
    b3 = [3., 0.] """

    nn = SequentialNN([2, 2, 2, 2])

    with torch.no_grad():
        parameter_list = list(nn.parameters())
        parameter_list[0].data = torch.tensor([[1, 1], [1, -1]], dtype=data_type).to(device)
        parameter_list[1].data = torch.tensor([0, 0], dtype=data_type).to(device)
        parameter_list[2].data = torch.tensor([[1, 1], [1, -1]], dtype=data_type).to(device)
        parameter_list[3].data = torch.tensor([-0.5, 0], dtype=data_type).to(device)
        parameter_list[4].data = torch.tensor([[-1, 1], [1, 1]], dtype=data_type).to(device)
        parameter_list[5].data = torch.tensor([3, 0], dtype=data_type).to(device)

    # nn.load_state_dict(torch.load("nn_2x2.pt"), strict=False)
    # train_sequential_2(nn, train_loader, ambient_loader, epochs=epochs)

    # matplotlib.use('TkAgg')

    # torch.save(nn.state_dict(), "nn_2x2.pt")

    test_image = torch.tensor([[0, 0]], dtype=data_type).to(device)

    icnns = []
    for i in range((len(parameter_list) - 2) // 2):
        layer_index = int(i / 2)
        icnn_input_size = nn.layer_widths[layer_index + 1]
        # next_net = ICNN([icnn_input_size, 10, 10, 10, 2 * icnn_input_size, 1], force_positive_init=False, init_scaling=10, init_all_with_zeros=False)
        next_net = ICNNLogical([icnn_input_size, 10, 10, 10, 1], force_positive_init=False, with_two_layers=False,
                               init_scaling=10, init_all_with_zeros=False)
        # next_net = ICNNApproxMax([icnn_input_size, 10, 10, 10, 1], maximum_function="SMU", function_parameter=0.1, force_positive_init=False, init_scaling=10, init_all_with_zeros=False)

        icnns.append(next_net)

    icnns = \
        dhov.start_verification(nn, test_image, icnns, eps=1, icnn_epochs=100, icnn_batch_size=1000, sample_count=1000,
                                sample_new=True, use_over_approximation=True,
                                sample_over_input_space=False, sample_over_output_space=True, force_inclusion_steps=0,
                                keep_ambient_space=False, data_grad_descent_steps=0, train_outer=False,
                                preemptive_stop=False,
                                even_gradient_training=False,
                                should_plot="none", optimizer="adam", init_network=True, adapt_lambda="none")

    milp_verifier = MILPVerifier(nn, test_image, 1)
    snv_verifier = SingleNeuronVerifier(nn, test_image, 1)
    dhov_verifier = DHOVVerifier(icnns, nn, test_image, 1)

    milp_verifier.generate_constraints_for_net()
    snv_verifier.generate_constraints_for_net()
    dhov_verifier.generate_constraints_for_net()

    input_flattened = torch.flatten(test_image)
    input_size = input_flattened.size(0)
    bounds_affine_out, bounds_layer_out = nn.calculate_box_bounds([input_flattened.add(-1), input_flattened.add(1)])

    test_space = torch.empty((0, input_flattened.size(0)), dtype=data_type).to(device)
    box_bound_output_space = ds.samples_uniform_over(test_space, 1000, bounds_layer_out[-1])

    in_snv = []
    in_milp = []
    in_dhov = []
    in_dhov_affine = []
    for i, sample in enumerate(box_bound_output_space):
        if i % 10 == 0:
            print(i)
        in_milp.append(milp_verifier.test_feasibility(sample))
        in_snv.append(snv_verifier.test_feasibility(sample))
        in_dhov.append(dhov_verifier.test_feasibility(sample))

    plt_inc_amb("milp", box_bound_output_space.detach().cpu().numpy(), in_milp)
    plt_inc_amb("snv", box_bound_output_space.detach().cpu().numpy(), in_snv)
    plt_inc_amb("dhov", box_bound_output_space.detach().cpu().numpy(), in_dhov)
    plt_inc_amb("dhov with affine", box_bound_output_space.detach().cpu().numpy(), in_dhov_affine)

    """A_snv, b_snv = to_A_b(snv_model)
    A_icnn, b_icnn = to_A_b(icnn_model)

    snv_polytope = pc.Polytope(A_snv, b_snv)
    icnn_polytope = pc.Polytope(A_icnn, b_icnn)

    snv_convex = pc.is_convex(pc.Region([snv_polytope]))
    icnn_convex = pc.is_convex(pc.Region([icnn_polytope]))

    snv_volume = pc.volume(pc.Region([snv_polytope]))
    icnn_volume = pc.volume(pc.Region([icnn_polytope]))
    snv_extreme = pc.extreme(snv_polytope)
    icnn_extreme = pc.extreme(icnn_polytope)

    print(snv_polytope)
    print(icnn_volume)"""


def cifar_net():
    batch_size = 10
    epochs = 30
    number_of_train_samples = 10000
    hyper_lambda = 1
    x_range = [-1.5, 1.5]
    y_range = [-1.5, 1.5]

    transform = Compose([ToTensor(),
                         Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
                        )

    training_data = CIFAR10(root="../../cifar",
                            train=True,
                            download=True,
                            transform=transform)
    images, labels = training_data.__getitem__(0)
    testimage, testlabel = torch.unsqueeze(images, 0).to(dtype=data_type).to(device), torch.unsqueeze(
        torch.tensor(labels), 0).to(dtype=data_type).to(device)

    nn = SequentialNN([32 * 32 * 3, 1024, 512, 10])
    nn.load_state_dict(torch.load("../../cifar_fc.pth", map_location=torch.device('cpu')), strict=False)

    # matplotlib.use('TkAgg')

    icnns, c_values = \
        dhov.start_verification(nn, testimage, eps=1, icnn_epochs=100, sample_new=True, use_over_approximation=True,
                                sample_over_input_space=False, sample_over_output_space=True,
                                keep_ambient_space=False, data_grad_descent_steps=False, train_outer=False,
                                should_plot="detailed")


def multi_net2D():
    """W1 = [1. 1.; 1. -1.]
    b1 = [0., 0.]
    W2 = [1. 1.; 1. -1.]
    b2 = [-0.5, 0.]
    W3 = [-1. 1.; 1. 1.]
    b3 = [3., 0.] """

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

    transform = Compose([ToTensor(),
                         Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
                        )

    training_data = CIFAR10(root="../../cifar", train=True, download=True, transform=transform)
    images, labels = training_data.__getitem__(0)
    test_image, test_label = torch.unsqueeze(images, 0).to(dtype=data_type).to(device), torch.unsqueeze(
        torch.tensor(labels), 0).to(dtype=data_type).to(device)

    #nn = SequentialNN([32 * 32 * 3, 1024, 512, 10])
    #nn.load_state_dict(torch.load("../../cifar_fc.pth", map_location=torch.device('cpu')), strict=False)
    nn = SequentialNN([300, 100, 50, 7])
    test_image = torch.zeros((1, 300), dtype=data_type).to(device)
    parameter_list = list(nn.parameters())

    group_size = 6
    icnns = []
    for i in range((len(parameter_list) - 2) // 2):
        layer_index = i
        in_size = nn.layer_widths[layer_index + 1]
        icnns.append([])
        for k in range(in_size // group_size):
            next_net = ICNNLogical([group_size, 10, 10, 10, 1], force_positive_init=False, with_two_layers=False, init_scaling=10,
                                     init_all_with_zeros=False)
            icnns[i].append(next_net)
        if in_size % group_size > 0:
            next_net = ICNNLogical([in_size % group_size, 10, 10, 10, 1], force_positive_init=False, with_two_layers=False,
                                   init_scaling=10,
                                   init_all_with_zeros=False)
            icnns[i].append(next_net)

    icnns = \
        multidhov.start_verification(nn, test_image, icnns, group_size, eps=1, icnn_epochs=10, icnn_batch_size=1000,
                                     sample_count=10, sample_new=False, use_over_approximation=True,
                                     sample_over_input_space=False, sample_over_output_space=True,
                                     force_inclusion_steps=0, preemptive_stop=False, even_gradient_training=False,
                                     keep_ambient_space=True, data_grad_descent_steps=0, train_outer=False,
                                     should_plot="none", optimizer="adam", init_network=True, adapt_lambda="none")

    return
    milp_verifier = MILPVerifier(nn, test_image, 1)
    snv_verifier = SingleNeuronVerifier(nn, test_image, 1)
    dhov_verifier = DHOVVerifier(icnns, group_size, nn, test_image, 1)

    milp_verifier.generate_constraints_for_net()
    snv_verifier.generate_constraints_for_net()
    dhov_verifier.generate_constraints_for_net()

    input_flattened = torch.flatten(test_image)
    input_size = input_flattened.size(0)
    bounds_affine_out, bounds_layer_out = nn.calculate_box_bounds([input_flattened.add(-1), input_flattened.add(1)])

    test_space = torch.empty((0, input_flattened.size(0)), dtype=data_type).to(device)
    box_bound_output_space = ds.samples_uniform_over(test_space, 1000, bounds_layer_out[-1])

    in_snv = []
    in_milp = []
    in_dhov = []
    for i, sample in enumerate(box_bound_output_space):
        if i % 100 == 0:
            print(i)
        in_milp.append(milp_verifier.test_feasibility(sample))
        in_snv.append(snv_verifier.test_feasibility(sample))
        in_dhov.append(dhov_verifier.test_feasibility(sample))

    plt_inc_amb("milp", box_bound_output_space.detach().cpu().numpy(), in_milp)
    plt_inc_amb("snv", box_bound_output_space.detach().cpu().numpy(), in_snv)
    plt_inc_amb("dhov", box_bound_output_space.detach().cpu().numpy(), in_dhov)

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


# cifar_net()

# net_2d()

multi_net2D()
