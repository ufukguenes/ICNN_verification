import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize

import DataSampling as ds
import DHOV as dhov
from torch.utils.data import DataLoader
from script.NeuralNets.Networks import SequentialNN, ICNN
from script.dataInit import Rhombus, ConvexDataset


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

    W = W.detach().numpy()
    b = b.detach().numpy()
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
        sol = torch.tensor(sol, dtype=torch.float64)
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

    W = W.detach().numpy()
    b = b.detach().numpy()
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

        sol = torch.tensor(sol, dtype=torch.float64)
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
        parameter_list[0].data = torch.tensor([[1, 1], [1, -1]], dtype=torch.float64)
        parameter_list[1].data = torch.tensor([0, 0], dtype=torch.float64)
        parameter_list[2].data = torch.tensor([[1, 1], [1, -1]], dtype=torch.float64)
        parameter_list[3].data = torch.tensor([-0.5, 0], dtype=torch.float64)
        parameter_list[4].data = torch.tensor([[-1, 1], [1, 1]], dtype=torch.float64)
        parameter_list[5].data = torch.tensor([3, 0], dtype=torch.float64)

    # nn.load_state_dict(torch.load("nn_2x2.pt"), strict=False)
    # train_sequential_2(nn, train_loader, ambient_loader, epochs=epochs)


    # matplotlib.use('TkAgg')

    # torch.save(nn.state_dict(), "nn_2x2.pt")


    test_image = torch.tensor([[0, 0]], dtype=torch.float64)
    icnns, c_values = \
        dhov.start_verification(nn, test_image, eps=1, icnn_epochs=100, sample_count=1000, sample_new=False, use_over_approximation=True,
                                sample_over_input_space=False, sample_over_output_space=True,
                                keep_ambient_space=False, use_grad_descent=True, train_outer=False, preemptive_stop=False,
                                should_plot="verification", optimizer="SdLBFGS", init_mode="logical", adapt_lambda="none")

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
    testimage, testlabel = torch.unsqueeze(images, 0).to(dtype=torch.float64), torch.unsqueeze(torch.tensor(labels), 0).to(dtype=torch.float64)

    nn = SequentialNN([32 * 32 * 3, 1024, 512, 10])
    nn.load_state_dict(torch.load("../../cifar_fc.pth", map_location=torch.device('cpu')), strict=False)


    # matplotlib.use('TkAgg')

    icnns, c_values = \
        dhov.start_verification(nn, testimage, eps=1, icnn_epochs=500, sample_new=True, use_over_approximation=True,
                                sample_over_input_space=False, sample_over_output_space=True,
                                keep_ambient_space=False, use_grad_descent=False, train_outer=False,
                                should_plot="detailed")

#cifar_net()
net_2d()