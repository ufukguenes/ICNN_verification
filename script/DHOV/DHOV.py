# DeepHull Over approximated Verification

from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from script.NeuralNets.Networks import SequentialNN, ICNN
from gurobipy import Model, GRB, max_, abs_
import torch
import numpy as np

from script.DHOV.Normalisation import get_std, get_mean, normalize_nn, normalize_data
import script.Verification.Verification as ver
import script.Verification.VerificationBasics as verbas
import script.DHOV.DataSampling as ds
from script.dataInit import ConvexDataset, Rhombus
from script.eval import Plots_for
from script.NeuralNets.trainFunction import train_icnn


def start_verification(nn: SequentialNN, input, eps=0.001, solver_time_limit=None, solver_bound=None,
                       icnn_batch_size=3000,
                       icnn_epochs=500, sample_count=2000, keep_ambient_space=False, sample_new=True,
                       sample_over_input_space=False,
                       sample_over_output_space=True):
    def plt_inc_amb(caption, inc, amb):
        plt.figure(figsize=(20, 10))
        plt.scatter(list(map(lambda x: x[0], amb)), list(map(lambda x: x[1], amb)), c="#ff7f0e")
        plt.scatter(list(map(lambda x: x[0], inc)), list(map(lambda x: x[1], inc)), c="#1f77b4")
        plt.title(caption)
        plt.show()

    # todo Achtung ich muss schauen, ob gurobi upper bound inklusive ist, da ich aktuell die upper bound mit eps nicht inklusive habe
    input_flattened = torch.flatten(input)
    eps_bounds = [input_flattened.add(-eps), input_flattened.add(eps)]
    box_bounds = verbas.calculate_box_bounds(nn, eps_bounds)  # todo abbrechen, wenn die box bounds schon die eigenschaft erfüllen

    included_space = torch.empty((0, input_flattened.size(0)), dtype=torch.float64)
    included_space = ds.samples_uniform_over(included_space, int(sample_count / 2), eps_bounds)

    ambient_space = torch.empty((0, input_flattened.size(0)), dtype=torch.float64)
    original_included_space, original_ambient_space = included_space, ambient_space

    """imshow_flattened(input_flattened, (3, 32, 32))
    imshow_flattened(included_space[0], (3, 32, 32))
    imshow_flattened(included_space[100], (3, 32, 32))
    imshow_flattened(ambient_space[0], (3, 32, 32))
    imshow_flattened(ambient_space[100], (3, 32, 32))"""

    if should_plot:
        plt_inc_amb("start", included_space, ambient_space)

    parameter_list = list(nn.parameters())

    icnns = []
    c_values = []
    center = input_flattened
    for i in range(0, len(parameter_list) - 2, 2):  # -2 because last layer has no ReLu activation
        current_layer_index = int(i / 2)
        icnn_input_size = nn.layer_widths[current_layer_index + 1]
        icnns.append(ICNN([icnn_input_size, 10, 10, icnn_input_size, 2 * icnn_input_size, 1]))
        current_icnn = icnns[current_layer_index]

        W, b = parameter_list[i], parameter_list[i + 1]

        if not keep_ambient_space:
            ambient_space = torch.empty((0, nn.layer_widths[current_layer_index]), dtype=torch.float64)

        if sample_over_input_space:
            if i == 0:
                ambient_space = ds.sample_uniform_excluding(ambient_space, int(sample_count / 2), eps_bounds, excluding_bound=eps_bounds, padding=0.5)
            else:
                ambient_space = ds.sample_uniform_excluding(ambient_space, int(sample_count / 2), box_bounds[current_layer_index-1], icnn_c=[icnns[current_layer_index-1], c_values[current_layer_index-1]], padding=0.5)  # todo test for when lower/upper bound is smaller then eps

        if should_plot:
            plt_inc_amb("affin e" + str(i), included_space.tolist(), ambient_space.tolist())

        included_space = ds.apply_affine_transform(W, b, included_space)
        if sample_over_input_space or keep_ambient_space:
            ambient_space = ds.apply_affine_transform(W, b, ambient_space)


        if should_plot:
            original_included_space = ds.apply_affine_transform(W, b, original_included_space)
            original_ambient_space = ds.apply_affine_transform(W, b, original_ambient_space)

            plt_inc_amb("affin e" + str(i), included_space.tolist(), ambient_space.tolist())
            plt_inc_amb("original affin e" + str(i), original_included_space.tolist(), original_ambient_space.tolist())


        included_space = ds.apply_ReLU_transform(included_space)
        if sample_over_input_space or keep_ambient_space:
            ambient_space = ds.apply_ReLU_transform(ambient_space)

        if should_plot:
            original_included_space = ds.apply_ReLU_transform(original_included_space)
            original_ambient_space = ds.apply_ReLU_transform(original_ambient_space)

            plt_inc_amb("relu " + str(i), included_space.tolist(), ambient_space.tolist())
            plt_inc_amb("original relu e" + str(i), original_included_space.tolist(), original_ambient_space.tolist())


        if sample_over_output_space:
            ambient_space = ds.samples_uniform_over(ambient_space, int(sample_count / 2),
                                                    box_bounds[current_layer_index], padding=0.5)
            original_ambient_space = ds.samples_uniform_over(original_ambient_space, int(sample_count / 2),
                                                             box_bounds[current_layer_index], padding=0.5)

        if should_plot:
            plt_inc_amb("enhanced ambient space " + str(i), included_space.tolist(), ambient_space.tolist())
            plt_inc_amb("original enhanced ambient space " + str(i), original_included_space.tolist(),
                        original_ambient_space.tolist())





        mean = get_mean(included_space, ambient_space)
        std = get_std(included_space, ambient_space)

        normalized_included_space, normalized_ambient_space = normalize_data(included_space, ambient_space, mean, std)
        dataset = ConvexDataset(data=normalized_included_space)
        train_loader = DataLoader(dataset, batch_size=icnn_batch_size, shuffle=True)
        dataset = ConvexDataset(data=normalized_ambient_space)
        ambient_loader = DataLoader(dataset, batch_size=icnn_batch_size, shuffle=True)

        # torch.save(included_space, "../included_space_after_relu.pt")
        # torch.save(ambient_space, "../ambient_space_after_relu.pt")

        ws = list(current_icnn.ws.parameters())
        us = list(current_icnn.us.parameters())
        low = box_bounds[current_layer_index][0]
        up = box_bounds[current_layer_index][1]
        low = torch.div(torch.add(low, -mean), std)
        up = torch.div(torch.add(up, -mean), std)
        init_icnn(current_icnn, [low, up])

        if should_plot:
            plots = Plots_for(0, current_icnn, normalized_included_space.detach(), normalized_ambient_space.detach(),
                              true_extremal_points,
                              [-1, 3], [-1, 3])
            plots.plt_dotted()
            plots.plt_mesh()

        # train icnn
        train_icnn(current_icnn, train_loader, ambient_loader, epochs=icnn_epochs, hyper_lambda=1)

        normalize_nn(current_icnn, mean, std, isICNN=True)

        # matplotlib.use("TkAgg")
        if should_plot:
            plots = Plots_for(0, current_icnn, included_space.detach(), ambient_space.detach(), true_extremal_points,
                              [-1, 3], [-1, 3])
            plots.plt_dotted()
            plots.plt_mesh()

        # verify and enlarge convex approximation
        if i == 0:
            adversarial_input, c = ver.verification(current_icnn,
                                                    center_eps_W_b=[center.detach().numpy(), eps, W.detach().numpy(),
                                                                    b.detach().numpy()], has_ReLU=True)
        else:
            prev_icnn = icnns[current_layer_index - 1]
            # prev_W, prev_b = parameter_list[i-2].detach().numpy(), parameter_list[i - 1].detach().numpy()
            prev_c = c_values[current_layer_index - 1]
            adversarial_input, c = ver.verification(current_icnn,
                                                    icnn_W_b_c=[prev_icnn, W.detach().numpy(), b.detach().numpy(),
                                                                prev_c], has_ReLU=True)
        c_values.append(c)
        if should_plot:
            plots.c = c
            plots.plt_dotted()
            plots.plt_mesh()

        c_values.append(0)

        # entweder oder:
        # nutze die samples weiter (dafür muss man dann das ReLU layer anwenden), und man muss schauen ob die
        # samples jetzt von ambient_space zu included_space gewechselt haben (wegen überapproximation)
        # damit könnte man gut den Fehler messen, da man ganz genau weiß wie viele elemente aus Ambient space in Icluded space übergegenagen sind

        # oder sample neue punkte

        if sample_new:
            included_space, ambient_space = ds.sample_max_radius(current_icnn, c, sample_count,
                                                                 box_bounds=box_bounds[current_layer_index])
            #todo wenn die box bounds besser als das icnn ist, beschreibt das icnn nicht mehr den included space,
            # man müsste dann noch mal nach dem Training das icnn mit boxbounds zusammenfügen, damit es das gleiche ist
            # dann muss man auch nicht mehr max_radius verwenden zum samplen

            if should_plot:
                plt_inc_amb("sampled new", included_space.tolist(), ambient_space.tolist())
                plt_inc_amb("original end of layer", original_included_space.tolist(), original_ambient_space.tolist())
        else:
            included_space, ambient_space = ds.regroup_samples(current_icnn, c, included_space, ambient_space)

    index = len(parameter_list) - 2
    W, b = parameter_list[index], parameter_list[index + 1]
    # last_layer_picture(icnns[-1], c_values[-1], W, b, 6, solver_time_limit, solver_bound)  # todo nicht hardcoden

    if should_plot:
        included_space = ds.apply_affine_transform(W, b, included_space)
        ambient_space = ds.apply_affine_transform(W, b, ambient_space)
        ambient_space = ds.samples_uniform_over(ambient_space, int(sample_count / 2), box_bounds[-1], padding=0.5)
        plt_inc_amb("output" + str(i), included_space.tolist(), ambient_space.tolist())

        original_included_space = ds.apply_affine_transform(W, b, original_included_space)
        original_ambient_space = ds.apply_affine_transform(W, b, original_ambient_space)
        original_ambient_space = ds.samples_uniform_over(original_ambient_space, int(sample_count / 2), box_bounds[-1])
        plt_inc_amb("original output" + str(i), original_included_space.tolist(), original_ambient_space.tolist())

    A_out, b_out = Rhombus().get_A(), Rhombus().get_b()
    last_layer_identity(icnns[-1], c_values[-1], W, b, A_out, b_out, box_bounds, solver_time_limit, solver_bound)


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


def init_icnn(icnn: ICNN, box_bounds):
    # todo check for the layer to have the right dimensions
    with torch.no_grad():
        k = len(icnn.ws)
        for i in range(0, len(icnn.ws)):
            ws = list(icnn.ws[i].parameters())
            # ws[0].data = torch.zeros_like(ws[0], dtype=torch.float64)

            """ For using box bounds constraints in second layer
            if i == 0:
                ws[0].data = torch.zeros_like(ws[0], dtype=torch.float64)
            else:
                num_rows = ws[0].size(0)
                num_cols = ws[0].size(1)
                max_num = max(num_cols, num_rows)
                t = torch.diag(torch.ones(max_num, dtype=torch.float64))
                t = t.split(num_cols, dim=1)
                t = t[0].split(num_rows, dim=0)
                ws[0].data = torch.diag(torch.ones(max_num, dtype=torch.float64)).split(num_cols, dim=1)[0].split(num_rows, dim=0)"""
            ws[1].data = torch.zeros_like(ws[1], dtype=torch.float64)

        for elem in icnn.us:
            w = list(elem.parameters())
            w[0].data = torch.zeros_like(w[0], dtype=torch.float64)

        """ For using box bounds constraints in second layer
        ws = list(icnn.ws[1].parameters()) 
        us = list(icnn.us[0].parameters())"""

        ws = list(icnn.ws[3].parameters())  # bias for first relu activation with weights from us (in second layer)
        us = list(icnn.us[2].parameters())  # us is used because values in ws are set to 0 when negative
        b = torch.zeros_like(ws[1], dtype=torch.float64)
        w = torch.zeros_like(us[0], dtype=torch.float64)
        for i in range(len(box_bounds)):
            w[2 * i][i] = 1
            w[2 * i + 1][i] = -1
            b[2 * i] = - box_bounds[1][i]  # upper bound
            b[2 * i + 1] = box_bounds[0][i]  # lower bound
        ws[0].data = torch.zeros_like(ws[0], dtype=torch.float64)
        ws[1].data = b
        us[0].data = w

        last = list(icnn.ws[4].parameters())
        last[0].data = torch.mul(torch.ones_like(last[0], dtype=torch.float64), 10)
        last[1].data = torch.zeros_like(last[1], dtype=torch.float64)


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


def imshow_flattened(img_flattened, shape):
    img = np.reshape(img_flattened, shape)
    img = img / 2 + .05  # revert normalization for viewing
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()


def plot_2d(nn, included, ambient):
    plt.figure(figsize=(20, 10))
    included_out_x = []
    included_out_y = []
    for x in included:
        x = torch.unsqueeze(x, dim=0)
        out = nn(x)
        included_out_x.append(out[0][0].item())
        included_out_y.append(out[0][1].item())
    ambient_out_x = []
    ambient_out_y = []
    for x in ambient:
        x = torch.unsqueeze(x, dim=0)
        out = nn(x)
        ambient_out_x.append(out[0][0].item())
        ambient_out_y.append(out[0][1].item())

    plt.scatter(included_out_x, included_out_y)
    plt.scatter(ambient_out_x, ambient_out_y)
    plt.show()


"""transform = Compose([ToTensor(),
                     Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
                    )

training_data = CIFAR10(root="../cifar",
                        train=True,
                        download=True,
                        transform=transform)
images, labels = training_data.__getitem__(0)
testimage, testlabel = torch.unsqueeze(images, 0), torch.unsqueeze(torch.tensor(labels), 0)

nn = SequentialNN([32 * 32 * 3, 1024, 512, 10])
nn.load_state_dict(torch.load("../cifar_fc.pth", map_location=torch.device('cpu')), strict=False)

torch.set_default_dtype(torch.float64)
start_verification(nn, images)"""

batch_size = 10
epochs = 30
number_of_train_samples = 10000
hyper_lambda = 1
x_range = [-1.5, 1.5]
y_range = [-1.5, 1.5]

included_space, ambient_space = Rhombus().get_uniform_samples(number_of_train_samples, x_range,
                                                              y_range)  # samples will be split in inside and outside the rhombus

true_extremal_points = Rhombus().get_extremal_points()
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
should_plot = True

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

# plot_2d(nn, included_space, ambient_space)
# torch.save(nn.state_dict(), "nn_2x2.pt")


test_image = torch.tensor([[0, 0]], dtype=torch.float64)
start_verification(nn, test_image, eps=1, sample_new=True,
                   sample_over_input_space=False, sample_over_output_space=True, keep_ambient_space=False)
