# DeepHull Over approximated Verification
import time
from functools import reduce

import matplotlib
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision as tv
from torchvision.transforms import Compose, ToTensor, Normalize

from script import dataInit
from script.Networks import SequentialNN, ICNN
from gurobipy import Model, GRB, max_
import torch
import numpy as np

from script.Verification import verification, add_non_sequential_constr, add_affine_constr
from script.dataInit import ConvexDataset, Rhombus
from script.eval import Plots_for
from script.trainFunction import train_icnn, train_sequential, train_sequential_2


def start_verification(nn: SequentialNN, input, eps=0.001, solver_time_limit=None, solver_bound=None, icnn_batch_size=4,
                       icnn_epochs=3, sample_count=10000, sample_new=False):
    # todo Achtung ich muss schauen, ob gurobi upper bound inklusive ist, da ich aktuell die upper bound mit eps nicht inklusive habe
    input_flattened = torch.flatten(input)

    included_space, ambient_space = sample_uniform_from(input_flattened, eps, sample_count, lower_bound=-0.003,
                                                        upper_bound=0.003)

    """imshow_flattened(input_flattened, (3, 32, 32))
    imshow_flattened(included_space[0], (3, 32, 32))
    imshow_flattened(included_space[100], (3, 32, 32))
    imshow_flattened(ambient_space[0], (3, 32, 32))
    imshow_flattened(ambient_space[100], (3, 32, 32))"""

    def plt_inc_amb(caption, inc, amb):
        plt.scatter(list(map(lambda x: x[0], amb)), list(map(lambda x: x[1], amb)))
        plt.scatter(list(map(lambda x: x[0], inc)), list(map(lambda x: x[1], inc)))
        plt.title(caption)
        plt.show()

    plt_inc_amb("start", included_space, ambient_space)

    parameter_list = list(nn.parameters())

    icnns = []
    c_values = []
    mean_values = []
    center = input_flattened
    for i in range(0, len(parameter_list)-2, 2):  # -2 because last layer has no ReLu activation
        icnn_input_size = nn.layer_widths[int(i / 2) + 1]
        icnns.append(ICNN([icnn_input_size, 10, 10, 1]))
        current_icnn = icnns[int(i / 2)]

        W, b = parameter_list[i], parameter_list[i + 1]

        included_space, ambient_space, center = apply_affine_transform(W, b, included_space, ambient_space, center)
        plt_inc_amb("affin e" + str(i), included_space.tolist(), ambient_space.tolist())
        included_space, ambient_space, center = apply_ReLU_transform(included_space, ambient_space, center)
        plt_inc_amb("relu " + str(i), included_space.tolist(), ambient_space.tolist())

        mean_inp = torch.mean(included_space, dim=0)
        mean_amp = torch.mean(ambient_space, dim=0)
        mean_scale_inc = len(included_space) / (len(included_space) + len(ambient_space))
        mean_scale_amb = len(included_space) / (len(included_space) + len(ambient_space))
        mean = mean_scale_amb * mean_amp + mean_scale_inc * mean_inp
        mean_values.append(mean)

        train_ambient_space = tv.transforms.Lambda(lambda x: 1000 * (x - mean))(ambient_space)
        train_included_space = tv.transforms.Lambda(lambda x: 1000 * (x - mean))(included_space)

        dataset = ConvexDataset(data=train_included_space.detach())
        train_loader = DataLoader(dataset, batch_size=icnn_batch_size, shuffle=True)
        dataset = ConvexDataset(data=train_ambient_space.detach())
        ambient_loader = DataLoader(dataset, batch_size=icnn_batch_size, shuffle=True)

        #torch.save(included_space, "../included_space_after_relu.pt")
        #torch.save(ambient_space, "../ambient_space_after_relu.pt")


        plots = Plots_for(0, current_icnn, train_included_space.detach(), train_ambient_space.detach(), true_extremal_points, x_range, y_range)
        # train icnn
        train_icnn(current_icnn, train_loader, ambient_loader, epochs=icnn_epochs, hyper_lambda=1)
        plots.plt_mesh()
        # verify and enlarge convex approximation

        #todo, ich darf nicht einfach center reingeben, dann ist gar nicht mehr garantiert, dass die eps umgebung um center
        # eine überrapproximation der eingabe region ist.
        adversarial_input, c = verification(current_icnn, center.detach().numpy(), eps)
        c_values.append(c)

        # entweder oder:
        # nutze die samples weiter (dafür muss man dann das ReLU layer anwenden), und man muss schauen ob die
        # samples jetzt von ambient_space zu included_space gewechselt haben (wegen überapproximation)
        # damit könnte man gut den Fehler messen, da man ganz genau weiß wie viele elemente aus Ambient space in Icluded space übergegenagen sind

        # oder sample neue punkte


        if sample_new:
            continue
        else:
            included_space, ambient_space = split_new(current_icnn, included_space, ambient_space, c)

    index = len(parameter_list) - 2
    W, b = parameter_list[index], parameter_list[index + 1]
    last_layer(icnns[-1], c_values[-1], W, b, 6, solver_time_limit, solver_bound)  # todo nicht hardcoden


def last_layer(last_icnn: ICNN, last_c, W, b, label, solver_time_limit, solver_bound):
    m = Model()

    if solver_time_limit is not None:
        m.setParam("TimeLimit", solver_time_limit)

    if solver_bound is not None:
        m.setParam("BestObjStop", solver_bound)

    icnn_input_size = last_icnn.layer_widths[0]
    output_of_penultimate_layer = m.addMVar(icnn_input_size, lb=-float('inf'))
    output_of_icnn = m.addMVar(1, lb=-float('inf'))
    add_non_sequential_constr(m, last_icnn, output_of_penultimate_layer, output_of_icnn)

    W = W.detach().numpy()
    b = b.detach().numpy()
    output_var = add_affine_constr(m, W, b, output_of_penultimate_layer, -1000, 1000)
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


def sample_uniform_from(input_flattened, eps, sample_size, icnn=None, lower_bound=None, upper_bound=None, ):
    input_size = input_flattened.size(dim=0)

    if lower_bound is None:
        lower_bound = - 3 * eps
    if upper_bound is None:
        upper_bound = 3 * eps

    if icnn is None:
        sample_size = int(sample_size / 2)
        included_space = torch.empty((sample_size, input_size), dtype=torch.float64)
        ambient_space = torch.empty((sample_size, input_size), dtype=torch.float64)

        lower = - eps
        upper = eps
        displacements_included_space = (upper - lower) * torch.rand((sample_size, input_size), dtype=torch.float64) + lower
        lower = lower_bound
        upper = upper_bound
        displacements_ambient_space = (upper - lower) * torch.rand((sample_size, input_size),
                                                                               dtype=torch.float64) + lower

        # making sure that at least one value is outside the ball with radius eps
        for i, displacement in enumerate(displacements_ambient_space):
            argmax_displacement = torch.argmax(displacement)
            argmin_displacement = torch.argmin(displacement)
            max_displacement = displacement[argmax_displacement]
            min_displacement = displacement[argmin_displacement]

            if max_displacement >= eps or min_displacement < -eps:
                continue
            if max_displacement < eps:
                displacement[argmax_displacement] = upper
                continue
            if min_displacement >= -eps:
                displacement[argmin_displacement] = lower

        for i in range(sample_size):
            included_space[i] = input_flattened + displacements_included_space[i]
            ambient_space[i] = input_flattened + displacements_ambient_space[i]

    else:
        included_space = torch.empty(0, dtype=torch.float64)
        ambient_space = torch.empty(0, dtype=torch.float64)
        displacements = (upper_bound - lower_bound) * torch.rand((sample_size, input_size),
                                                                 dtype=torch.float64) + lower_bound
        for i in range(sample_size):
            disp = input_flattened + displacements[i]
            out = icnn(disp)
            if out <= 0:  # todo hier mus <= c sein und nicht kleiner 0
                included_space = torch.cat([included_space, disp], dim=0)
            else:
                ambient_space = torch.cat([ambient_space, disp], dim=0)

    return included_space, ambient_space


def apply_affine_transform(W, b, included_space, ambient_space, center):
    t = time.time()
    affine_inc2 = torch.empty((included_space.shape[0], b.shape[0]), dtype=torch.float64)
    affine_amb2 = torch.empty((ambient_space.shape[0], b.shape[0]), dtype=torch.float64)

    for i in range(included_space.shape[0]):
        affine_inc2[i] = torch.matmul(W, included_space[i]).add(b)
    for i in range(ambient_space.shape[0]):
        affine_amb2[i] = torch.matmul(W, ambient_space[i]).add(b)

    center = torch.matmul(W, center).add(b)
    t = time.time() - t
    print("Time: {}".format(t))

    return affine_inc2, affine_amb2, center


def apply_ReLU_transform(included_space, ambient_space, center):
    relu = torch.nn.ReLU()
    included_space = relu(included_space)
    ambient_space = relu(ambient_space)
    center = relu(center)
    return included_space, ambient_space, center


def split_new(icnn, included_space, ambient_space, c):
    moved = 0

    in_outs = []
    for elem in included_space:
        elem = torch.unsqueeze(elem, 0)
        out = icnn(elem)
        in_outs.append(out.item())

    outputs = []
    for i, elem in enumerate(ambient_space):
        elem = torch.unsqueeze(elem, 0)
        output = icnn(elem)
        outputs.append(output.item())
        if output <= c:
            included_space = torch.cat([included_space, elem], dim=0)
            ambient_space = torch.cat([ambient_space[:i - moved], ambient_space[i + 1 - moved:]])
            moved += 1
    mini = min(outputs)
    maxi = max(in_outs)

    return included_space, ambient_space


def imshow_flattened(img_flattened, shape):
    img = np.reshape(img_flattened, shape)
    img = img / 2 + .05  # revert normalization for viewing
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()


def plot_2d(nn, included, ambient):
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

#matplotlib.use('TkAgg')

batch_size = 10
epochs = 2
number_of_train_samples = 10000
hyper_lambda = 1
x_range = [-1.5, 1.5]
y_range = [-1.5, 1.5]

included_space, ambient_space = Rhombus().get_uniform_samples(number_of_train_samples, x_range,
                                                              y_range)  # samples will be split in inside and outside the rhombus
"""plt.scatter(list(map(lambda x: x[0], ambient_space)), list(map(lambda x: x[1], ambient_space)))
plt.scatter(list(map(lambda x: x[0], included_space)), list(map(lambda x: x[1], included_space)))
plt.show()"""


true_extremal_points = Rhombus().get_extremal_points()
dataset_in = ConvexDataset(data=included_space)
train_loader = DataLoader(dataset_in, batch_size=batch_size, shuffle=True)
dataset = ConvexDataset(data=ambient_space)
ambient_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

nn = SequentialNN([2, 2, 2])


train_sequential_2(nn, train_loader, ambient_loader, epochs=epochs)

plot_2d(nn, included_space, ambient_space)

parameter_list = list(nn.parameters())
W = parameter_list[0]
b = parameter_list[1]
print(W)
print(b)

mean_inp = torch.mean(included_space, dim=0)
mean_amp = torch.mean(ambient_space, dim=0)
mean = (mean_amp + mean_inp) / 2
print("mean {}, mean_inc {}, mean_amb {}".format(mean, mean_inp, mean_amp))

mean = mean.detach().requires_grad_(True)

std_inp = torch.std(included_space, dim=0)
std_amp = torch.std(ambient_space, dim=0)
std = (std_amp + std_inp) / 2
print("std {}, std_inc {}, std_amb {}".format(std, std_inp, std_amp))
std = std.detach().requires_grad_(True)

parameter_list = list(nn.parameters())
p0 = parameter_list[0]
p1 = parameter_list[1]
test = torch.div(parameter_list[0], std)
test2 = torch.matmul(parameter_list[0], mean) + parameter_list[1]
with torch.no_grad():
    for i, p in enumerate(nn.parameters()):
        if i == 0:
            p.data = torch.div(p, std)
            w = p
        elif i == 1:
            p.data = torch.matmul(w, mean) + p
        else:
            break

parameter_list = list(nn.parameters())
W = parameter_list[0]
b = parameter_list[1]
print(W)
print(b)

test_image = torch.tensor([[0,0]])
start_verification(nn, test_image)
