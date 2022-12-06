# DeepHull Over approximated Verification
import time
from functools import reduce

from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize

from Networks import SequentialNN, ICNN
from gurobipy import Model, GRB
import torch
import numpy as np

from script.Verification import verification
from script.dataInit import ConvexDataset
from script.eval import Plots_for
from script.trainFunction import train_icnn


def start_verification(nn: SequentialNN, input, eps=0.07, solver_time_limit=None, solver_bound=None, icnn_batch_size=4,
                       icnn_epochs=5, sample_count=10000, sample_new=False):
    # todo Achtung ich muss schauen, ob gurobi upper bound inklusive ist, da ich aktuell die upper bound mit eps nicht inklusive habe
    m = Model()
    input_flattened = torch.flatten(input)



    if solver_time_limit is not None:
        m.setParam("TimeLimit", solver_time_limit)

    if solver_bound is not None:
        m.setParam("BestObjStop", solver_bound)

    included_space, ambient_space = sample_uniform_from(input_flattened, eps, sample_count, lower_bound=-0.1, upper_bound=0.1)

    """ imshow_flattened(input_flattened, (3, 32, 32))
    imshow_flattened(included_space[0], (3, 32, 32))
    imshow_flattened(ambient_space[0], (3, 32, 32))"""
    parameter_list = list(nn.parameters())

    icnns = []
    enlarge_values = []
    center = input_flattened
    for i in range(0, len(parameter_list), 2):
        icnns.append(ICNN([1024, 100, 10, 1]))
        current_icnn = icnns[int(i / 2)]
        W, b = parameter_list[i], parameter_list[i + 1]

        included_space, ambient_space, center = apply_affine_transform(W, b, included_space, ambient_space, center)
        included_space, ambient_space, center = apply_ReLU_transform(included_space, ambient_space, center)


        dataset = ConvexDataset(data=included_space.detach())
        train_loader = DataLoader(dataset, batch_size=icnn_batch_size, shuffle=True)
        dataset = ConvexDataset(data=ambient_space.detach())
        ambient_loader = DataLoader(dataset, batch_size=icnn_batch_size, shuffle=True)

        # train icnn
        train_icnn(current_icnn, train_loader, ambient_loader, epochs=icnn_epochs)

        #verify and enlarge convex approxmation

        adversarial_input, enlarge = verification(current_icnn, center.detach().numpy(), eps)
        enlarge_values.append(enlarge)
        # entweder oder:
        # nutze die samples weiter (dafür muss man dann das ReLU layer anwenden), und man muss schauen ob die
        # samples jetzt von ambient_space zu included_space gewechselt haben (wegen überapproximation)
        # damit könnte man gut den Fehler messen, da man ganz genau weiß wie viele elemente aus Ambient space in Icluded space übergegenagen sind

        # oder sample neue punkte

        if sample_new:
            continue
        else:
            included_space, ambient_space = split_new(current_icnn, included_space, ambient_space, enlarge)

        break



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

        displacements_included_space = (2 * eps) * torch.rand((sample_size, input_size), dtype=torch.float64) - eps
        displacements_ambient_space = (upper_bound - lower_bound) * torch.rand((sample_size, input_size), dtype=torch.float64) - lower_bound

        # making sure that at least one value is outside the ball with radius eps
        for i, displacement in enumerate(displacements_ambient_space):
            argmax_displacement = torch.argmax(displacement)
            argmin_displacement = torch.argmin(displacement)
            max_displacement = displacement[argmax_displacement]
            min_displacement = displacement[argmin_displacement]

            if max_displacement >= eps or min_displacement < -eps:
                continue
            if max_displacement < eps:
                displacement[argmax_displacement] += eps
                continue
            if min_displacement >= -eps:
                displacement[argmin_displacement] -= eps

        for i in range(sample_size):
            included_space[i] = input_flattened + displacements_included_space[i]
            ambient_space[i] = input_flattened + displacements_ambient_space[i]

    else:
        included_space = torch.empty(0, dtype=torch.float64)
        ambient_space = torch.empty(0, dtype=torch.float64)
        displacements = (upper_bound - lower_bound) * torch.rand((sample_size, input_size), dtype=torch.float64) - lower_bound
        for i in range(sample_size):
            disp = input_flattened + displacements[i]
            icnn.double()
            out = icnn(disp)
            if out <= 0:
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


    center = center.double()
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
    for i, elem in enumerate(ambient_space):
        icnn.double()
        elem = torch.unsqueeze(elem, 0)
        output = icnn(elem)
        if output <= c:
            included_space = torch.cat([included_space, elem], dim=0)
            ambient_space = torch.cat([ambient_space[:i - moved], ambient_space[i+1 - moved:]])
            moved += 1

    return included_space, ambient_space

def imshow_flattened(img_flattened, shape):
    img = np.reshape(img_flattened, shape)
    img = img / 2 + .05  # revert normalization for viewing
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()


transform = Compose([ToTensor(),
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
start_verification(nn, images)
