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


def start_verification(nn: SequentialNN, input, eps=0.001, solver_time_limit=None, solver_bound=None, icnn_batch_size=1,
                       icnn_epochs=5, sample_new=False):
    # todo Achtung ich muss schauen, ob gurobi upper bound inklusive ist, da ich aktuell die upper bound mit eps nicht inklusive habe
    m = Model()
    input_flattened = torch.flatten(input).numpy()



    if solver_time_limit is not None:
        m.setParam("TimeLimit", solver_time_limit)

    if solver_bound is not None:
        m.setParam("BestObjStop", solver_bound)

    included_space, ambient_space = sample_uniform_from(input_flattened, eps, 100, lower_bound=-0.1, upper_bound=0.1)

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
        W, b = parameter_list[i].detach().numpy(), parameter_list[i + 1].detach().numpy()

        included_space, ambient_space, center = apply_affine_transform(W, b, included_space, ambient_space, center)
        included_space, ambient_space, center = apply_ReLU_transform(included_space, ambient_space, center)

        included_space_tensor = torch.from_numpy(included_space).to(torch.float32)
        ambient_space_tensor = torch.from_numpy(ambient_space).to(torch.float32)

        dataset = ConvexDataset(data=included_space_tensor)
        train_loader = DataLoader(dataset, batch_size=icnn_batch_size, shuffle=True)
        dataset = ConvexDataset(data=ambient_space_tensor)
        ambient_loader = DataLoader(dataset, batch_size=icnn_batch_size, shuffle=True)

        # train icnn
        train_icnn(current_icnn, train_loader, ambient_loader, epochs=icnn_epochs)

        #verify and enlarge convex approxmation

        upper_bound_b = np.full(center.size, eps)
        lower_bound_b = np.full(center.size, -eps)

        inc_res1 = []
        inc_res2 = []
        for inc in included_space:
            res = center - inc
            if (res >= upper_bound_b).all or (res <= lower_bound_b).all:
                inc_res1.append(res)
            else:
                inc_res2.append(res)

        inc_res1 = []
        inc_res2 = []
        for inc in ambient_space:
            res = center - inc
            if (res >= upper_bound_b).all or (res <= lower_bound_b).all:
                inc_res1.append(res)
            else:
                inc_res2.append(res)

        adversarial_input, enlarge = verification(current_icnn, center, eps)
        if enlarge < 0:
            enlarge = 0
        enlarge_values.append(enlarge)
        # entweder oder:
        # nutze die samples weiter (dafür muss man dann das ReLU layer anwenden), und man muss schauen ob die
        # samples jetzt von ambient_space zu included_space gewechselt haben (wegen überapproximation)
        # damit könnte man gut den Fehler messen, da man ganz genau weiß wie viele elemente aus Ambient space in Icluded space übergegenagen sind

        # oder sample neue punkte

        if sample_new:
            continue
        else:
            included_space, ambient_space = split_new(current_icnn, enlarge, included_space, ambient_space)


# const = model.addConstrs(out_vars[i] == output_vars[i] for i in range(out_fet))


def sample_uniform_from(input_flattened, eps, sample_size, icnn=None, lower_bound=None, upper_bound=None, ):
    input_size = input_flattened.size

    if lower_bound is None:
        lower_bound = - 3 * eps
    if upper_bound is None:
        upper_bound = 3 * eps

    if icnn is None:
        sample_size = int(sample_size / 2)
        included_space = np.empty((sample_size, input_size))
        ambient_space = np.empty((sample_size, input_size))

        displacements_included_space = (2 * eps) * np.random.random_sample((sample_size, input_size)) - eps
        displacements_ambient_space = (upper_bound - lower_bound) * np.random.random_sample(
            (sample_size, input_size)) + lower_bound

        # making sure that at least one value is outside the ball with radius eps
        for i, displacement in enumerate(displacements_ambient_space):
            argmax_displacement = np.argmax(displacement)
            argmin_displacement = np.argmin(displacement)
            max_displacement = displacement[argmin_displacement]
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
            ambient_space[i] = input_flattened + ambient_space[i]

    else:
        included_space = np.empty(0)
        ambient_space = np.empty(0)
        displacements = (upper_bound - lower_bound) * np.random.random_sample(
            (sample_size, input_size)) + lower_bound
        for i in range(sample_size):
            disp = input_flattened + displacements[i]
            out = icnn(disp)
            if out <= 0:
                np.append(included_space, disp, axis=0)
            else:
                np.append(ambient_space, disp, axis=0)

    return included_space, ambient_space


def apply_affine_transform(W, b, included_space, ambient_space, center):
    def func(x):
       # res = np.einsum('ji,i->j', W, x) + b
        res = np.matmul(W, x) + b
        return res
    """included_space = map(lambda x: np.dot(W, x) + b, included_space)
    ambient_space = map(lambda x: np.dot(W, x) + b, ambient_space)"""

    output_size = W.shape[0]

    inc_len = included_space.shape[0]
    affine_inc = np.empty((inc_len, output_size))
    t = time.time()
    for i in range(inc_len):
        inp = included_space[i]
        affine_inc[i] = func(inp)

    amb_len = ambient_space.shape[0]
    affine_amb = np.empty((amb_len, output_size))
    for i in range(inc_len):
        inp = ambient_space[i]
        affine_amb[i] = func(inp)

    center = np.dot(W, center) + b
    t = time.time() - t
    print("time for dot = {}".format(t))

    t = time.time()
    W_ = torch.tensor(W, dtype=torch.float64)
    b_ = torch.tensor(b, dtype=torch.float64)
    inc_ = torch.tensor(included_space, dtype=torch.float64)
    amb_ = torch.tensor(ambient_space, dtype=torch.float64)
    affine_inc2 = torch.empty((included_space.shape[0], b.shape[0]), dtype=torch.float64)
    affine_amb2 = torch.empty((ambient_space.shape[0],b.shape[0]), dtype=torch.float64)


    for i in range(included_space.shape[0]):
        affine_inc2[i] = torch.matmul(W_, inc_[i]).add(b_)
    for i in range(ambient_space.shape[0]):
        affine_amb2[i] = torch.matmul(W_, amb_[i]).add(b_)

    #center = np.dot(W, center) + b
    affine_inc2, affine_amb2 = affine_inc2.detach().numpy(), affine_amb2.detach().numpy()
    t = time.time() - t
    print("Time: {}".format(t))

    for i in range(50):
        for k in range(1024):
            if abs(affine_inc[i][k]-affine_inc2[i][k]) > 0.00000000000001:
                print("help")
            if abs(affine_amb[i][k]-affine_amb2[i][k]) > 0.00000000000001:
                print("help 2")
    print(np.array_equal(affine_inc, affine_inc2))
    print(np.array_equal(affine_amb, affine_amb2))
    return ret



def apply_ReLU_transform(included_space, ambient_space, center):
    shape = np.shape(included_space[0])
    zeros = np.zeros(shape)
    included_space = map(lambda x: np.maximum(x, zeros), included_space)
    ambient_space = map(lambda x: np.maximum(x, zeros), ambient_space)
    center = np.maximum(center, zeros)
    return np.asarray(list(included_space)), np.asarray(list(ambient_space)), center


def split_new(icnn, c, included_space, ambient_space):
    out_indices = []
    out_values = []
    outs = []
    for i, x in enumerate(ambient_space):
        inp = torch.tensor([x], dtype=torch.float32)
        out = icnn(inp)
        out = out.detach().numpy()[0][0]
        outs.append(out)
        if out < c:
            out_values.append(x)
            out_indices.append(i)

    outs = np.asarray(outs)
    max = outs.max()
    out_values = np.asarray(out_values)
    included_space = np.append(included_space, out_values, axis=0)
    ambient_space = np.delete(ambient_space, out_indices, axis=0)

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

start_verification(nn, images)
