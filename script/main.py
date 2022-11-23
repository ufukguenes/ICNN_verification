import random

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torchvision.transforms import Normalize, Compose
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
from gurobipy import *
from gurobi_ml.torch import add_sequential_constr

import time

from script.Networks import *
from script.dataInit import *
from script.trainFunction import *
from eval import *
from dataInit import *
import Verification as ver



def start_training(train_loader, ambient_space, epochs, sequential=False):
    if not sequential:
        history = train_icnn(icnn, train_loader, ambient_space, epochs=epochs)
    else:
        history = train_sequential_icnn(icnn, train_loader, ambient_space, epochs=epochs)
    torch.save(icnn.state_dict(), "../convexHullModel.pth")
    return history


def start_adversarial_training(train_loader, adversarial_loader, epochs):
    history = train_icnn_adversarial(icnn, adversarial, train_loader, adversarial_loader, epochs=epochs)
    torch.save(icnn.state_dict(), "../convexHullModel_adv.pth")
    torch.save(adversarial.state_dict(), "../adversarialHullModel_adv.pth")
    return history


def continue_training(train_loader):
    icnn.load_state_dict(torch.load("../convexHullModel.pth"), strict=False)
    history = []
    # history = train_icnn(icnn, train_loader, ambient_space, epochs=3)
    # history = train_sequential_icnn(icnn, train_loader, ambient_space, epochs=3)
    torch.save(icnn.state_dict(), "../convexHullModel.pth")
    return history


def adv_setUp(epoch, extremal_points):
    rand = []
    for i in range(1000):
        x = np.random.default_rng().uniform(low=x_range[0], high=x_range[1])
        y = np.random.default_rng().uniform(low=y_range[0], high=y_range[1])
        rand.append([[x, y]])

    value = torch.tensor(rand, requires_grad=True, dtype=torch.float64)
    adversarial_set = ConvexDataset(data=value)
    adversarial_loader = DataLoader(adversarial_set, batch_size=1, shuffle=True)

    plots = Plots_for(0, icnn, included_space, ambient_space, true_extremal_points, x_range, y_range)
    plots.plt_initial()
    plots.plt_mesh()

    plots.plt_advers()
    plots.plt_advers_value()

    history = start_adversarial_training(train_loader, adversarial_loader, epoch)

    plots.plt_mesh()
    plots.plt_advers()
    plots.plt_advers_value()



def verify(sequential=False):
    ver.load(icnn)
    ver.verification(icnn, sequential=sequential)

sample_set_size = 3000

icnn = ICNN([2, 10, 10, 1])
#icnn = SequentialNN([2, 10, 10, 1])

adversarial = SequentialNN([2, 10, 10, 2])
x_range = [-1.5, 1.5]
y_range = [-1.5, 1.5]
included_space, ambient_space = Rhombus().get_uniform_samples(sample_set_size, x_range, y_range)

dataset = ConvexDataset(data=included_space)
train_loader = DataLoader(dataset, batch_size=1, shuffle=True)

dataset = ConvexDataset(data=ambient_space)
ambient_loader = DataLoader(dataset, batch_size=1, shuffle=True)

sequential = False
#icnn.load_state_dict(torch.load("../convexHullModel.pth"), strict=False)
#adv_setUp(1, Rhombus().get_extremal_points())
#verify(sequential)



true_extremal_points = Rhombus().get_extremal_points()
epoch = 1

plots = Plots_for(0, icnn, included_space, ambient_space, true_extremal_points, x_range, y_range)
plots.plt_initial()
plots.plt_mesh()

adv_setUp(epoch, true_extremal_points)

#plots = Plots_for(0, icnn, included_space, ambient_space, true_extremal_points, x_range, y_range)
plots.plt_initial()
plots.plt_mesh()

#icnn.load_state_dict(torch.load("../convexHullModel.pth"), strict=False)
# icnn.load_state_dict(torch.load("../convexHullModel_adv.pth"), strict=False)
# adversarial.load_state_dict(torch.load("../adversarialHullModel_adv.pth"),strict=False )
#included_space = torch.load("../included_space.pt")
#ambient_space = torch.load("../ambient_space.pt")


"""rom = Rhombus().get_extremal_points()
init_for(3.01, icnn, included_space, ambient_space, rom)
plt_learned_dot()
plt_learned_mesh()
print(icnn(torch.tensor([[-1.0, 0]], dtype=torch.float64)))
print(icnn(torch.tensor([[0, 1]], dtype=torch.float64)))
"""