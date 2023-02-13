import numpy as np
from matplotlib import pyplot as plt
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize
import gurobipy as grp
import torch

import DataSampling as ds
from torch.utils.data import DataLoader
from script.NeuralNets.Networks import SequentialNN
from script.Verification.Verifier import SingleNeuronVerifier, MILPVerifier
from script.settings import device, data_type


def imshow(img):
    img = img / 2 + .05  # revert normalization for viewing
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def add_to_model(verifier, labels):
    output_var = verifier.output_vars
    m = verifier.model
    input_var = verifier.input_vars

    difference = m.addVars(9, lb=-float('inf'))
    m.addConstrs(difference[i] == output_var.tolist()[i] - output_var.tolist()[labels] for i in range(0, labels))
    m.addConstrs(
        difference[i - 1] == output_var.tolist()[i] - output_var.tolist()[labels] for i in range(labels + 1, 10))

    # m.addConstrs(difference[i] == output_var.tolist()[i] - output_var.tolist()[label] for i in range(10))
    max_var = m.addVar(lb=-float('inf'), ub=10)
    m.addConstr(max_var == grp.max_(difference))
    m.addConstr(max_var >= 0)

def net_cifar():
    transform = Compose([ToTensor(),
                         Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
                        )

    training_data = CIFAR10(root="../../cifar", train=True, download=True, transform=transform)

    test_data = CIFAR10(root="../../cifar", train=False, download=True, transform=transform)

    batch_size = 4
    train_dataloader = DataLoader(training_data,  batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    classes = training_data.classes

    nn = SequentialNN([32 * 32 * 3, 1024, 512, 10])
    nn.load_state_dict(torch.load("../../cifar_fc.pth", map_location=torch.device('cpu')), strict=False)

    images, labels = training_data.__getitem__(0)
    testimage, testlabel = torch.unsqueeze(images, 0).to(torch.float64), torch.unsqueeze(torch.tensor(labels), 0).to(
        torch.float64)
    #imshow(images)

    print("label is {} with index {}".format(classes[labels], labels))
    pred = nn(testimage)
    print("prediction is {} with output {} ".format(classes[pred.argmax()], pred))

    """m.computeIIS()
    print("constraint")
    all_constr = m.getConstrs()

    for const in all_constr:
        if const.IISConstr:
            print(const)

    print("lower bound")
    all_var = m.getVars()
    for var in all_var:
        if var.IISLB:
            print(var)

    print("upper bound")
    all_var = m.getVars()
    for var in all_var:
        if var.IISUB:
            print(var)"""

    eps = 0.001
    input_flattened = torch.flatten(testimage)
    new = input_flattened.add(-eps)
    new2 = input_flattened.add(eps)
    bounds = nn.calculate_box_bounds([input_flattened.add(-eps), input_flattened.add(eps)], with_relu=True)

    milp_verifier = MILPVerifier(nn, testimage, eps, print_log=False)
    snv_verifier = SingleNeuronVerifier(nn, testimage, eps, print_log=False)

    milp_verifier.generate_constraints_for_net()
    snv_verifier.generate_constraints_for_net()

    add_to_model(milp_verifier, labels)
    add_to_model(snv_verifier, labels)

    test_space = torch.empty((0, 10), dtype=data_type).to(device)
    box_bound_output_space = ds.samples_uniform_over(test_space, 10, bounds[-1])

    in_snv = []
    in_milp = []
    for i, sample in enumerate(box_bound_output_space):
        if i % 1 == 0:
            print(i)
        in_milp.append(milp_verifier.test_feasibility(sample))
        in_snv.append(snv_verifier.test_feasibility(sample))

    num_in_milp = 0
    num_out_milp = 0
    for val in in_milp:
        if val:
            num_in_milp += 1
        else:
            num_out_milp += 1

    num_in_snv = 0
    num_out_snv = 0
    for val in in_snv:
        if val:
            num_in_snv += 1
        else:
            num_out_snv += 1


    not_not = 0
    not_in = 0
    in_not = 0
    in_in = 0
    for i in range(len(in_milp)):
        if not in_milp and not in_snv:
            not_not += 1
        elif not in_milp[i] and in_snv[i]:
            not_in += 1
        elif in_milp[i] and not in_snv[i]:
            in_not += 1
        elif in_milp[i] and in_snv[i]:
            in_in += 1


    print("samples MILP - in: {}, out: {}".format(num_in_milp, num_out_milp))
    print("samples SNV  - in: {}, out: {}".format(num_in_snv, num_out_snv))
    print("both out     - {}".format(not_not))
    print("both in      - {}".format(in_in))
    print("only in snv (due to over approximation) - {}".format(not_in))
    print("only in milp (due to false approximation) - {}".format(in_not))



net_cifar()