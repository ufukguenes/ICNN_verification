# DeepHull Over approximated Verification

from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from script.NeuralNets.Networks import SequentialNN, ICNN
import torch
import numpy as np

from script.DHOV.Normalisation import get_std, get_mean, normalize_nn, normalize_data
import script.Verification.Verification as ver
import script.Verification.VerificationBasics as verbas
import script.DHOV.DataSampling as ds
import script.DHOV.DataOptimization as dop
from script.dataInit import ConvexDataset, Rhombus
from script.eval import Plots_for
from script.NeuralNets.trainFunction import train_icnn, train_icnn_outer

"""
should_plot has values: none, simple, detailed
optimizer has values: adam, LBFGS, None. If None, adam will be used
"""
def start_verification(nn: SequentialNN, input, eps=0.001, solver_time_limit=None, solver_bound=None,
                       icnn_batch_size=1000,
                       icnn_epochs=100, sample_count=1000, keep_ambient_space=False, sample_new=True,
                       use_over_approximation=True, sample_over_input_space=False,
                       sample_over_output_space=True, use_grad_descent=False, train_outer=False, init_box_bounds=True, should_plot='none', optimizer="adam"):

    # todo Achtung ich muss schauen, ob gurobi upper bound inklusive ist, da ich aktuell die upper bound mit eps nicht inklusive habe
    input_flattened = torch.flatten(input)
    eps_bounds = [input_flattened.add(-eps), input_flattened.add(eps)]
    box_bounds = verbas.calculate_box_bounds(nn, eps_bounds, is_sequential=True)  # todo abbrechen, wenn die box bounds schon die eigenschaft erfüllen

    included_space = torch.empty((0, input_flattened.size(0)), dtype=torch.float64)
    included_space = ds.samples_uniform_over(included_space, int(sample_count / 2), eps_bounds)

    ambient_space = torch.empty((0, input_flattened.size(0)), dtype=torch.float64)
    original_included_space, original_ambient_space = included_space, ambient_space

    parameter_list = list(nn.parameters())

    icnns = []
    c_values = []
    center = input_flattened
    for i in range(0, len(parameter_list) - 2, 2):  # -2 because last layer has no ReLu activation
        current_layer_index = int(i / 2)
        icnn_input_size = nn.layer_widths[current_layer_index + 1]
        icnns.append(ICNN([icnn_input_size, 10, 10, 10, 2 * icnn_input_size, 1]))
        current_icnn = icnns[current_layer_index]

        W, b = parameter_list[i], parameter_list[i + 1]

        if not keep_ambient_space:
            ambient_space = torch.empty((0, nn.layer_widths[current_layer_index]), dtype=torch.float64)

        if sample_over_input_space:
            if i == 0:
                ambient_space = ds.sample_uniform_excluding(ambient_space, int(sample_count / 2), eps_bounds,
                                                            excluding_bound=eps_bounds, padding=0.5)
            else:
                ambient_space = ds.sample_uniform_excluding(ambient_space, int(sample_count / 2),
                                                            box_bounds[current_layer_index - 1],
                                                            icnn_c=[icnns[current_layer_index - 1],
                                                                    c_values[current_layer_index - 1]],
                                                            padding=0.5)  # todo test for when lower/upper bound is smaller then eps

        if should_plot == "detailed":
            plt_inc_amb("start " + str(i), included_space.tolist(), ambient_space.tolist())

        included_space = ds.apply_affine_transform(W, b, included_space)
        ambient_space = ds.apply_affine_transform(W, b, ambient_space)

        if should_plot == "simple" or should_plot == "detailed":
            original_included_space = ds.apply_affine_transform(W, b, original_included_space)
            original_ambient_space = ds.apply_affine_transform(W, b, original_ambient_space)
            if should_plot == "detailed":
                plt_inc_amb("affin e" + str(i), included_space.tolist(), ambient_space.tolist())
                plt_inc_amb("original affin e" + str(i), original_included_space.tolist(), original_ambient_space.tolist())

        included_space = ds.apply_ReLU_transform(included_space)
        ambient_space = ds.apply_ReLU_transform(ambient_space)

        if should_plot == "simple" or should_plot == "detailed":
            original_included_space = ds.apply_ReLU_transform(original_included_space)
            original_ambient_space = ds.apply_ReLU_transform(original_ambient_space)
            if should_plot == "detailed":
                plt_inc_amb("relu " + str(i), included_space.tolist(), ambient_space.tolist())
                plt_inc_amb("original relu e" + str(i), original_included_space.tolist(), original_ambient_space.tolist())

        if sample_over_output_space:
            ambient_space = ds.samples_uniform_over(ambient_space, int(sample_count / 2),
                                                    box_bounds[current_layer_index], padding=0.5)
            if should_plot == "simple" or should_plot == "detailed":
                original_ambient_space = ds.samples_uniform_over(original_ambient_space, int(sample_count / 2),
                                                                 box_bounds[current_layer_index], padding=0.5)

        if should_plot == "detailed":
            plt_inc_amb("enhanced ambient space " + str(i), included_space.tolist(), ambient_space.tolist())
            plt_inc_amb("original enhanced ambient space " + str(i), original_included_space.tolist(),
                        original_ambient_space.tolist())

        mean = get_mean(included_space, ambient_space)
        std = get_std(included_space, ambient_space)
        normalized_included_space, normalized_ambient_space = normalize_data(included_space, ambient_space, mean, std)

        if optimizer == "LBFGS": #todo does LBFGS support minibatch? I dont think so!
            icnn_batch_size = len(normalized_ambient_space) + len(normalized_included_space)
        dataset = ConvexDataset(data=normalized_included_space)
        train_loader = DataLoader(dataset, batch_size=icnn_batch_size, shuffle=False)
        dataset = ConvexDataset(data=normalized_ambient_space)
        ambient_loader = DataLoader(dataset, batch_size=icnn_batch_size, shuffle=False) # todo shuffel alle wieder auf true setzen

        if init_box_bounds:
            low = box_bounds[current_layer_index][0]
            up = box_bounds[current_layer_index][1]
            low = torch.div(torch.add(low, -mean), std)
            up = torch.div(torch.add(up, -mean), std)

            init_icnn_box_bounds(current_icnn, [low, up])

        # train icnn
        untouched_normalized_ambient_space = normalized_ambient_space.detach().clone()
        if use_grad_descent:
            if should_plot == "simple" or should_plot == "detailed":
                plt_inc_amb("without gradient descent", normalized_included_space.tolist(),
                            normalized_ambient_space.tolist())

            num_optimizations = 1
            optimization_steps = 1000

            epochs_per_optimization = icnn_epochs // (num_optimizations + 1)
            modulo_epochs = icnn_epochs % (num_optimizations + 1)

            for h in range(num_optimizations + 1):

                if h < num_optimizations:
                    epochs_in_run = epochs_per_optimization
                else:
                    epochs_in_run = epochs_per_optimization + modulo_epochs

                train_icnn(current_icnn, train_loader, ambient_loader, epochs=epochs_in_run, hyper_lambda=0.5, optimizer=optimizer)

                if h < num_optimizations:
                    for k in range(optimization_steps):
                        #normalized_ambient_space = dop.gradient_descent_data_optim(current_icnn, normalized_ambient_space.detach())
                        normalized_ambient_space = dop.adam_data_optim(current_icnn, normalized_ambient_space.detach())
                    dataset = ConvexDataset(data=torch.cat([normalized_ambient_space.detach(), untouched_normalized_ambient_space.detach()]))
                    #todo hier muss ich noch verwalten was passiert wenn ambient space in die nächste runde übernommen wird
                    if optimizer == "LBFGS":
                        icnn_batch_size = len(torch.cat([normalized_ambient_space.detach(), untouched_normalized_ambient_space.detach()])) + len(normalized_included_space)
                    ambient_loader = DataLoader(dataset, batch_size=icnn_batch_size, shuffle=False)

            if should_plot == "simple" or should_plot == "detailed":
                plt_inc_amb("with gradient descent", normalized_included_space.tolist(),
                            torch.cat([normalized_ambient_space.detach(), untouched_normalized_ambient_space.detach()]).tolist())
                plots = Plots_for(0, current_icnn, normalized_included_space.detach(), torch.cat([normalized_ambient_space.detach(), untouched_normalized_ambient_space.detach()]).detach(),
                                  [-2, 3], [-2, 3])
                plots.plt_mesh()
        else:
            train_icnn(current_icnn, train_loader, ambient_loader, epochs=icnn_epochs, hyper_lambda=1, optimizer=optimizer)

        if train_outer:
            lam = 10

            with torch.no_grad():
                for i in range(0, len(current_icnn.ws)):
                    ws = list(current_icnn.ws[i].parameters())
                    ws[1].data = torch.mul(ws[1], lam)

                ws = list(current_icnn.ws[-1].parameters())
                ws[0].data = torch.div(ws[0].data, lam)
                ws[1].data = torch.div(ws[1].data, lam)
                us = list(current_icnn.us[-1].parameters())
                us[0].data = torch.div(us[0].data, lam)

            plots = Plots_for(0, current_icnn, normalized_included_space.detach(),
                              untouched_normalized_ambient_space.detach(),
                              [-1*lam, 3*lam], [-1*lam, 3*lam])
            plots.plt_dotted()
            plots.plt_mesh()

        normalize_nn(current_icnn, mean, std, isICNN=True)

        # matplotlib.use("TkAgg")
        if should_plot == "detailed":
            plots = Plots_for(0, current_icnn, included_space.detach(), ambient_space.detach(), [-1, 3], [-1, 3])
            plots.plt_dotted()
            plots.plt_mesh()
        elif should_plot == "simple":
            plots = Plots_for(0, current_icnn, included_space.detach(), ambient_space.detach(), [-1, 3], [-1, 3])
            plots.plt_dotted()

        # verify and enlarge convex approximation
        if use_over_approximation:
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
            if should_plot == "detailed":
                plots.c = c
                plots.plt_dotted()
                plots.plt_mesh()
            elif should_plot == "simple":
                plots.c = c
                plots.plt_dotted()
        else:
            c = 0

        c_values.append(c)



        # entweder oder:
        # nutze die samples weiter (dafür muss man dann das ReLU layer anwenden), und man muss schauen ob die
        # samples jetzt von ambient_space zu included_space gewechselt haben (wegen überapproximation)
        # damit könnte man gut den Fehler messen, da man ganz genau weiß wie viele elemente aus Ambient space in Icluded space übergegenagen sind

        # oder sample neue punkte

        if sample_new:
            included_space, ambient_space = ds.sample_max_radius(current_icnn, c, sample_count,
                                                                 box_bounds=box_bounds[current_layer_index])
            # todo wenn die box bounds besser als das icnn ist, beschreibt das icnn nicht mehr den included space,
            # man müsste dann noch mal nach dem Training das icnn mit boxbounds zusammenfügen, damit es das gleiche ist
            # dann muss man auch nicht mehr max_radius verwenden zum samplen

        else:
            included_space, ambient_space = ds.regroup_samples(current_icnn, c, included_space, ambient_space)

        if should_plot == "detailed":
            plt_inc_amb("sampled new", included_space.tolist(), ambient_space.tolist())
            plt_inc_amb("original end of layer", original_included_space.tolist(), original_ambient_space.tolist())

    if should_plot == "simple" or should_plot == "detailed":
        index = len(parameter_list) - 2
        W, b = parameter_list[index], parameter_list[index + 1]
        included_space = ds.apply_affine_transform(W, b, included_space)
        ambient_space = ds.apply_affine_transform(W, b, ambient_space)
        original_included_space = ds.apply_affine_transform(W, b, original_included_space)
        original_ambient_space = ds.apply_affine_transform(W, b, original_ambient_space)
        plt_inc_amb("output approx", included_space.tolist(), ambient_space.tolist())
        plt_inc_amb("output exact", original_included_space.tolist(), original_ambient_space.tolist())
        plots = Plots_for(0, current_icnn, included_space.detach(), ambient_space.detach(), [-1, 3], [-1, 3], extr=original_included_space.detach())
        plots.plt_initial()

    return icnns, c_values



def init_icnn_box_bounds(icnn: ICNN, box_bounds):
    # todo check for the layer to have the right dimensions
    with torch.no_grad():
        for i in range(0, len(icnn.ws)):
            ws = list(icnn.ws[i].parameters())
            ws[1].data = torch.zeros_like(ws[1], dtype=torch.float64)
            ws[0].data = torch.zeros_like(ws[0], dtype=torch.float64)

        for elem in icnn.us:
            w = list(elem.parameters())
            w[0].data = torch.zeros_like(w[0], dtype=torch.float64)

        ws = list(icnn.ws[3].parameters())  # bias for first relu activation with weights from us (in second layer)
        us = list(icnn.us[2].parameters())  # us is used because values in ws are set to 0 when negative
        b = torch.zeros_like(ws[1], dtype=torch.float64)
        u = torch.zeros_like(us[0], dtype=torch.float64)
        for i in range(len(box_bounds)):
            u[2 * i][i] = 1
            u[2 * i + 1][i] = -1
            b[2 * i] = - box_bounds[1][i]  # upper bound
            b[2 * i + 1] = box_bounds[0][i]  # lower bound
        ws[0].data = torch.zeros_like(ws[0], dtype=torch.float64)
        ws[1].data = b
        us[0].data = u

        last_us = list(icnn.us[3].parameters())[0]
        last_us.data = torch.zeros_like(last_us, dtype=torch.float64)

        last = list(icnn.ws[4].parameters())
        last[0].data = torch.mul(torch.ones_like(last[0], dtype=torch.float64), 10)
        last[1].data = torch.zeros_like(last[1], dtype=torch.float64)


def init_icnn_prev_icnn(current_icnn, prev_icnn):
    current_icnn.load_state_dict(prev_icnn.state_dict())


def imshow_flattened(img_flattened, shape):
    img = np.reshape(img_flattened, shape)
    img = img / 2 + .05  # revert normalization for viewing
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()


def plt_inc_amb(caption, inc, amb):
    plt.figure(figsize=(20, 10))
    plt.scatter(list(map(lambda x: x[0], amb)), list(map(lambda x: x[1], amb)), c="#ff7f0e")
    plt.scatter(list(map(lambda x: x[0], inc)), list(map(lambda x: x[1], inc)), c="#1f77b4")
    plt.title(caption)
    plt.show()


