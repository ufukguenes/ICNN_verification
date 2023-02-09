# DeepHull Over approximated Verification
import random

from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from script.NeuralNets.Networks import SequentialNN
import torch
import numpy as np

import script.DHOV.Normalisation as norm
import script.Verification.Verification as ver
import script.DHOV.DataSampling as ds
import script.DHOV.DataOptimization as dop
from script.dataInit import ConvexDataset
from script.eval import Plots_for
from script.NeuralNets.trainFunction import train_icnn, train_icnn_outer

# todo zu Klasse umwandeln
"""
should_plot has values: none, simple, detailed, verification
optimizer has values: adam, LBFGS, SdLBFGS
adapt_lambda has values: none, high_low, included
"""


def start_verification(nn: SequentialNN, input, icnns, eps=0.001, icnn_batch_size=1000, icnn_epochs=100,
                       sample_count=1000,
                       keep_ambient_space=False, sample_new=True, use_over_approximation=True,
                       sample_over_input_space=False, sample_over_output_space=True, use_grad_descent=False,
                       train_outer=False, preemptive_stop=True, even_gradient_training=False, force_inclusion=1,
                       init_network=False, adapt_lambda="none", should_plot='none', optimizer="adam"):
    valid_adapt_lambda = ["none", "high_low", "included"]
    valid_should_plot = ["none", "simple", "detailed", "verification"]
    valid_optimizer = ["adam", "LBFGS", "SdLBFGS"]

    parameter_list = list(nn.parameters())

    if adapt_lambda not in valid_adapt_lambda:
        raise AttributeError(
            "Expected adaptive lambda mode one of: {}, actual: {}".format(valid_adapt_lambda, adapt_lambda))
    if should_plot not in valid_should_plot:
        raise AttributeError("Expected plotting mode one of: {}, actual: {}".format(valid_should_plot, should_plot))
    if optimizer not in valid_optimizer:
        raise AttributeError("Expected optimizer one of: {}, actual: {}".format(valid_optimizer, optimizer))
    if force_inclusion <= 0:
        raise AttributeError("Expected force_inclusion to be:  > 0 , got: {}".format(force_inclusion))
    if len(icnns) != (len(parameter_list) - 2) / 2:
        raise AttributeError("For each layer one ICNN is needed to be trained. "
                             "Amount provided: {}, expected: {}".format(len(icnns), (len(parameter_list) - 2) / 2))

    input_flattened = torch.flatten(input)
    eps_bounds = [input_flattened.add(-eps), input_flattened.add(eps)]
    box_bounds = nn.calculate_box_bounds(
        eps_bounds)  # todo abbrechen, wenn die box bounds schon die eigenschaft erfüllen

    included_space = torch.empty((0, input_flattened.size(0)), dtype=torch.float64)
    included_space = ds.samples_uniform_over(included_space, int(sample_count / 2), eps_bounds)
    ambient_space = torch.empty((0, input_flattened.size(0)), dtype=torch.float64)
    original_included_space = torch.empty((0, input_flattened.size(0)), dtype=torch.float64)
    original_ambient_space = torch.empty((0, input_flattened.size(0)), dtype=torch.float64)

    if should_plot in valid_should_plot:
        original_included_space, original_ambient_space = included_space, ambient_space

    center = input_flattened
    current_icnn = icnns[0]

    for i in range(0, len(parameter_list) - 2, 2):  # -2 because last layer has no ReLu activation
        current_layer_index = int(i / 2)
        current_icnn = icnns[current_layer_index]

        affine_w, affine_b = parameter_list[i], parameter_list[i + 1]

        if not keep_ambient_space:
            ambient_space = torch.empty((0, nn.layer_widths[current_layer_index]), dtype=torch.float64)

        if sample_over_input_space:
            if i == 0:
                ambient_space = ds.sample_uniform_excluding(ambient_space, int(sample_count / 2), eps_bounds,
                                                            excluding_bound=eps_bounds, padding=0.5)
            else:
                # todo test for when lower/upper bound is smaller then eps
                ambient_space = ds.sample_uniform_excluding(ambient_space, int(sample_count / 2),
                                                            box_bounds[current_layer_index - 1],
                                                            icnn=icnns[current_layer_index - 1],
                                                            padding=0.5)

        if should_plot == "detailed":
            plt_inc_amb("start " + str(i), included_space.tolist(), ambient_space.tolist())

        included_space = ds.apply_affine_transform(affine_w, affine_b, included_space)
        ambient_space = ds.apply_affine_transform(affine_w, affine_b, ambient_space)

        if should_plot in valid_should_plot:
            original_included_space = ds.apply_affine_transform(affine_w, affine_b, original_included_space)
            original_ambient_space = ds.apply_affine_transform(affine_w, affine_b, original_ambient_space)
            if should_plot == "detailed":
                plt_inc_amb("affin e" + str(i), included_space.tolist(), ambient_space.tolist())
                plt_inc_amb("original affin e" + str(i), original_included_space.tolist(),
                            original_ambient_space.tolist())

        included_space = ds.apply_ReLU_transform(included_space)
        ambient_space = ds.apply_ReLU_transform(ambient_space)

        if should_plot in valid_should_plot:
            original_included_space = ds.apply_ReLU_transform(original_included_space)
            original_ambient_space = ds.apply_ReLU_transform(original_ambient_space)
            if should_plot == "detailed":
                plt_inc_amb("relu " + str(i), included_space.tolist(), ambient_space.tolist())
                plt_inc_amb("original relu e" + str(i), original_included_space.tolist(),
                            original_ambient_space.tolist())

        if sample_over_output_space:
            ambient_space = ds.samples_uniform_over(ambient_space, int(sample_count / 2),
                                                    box_bounds[current_layer_index], padding=0.5)
            if should_plot in ["simple", "detailed"]:
                original_ambient_space = ds.samples_uniform_over(original_ambient_space, int(sample_count / 2),
                                                                 box_bounds[current_layer_index], padding=0.5)

        if should_plot == "detailed":
            plt_inc_amb("enhanced ambient space " + str(i), included_space.tolist(), ambient_space.tolist())
            plt_inc_amb("original enhanced ambient space " + str(i), original_included_space.tolist(),
                        original_ambient_space.tolist())

        mean = norm.get_mean(included_space, ambient_space)
        std = norm.get_std(included_space, ambient_space)
        normalized_included_space, normalized_ambient_space = norm.normalize_data(included_space, ambient_space, mean,
                                                                                  std)

        if optimizer == "LBFGS":  # todo does LBFGS support minibatch? I dont think so!
            icnn_batch_size = len(normalized_ambient_space) + len(normalized_included_space)
        dataset = ConvexDataset(data=normalized_included_space)
        train_loader = DataLoader(dataset, batch_size=icnn_batch_size, shuffle=False)
        dataset = ConvexDataset(data=normalized_ambient_space)
        ambient_loader = DataLoader(dataset, batch_size=icnn_batch_size,
                                    shuffle=False)  # todo shuffel alle wieder auf true setzen

        if init_network:
            low = box_bounds[current_layer_index][0]
            up = box_bounds[current_layer_index][1]
            low = torch.div(torch.add(low, -mean), std)
            up = torch.div(torch.add(up, -mean), std)
            current_icnn.init(low, up)

        # train icnn
        epochs_per_inclusion = icnn_epochs // force_inclusion
        epochs_in_last_inclusion = icnn_epochs % force_inclusion
        for k in range(force_inclusion):
            if k > 0:
                if k == force_inclusion - 1 and epochs_in_last_inclusion > 0:
                    epochs_per_inclusion = epochs_in_last_inclusion

                out = current_icnn(normalized_included_space)
                max_out = torch.max(out)
                current_icnn.apply_enlargement(max_out)

            if use_grad_descent:
                # todo sollte ich das noch dazu nehmen, dann wird ja ggf die anzahl samples
                #  doppelt so groß und es dauert länger
                untouched_normalized_ambient_space = normalized_ambient_space.detach().clone()
                if should_plot in ["simple", "detailed"]:
                    plt_inc_amb("without gradient descent", normalized_included_space.tolist(),
                                normalized_ambient_space.tolist())

                num_optimizations = 1
                optimization_steps = 1000

                epochs_per_optimization = epochs_per_inclusion // (num_optimizations + 1)
                modulo_epochs = epochs_per_inclusion % (num_optimizations + 1)

                for h in range(num_optimizations + 1):

                    if h < num_optimizations:
                        epochs_in_run = epochs_per_optimization
                    else:
                        epochs_in_run = epochs_per_optimization + modulo_epochs

                    train_icnn(current_icnn, train_loader, ambient_loader, epochs=epochs_in_run, hyper_lambda=0.5,
                               optimizer=optimizer, adapt_lambda=adapt_lambda, preemptive_stop=preemptive_stop)

                    if h < num_optimizations:
                        for v in range(optimization_steps):
                            # normalized_ambient_space =
                            # dop.gradient_descent_data_optim(current_icnn, normalized_ambient_space.detach())
                            normalized_ambient_space = dop.adam_data_optim(current_icnn,
                                                                           normalized_ambient_space.detach())
                        dataset = ConvexDataset(data=torch.cat(
                            [normalized_ambient_space.detach(), untouched_normalized_ambient_space.detach()]))
                        # todo hier muss ich noch verwalten was passiert wenn ambient space in die nächste runde
                        #  übernommen wird
                        if optimizer == "LBFGS":
                            icnn_batch_size = len(torch.cat([normalized_ambient_space.detach(),
                                                             untouched_normalized_ambient_space.detach()])) + len(
                                normalized_included_space)
                        ambient_loader = DataLoader(dataset, batch_size=icnn_batch_size, shuffle=False)

                if should_plot in ["simple", "detailed"]:
                    plt_inc_amb("with gradient descent", normalized_included_space.tolist(),
                                torch.cat([normalized_ambient_space.detach(),
                                           untouched_normalized_ambient_space.detach()]).tolist())
                    plots = Plots_for(0, current_icnn, normalized_included_space.detach(), torch.cat(
                        [normalized_ambient_space.detach(), untouched_normalized_ambient_space.detach()]).detach(),
                                      [-2, 3], [-2, 3])
                    plots.plt_mesh()
                normalized_ambient_space = untouched_normalized_ambient_space  # todo das ist vielleicht unnötig
            else:
                train_icnn(current_icnn, train_loader, ambient_loader, epochs=epochs_per_inclusion, hyper_lambda=1,
                           optimizer=optimizer, adapt_lambda=adapt_lambda, preemptive_stop=preemptive_stop)

        if train_outer:  # todo will ich train outer behalten oder einfach verwerfen?
            for k in range(icnn_epochs):
                train_icnn_outer(current_icnn, train_loader, ambient_loader, epochs=1)
                if k % 10 == 0:
                    plots = Plots_for(0, current_icnn, normalized_included_space.detach(),
                                      normalized_ambient_space.detach(),
                                      [-2, 3], [-2, 3])
                    plots.plt_mesh()

        current_icnn.apply_normalisation(mean, std)

        current_icnn.use_training_setup = False  # todo richtig integrieren in den code

        # matplotlib.use("TkAgg")
        if should_plot == "detailed":
            plots = Plots_for(0, current_icnn, included_space.detach(), ambient_space.detach(), [-1, 3], [-1, 3])
            plots.plt_dotted()
            plots.plt_mesh()
        elif should_plot == "simple":
            plots = Plots_for(0, current_icnn, included_space.detach(), ambient_space.detach(), [-1, 3], [-1, 3])
            plots.plt_dotted()
        elif should_plot == "verification":
            plots = Plots_for(0, current_icnn, included_space.detach(), ambient_space.detach(), [-1, 3], [-1, 3])
            plots.plt_mesh()

        # verify and enlarge convex approximation
        if use_over_approximation:
            if i == 0:
                adversarial_input, c = ver.verification(current_icnn, has_ReLU=True,
                                                        center_eps_W_b=[center.detach().numpy(), eps,
                                                                        affine_w.detach().numpy(),
                                                                        affine_b.detach().numpy()])
            else:
                prev_icnn = icnns[current_layer_index - 1]
                # prev_W, prev_b = parameter_list[i-2].detach().numpy(), parameter_list[i - 1].detach().numpy()
                adversarial_input, c = ver.verification(current_icnn, has_ReLU=True,
                                                        icnn_W_b_c=[prev_icnn, affine_w.detach().numpy(),
                                                                    affine_b.detach().numpy()])

            current_icnn.apply_enlargement(c)

            if should_plot == "detailed":
                plots = Plots_for(0, current_icnn, included_space.detach(), ambient_space.detach(), [-1, 3], [-1, 3])
                plots.plt_dotted()
                plots.plt_mesh()
            elif should_plot == "simple":
                plots = Plots_for(0, current_icnn, included_space.detach(), ambient_space.detach(), [-1, 3], [-1, 3])
                plots.plt_dotted()
            if should_plot == "verification":
                plots = Plots_for(0, current_icnn, included_space.detach(), ambient_space.detach(), [-1, 3], [-1, 3])
                plots.plt_mesh()

        # entweder oder:
        # nutze die samples weiter (dafür muss man dann das ReLU layer anwenden), und man muss schauen ob die
        # samples jetzt von ambient_space zu included_space gewechselt haben (wegen überapproximation)
        # damit könnte man gut den Fehler messen, da man ganz genau weiß wie viele elemente aus Ambient space
        # in Icluded space übergegenagen sind

        # oder sample neue punkte

        if sample_new:
            # todo entweder über box bounds sampeln oder über maximum radius
            included_space, ambient_space = ds.sample_max_radius(current_icnn, sample_count)

        else:
            included_space, ambient_space = ds.regroup_samples(current_icnn, included_space, ambient_space)

        if should_plot == "detailed":
            plt_inc_amb("sampled new", included_space.tolist(), ambient_space.tolist())
            plt_inc_amb("original end of layer", original_included_space.tolist(), original_ambient_space.tolist())

    if should_plot in valid_should_plot:
        index = len(parameter_list) - 2
        affine_w, affine_b = parameter_list[index], parameter_list[index + 1]
        included_space = ds.apply_affine_transform(affine_w, affine_b, included_space)
        ambient_space = ds.apply_affine_transform(affine_w, affine_b, ambient_space)
        original_included_space = ds.apply_affine_transform(affine_w, affine_b, original_included_space)
        plots = Plots_for(0, current_icnn, included_space.detach(), ambient_space.detach(), [-1, 3], [-1, 3],
                          extr=original_included_space.detach())
        plots.plt_initial()

    return icnns


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
