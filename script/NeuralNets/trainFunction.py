import time

import torch

from script.NeuralNets.lossFunction import deep_hull_simple_loss, deep_hull_loss
from script.NeuralNets.testFunction import test_icnn
from script.Optimizer.sdlbfgs import SdLBFGS
from script.settings import device, data_type

def train_icnn(model, train_loader, ambient_loader, epochs=10, optimizer="adam", return_history=False,
               sequential=False, force_convex=True,  adapt_lambda="none",  hyper_lambda=1, preemptive_stop=True, min_loss_change=1e-3, verbose=False, print_last_loss=False):
    history = []

    params_to_train = model.parameters()
    if optimizer == "adam":
        opt = torch.optim.Adam(params_to_train)
    elif optimizer == "LBFGS":
        opt = torch.optim.LBFGS(params_to_train)
    elif optimizer == "SdLBFGS":
        opt = SdLBFGS(params_to_train)

    loss = 0
    low = True
    window_size = 10
    moving_avg_loss = torch.zeros(window_size)
    zero_tensor = torch.Tensor([0]).to(device)
    for epoch in range(epochs):
        train_loss = 0
        train_n = 0
        if verbose:
            print("=== Epoch: {}===".format(epoch))
        epoch_start_time = time.time()
        for i, (X, X_ambient) in enumerate(zip(train_loader, ambient_loader)):
            if optimizer in ["LBFGS", "SdLBFGS"]:
                def closure():
                    opt.zero_grad()
                    prediction_ambient = model(X_ambient)
                    output = model(X)
                    loss = deep_hull_simple_loss(output, prediction_ambient, hyper_lambda=hyper_lambda)
                    loss.backward()
                    return loss
                loss = closure()
                opt.step(closure)

            else:
                prediction_ambient = model(X_ambient)
                output = model(X)
                loss = deep_hull_simple_loss(output, prediction_ambient, hyper_lambda=hyper_lambda)
                opt.zero_grad()
                loss.backward()
                opt.step()

            if force_convex:
                if not sequential:
                    with torch.no_grad():
                        for w in model.ws:
                            for p in w.parameters():
                                if len(p.size()) > 1:  # we have a matrix
                                    # only want positive entries
                                    p[:] = torch.maximum(zero_tensor, p)
                else:
                    with torch.no_grad():
                        for p in model.parameters():
                            if len(p.size()) > 1:  # we have a matrix
                                # only want positive entries
                                p[:] = torch.maximum(zero_tensor, p)

            last_loss = loss

            train_loss += loss.item()
            train_n += 1

            if return_history:
                history.append(train_loss / train_n)
            if verbose and i % 100 == 0:
                    print("batch = {}, mean loss = {}".format(i, train_loss / train_n))

        if train_n == 0:
            train_n = 1
        if verbose and i % 100 != 0:
            print("batch = {}, mean loss = {}".format(len(train_loader), train_loss / train_n))
            print("time per epoch: {}".format(time.time() - epoch_start_time))

        cyclic_index = epoch % window_size
        moving_avg_loss[cyclic_index] = loss
        avg_loss = torch.abs(moving_avg_loss - moving_avg_loss.mean()).max()
        if preemptive_stop and avg_loss <= min_loss_change:
            # print("preemptive stop at epoch {}".format(epoch))
            break


        if adapt_lambda == "none":
            continue
        elif adapt_lambda == "included":
            count_of_outside = 0
            for x in train_loader:
                out = model(x)
                for y in out:
                    if y > 0:
                        count_of_outside += 1

            percentage = count_of_outside / len(train_loader.dataset)

            if count_of_outside > 0:
                hyper_lambda = 1 - percentage
            else:
                hyper_lambda = 1
        elif adapt_lambda == "high_low":
            if low:
                hyper_lambda = 0.7
                low = False
            else:
                hyper_lambda = 1.2
                low = True

    if verbose or print_last_loss:
        print("Stop after {} Epochs".format(epoch))
        print("test for test setup")
        test_icnn(model, train_loader, ambient_loader, critic=deep_hull_simple_loss, hyper_lambda=hyper_lambda)

        print("test for without test setup")
        old_state = model.use_training_setup
        model.use_training_setup = False
        test_icnn(model, train_loader, ambient_loader, critic=deep_hull_simple_loss, hyper_lambda=hyper_lambda)
        model.use_training_setup = old_state
    if return_history:
        return history
