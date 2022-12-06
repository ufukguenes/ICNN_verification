import time

from script.lossFunction import *
import torch.nn as nn


def train_icnn(model, train_loader, ambient_loader, epochs=10, opt=None,
               return_history=False, sequential=False, hyper_lambda=1):
    history = []
    if opt is None:
        opt = torch.optim.Adam(model.parameters())

    for epoch in range(epochs):
        train_loss = 0
        train_n = 0

        print("=== Epoch: {}===".format(epoch))


        for i, (X, X_ambient) in enumerate(zip(train_loader, ambient_loader)):
            model.double()
            X, X_ambient = X.double(), X_ambient.double()
            prediction_ambient = model(X_ambient)
            output = model(X)
            loss = deep_hull_simple_loss(output, prediction_ambient, hyper_lambda=hyper_lambda)
            opt.zero_grad()
            loss.backward()
            model.float()
            opt.step()

            if not sequential:
                with torch.no_grad():
                    for w in model.ws:
                        for p in w.parameters():
                            if len(p.size()) > 1:  # we have a matrix
                                # only want positive entries
                                p[:] = torch.maximum(torch.Tensor([0]), p)
            else:
                with torch.no_grad():
                    for p in model.parameters():
                        if len(p.size()) > 1:  # we have a matrix
                            # only want positive entries
                            p[:] = torch.maximum(torch.Tensor([0]), p)

            train_loss += loss.item()
            train_n += 1

            if return_history:
                history.append(train_loss / train_n)

            if i % 100 == 0:
                print("batch = {}, mean loss = {}".format(i, train_loss / train_n))

        print("batch = {}, mean loss = {}".format(len(train_loader), train_loss / train_n))

    if return_history:
        return history


def train_icnn_adversarial(model, adversarial, train_loader, adversarial_loader, epochs=10, train_ICNN=True,
                           opt_model=None, opt_adv=None, return_history=False, hyper_lambda=1, use_max_distance=False):
    history = []
    model.train()
    adversarial.train()
    if opt_model is None:
        opt_model = torch.optim.Adam(model.parameters())

    if opt_adv is None:
        opt_adv = torch.optim.Adam(adversarial.parameters())

    for epoch in range(epochs):
        train_loss = 0
        train_n = 0
        l_pos, l_neg, l_gen = 0, 0, 0

        print("=== Epoch: {}===".format(epoch))

        for i, (X, X_adv) in enumerate(zip(train_loader, adversarial_loader)):
            output_adv = adversarial(X_adv)

            prediction_from_adv = model(output_adv)
            output = model(X)

            loss, a, b, c = deep_hull_loss(output, output_adv, prediction_from_adv, hyper_lambda=hyper_lambda, use_max_distance=use_max_distance)

            opt_model.zero_grad()
            opt_adv.zero_grad()

            loss.backward()

            opt_adv.step()

            if train_ICNN:
                opt_model.step()
                with torch.no_grad():
                    for w in model.ws:
                        for p in w.parameters():
                            if len(p.size()) > 1:  # we have a matrix
                                # only want positive entries
                                p[:] = torch.maximum(torch.Tensor([0]), p)

            train_loss += loss.item()
            train_n += 1
            l_pos += a.item()
            l_neg += b.item()
            l_gen += c.item()

            if return_history:
                history.append(train_loss / train_n)

            if i % 100 == 0:
                print("batch = {}, mean loss = {}, l_pos = {}, l_neg = {}, l_gen = {}".
                      format(i, train_loss / train_n, l_pos / train_n, l_neg / train_n, l_gen / train_n))

        print("batch = {}, mean loss = {}, l_pos = {}, l_neg = {}, l_gen = {}".
              format(len(train_loader), train_loss / train_n, l_pos / train_n, l_neg / train_n, l_gen / train_n))

    if return_history:
        return history


def train_sequential(model, train_data, test_data, loss_fn=nn.CrossEntropyLoss(), optimizer=None, epochs=10):
    if optimizer is None:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    size = len(train_data.dataset)
    model.train()

    for epoch in range(epochs):

        print("=== Epoch: {}===".format(epoch))
        current = time.time()
        for batch, (X, y) in enumerate(train_data):
            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch % 2000 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        test(model, test_data, loss_fn=loss_fn)
        current = time.time() - current
        print("Time: {}".format(current))


def test(model, dataloader, loss_fn=nn.CrossEntropyLoss()):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
