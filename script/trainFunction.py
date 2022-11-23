from script.lossFunction import *


def train_icnn(model, train_loader, ambient_loader, criterion=deep_hull_simple_loss, epochs=10, opt=None,
               return_history=False, sequential=False):
    history = []
    if opt is None:
        opt = torch.optim.Adam(model.parameters())

    for epoch in range(epochs):
        train_loss = 0
        train_n = 0

        print("=== Epoch: {}===".format(epoch))

        for i, (X, X_ambient) in enumerate(zip(train_loader, ambient_loader)):

            prediction_ambient = model(X_ambient)
            output = model(X)
            loss = criterion(output, prediction_ambient)
            opt.zero_grad()
            loss.backward()
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


def train_icnn_adversarial(model, adversarial, train_loader, adversarial_loader,
                           criterion=deep_hull_loss, epochs=10,
                           opt_model=None, opt_adv=None, return_history=False):
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

            loss, a, b, c = criterion(output, prediction_from_adv)

            opt_model.zero_grad()
            opt_adv.zero_grad()

            loss.backward()

            opt_model.step()
            opt_adv.step()

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


