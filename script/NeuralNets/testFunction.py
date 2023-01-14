import torch
from torch import nn

from script.NeuralNets.lossFunction import deep_hull_simple_loss, identity_loss


def test_icnn(model, train_loader, ambient_loader, critic=deep_hull_simple_loss, hyper_lambda=1):
    test_loss = 0
    test_n = 0
    model.eval()
    with torch.no_grad():
        for i, (X, X_ambient) in enumerate(zip(train_loader, ambient_loader)):
            prediction_ambient = model(X_ambient)
            output = model(X)
            loss = critic(output, prediction_ambient, hyper_lambda=hyper_lambda)

            test_loss += loss.item()
            test_n += 1

    print("test run {} iterations,  with mean loss = {}".format(test_n, test_loss / test_n))


def test_sequential(model, train_loader, ambient_loader):
    test_loss = 0
    test_n = 0
    model.eval()
    with torch.no_grad():
        for i, (x_included, x_ambient) in enumerate(zip(train_loader, ambient_loader)):
            output_included = model(x_included)
            output_ambient = model(x_ambient)

            loss = identity_loss(output_included, output_ambient, x_included, x_ambient)

            test_loss += loss.item()
            test_n += 1

    print("test run {} iterations,  with mean loss = {}".format(test_n, test_loss / test_n))


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
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
