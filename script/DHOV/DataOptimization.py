import torch
from torch.utils.data import DataLoader

from script.dataInit import ConvexDataset


def gradient_descent_data_optim(icnn, samples):
    new_samples = torch.empty_like(samples, dtype=torch.float64)
    for h, elem in enumerate(samples):
        new_elem = torch.tensor(elem, dtype=torch.float64, requires_grad=True)
        inp = torch.unsqueeze(new_elem, dim=0)
        output = icnn(inp)
        target = torch.tensor([[0]], dtype=torch.float64)
        loss = torch.nn.MSELoss()(output, target)
        grad = torch.autograd.grad(loss, inp)
        lr = 0.001
        grad = torch.mul(grad[0], lr)
        new = torch.sub(inp, grad[0])
        new_samples[h] = new[0]
    return new_samples

def adam_data_optim(icnn, samples):

    #samples = [torch.tensor(samples.detach(), dtype=torch.float64, requires_grad=True)]
    new_samples = []
    for elem in samples:
        new_samples.append(torch.tensor([elem.tolist()], dtype=torch.float64, requires_grad=True))
    optimizer = torch.optim.Adam(new_samples)
    for h, elem in enumerate(new_samples):
        output = icnn(elem)
        target = torch.zeros_like(output, dtype=torch.float64)
        loss = torch.nn.MSELoss()(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    ret_samples = torch.empty_like(samples, dtype=torch.float64)
    for k, elem in enumerate(new_samples):
        ret_samples[k] = elem
    return ret_samples

