import torch

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


def sgd_data_optim(icnn, samples):
    samples = [torch.tensor(samples.detach(), dtype=torch.float64, requires_grad=True)]
    optimizer = torch.optim.SGD(samples, lr=0.001)
    for h, elem in enumerate(samples):
        output = icnn(elem)
        target = torch.zeros_like(output, dtype=torch.float64)
        loss = torch.nn.MSELoss()(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return samples[0]


def adam_data_optim(icnn, samples):

    samples = [torch.tensor(samples.detach(), dtype=torch.float64, requires_grad=True)]
    optimizer = torch.optim.Adam(samples)
    for h, elem in enumerate(samples):
        output = icnn(elem)
        target = torch.zeros_like(output, dtype=torch.float64)
        loss = torch.nn.MSELoss()(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return samples[0]

