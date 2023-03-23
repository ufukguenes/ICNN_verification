import random
import torch
from script.settings import device, data_type


def gradient_descent_data_optim(icnn, samples):
    new_samples = torch.empty_like(samples, dtype=data_type).to(device)
    for h, elem in enumerate(samples):
        new_elem = torch.tensor(elem, dtype=data_type, requires_grad=True).to(device)
        inp = torch.unsqueeze(new_elem, dim=0)
        output = icnn(inp)
        target = torch.tensor([[0]], dtype=data_type).to(device)
        loss = torch.nn.MSELoss()(output, target)
        grad = torch.autograd.grad(loss, inp)
        lr = 0.001
        grad = torch.mul(grad[0], lr)
        new = torch.sub(inp, grad[0])
        new_samples[h] = new[0]
    return new_samples


def sgd_data_optim(icnn, samples):
    samples = [torch.tensor(samples.detach(), dtype=data_type, requires_grad=True).to(device)]
    optimizer = torch.optim.SGD(samples, lr=0.001)
    for h, elem in enumerate(samples):
        output = icnn(elem)
        target = torch.zeros_like(output, dtype=data_type).to(device)
        loss = torch.nn.MSELoss()(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return samples[0]


def adam_data_optim(icnn, samples):
    samples = [torch.tensor(samples.detach(), dtype=data_type, requires_grad=True).to(device)]
    optimizer = torch.optim.Adam(samples)
    for h, elem in enumerate(samples):
        output = icnn(elem)
        target = torch.zeros_like(output, dtype=data_type).to(device)
        loss = torch.nn.MSELoss()(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return samples[0]


def test_all(icnn, all_sample, threshold=0.02):
    avg = torch.zeros(0, dtype=data_type).to(device)
    num_sample = 10
    # num_sample = len(all_sample)

    for i in range(num_sample):
        rand = random.randint(0, len(all_sample) - 1)
        rand_sample = all_sample[rand]
        is_, val = even_gradient(icnn, rand_sample, threshold=0.02)
        if is_ or val > 0:
            avg = torch.cat([avg, val])
    avg_out = avg.sum() / len(avg)
    if avg.size(0) < num_sample:
        return False, avg_out
    out = avg_out < threshold
    return out, avg_out


def even_gradient(icnn, sample, threshold=0.001):
    sample = torch.unsqueeze(sample, dim=0)
    original_grad, original_out = get_grad_output(icnn, sample)

    if original_out >= 0:
        return False, 0

    first_step_size = 1 / torch.linalg.norm(original_grad[0])
    point_1 = line_search(icnn, original_grad[0], sample, low=0, up=first_step_size)
    point_2 = line_search(icnn, -1 * original_grad[0], sample, low=0, up=first_step_size)

    grad_1, out_1 = get_grad_output(icnn, point_1)
    grad_2, out_2 = get_grad_output(icnn, point_2)

    lr = 10
    grad_1 = torch.mul(grad_1[0], lr)
    grad_2 = torch.mul(grad_2[0], lr)
    new_1 = torch.add(point_1, grad_1[0])
    new_2 = torch.add(point_2, grad_2[0])
    new_out_1 = icnn(new_1)
    new_out_2 = icnn(new_2)
    # return True, ((point_1, new_1), (point_2, new_2))

    out = abs(new_out_1 - new_out_2) < threshold
    loss = torch.nn.MSELoss()(new_out_1 - new_out_2, torch.zeros_like(new_out_1, dtype=data_type).to(device))
    return out, torch.unsqueeze(loss, dim=0)


def get_grad_output(icnn, sample):
    output = icnn(sample)
    target = torch.tensor([[0]], dtype=data_type).to(device)
    loss = torch.nn.MSELoss()(output, target)
    grad = torch.autograd.grad(loss, sample)

    return grad, output


def line_search(icnn, grad, x, low=0.0, up=1.0):
    new_grad = torch.mul(grad, up)
    new_x = torch.add(new_grad, x)
    new_out = icnn(new_x)
    if new_out < 0:
        return line_search(icnn, grad, x, low=up, up=2 * up)

    new_grad = torch.mul(grad, low)
    new_x = torch.add(new_grad, x)
    new_out = icnn(new_x)
    if new_out > 0:
        return line_search(icnn, grad, x, low=low / 2, up=low)

    middle = (up + low) / 2
    new_grad = torch.mul(grad, middle)
    new_x = torch.add(new_grad, x)
    new_out = icnn(new_x)

    if 0 < new_out < 0 + 1e-2:
        return new_x
    if new_out < 0:
        low = middle
        return line_search(icnn, grad, x, low=low, up=up)
    if new_out > 0:
        up = middle
        return line_search(icnn, grad, x, low=low, up=up)
