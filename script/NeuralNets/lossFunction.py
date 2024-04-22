import torch
from script.settings import device, data_type

def deep_hull_loss(model_output, output_adv, adversarial_output, hyper_lambda=1, use_max_distance=False):
    def l_pos(x):
        loss = - torch.log(1 - torch.sigmoid(x))
        loss = torch.sum(loss)
        ret = loss / len(x)
        return ret

    def l_neg(z):
        loss = - torch.log(torch.sigmoid(z))
        loss = torch.sum(loss)
        ret = loss / len(z)
        return ret

    def l_generative_deep_hull(z):
        sigmoid = torch.sigmoid(z)
        loss = torch.sum(sigmoid) / len(z)
        return loss

    def l_generative_max_distance(z, x):
        sqrt = torch.sqrt(torch.abs(z))
        flipped = torch.flip(x, [0])
        distance = torch.split(torch.cdist(x, flipped), 2)
        log = - torch.log(torch.sigmoid(distance[0]))
        add = torch.sum(sqrt) + 10 * torch.sum(log)
        return add / len(z)

    a = l_pos(model_output)
    if use_max_distance:
        b = torch.tensor([0], dtype=data_type).to(device)
        c = l_generative_max_distance(adversarial_output, output_adv)
    else:
        b = hyper_lambda * l_neg(adversarial_output)
        c = l_generative_deep_hull(adversarial_output)

    out = a + b + c, a, b, c
    return out


def deep_hull_simple_loss(model_output, ambient_space, hyper_lambda=1):
    def l_pos(x):
        sig = torch.sigmoid(x)
        max_value = torch.zeros_like(sig).add(1 - 1e-12)
        sig = torch.minimum(sig, max_value)

        loss = - torch.log(1 - sig)
        loss = torch.sum(loss)
        ret = loss / len(x)
        return ret

    def l_neg(z):
        sig = torch.sigmoid(z)
        min_value = torch.zeros_like(sig).add(1e-12)
        sig = torch.maximum(sig, min_value)

        loss = - torch.log(sig)
        loss = torch.sum(loss)
        ret = loss / len(z)
        return ret

    a = l_pos(model_output)
    b = hyper_lambda * l_neg(ambient_space)
    out = a + b

    return out
