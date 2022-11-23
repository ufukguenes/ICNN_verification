import torch


def deep_hull_loss(model_output, adversarial_output, hyper_lambda=1):
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

    def l_generative(z):
        sigmoid = torch.sigmoid(z)
        loss = torch.sum(sigmoid) / len(z)
        return loss

    a = l_pos(model_output)
    b = hyper_lambda * l_neg(adversarial_output)
    c = l_generative(adversarial_output)
    out = a + b + c, a, b, c
    return out


def deep_hull_simple_loss(model_output, ambient_space, hyper_lambda=1):
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

    z = ambient_space
    a = l_pos(model_output)
    b = hyper_lambda * l_neg(z)
    out = a + b

    return out
