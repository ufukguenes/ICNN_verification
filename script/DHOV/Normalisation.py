import torch
import torchvision as tv


def get_mean(included_space, ambient_space):
    mean = torch.mean(torch.cat((included_space, ambient_space)), dim=0)
    mean = mean.detach().requires_grad_(True)
    return mean


def get_std(included_space, ambient_space):
    std = torch.std(torch.cat((included_space, ambient_space)), dim=0)
    std = std.detach().requires_grad_(True)
    return std


def normalize_data(included_space, ambient_space, mean, std):
    std = std.detach()
    for i, val in enumerate(std):
        if val == 0:
            std[i] = 1

    ambient_space_transform = tv.transforms.Lambda(lambda x: (x - mean) / std)(ambient_space)
    included_space_transform = tv.transforms.Lambda(lambda x: (x - mean) / std)(included_space)
    return included_space_transform.detach().requires_grad_(True), ambient_space_transform.detach().requires_grad_(True)
