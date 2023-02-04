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


def normalize_nn(nn, mean, std, isICNN=False, with_logical=False):
    if not isICNN:
        with torch.no_grad():
            parameter_list = list(nn.parameters())
            parameter_list[0].data = torch.div(parameter_list[0], std)
            parameter_list[1].data = torch.add(- torch.matmul(parameter_list[0], mean), parameter_list[1])

    else:
        with torch.no_grad():
            parameter_list = list(nn.ws[0].parameters())
            parameter_list[0].data = torch.div(parameter_list[0], std)
            parameter_list[1].data = torch.add(- torch.matmul(parameter_list[0], mean), parameter_list[1])

            k = len(nn.us)
            l = len(nn.ws)
            for i in range(len(nn.us)):
                parameter_list = list(nn.us[i].parameters())
                parameter_list[0].data = torch.div(parameter_list[0], std)

                internal_parameter_list = list(nn.ws[i + 1].parameters())
                internal_parameter_list[1].data = torch.add(- torch.matmul(parameter_list[0], mean),
                                                            internal_parameter_list[1])
            if with_logical:
                parameter_list = list(nn.ls[0].parameters())
                parameter_list[0].data = torch.div(parameter_list[0], std)
                parameter_list[1].data = torch.add(- torch.matmul(parameter_list[0], mean), parameter_list[1])


def normalize_data(included_space, ambient_space, mean, std):
    std = std.detach()
    for i, val in enumerate(std):
        if val == 0:
            std[i] = 1 # todo this cant be right

    ambient_space_transform = tv.transforms.Lambda(lambda x: (x - mean) / std)(ambient_space)
    included_space_transform = tv.transforms.Lambda(lambda x: (x - mean) / std)(included_space)
    return included_space_transform.detach().requires_grad_(True), ambient_space_transform.detach().requires_grad_(True)