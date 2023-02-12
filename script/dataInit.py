import numpy as np
import torch
from scipy.spatial import ConvexHull

from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from script.settings import device, data_type

class Multivariate():
    def get_samples(self, offset, number_of_samples, x_range, y_range):
        rng = np.random.default_rng()
        included_space = np.empty((0, 2))
        subtract_space = None

        # Gaußverteilung mit Loch in der Seite
        # included_space = rng.multivariate_normal([0, 0], [[1, 0], [0, 1]], size=number_of_included_samples)
        # included_space = np.append(included_space,
        #                           rng.multivariate_normal([5, 5], [[1, 0], [0, 1]], size=number_of_included_samples),
        #                          axis=0)

        # Gaußverteilung mit Loch in der Mitte
        # included_space = rng.multivariate_normal([0, 0], [[5, 0], [0, 10]], size=number_of_included_samples)
        # subtract_space = rng.multivariate_normal([3, 3], [[3, 0], [0, 3]], size=number_of_included_samples)

        included_space = rng.multivariate_normal([offset, offset], [[5, 0], [0, 10]], size=number_of_samples)
        subtract_space = rng.multivariate_normal([offset, offset], [[1, 0], [0, 1]], size=int(number_of_samples/10))

        label_in = [[1,0]]
        label_out = [[0,1]]

        subtract_hull = ConvexHull(subtract_space)
        A2, b2 = subtract_hull.equations[:, :-1], subtract_hull.equations[:, -1:]

        if subtract_space is not None:
            deleted_points = []
            for i, x in enumerate(included_space):
                if self.contained(x, A2, b2):
                    deleted_points.append(i)
            included_space = np.delete(included_space, deleted_points, axis=0)

        hull = ConvexHull(included_space)
        A, b = hull.equations[:, :-1], hull.equations[:, -1:]
        ambient = []
        rng = np.random.default_rng()
        while len(ambient) < number_of_samples:
            x_cord = rng.uniform(*x_range)
            y_cord = rng.uniform(*y_range)
            if not self.contained([[x_cord, y_cord]], A, b):
                ambient.append([x_cord, y_cord])

        ambient_space = np.array(ambient)

        included_space = torch.from_numpy(included_space).to(data_type).to(device)
        ambient_space = torch.from_numpy(ambient_space).to(data_type).to(device)

        return included_space, ambient_space

    def contained(self, x, A, b):
        eps = np.finfo(np.float32).eps

        return np.all(np.asarray(x) @ A.T + b.T < eps, axis=1)


class Polytope(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_A(self):
        pass

    @abstractmethod
    def get_b(self):
        pass

    @abstractmethod
    def get_dimension(self):
        pass

    @abstractmethod
    def get_uniform_samples(self, number_of_included_samples, x_range, y_range):
        pass

    def get_extremal_points(self):
        pass


class Rhombus(Polytope):
    def __init__(self):
        self.A = np.array([
            [1, 1],
            [-1, -1],
            [-1, 1],
            [1, -1]
        ])

        self.b = np.array([1, 1, 1, 1])

    def get_A(self):
        return self.A

    def get_b(self):
        return self.b

    def get_dimension(self):
        return 2

    def get_uniform_samples(self, number_of_samples, x_range, y_range):
        xs = np.random.uniform([x_range[0], y_range[0]], [x_range[1], y_range[1]], (number_of_samples, 2))

        included_space = np.empty((0, 2))
        ambient_space = np.empty((0, 2))

        for x in xs:
            if self.f(x):
                included_space = np.append(included_space, [x], axis=0)
            else:
                ambient_space = np.append(ambient_space, [x], axis=0)

        included_space = torch.from_numpy(included_space).to(torch.float64)
        ambient_space = torch.from_numpy(ambient_space).to(torch.float64)

        return included_space, ambient_space

    def f(self, x):
        solution = self.A.dot(x) <= self.b

        for sol in solution:
            if not sol:
                return False

        return True

    def get_extremal_points(self):
        return [[-1, 0], [0, -1], [1, 0], [0, 1]]


class ConvexDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]
