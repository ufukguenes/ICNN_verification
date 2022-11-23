import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull

class Plots_for():
    # matplotlib.use('TkAgg')

    in_convex = []
    not_in_convex = []
    c = 0
    hull = 0
    ambient_space = []
    included_space = []
    model: torch.nn.Module
    advers_model: torch.nn.Module
    true_extremal = 0
    x_range = []
    y_range = []
    adversarial = None
    adversarial_values = None

    def __init__(self, c_val, model_local, inc_space, amb_space, extr, range_x, range_y, adversarial=None, adversarial_values=None):

        self.x_range = range_x
        self.y_range = range_y
        self.c = c_val
        self.model = model_local
        self.ambient_space = amb_space
        self.included_space = inc_space
        self.true_extremal = extr

        self.hull = ConvexHull(self.included_space)

        self.in_convex = []
        self.not_in_convex = []

        if adversarial is not None:
            self.adversarial = adversarial

        if adversarial_values is not None:
            self.adversarial_values = adversarial_values



    def _create_plot_convex_hull(self):
        for simplex in self.hull.simplices:
            plt.plot(self.included_space[simplex, 0], self.included_space[simplex, 1], 'k-')

    def _create_plot_model(self):
        self.in_convex = []
        self.not_in_convex = []

        def test_contained(x):
            bool_val = False
            if torch.is_tensor(self.c):
                bool_val = self.model(x).less_equal(self.c)
                for val in torch.flatten(bool_val):
                    if not val:
                        bool_val = False
                        break
                bool_val = True
            else:
                bool_val = self.model(x) <= self.c
            return bool_val

        for x in self.included_space:
            x = torch.unsqueeze(x, 0)
            bool_val = test_contained(x)

            if bool_val:
                self.in_convex.append(x.numpy())
            else:
                self.not_in_convex.append(x.numpy())

        for x in self.ambient_space:
            x = torch.unsqueeze(x, 0)
            bool_val = test_contained(x)
            if bool_val:
                self.in_convex.append(x.numpy())
            else:
                self.not_in_convex.append(x.numpy())

        self.in_convex = np.asarray(self.in_convex)
        self.not_in_convex = np.asarray(self.not_in_convex)

        if self.in_convex.size > 0:
            plt.scatter(self.in_convex[:, 0, 0], self.in_convex[:, 0, 1])
        if self.not_in_convex.size > 0:
            plt.scatter(self.not_in_convex[:, 0, 0], self.not_in_convex[:, 0, 1])

    def _create_plot_true_convex_hull(self):
        x = []
        y = []
        for vertex in self.true_extremal:
            x.append(vertex[0])
            y.append(vertex[1])
        x.append(self.true_extremal[0][0])
        y.append(self.true_extremal[0][1])
        plt.plot(x, y, 'ro-')

    def plt_initial(self):
        fig = plt.figure(figsize=(20, 10))

        self._create_plot_convex_hull()
        plt.scatter(self.included_space[:, 0], self.included_space[:, 1])
        plt.scatter(self.ambient_space[:, 0], self.ambient_space[:, 1])
        self._create_plot_true_convex_hull()

        plt.show()

    def plt_dotted(self):
        in_convex = np.asarray(self.in_convex)
        not_in_convex = np.asarray(self.not_in_convex)

        fig = plt.figure(figsize=(20, 10))

        self._create_plot_convex_hull()
        self._create_plot_model()
        self._create_plot_true_convex_hull()
        plt.show()

    def plt_mesh(self):
        fig = plt.figure(figsize=(20, 10), facecolor="w")
        x = np.linspace(*self.x_range, 500)
        y = np.linspace(*self.y_range, 500)
        xx, yy = np.meshgrid(x, y)
        x_in = torch.tensor(np.c_[xx.ravel(), yy.ravel()])

        y_pred = self.model(x_in)
        y_pred = np.round(y_pred.detach().numpy(), decimals=5).reshape(xx.shape)

        plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)

        plt.colorbar()

        self._create_plot_convex_hull()
        self._create_plot_model()
        self._create_plot_true_convex_hull()
        plt.show()

    def plt_adversarial_initial(self):
        if self.adversarial is None or self.adversarial_values is None:
            return

        pred_x_arr = []
        pred_y_arr = []
        fig = plt.figure(figsize=(20, 10))
        ax = plt.axes()
        ax.set_xlim([self.x_range[0], self.x_range[1]])
        ax.set_ylim([self.y_range[0], self.y_range[1]])

        for v in self.adversarial_values:
            v = torch.asarray(torch.flatten(v))
            pred_x_arr.append(v[0])
            pred_y_arr.append(v[1])

        plt.scatter(pred_x_arr, pred_y_arr)
        plt.show()

    def plt_adversarial_dotted(self):
        if self.adversarial is None or self.adversarial_values is None:
            return

        pred_x_arr = []
        pred_y_arr = []
        fig = plt.figure(figsize=(20, 10))

        for x in self.adversarial_values:
            pred = self.adversarial(x)
            pred = torch.asarray(torch.flatten(pred))
            pred_x_arr.append(pred[0].item())
            pred_y_arr.append(pred[1].item())

        plt.scatter(pred_x_arr, pred_y_arr)

        x = np.linspace(*self.x_range, 100)
        y = np.linspace(*self.y_range, 100)
        xx, yy = np.meshgrid(x, y)
        x_in = torch.tensor(np.c_[xx.ravel(), yy.ravel()])

        y_pred = self.adversarial(x_in)
        rounded = np.round(y_pred.detach().numpy(), decimals=5)
        x_s = rounded[:, 0]
        y_s = rounded[:, 1]

        plt.scatter(x_s, y_s)

        plt.show()
