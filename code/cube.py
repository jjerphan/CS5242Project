import logging

import matplotlib.pyplot as plt
import numpy as np

from settings import LENGTH_CUBE_SIDE

logger = logging.getLogger('cnn.discretization')
logger.addHandler(logging.NullHandler())


class Cube:
    """
    Cube object consist of x, y, z coordinates, atom type (hydrophobic, polar) and origin (protein, ligand).
    """

    def __init__(self, protein_file, resolution):
        self.x = protein_file[:, 0]
        self.y = protein_file[:, 1]
        self.z = protein_file[:, 2]
        self.atom_features = protein_file[:, 3:]
        self.resolution = resolution

    def find_coord_min_max(self):
        """
        Get minimum and maximum values of x, y, z. Then the range of x, y, z.
        :return: x_min, y_min, z_min, x_range, y_range, z_range
        """
        x_min = np.min(self.x)
        x_max = np.max(self.x)
        y_min = np.min(self.y)
        y_max = np.max(self.y)
        z_min = np.min(self.z)
        z_max = np.max(self.z)
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min
        return (x_min, y_min, z_min, x_range, y_range, z_range)

    def __scale(self):
        x_min, y_min, z_min, x_range, y_range, z_range = self.find_coord_min_max()
        eps = 10e-4
        self.x = np.floor((self.x - x_min) / (x_range + eps) * self.resolution).astype(int)
        self.y = np.floor((self.y - y_min) / (y_range + eps) * self.resolution).astype(int)
        self.z = np.floor((self.z - z_min) / (z_range + eps) * self.resolution).astype(int)

    def make_cube(self):
        self.__scale()
        cube = np.zeros((self.resolution, self.resolution, self.resolution, self.atom_features.shape[1]))
        cube[self.x[:], self.y[:], self.z[:]] = self.atom_features
        return cube

    def plot_cube(self):
        print("Plotting the cube")
        atom_type = self.atom_features[:, 0]
        origin = self.atom_features[:, 1]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(self.x, self.y, self.z, c=origin, marker="o")

        ax.set_xlim((0, LENGTH_CUBE_SIDE))
        ax.set_ylim((0, LENGTH_CUBE_SIDE))
        ax.set_zlim((0, LENGTH_CUBE_SIDE))

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        fig.show()
