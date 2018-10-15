import numpy as np
import os
import matplotlib.pyplot as plt
import logging
from mpl_toolkits.mplot3d import Axes3D  # needed by the 3D plotter

from discretization import load_nparray
from settings import float_type, comment_delimiter, training_examples_folder, length_cube_side, nb_features

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

        ax.set_xlim((0, length_cube_side))
        ax.set_ylim((0, length_cube_side))
        ax.set_zlim((0, length_cube_side))

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        fig.show()


if __name__ == "__main__":

    # Just to test
    examples_files = sorted(os.listdir(training_examples_folder))
    for ex_file in examples_files:
        file_name = os.path.join(training_examples_folder, ex_file)
        example = load_nparray(os.path.join(training_examples_folder, ex_file))

        cube = Cube(example, length_cube_side)
        cube.make_cube()
        cube.plot_cube()

        input()
