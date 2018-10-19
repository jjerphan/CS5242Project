import numpy as np
import os
import matplotlib.pyplot as plt
import logging
from abc import ABC, abstractmethod

from mpl_toolkits.mplot3d import Axes3D

from settings import float_type, comment_delimiter, training_examples_folder, length_cube_side, nb_features, \
    indices_features

logger = logging.getLogger('cnn.discretization')
logger.addHandler(logging.NullHandler())


class CubeRepresentation(ABC):

    def __init__(self, length_cube_side, use_rotation_invariance, translate_ligand, verbose):
        self._length_cube_side = length_cube_side
        self._use_rotation_invariance = use_rotation_invariance
        self._translate_ligand = translate_ligand
        self._verbose = verbose

    def _representation_invariance(self, original_coords, is_from_protein_indices):
        """
        Return an representation invariant of the system.

        The rotation is found using PCA on the protein molecules
        and is applied on all the atoms then.

        This way we are guaranteed that we can find a canonical representation
        of the system for all initial positions.

        There should be no pro

        :return: the canonical representation of the protein-ligand system
        """
        protein_coords = original_coords[is_from_protein_indices]
        columns_means = np.mean(protein_coords, axis=0)
        centered_protein_coords = protein_coords - columns_means

        # Getting the rotation matrix by diagonalizing the
        # covariance matrix of the centered protein coordinates
        cov_matrix = np.cov(centered_protein_coords.T)
        assert cov_matrix.shape == (3, 3)
        eigen_vals, rotation_mat = np.linalg.eig(cov_matrix)

        # Applying this rotation matrix on all points
        # Note : we should not transpose the rotation matrix (tested)
        new_coords = original_coords.dot(rotation_mat)

        # Should have the same shape
        assert new_coords.shape == original_coords.shape

        if self._verbose:
            print("Eigen Values (~ scale factor of molecules):"),
            print("x factor : ", eigen_vals[0])
            print("y factor : ", eigen_vals[1])
            print("z factor : ", eigen_vals[2])
            print("Rotation matrix")
            print(rotation_mat)

        return new_coords

    @staticmethod
    def translate_ligand_on_protein(original_coords, is_from_protein_indices):
        """
        Translate a ligand to the center of the protein.

        :param original_coords: the coordinates of the system
        :param is_from_protein_indices: the set of indices of the protein
        :return:
        """
        center_protein = np.mean(original_coords[is_from_protein_indices], axis=0)
        center_ligand = np.mean(original_coords[~is_from_protein_indices], axis=0)

        vector = center_ligand - center_protein

        original_coords[~is_from_protein_indices] = original_coords[~is_from_protein_indices] - vector

        return original_coords

    @staticmethod
    def plot_cube(cube):
        """
        Plot the cube representation of a given system of protein and ligand.

        :param cube: np.ndarray of size (res,res,res,2)
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        xs = []
        ys = []
        zs = []
        cs = []
        for x in range(length_cube_side):
            for y in range(length_cube_side):
                for z in range(length_cube_side):
                    is_from_protein_pos = indices_features["is_from_protein"] - 3
                    is_atom_in_voxel = cube[x, y, z, is_from_protein_pos] != 0
                    if is_atom_in_voxel:
                        # Plotting accordingly to the molecule type
                        color = 2 * cube[x, y, z, is_from_protein_pos] - 1
                        xs.append(x)
                        ys.append(y)
                        zs.append(z)
                        cs.append(color)

        ax.scatter(xs, ys, zs, c=cs, marker="o")

        ax.set_xlim((0, length_cube_side))
        ax.set_ylim((0, length_cube_side))
        ax.set_zlim((0, length_cube_side))

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        plt.show()

    @abstractmethod
    def make_cube(self, system: np.ndarray):
        pass


class AbsoluteCubeRepresentation(CubeRepresentation):
    """
    Class that construct an absolute cube representation.

    """

    def __init__(self, length_cube_side, use_rotation_invariance, translate_ligand, verbose, cube_resolution=1.0):
        """

        :param length_cube_side: length of the side of the cube to create
        :param use_rotation_invariance:  perform the canonical rotation of the PCA on the given points
        :param translate_ligand:
        :param verbose: utputs info about transformation
        :param cube_resolution: the resolution used for the cube (1 Å)
        """
        super.__init__(length_cube_side, use_rotation_invariance, translate_ligand, verbose)
        self._cube_resolution = cube_resolution
        self._cube_resolution = float(self._cube_resolution)

    def make_cube(self, system: np.ndarray):
        """
        Convert atom coordinates and features represented as 2D arrays into a
        fixed-sized 3D box.

        :param system: the protein-ligand system
        :return: a cube 4D np.ndarray of size (res, res, res, nb_features)
        """

        # Spatial coordinates of atoms
        original_coords = system[:, 0:3]

        if self._use_rotation_invariance:
            is_from_protein_column = indices_features["is_from_protein"]
            is_from_protein_indices = np.where(system[:, is_from_protein_column] == 1.)
            coords = self._representation_invariance(original_coords, is_from_protein_indices)
        else:
            coords = original_coords

        atom_features = system[:, 3:]
        nb_feat = atom_features.shape[1]

        assert nb_feat + coords.shape[1] == nb_features

        length_cube_side_int = int(length_cube_side)
        length_cube_side_float = float(length_cube_side)

        center = np.mean(coords, axis=0)

        centered_coords = coords - center

        translation_distance = length_cube_side_float / 2 * self._cube_resolution

        scaled_coords = (centered_coords + translation_distance) / self._cube_resolution
        scaled_coords = scaled_coords.round().astype(int)

        # Just keeping atom that are in the box
        in_box = ((scaled_coords >= 0) & (scaled_coords < length_cube_side)).all(axis=1)
        cube = np.zeros((length_cube_side_int, length_cube_side_int, length_cube_side_int, nb_feat), dtype=np.float32)
        for (x, y, z), f in zip(scaled_coords[in_box], atom_features[in_box]):
            cube[x, y, z] += f

        return cube


class RelativeCubeRepresentation(CubeRepresentation):
    """
    Class that constructs a relative cube representation.


    """

    def __init__(self, keep_proportions):
        """

        :param keep_proportions: should keep the proportion
        """
        self._keep_proportions = keep_proportions

    @staticmethod
    def _values_range(spatial_coordinates):
        """
        Return the extreme values for atoms coordinates.

        :param spatial_coordinates: np.ndarray of size (nb_atoms, 3)
        :return: extreme values for each coordinates x, y, z
        """
        x_min = np.min(spatial_coordinates[:, 0])
        y_min = np.min(spatial_coordinates[:, 1])
        z_min = np.min(spatial_coordinates[:, 2])

        x_max = np.max(spatial_coordinates[:, 0])
        y_max = np.max(spatial_coordinates[:, 1])
        z_max = np.max(spatial_coordinates[:, 2])

        return x_min, x_max, y_min, y_max, z_min, z_max

    def make_cube(self, system: np.ndarray):
        """
            Creating a cube from a system.

            Using a rotation invariance representation and keeping the proportions gives better results normaly.

            :param system: the protein-ligand system
            :return: a cube 4D np.ndarray of size (res, res, res, nb_features)
            """

        # Spatial coordinates of atoms
        original_coords = system[:, 0:3]

        if self._use_rotation_invariance:
            is_from_protein_column = indices_features["is_from_protein"]
            is_from_protein_indices = np.where(system[:, is_from_protein_column] == 1.)
            coords = self._representation_invariance(original_coords, is_from_protein_indices)
        else:
            coords = original_coords

        atom_features = system[:, 3:]
        nb_feat = atom_features.shape[1]

        assert nb_feat + coords.shape[1] == nb_features

        # Getting extreme values
        x_min, x_max, y_min, y_max, z_min, z_max = self._values_range(coords)

        # Finding the maximum range between extreme points on each coordinates
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min

        # If we want to keep the proportion, we can scale adequately
        if self._keep_proportions:
            max_range = max([x_range, y_range, z_range])
            x_range = max_range
            y_range = max_range
            z_range = max_range

        # Scaling coordinates to be in the cube [0,res]^3 then flooring

        scaled_coords = (coords * 0).astype(int)
        eps = 10e-4  # To be sure to round down on exact position
        scaled_coords[:, 0] = np.floor((coords[:, 0] - x_min) / (x_range + eps) * length_cube_side).astype(int)
        scaled_coords[:, 1] = np.floor((coords[:, 1] - y_min) / (y_range + eps) * length_cube_side).astype(int)
        scaled_coords[:, 2] = np.floor((coords[:, 2] - z_min) / (z_range + eps) * length_cube_side).astype(int)

        cube = np.zeros((length_cube_side, length_cube_side, length_cube_side, nb_feat))
        logger.debug("Cube size is %s", str(cube.shape))

        # Filling the cube with the features
        for (x, y, z), f in zip(scaled_coords, atom_features):
            cube[x, y, z] += f

        return cube


def load_nparray(file_name: str):
    """
    Loads an numpy ndarray stored in given file
    :param file_name: the file to use
    :return:
    """

    example = np.loadtxt(file_name, dtype=float_type, comments=comment_delimiter)
    # If it's a vector (i.e if there is just one atom),
    # we reshape it into a (1,n nb_features) array
    if len(example.shape) == 1:
        example = example.reshape(1, -1)

    return example