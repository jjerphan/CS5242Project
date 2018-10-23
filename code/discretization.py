import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

from settings import LENGTH_CUBE_SIDE, NB_FEATURES, \
    INDICES_FEATURES


class CubeRepresentation(ABC):
    """
    Class that is used to return a specific cube representation of an example of
    a protein-ligand system.

    A cube is composed of voxels storing information about atom of the system present in it.
    The number of voxels is here parametrized by at one parameter: the length of its sides `length_cube_side`.

    `length_cube_side` represent the number of voxel on one direction.

    Hence, as we are using a 3D cube, there is exactly length_cube_side^3 voxels in the representation.

    The representation makes a cube using the method make_cube.

    When making such a cube:
     - the system can be arranged in a rotation invariance position (that is, if identical system that are rotated
     initially differently oriented will be it represented identically as the end)
     - the ligand can be centered on the molecule
     - output of the result

    This behaviour can parameterized with the booleans `use_rotation_invariance`, `translate_ligand` and `verbose`
    respectively on the constructor.

    """

    name = "abstract"

    def __init__(self, length_cube_side: int, use_rotation_invariance: bool, translate_ligand: bool, verbose: bool):
        """

        :param length_cube_side: the length of the cube (number of voxel on one dimension)
        :param use_rotation_invariance: to perform the PCA rotation and get a fixed position w.r.t the protein
        :param translate_ligand: to translated the ligand at the center of the protein
        :param verbose: to output information about the cube created
        """
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
    def _translate_ligand_on_protein(original_coords, is_from_protein_indices):
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
        for x in range(LENGTH_CUBE_SIDE):
            for y in range(LENGTH_CUBE_SIDE):
                for z in range(LENGTH_CUBE_SIDE):
                    is_from_protein_pos = INDICES_FEATURES["is_from_protein"] - 3
                    is_atom_in_voxel = cube[x, y, z, is_from_protein_pos] != 0
                    if is_atom_in_voxel:
                        # Plotting accordingly to the molecule type
                        color = 2 * cube[x, y, z, is_from_protein_pos] - 1
                        xs.append(x)
                        ys.append(y)
                        zs.append(z)
                        cs.append(color)

        ax.scatter(xs, ys, zs, c=cs, marker="o")

        ax.set_xlim((0, LENGTH_CUBE_SIDE))
        ax.set_ylim((0, LENGTH_CUBE_SIDE))
        ax.set_zlim((0, LENGTH_CUBE_SIDE))

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        plt.show()

    @abstractmethod
    def make_cube(self, system: np.ndarray):
        pass


class AbsoluteCubeRepresentation(CubeRepresentation):
    """
    Class that construct an absolute cube representation (see `CubeRepresentation` for an general overview of cube
    representations).

    This representation construct a box using a fixed resolution in angstrom (by default 3 angstrom)
    given by the `cube_resolution`.
    Then it represents the system by placing the center of the system in the center of the cube.

    Atom that are outside of the boxes get discarded

    Similarly to all the `CubeRepresentations`, the representation can be rotation invariant, the ligand can
    be translated on the protein, constructing the cube can be verbose.


    """
    name = "absolute"

    def __init__(self, length_cube_side, cube_resolution=3.0, use_rotation_invariance=True, translate_ligand=False,
                 verbose=False):
        """

        :param length_cube_side: length of the side of the cube to create
        :param use_rotation_invariance:  perform the canonical rotation of the PCA on the given points
        :param translate_ligand:
        :param verbose: utputs info about transformation
        :param cube_resolution: the resolution used for the cube (1 Å)
        """
        super().__init__(length_cube_side, use_rotation_invariance, translate_ligand, verbose)
        self._cube_resolution = float(cube_resolution)

    def make_cube(self, system: np.ndarray):
        """
        Creating a cube from a system.

        :param system: the protein-ligand system
        :return: a cube 4D np.ndarray of size (res, res, res, nb_features)
        """

        # Spatial coordinates of atoms
        original_coords = system[:, 0:3]

        if self._translate_ligand:
            is_from_protein_column = INDICES_FEATURES["is_from_protein"]
            is_from_protein_indices = np.where(system[:, is_from_protein_column] == 1.)
            original_coords = self._translate_ligand_on_protein(original_coords, is_from_protein_indices)

        if self._use_rotation_invariance:
            is_from_protein_column = INDICES_FEATURES["is_from_protein"]
            is_from_protein_indices = np.where(system[:, is_from_protein_column] == 1.)
            coords = self._representation_invariance(original_coords, is_from_protein_indices)
        else:
            coords = original_coords

        atom_features = system[:, 3:]
        nb_feat = atom_features.shape[1]

        assert nb_feat + coords.shape[1] == NB_FEATURES

        length_cube_side_int = int(LENGTH_CUBE_SIDE)
        length_cube_side_float = float(LENGTH_CUBE_SIDE)

        center = np.mean(coords, axis=0)

        centered_coords = coords - center

        translation_distance = length_cube_side_float / 2 * self._cube_resolution

        scaled_coords = (centered_coords + translation_distance) / self._cube_resolution
        scaled_coords = scaled_coords.round().astype(int)

        # Just keeping atom that are in the box
        in_box = ((scaled_coords >= 0) & (scaled_coords < LENGTH_CUBE_SIDE)).all(axis=1)
        cube = np.zeros((length_cube_side_int, length_cube_side_int, length_cube_side_int, nb_feat), dtype=np.float32)
        for (x, y, z), f in zip(scaled_coords[in_box], atom_features[in_box]):
            cube[x, y, z] += f

        return cube


class RelativeCubeRepresentation(CubeRepresentation):
    """
    Class that construct a relative cube representation (see `CubeRepresentation` for an general overview of cube
    representations).

    This representation find the bounding boxes of the system of protein-ligand. By default it keeps the proportions
    to represent the system in this box : this can be changed with the `keep_proportions` parameter.

    Similarly to all the `CubeRepresentations`, the representation can be rotation invariant, the ligand can
    be translated on the protein, constructing the cube can be verbose.

    """

    name = "relative"

    def __init__(self, length_cube_side, use_rotation_invariance=True, translate_ligand=False, verbose=False,
                 keep_proportions=True):
        """

        :param length_cube_side:
        :param use_rotation_invariance:
        :param translate_ligand:
        :param verbose:
        :param keep_proportions:
        """
        super().__init__(length_cube_side, use_rotation_invariance, translate_ligand, verbose)
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

        :param system: the protein-ligand system
        :return: a cube 4D np.ndarray of size (res, res, res, nb_features)
        """

        # Spatial coordinates of atoms
        original_coords = system[:, 0:3]

        if self._translate_ligand:
            is_from_protein_column = INDICES_FEATURES["is_from_protein"]
            is_from_protein_indices = np.where(system[:, is_from_protein_column] == 1.)
            original_coords = self._translate_ligand_on_protein(original_coords, is_from_protein_indices)

        if self._use_rotation_invariance:
            is_from_protein_column = INDICES_FEATURES["is_from_protein"]
            is_from_protein_indices = np.where(system[:, is_from_protein_column] == 1.)
            coords = self._representation_invariance(original_coords, is_from_protein_indices)
        else:
            coords = original_coords

        atom_features = system[:, 3:]
        nb_feat = atom_features.shape[1]

        assert nb_feat + coords.shape[1] == NB_FEATURES

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
        scaled_coords[:, 0] = np.floor((coords[:, 0] - x_min) / (x_range + eps) * LENGTH_CUBE_SIDE).astype(int)
        scaled_coords[:, 1] = np.floor((coords[:, 1] - y_min) / (y_range + eps) * LENGTH_CUBE_SIDE).astype(int)
        scaled_coords[:, 2] = np.floor((coords[:, 2] - z_min) / (z_range + eps) * LENGTH_CUBE_SIDE).astype(int)

        cube = np.zeros((LENGTH_CUBE_SIDE, LENGTH_CUBE_SIDE, LENGTH_CUBE_SIDE, nb_feat))

        # Filling the cube with the features
        for (x, y, z), f in zip(scaled_coords, atom_features):
            cube[x, y, z] += f

        return cube
