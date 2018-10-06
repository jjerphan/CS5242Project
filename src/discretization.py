import numpy as np
import os
import matplotlib.pyplot as plt
import logging
from mpl_toolkits.mplot3d import Axes3D  # needed by the 3D plotter

from settings import float_type, comment_delimiter, training_examples_folder, resolution_cube, nb_features, \
    indices_features

logger = logging.getLogger('cnn.discretization')
logger.addHandler(logging.NullHandler())

def representation_invariance(original_coords, is_from_protein_indices, verbose=False):
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

    if verbose:
        print("Eigen Values (~ scale factor of molecules):"),
        print("x factor : ", eigen_vals[0])
        print("y factor : ", eigen_vals[1])
        print("z factor : ", eigen_vals[2])
        print("Rotation matrix")
        print(rotation_mat)

    return new_coords


def make_cube(system: np.ndarray, resolution, use_rotation_invariance=True, keep_proportions=True,
              verbose=False) -> np.ndarray:
    """
    Creating a cube from a system.

    Using a rotation invariance representation and keeping the proportions gives better results normaly.

    :param system: the protein-ligand system
    :param resolution: resolution for the discretization
    :param use_rotation_invariance: perform the canonical rotation of the PCA on the given points
    :param keep_proportions: should keep the proportion
    :param verbose: outputs info about transformation
    :return: a cube 4D np.ndarray of size (res, res, res, nb_features)
    """

    # Spatial coordinates of atoms
    original_coords = system[:, 0:3]

    is_from_protein_column = indices_features["is_from_protein"]
    is_from_protein_indices = np.where(system[:, is_from_protein_column] == 1.)

    if use_rotation_invariance:
        coords = representation_invariance(original_coords, is_from_protein_indices, verbose=verbose)
    else:
        coords = original_coords

    atom_features = system[:, 3:]
    nb_feat = atom_features.shape[1]

    assert nb_feat + coords.shape[1] == nb_features

    # Getting extreme values
    x_min, x_max, y_min, y_max, z_min, z_max = values_range(coords)

    # Finding the maximum range between extreme points on each coordinates
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min

    # If we want to keep the proportion, we can scale adequately
    if keep_proportions:
        max_range = max([x_range, y_range, z_range])
        x_range = max_range
        y_range = max_range
        z_range = max_range

    # Scaling coordinates to be in the cube [0,res]^3 then flooring
    scaled_coords = (coords * 0).astype(int)
    eps = 10e-4  # To be sure to round down on exact position
    scaled_coords[:, 0] = np.floor((coords[:, 0] - x_min) / (x_range + eps) * resolution).astype(int)
    scaled_coords[:, 1] = np.floor((coords[:, 1] - y_min) / (y_range + eps) * resolution).astype(int)
    scaled_coords[:, 2] = np.floor((coords[:, 2] - z_min) / (z_range + eps) * resolution).astype(int)

    cube = np.zeros((resolution, resolution, resolution, nb_feat))
    logger.debug("Cube size is %s", str(cube.shape))

    # Filling the cube with the features
    cube[scaled_coords[:, 0], scaled_coords[:, 1], scaled_coords[:, 2]] = atom_features

    return cube


def is_positive(name):
    """
    name is of the form "xxxx_yyyy[.csv]"

    'xxxx' corresponds to the protein.
    'yyyy' corresponds to the ligands

    Return xxxx == yyyy (both molecules bind together)

    :param name: the name of a file of the form "xxxx_yyyy[.csv]"
    :return:
    """
    systems = name.replace(".csv", "").split("_")
    return systems[0] == systems[1]

def is_negative(name):
    """
    name is of the form "xxxx_yyyy[.csv]"

    'xxxx' corresponds to the protein.
    'yyyy' corresponds to the ligands

    Return xxxx != yyyy (both molecules don't bind together)

    :param name: the name of a file of the form "xxxx_yyyy[.csv]"
    :return:
    """
    return not(is_positive(name))


def only_positive_examples(system_names):
    """
    Filter the names to return the ones of positive examples solely (i.e. with same)

    :param system_names: name of the form "xxxx_yyyy" where xxxx is the system of the protein and yyyy of the ligand
    :return: the list of system names of the form "xxxx_xxxx"
    """

    return list(filter(is_positive, system_names))


def values_range(spatial_coordinates):
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
    for x in range(resolution_cube):
        for y in range(resolution_cube):
            for z in range(resolution_cube):
                is_from_protein_pos = indices_features["is_from_protein"]-3
                is_atom_in_voxel = cube[x, y, z, is_from_protein_pos] != 0
                if is_atom_in_voxel:
                    # Plotting accordingly to the molecule type
                    color = 2 * cube[x, y, z, is_from_protein_pos] - 1
                    xs.append(x)
                    ys.append(y)
                    zs.append(z)
                    cs.append(color)

    ax.scatter(xs, ys, zs, c=cs, marker="o")

    ax.set_xlim((0, resolution_cube))
    ax.set_ylim((0, resolution_cube))
    ax.set_zlim((0, resolution_cube))

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()


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


if __name__ == "__main__":

    # Just to test
    examples_files = sorted(os.listdir(training_examples_folder))
    plt.ion()
    plt.show()
    for ex_file in examples_files:
        plt.close('all')
        print(f"System {ex_file}")
        file_name = os.path.join(training_examples_folder, ex_file)
        example = load_nparray(os.path.join(training_examples_folder, ex_file))

        print("Compressed Representation")
        cube = make_cube(example, resolution_cube,
                         use_rotation_invariance=False,
                         keep_proportions=False,
                         verbose=True)
        plot_cube(cube)
        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(0, 100, 640, 545)
        plt.pause(0.003)

        print("Properly Scaled Representation")
        cube = make_cube(example, resolution_cube,
                         use_rotation_invariance=False,
                         keep_proportions=True,
                         verbose=True)
        plot_cube(cube)
        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(640, 100, 640, 545)
        plt.pause(1)

        print("Properly Scaled Rotation Invariant Representation")
        cube = make_cube(example, resolution_cube,
                         use_rotation_invariance=True,
                         keep_proportions=True,
                         verbose=True)

        plot_cube(cube)
        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(2 * 640, 100, 640, 545)
        plt.pause(2)
        input()

