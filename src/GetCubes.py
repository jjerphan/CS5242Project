import os
import numpy as np
from discretization import load_nparray, make_cube, is_positive
from settings import training_examples_folder, resolution_cube

def get_cubes(nb_examples=128):
    """
    Return the first nb_examples cubes with their ys.
    :param nb_examples:
    :return: list of cubes and list of their ys
    """
    examples_files = sorted(os.listdir(training_examples_folder))[0:nb_examples]

    cubes = []
    ys = []
    for index, ex_file in enumerate(examples_files):
        file_name = os.path.join(training_examples_folder, ex_file)
        example = load_nparray(file_name)

        cube = make_cube(example, resolution_cube)
        y = 1 * is_positive(ex_file)

        cubes.append(cube)
        ys.append(y)

    # Conversion to np.ndarrays with the first axes used for examples
    cubes = np.array(cubes)
    ys = np.array(ys)
    assert(ys.shape[0] == nb_examples)
    assert(cubes.shape[0] == nb_examples)

    return cubes, ys