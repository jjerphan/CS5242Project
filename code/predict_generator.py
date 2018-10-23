import os
from discretization import RelativeCubeRepresentation, CubeRepresentation
from pipeline_fixtures import load_nparray
from settings import LENGTH_CUBE_SIDE
import numpy as np


def PredictGenerator(examples_folder,
                     representation: CubeRepresentation= RelativeCubeRepresentation(length_cube_side=LENGTH_CUBE_SIDE)):
    """
    A Generator that return examples in a specific examples_folder with the id of the protein and of the ligand.

    :param examples_folder: the folder where files are
    :param representation: the representation to use
    :return:
    """
    examples_files = os.listdir(examples_folder)
    for file in examples_files:
        example = load_nparray(os.path.join(examples_folder, file))
        cube = representation.make_cube(example)
        cube = np.array([cube])
        protein = file.split('_')[0]
        ligand = file.split('_')[1].split('.')[0]

        yield (protein, ligand, cube)
