import os
from discretization import RelativeCubeRepresentation
from pipeline_fixtures import load_nparray
from settings import LENGTH_CUBE_SIDE
import numpy as np


def PredictGenerator(examples_folder):
    examples_files = os.listdir(examples_folder)
    relative_representation = RelativeCubeRepresentation(length_cube_side=LENGTH_CUBE_SIDE)
    for file in examples_files:
        example = load_nparray(os.path.join(examples_folder, file))
        cube = relative_representation.make_cube(example)
        cube = np.array([cube])
        protein = file.split('_')[0]
        ligand = file.split('_')[1].split('.')[0]

        yield (protein, ligand, cube)
