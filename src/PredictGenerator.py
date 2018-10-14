import os
from discretization import load_nparray, make_cube
from settings import resolution_cube
import numpy as np



def PredictGenerator(examples_folder):
    examples_files = os.listdir(examples_folder)
    for file in examples_files:
        example = load_nparray(os.path.join(examples_folder, file))
        cube = make_cube(example, resolution_cube)
        cube = np.array([cube])
        protein = file.split('_')[0]
        ligend = file.split('_')[1].split('.')[0]

        yield (protein, ligend, cube)