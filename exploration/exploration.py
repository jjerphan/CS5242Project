import os

import numpy as np

from extraction_data import read_pdb
from settings import ORIGINAL_GIVEN_DATA_FOLDER
from pipeline_fixtures import show_progress


def values_range():
    """
    :return: global values of coordinates and atoms type present in the original files
    """
    x_min = np.inf
    y_min = np.inf
    z_min = np.inf

    x_max = - np.inf
    y_max = - np.inf
    z_max = - np.inf

    atom_types = set()
    empty_files = set()

    for pdb_original_file in show_progress(sorted(os.listdir(ORIGINAL_GIVEN_DATA_FOLDER))):
        pdb_original_file_path = os.path.join(ORIGINAL_GIVEN_DATA_FOLDER, pdb_original_file)

        x_list, y_list, z_list, atom_type_list = read_pdb(pdb_original_file_path)

        if len(x_list) == 0:
            empty_files.add(pdb_original_file)
            continue

        x_min = min(x_min, min(x_list))
        y_min = min(y_min, min(y_list))
        z_min = min(z_min, min(z_list))

        x_max = max(x_max, max(x_list))
        y_max = max(y_max, max(y_list))
        z_max = max(z_max, max(y_list))

        atom_types = atom_types.union(set(atom_type_list))

    return x_min, x_max, y_min, y_max, z_min, z_max, atom_types, empty_files


if __name__ == "__main__":
    x_min, x_max, y_min, y_max, z_min, z_max, atom_types, empty_files = values_range()
    empty_proteins = set(map(lambda x: "pro" in x, empty_files))
    empty_ligands = set(map(lambda x: "lig" in x, empty_files))

    print("Range of values")
    print(f"x : [{x_min}, {x_max}]")
    print(f"y : [{y_min}, {y_max}]")
    print(f"z : [{z_min}, {z_max}]")
    print(f"atomtypes : {atom_types}")
    print(f"Number of empty files : {len(empty_files)}")

    print("Empty proteins :")
    for p in empty_proteins:
        print(" > ", p)

    print("Empty ligands :")
    for p in empty_ligands:
        print(" > ", p)
