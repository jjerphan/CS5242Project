import os
import numpy as np
from settings import extract_id, progress, extracted_protein_suffix, extracted_ligand_suffix, resolution_cube
from discretization import load_nparray, make_cube



def examples_iterator(data_folder) -> (np.ndarray, str, str):
    """
    Construct all the examples in the given folder

    :param data_folder:
    :return: a example (cube) at the time with the system used to construct the cube
    """
    # Getting all the systems
    list_systems_ids = set(list(map(extract_id, os.listdir(data_folder))))

    # For each system, we create the associated positive example and we generate some negative examples
    for system_id in progress(sorted(list_systems_ids)):
        protein = load_nparray(os.path.join(data_folder, system_id + extracted_protein_suffix))
        ligand = load_nparray(os.path.join(data_folder, system_id + extracted_ligand_suffix))

        # Yielding first positive example
        positive_example = np.concatenate((protein, ligand), axis=0)
        cube_pos_example = make_cube(positive_example, resolution_cube)
        yield cube_pos_example, system_id, system_id

        # Yielding all the others negatives examples with the same protein
        others_system = sorted(list(list_systems_ids.difference(set(system_id))))
        for other_system in others_system:
            bad_ligand = load_nparray(os.path.join(data_folder, other_system + extracted_ligand_suffix))

            # Saving negative example
            negative_example = np.concatenate((protein, bad_ligand), axis=0)
            cube_neg_example = make_cube(negative_example, resolution_cube)
            yield cube_neg_example, system_id, other_system