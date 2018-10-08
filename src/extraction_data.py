import os
import re
import numpy as np
import logging
from concurrent import futures

from settings import original_data_folder, extracted_data_folder, hydrophobic_types, float_type, \
    formatter, nb_features, extracted_data_train_folder, extracted_data_test_folder, \
    train_indices, original_predict_folder, extracted_predict_folder

logger = logging.getLogger('__main__.extract_data')
logger.addHandler(logging.NullHandler())

def read_pdb(file_name) -> (list, list, list, list):
    """
    Read a original pdb file and return the data.
    The lists contains ordered information about each atom.

    :param file_name: the file to extract data
    :return: lists of coordinates and atom type for each atom
    """
    x_list = list()
    y_list = list()
    z_list = list()
    atom_type_list = list()

    with open(file_name, 'r') as file:
        for num_line, strline in enumerate(file.readlines()):
            # removes all whitespace at the start and end, including spaces, tabs, newlines and carriage returns
            stripped_line = strline.strip()

            # Extracting the information here

            # First line of 0001_lig_cg.pdb as an example:
            #
            # Features     |                                        x       y       z               atom_type
            # Line in file |ATOM      2  CA  HIS A   0      17.186 -28.155 -12.495  1.00 26.12           C
            #               ^                             ^       ^       ^       ^                     ^
            # Position     |0                            30      38      46      54                    76

            x_list.append(float_type(stripped_line[30:38].strip()))
            y_list.append(float_type(stripped_line[38:46].strip()))
            z_list.append(float_type(stripped_line[46:54].strip()))
            atom_type_list.append(stripped_line[76:].strip())

    assert len(x_list) == len(y_list)
    assert len(x_list) == len(z_list)
    assert len(x_list) == len(atom_type_list)
    return x_list, y_list, z_list, atom_type_list


def extract_molecule(x_list: list, y_list: list, z_list: list, atom_type_list: list, molecule_is_protein:bool) -> np.array:
    """
    Convert the data extract from file into a np.ndarray.

    The information of one atom is represented as a line in the array.

    See settings.py for values used to represented categorical features (molecule type and atom type)

    :param x_list: list of x coordinates
    :param y_list: list of y coordinates
    :param z_list: list of z coordinates
    :param atom_type_list: list of atom type (string)
    :param molecule_is_protein: boolean
    :return: np.ndarray of dimension (nb_atoms, 3 + nb_atom_features)
    """
    nb_atoms = len(x_list)
    # One hot encoding for atom type and molecule types
    is_hydrophobic = np.array([1 if type in hydrophobic_types else 0 for type in atom_type_list])
    #is_polar = 1 - is_hydrophobic

    is_from_protein = (1 * molecule_is_protein) * np.ones((nb_atoms,))
    #is_from_ligand = 1 - is_from_protein
    is_hydrophobic_list = np.array([1 if type in hydrophobic_types else -1 for type in atom_type_list])

    is_from_protein_list = (2 * molecule_is_protein) * np.ones((nb_atoms,)) - 1

    # See `features_names` in settings to see how the features are organised
    #formated_molecule = np.array([x_list, y_list, z_list, is_hydrophobic, is_from_protein]).T
    formated_molecule = np.array([x_list, y_list, z_list, is_hydrophobic_list, is_from_protein_list]).T

    assert(formated_molecule.shape == (nb_atoms, nb_features))

    return formated_molecule


def extract_data_bak():
    logger.debug('Checking if extracted_data, extracted_data_train, extracted_data_test folder exists.')
    if not(os.path.exists(extracted_data_folder)):
        logger.debug('The extracted_data folder does not exist. Creating the three folders.')
        os.makedirs(extracted_data_folder)
        os.makedirs(extracted_data_train_folder)
        os.makedirs(extracted_data_test_folder)

    logger.info('Checking if extracted_data, extracted_data_train, extracted_data_test folder exists.')

    # Creating folders if they don't exist
    for folder in [extracted_data_folder, extracted_data_test_folder, extracted_data_train_folder]:
        if not(os.path.exists(folder)):
            logger.debug('The %s folder does not exist. Creating it.', folder)
            os.makedirs(folder)

    # For each file, we extract the info and save it into a file in a specified folders
    logger.debug('Read orginal pdb files from %s.', original_data_folder)
    logger.debug('Total files are %d', len(os.listdir(original_data_folder)))

    with futures.ProcessPoolExecutor(max_workers=6) as executor:
        for pdb_original_file in sorted(os.listdir(original_data_folder)):
            executor.submit(save_data, pdb_original_file)

    logger.debug('Molecules saved into train and test folder in csv.')


def save_data(pdb_original_file, split_training):
    pdb_original_file_path = os.path.join(original_data_folder, pdb_original_file)

    # Extract features from pdb files.
    x_list, y_list, z_list, atom_type_list = read_pdb(pdb_original_file_path)

    is_protein = "pro" in pdb_original_file

    molecule = extract_molecule(x_list, y_list, z_list, atom_type_list, is_protein)

    # Saving the data is a csv file with the same name
    # Choosing the appropriate folder using the split index
    molecule_index = int(re.match("\d+", pdb_original_file).group())

    if not split_training:
        extracted_file_path = os.path.join(extracted_predict_folder, pdb_original_file.replace(".pdb", ".csv"))
    else:
        if molecule_index in train_indices:
            extracted_file_path = os.path.join(extracted_data_train_folder, pdb_original_file.replace(".pdb", ".csv"))
        else:
            extracted_file_path = os.path.join(extracted_data_test_folder, pdb_original_file.replace(".pdb", ".csv"))

    np.savetxt(fname=extracted_file_path, X=molecule, fmt=formatter)

def extract_data(pdb_folder, split_training=True):
    for folder in [extracted_data_folder, extracted_data_test_folder, extracted_data_train_folder, extracted_predict_folder]:
        if not(os.path.exists(folder)):
            logger.debug('The %s folder does not exist. Creating it.', folder)
            os.makedirs(folder)

    logger.debug('Read orginal pdb files from %s.', pdb_folder)
    logger.debug('Total files are %d', len(os.listdir(pdb_folder)))

    with futures.ProcessPoolExecutor(max_workers=6) as executor:
        for pdb_original_file in sorted(os.listdir(pdb_folder)):
            executor.submit(save_data, pdb_original_file, split_training)

    logger.debug('Molecules saved into folders in csv format.')

if __name__ == "__main__":
    extract_data(original_data_folder, split_training=True)
    extract_data(original_predict_folder, split_training=False)
