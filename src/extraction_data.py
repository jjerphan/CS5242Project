import os
import numpy as np
import logging
from concurrent import futures
from random import Random

from settings import hydrophobic_types, float_type, formatter, nb_features, percent_train, percent_test, nb_workers, \
    extracted_given_data_train_folder, extracted_given_data_validation_folder, original_given_data_folder, \
    extracted_given_data_folder, \
    original_predict_data_folder, extracted_predict_data_folder, extracted_given_data_test_folder

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


def read_pdb_predict(file_name) -> (list, list, list, list):
    """
    Read a original pdb file given for prediction and return the data.
    The lists contains ordered information about each atom.
    :param file_name: the file to extract data
    :return: lists of coordinates and atom type for each atom
    """
    with open(file_name, 'r') as file:
        strline_L = file.readlines()

    x_list = list()
    y_list = list()
    z_list = list()
    atom_type_list = list()
    for strline in strline_L:
        stripped_line = strline.strip()

        splitted_line = stripped_line.split('\t')

        x_list.append(float_type(splitted_line[0]))
        y_list.append(float_type(splitted_line[1]))
        z_list.append(float_type(splitted_line[2]))
        atom_type_list.append(str(splitted_line[3]))

    return x_list, y_list, z_list, atom_type_list


def build_molecule_features(x_list: list, y_list: list, z_list: list, atom_type_list: list,
                            molecule_is_protein: bool) -> np.array:
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
    is_hydrophobic_list = np.array([1 if atom_type in hydrophobic_types else -1 for atom_type in atom_type_list])

    is_from_protein_list = (2 * molecule_is_protein) * np.ones((nb_atoms,)) - 1

    # See `features_names` in settings to see how the features are organised
    formated_molecule = np.array([x_list, y_list, z_list, is_hydrophobic_list, is_from_protein_list]).T

    assert (formated_molecule.shape == (nb_atoms, nb_features))

    return formated_molecule


def save_given_data(pdb_file, group_indices):
    """
    Save the data for a file from the given data set.

    :param pdb_file:
    :return:
    """
    pdb_original_file_path = os.path.join(original_given_data_folder, pdb_file)
    # Extract features from pdb files.
    x_list, y_list, z_list, atom_type_list = read_pdb(pdb_original_file_path)

    is_protein = "pro" in pdb_file

    molecule = build_molecule_features(x_list, y_list, z_list, atom_type_list, is_protein)

    # Saving the data is a csv file with the same name
    # Choosing the appropriate folder using the split index
    molecule_index = pdb_file.split("_")[0]

    pdb_file_csv = pdb_file.replace(".pdb", ".csv")

    assert len(group_indices) > 0
    if molecule_index in group_indices[0]:
        extracted_file_path = os.path.join(extracted_given_data_train_folder, pdb_file_csv)
    elif molecule_index in group_indices[1]:
        extracted_file_path = os.path.join(extracted_given_data_validation_folder, pdb_file_csv)
    elif molecule_index in group_indices[2]:
        extracted_file_path = os.path.join(extracted_given_data_test_folder, pdb_file_csv)
    else:
        logger.debug("Not inside indices. Something went wrong")
        raise()

    np.savetxt(fname=extracted_file_path, X=molecule, fmt=formatter)


def save_predict_data(pdb_file):
    """
    Save the data for a file from the data set for final prediction.

    :param pdb_file:
    :return:
    """
    pdb_original_file_path = os.path.join(original_predict_data_folder, pdb_file)
    # Extract features from pdb files.
    x_list, y_list, z_list, atom_type_list = read_pdb_predict(pdb_original_file_path)

    is_protein = "pro" in pdb_file

    molecule = build_molecule_features(x_list, y_list, z_list, atom_type_list, is_protein)

    # Saving the data is a csv file with the same name
    # Choosing the appropriate folder using the split index
    pdb_file_csv = pdb_file.replace(".pdb", ".csv")

    extracted_file_path = os.path.join(extracted_predict_data_folder, pdb_file_csv)

    np.savetxt(fname=extracted_file_path, X=molecule, fmt=formatter)


def extract_predict_data():
    """
    Extract the original data given for final prediction.

    :return:
    """
    if not(os.path.exists(extracted_predict_data_folder)):
        logger.debug('The %s folder does not exist. Creating it.', extracted_predict_data_folder)
        os.makedirs(extracted_predict_data_folder)

    original_files = sorted(os.listdir(original_predict_data_folder))

    logger.debug('Read original pdb files from %s.', original_predict_data_folder)
    logger.debug('Total files are %d', len(original_files))

    with futures.ProcessPoolExecutor(max_workers=nb_workers) as executor:
        for pdb_original_file in original_files:
            executor.submit(save_predict_data, pdb_original_file)

    logger.debug('Molecules saved into folders in csv format.')


def extract_given_data():
    """
    Extract the original data given for training.

    :return:
    """
    for folder in [extracted_given_data_folder, extracted_given_data_validation_folder, extracted_given_data_test_folder, extracted_given_data_train_folder]:
        if not(os.path.exists(folder)):
            logger.debug('The %s folder does not exist. Creating it.', folder)
            os.makedirs(folder)

    logger.debug('Splitting data into 80% training, 10% validation, 10% testing.')

    original_files = sorted(os.listdir(original_given_data_folder))

    indices = sorted(set(map(lambda x: x.split('_')[0], original_files)))
    Random(48).shuffle(indices)

    total = len(indices)

    test_split_index = int(total * percent_train)
    pred_split_index = int(total * (percent_test + percent_train))
    training_indices = indices[:test_split_index]
    validation_indices = indices[test_split_index:pred_split_index]
    test_indices = indices[pred_split_index:]

    group_indices = [training_indices, validation_indices, test_indices]

    logger.debug('Read original pdb files from %s.', original_given_data_folder)
    logger.debug('Total files are %d', len(original_files))

    with futures.ProcessPoolExecutor(max_workers=nb_workers) as executor:
        for pdb_original_file in original_files:
            executor.submit(save_given_data, pdb_original_file, group_indices)

    logger.debug('Molecules saved into folders in csv format.')


if __name__ == "__main__":
    print("Extracting the given data")
    extract_given_data()
    print("Extracting the data for prediction")
    extract_predict_data()

