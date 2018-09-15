import os
import numpy as np
import progressbar

from settings import original_data, extracted_data, hydrophobic_types, float_type, \
    formatter, widgets_progressbar, nb_features


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
    is_polar = 1 - is_hydrophobic

    is_from_protein = (1 * molecule_is_protein) * np.ones((nb_atoms,))
    is_from_ligand = 1 - is_from_protein

    formated_molecule = np.array([x_list, y_list, z_list, is_hydrophobic, is_polar, is_from_protein, is_from_ligand]).T

    assert(formated_molecule.shape == (nb_atoms, nb_features))

    return formated_molecule


if __name__ == "__main__":
    if not(os.path.exists(extracted_data)):
        os.makedirs(extracted_data)

    # For each file, we extract the info and save it into a file in a specified folders
    for pdb_original_file in progressbar.progressbar(sorted(os.listdir(original_data)), widgets=widgets_progressbar,
                                                     redirect_stdout=True):
        pdb_original_file_path = os.path.join(original_data, pdb_original_file)

        x_list, y_list, z_list, atom_type_list = read_pdb(pdb_original_file_path)

        is_protein = "pro" in pdb_original_file

        molecule = extract_molecule(x_list, y_list, z_list, atom_type_list, is_protein)

        # Saving the data is a csv file with the same name
        extracted_file_path = os.path.join(extracted_data, pdb_original_file.replace(".pdb", ".csv"))
        np.savetxt(fname=extracted_file_path, X=molecule, fmt=formatter)
