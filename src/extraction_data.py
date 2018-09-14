import os
import numpy as np
import progressbar

from settings import original_data, extracted_data, hydrophobic_value, polar_value, hydrophobic_types, float_type, \
    formatter, protein_value, ligand_value, protein_suffix, widgets_progressbar


def values_range():
    x_min = np.inf
    y_min = np.inf
    z_min = np.inf

    x_max = - np.inf
    y_max = - np.inf
    z_max = - np.inf

    atom_types = set()

    for pdb_original_file in progressbar.progressbar(sorted(os.listdir(original_data))):
        pdb_original_file_path = os.path.join(original_data, pdb_original_file)

        x_list, y_list, z_list, atom_type_list = read_pdb(pdb_original_file_path)

        if len(x_list) == 0:
            print(f"{pdb_original_file} is empty")
            continue

        x_min = min(x_min, min(x_list))
        y_min = min(y_min, min(y_list))
        z_min = min(z_min, min(z_list))

        x_max = max(x_max, max(x_list))
        y_max = max(y_max, max(y_list))
        z_max = max(z_max, max(y_list))

        atom_types = atom_types.union(set(atom_type_list))

    print("Range of values")
    print(f"x : [{x_min}, {x_max}]")
    print(f"y : [{y_min}, {y_max}]")
    print(f"z : [{z_min}, {z_max}]")
    print(f"atomtypes : {atom_types}")
    return x_min, x_max, y_min, y_max, z_min, z_max, atom_types


def read_pdb(file_name) -> (list, list, list, list):
    x_list = list()
    y_list = list()
    z_list = list()
    atom_type_list = list()

    # First line of 0001_lig_cg.pdb for example:
    #                                      x       y       z               atom_type
    # ATOM      2  CA  HIS A   0      17.186 -28.155 -12.495  1.00 26.12           C
    #
    normal_len = 78

    with open(file_name, 'r') as file:
        for num_line, strline in enumerate(file.readlines()):
            # removes all whitespace at the start and end, including spaces, tabs, newlines and carriage returns
            stripped_line = strline.strip()

            line_length = len(stripped_line)
            # if line_length != normal_len:
            #     print(
            #         f'ERROR: line {num_line+1} length is different in file {file_name} .
            # Expected={normal_len}, current={line_length}')
            #     print(strline)

            x_list.append(float_type(stripped_line[30:38].strip()))
            y_list.append(float_type(stripped_line[38:46].strip()))
            z_list.append(float_type(stripped_line[46:54].strip()))
            atom_type_list.append(stripped_line[76:].strip())

    assert len(x_list) == len(y_list)
    assert len(x_list) == len(z_list)
    assert len(x_list) == len(atom_type_list)
    return x_list, y_list, z_list, atom_type_list


def create_example(x_list: list, y_list: list, z_list: list, atom_type_list: list, molecule_value) -> np.array:
    molecule_values = [molecule_value] * len(x_list)
    encoded_ato_type_list = [hydrophobic_value if type in hydrophobic_types else polar_value for type in atom_type_list]
    return np.array([x_list, y_list, z_list, encoded_ato_type_list, molecule_values]).T


if __name__ == "__main__":
    if not(os.path.exists(extracted_data)):
        os.makedirs(extracted_data)

    for pdb_original_file in progressbar.progressbar(sorted(os.listdir(original_data)), widgets=widgets_progressbar,
                                                     redirect_stdout=True):
        pdb_original_file_path = os.path.join(original_data, pdb_original_file)

        x_list, y_list, z_list, atom_type_list = read_pdb(pdb_original_file_path)

        molecule_value = protein_value if "pro" in pdb_original_file else ligand_value

        example = create_example(x_list, y_list, z_list, atom_type_list, molecule_value)

        extracted_file_path = os.path.join(extracted_data, pdb_original_file.replace(".pdb", ".csv"))
        np.savetxt(fname=extracted_file_path, X=example, fmt=formatter)
