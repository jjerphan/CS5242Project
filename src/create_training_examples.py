import numpy as np
import os
import shutil

from discretization import load_nparray
from settings import training_examples_folder, extracted_data_train_folder, extracted_protein_suffix, \
    extracted_ligand_suffix, comment_delimiter, extract_id, progress, nb_neg_ex_per_pos


def save_example(folder: str, protein: np.ndarray, ligand: np.ndarray, system_protein: str, system_ligand: str):
    file_name = system_protein + "_" + system_ligand + ".csv"
    file_path = os.path.join(folder, file_name)

    # We concatenate the molecule vertically
    if len(protein.shape) < 2 or len(ligand.shape) < 2:
        print("\nOne molecule is empty")
        print(f"Protein {system_protein}: {len(protein.shape)}")
        print(f"Ligand  {system_ligand}: {len(ligand.shape)}")
        return

    if protein.shape[1] != ligand.shape[1]:
        print("Different dimension")
        print(f"Protein {system_protein}: {protein.shape}")
        print(f"Ligand  {system_ligand}: {ligand.shape}")
        return

    # Merging the protein and the ligand together
    example = np.concatenate((protein, ligand), axis=0)

    type_example = ("Positive" if system_protein == system_ligand else "Negative") + " example "

    comments = [f"{type_example} of (Protein, Ligand) : ({system_protein},{system_ligand})",
                f" - Number of atoms in Protein: {protein.shape[0]}",
                f" - Number of atoms in Ligand : {ligand.shape[0]}"]

    comment = comment_delimiter + f"\n{comment_delimiter} ".join(comments) + "\n"

    with open(file_path, "w") as f:
        f.write(comment)
        np.savetxt(fname=f, X=example)


# To get reproducible generations of examples
np.random.seed(1337)

if __name__ == "__main__":

    # Deleting the folders of examples and recreating it
    if os.path.exists(training_examples_folder):
        shutil.rmtree(training_examples_folder)
    os.makedirs(training_examples_folder)

    # Getting all the systems
    list_systems = set(list(map(extract_id, os.listdir(extracted_data_train_folder))))

    # For each system, we create the associated positive example and we generate some negative examples
    for system in progress(sorted(list_systems)):
        protein = load_nparray(os.path.join(extracted_data_train_folder, system + extracted_protein_suffix))
        ligand = load_nparray(os.path.join(extracted_data_train_folder, system + extracted_ligand_suffix))

        # Saving positive example
        save_example(training_examples_folder, protein, ligand, system, system)

        # Creating false example using nb_neg_ex negatives examples
        others_system = sorted(list(list_systems.difference(set(system))))
        some_others_systems_indices = np.random.randint(0, len(others_system), nb_neg_ex_per_pos)
        for other_system in map(lambda index: others_system[index], some_others_systems_indices):
            bad_ligand = load_nparray(os.path.join(extracted_data_train_folder, other_system + extracted_ligand_suffix))

            # Saving negative example
            save_example(training_examples_folder, protein, bad_ligand, system, other_system)
