import numpy as np
import os
import shutil

from discretization import load_nparray
from settings import training_examples_folder, extracted_data_train_folder, extracted_protein_suffix, \
    extracted_ligand_suffix, comment_delimiter, extract_id, progress, nb_neg_ex_per_pos, nb_systems, features_names


def save_example(folder: str, protein: np.ndarray, ligand: np.ndarray, protein_system: str, ligand_system: str):
    """
    Save an example of a system using representations of a protein and of a ligand.

    Files are saved in the `folder` with the naming convention:

        `xxxx_yyyy.csv`

    where `xxxx` denotes the `protein_system` and `yyyy` denotes `ligand_system`.

    Hence `xxxx` == `yyyy` if and only if the system is a positive example.

    :param folder: the folder where the training examples are saved
    :param protein: the representation of the protein
    :param ligand: the representation of the ligand
    :param protein_system: the ID of the system of the protein
    :param ligand_system: the ID of the system of the ligand
    :return:
    """
    file_name = protein_system + "_" + ligand_system + ".csv"
    file_path = os.path.join(folder, file_name)

    if len(protein.shape) < 2 or len(ligand.shape) < 2:
        print("\nOne molecule is empty")
        print(f"Protein {protein_system}: {len(protein.shape)}")
        print(f"Ligand  {ligand_system}: {len(ligand.shape)}")
        return

    if protein.shape[1] != ligand.shape[1]:
        print("Different dimension")
        print(f"Protein {protein_system}: {protein.shape}")
        print(f"Ligand  {ligand_system}: {ligand.shape}")
        return

    # Merging the protein and the ligand together
    # We concatenate the molecule vertically
    example = np.concatenate((protein, ligand), axis=0)

    # We add a comment at the beginning of the file
    type_example = (" Positive" if protein_system == ligand_system else " Negative") + " example "

    comments = [f"{type_example} of (Protein, Ligand) : ({protein_system},{ligand_system})",
                f" - Number of atoms in Protein: {protein.shape[0]}",
                f" - Number of atoms in Ligand : {ligand.shape[0]}",
                ','.join(features_names)]

    comment = comment_delimiter + f"\n{comment_delimiter} ".join(comments) + "\n"

    with open(file_path, "w") as f:
        f.write(comment)
        np.savetxt(fname=f, X=example)


# To get reproducible generations of examples
np.random.seed(1337)


def create_training_examples(nb_neg: int):
    """
    Create training examples and saves them in files in the
    `extracted_data_train_folder` folder.

    Here, we create both positive and negative examples.

    We are given `nb_systems` that binds ; so this makes `nb_systems` positive examples.

    We can then each protein and some others ligands to create negative examples (ie examples of systems that don't
    bind with each others). Those examples are created randomly taking some others ligand that are not binding.
    This is made reproducible using a seed.

    If we note `nb_neg` the number of negative examples created per positive examples, we have exactly :

        `nb_systems` * `nb_neg` negatives examples

    Hence this procedure creates `n_systems` * (1 + `nb_neg`) examples, that is at max `nb_systems^2` examples.

    :type nb_neg: the number of negative example to create per positive example
    :return:
    """
    if nb_neg > nb_systems:
        raise RuntimeError(f"Cannot create more than {nb_systems-1} negatives examples per positive examples (actual "
                           f"value = {nb_neg}")

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
        other_systems = sorted(list(list_systems.difference({system})))
        some_others_systems_indices = np.random.permutation(len(other_systems))[0:nb_neg]

        for other_system in map(lambda index: other_systems[index], some_others_systems_indices):
            bad_ligand = load_nparray(os.path.join(extracted_data_train_folder, other_system + extracted_ligand_suffix))

            if other_system == system:
                raise RuntimeError(f"other_system = {other_system} shoud be != system = {system}")

            nb_examples = len(os.listdir(training_examples_folder))
            # Saving negative example
            save_example(training_examples_folder, protein, bad_ligand, system, other_system)
            new_nb_examples = len(os.listdir(training_examples_folder))
            if nb_examples == new_nb_examples:
                print(f"Example {system}-{other_system} not created")


if __name__ == "__main__":
    create_training_examples(nb_neg_ex_per_pos)
