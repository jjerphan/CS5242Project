import numpy as np
import os
import shutil
import logging
from discretization import load_nparray
from concurrent import futures
from settings import extracted_data_train_folder, extracted_data_test_folder, extracted_protein_suffix, \
    extracted_ligand_suffix, comment_delimiter, extract_id, nb_neg_ex_per_pos, features_names, training_examples_folder,\
    testing_examples_folder, extracted_predict_folder, predict_examples_folder


logger = logging.getLogger('__main__.create_training_example')
logger.addHandler(logging.NullHandler())


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


def save_examples(system, list_systems, nb_neg, data_folder, save_folder):
    folder_path = data_folder

    try:
        protein = load_nparray(os.path.join(folder_path, system + extracted_protein_suffix))
        ligand = load_nparray(os.path.join(folder_path, system + extracted_ligand_suffix))
    except:
        logger.debug(f"Loading protein/ligand failed. Protein folder {os.path.join(folder_path, system + extracted_protein_suffix)}, Ligand folder {os.path.join(folder_path, system + extracted_ligand_suffix)}")
        print(f"Loading protein/ligand failed. Protein folder {os.path.join(folder_path, system + extracted_protein_suffix)}, Ligand folder {os.path.join(folder_path, system + extracted_ligand_suffix)}")
        raise

    # Saving positive example
    save_example(save_folder, protein, ligand, system, system)

    # Creating false example using nb_neg_ex negatives examples
    other_systems = sorted(list(list_systems.difference({system})))
    some_others_systems_indices = np.random.permutation(len(other_systems))[0:nb_neg]

    for other_system in map(lambda index: other_systems[index], some_others_systems_indices):
        bad_ligand = load_nparray(os.path.join(folder_path, other_system + extracted_ligand_suffix))

        if other_system == system:
            raise RuntimeError(f"other_system = {other_system} shoud be != system = {system}")

        # Saving negative example
        try:
            save_example(save_folder, protein, bad_ligand, system, other_system)
        except:
            logger.debug(f'Save failed to {save_folder}')
            raise


def create_examples(data_folder, example_folder, nb_neg: int=-1):
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

    :type nb_neg: the number of negative example to create per positive example. Default -1 means maximum.
    :return:
    """
    # Getting all the systems
    list_systems = set(list(map(extract_id, os.listdir(data_folder))))
    logger.debug(f'Get system id from {data_folder}')

    nb_systems = len(list_systems)

    if nb_neg > nb_systems:
        raise RuntimeError(f"Cannot create more than {nb_systems-1} negatives examples per positive examples (actual "
                           f"value = {nb_neg}")
    elif nb_neg == -1:
        nb_neg = nb_systems - 1

    # Deleting the folders of examples and recreating it
    if os.path.exists(example_folder):
        logger.debug(f'Delete {example_folder} examples folder.')
        shutil.rmtree(example_folder)
    os.makedirs(example_folder)
    logger.debug(f'Create new {example_folder} examples folder.')

    # For each system, we create the associated positive example and we generate some negative examples
    logger.debug('Create 1 positive binding and %d random negative protein-ligand bindings.', nb_neg)
    with futures.ProcessPoolExecutor(max_workers=6) as executor:
        for system in sorted(list_systems):
            executor.submit(save_examples, system, list_systems, nb_neg, data_folder, example_folder)

    logger.debug(f'Create {example_folder} examples done.')


if __name__ == "__main__":
    create_examples(extracted_data_train_folder, training_examples_folder, nb_neg_ex_per_pos)
    create_examples(extracted_data_test_folder, testing_examples_folder, nb_neg_ex_per_pos)
    create_examples(extracted_predict_folder, predict_examples_folder)
