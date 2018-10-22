import numpy as np
import os
import logging
from concurrent import futures
from pipeline_fixtures import get_current_timestamp, load_nparray

from settings import EXTRACTED_GIVEN_DATA_TRAIN_FOLDER, extracted_given_data_validation_folder, \
    EXTRACTED_PROTEIN_SUFFIX, \
    EXTRACTED_LIGAND_SUFFIX, COMMENT_DELIMITER, FEATURES_NAMES, TRAINING_EXAMPLES_FOLDER, \
    VALIDATION_EXAMPLES_FOLDER, NB_WORKERS, MAX_NB_NEG_PER_POS, TESTING_EXAMPLES_FOLDER, EXTRACTED_PREDICT_DATA_FOLDER, \
    PREDICT_EXAMPLES_FOLDER, extracted_given_data_test_folder, LOGS_FOLDER


def save_example(examples_folder: str, protein: np.ndarray, ligand: np.ndarray,
                 protein_system: str, ligand_system: str):
    """
    Save an example of a system using representations of a protein and of a ligand.

    Files are saved in the `folder` with the naming convention:

        `xxxx_yyyy.csv`

    where `xxxx` denotes the `protein_system` and `yyyy` denotes `ligand_system`.

    Hence `xxxx` == `yyyy` if and only if the system is a positive example.

    :param examples_folder: the folder where the examples are saved
    :param protein: the representation of the protein
    :param ligand: the representation of the ligand
    :param protein_system: the ID of the system of the protein
    :param ligand_system: the ID of the system of the ligand
    :return:
    """
    file_name = protein_system + "_" + ligand_system + ".csv"
    file_path = os.path.join(examples_folder, file_name)

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
                ','.join(FEATURES_NAMES)]

    comment = COMMENT_DELIMITER + f"\n{COMMENT_DELIMITER} ".join(comments) + "\n"

    with open(file_path, "w") as f:
        f.write(comment)
        np.savetxt(fname=f, X=example)


# To get reproducible generations of examples
np.random.seed(1337)


def save_system_examples(system, list_systems, nb_neg, extracted_data_folder, examples_folder):
    """
    For one system in the `extracted_data_folder`, save its positive example and `nb_neg` random negative example.
    Example get saved in `examples_folder`.


    :param system: an id xxxx
    :param list_systems: the complete list of id in data folder
    :param nb_neg: the number of negative examples to create
    :param extracted_data_folder: where the original data is
    :param examples_folder: where to save the new data
    :return:
    """

    try:
        system_protein = load_nparray(os.path.join(extracted_data_folder, system + EXTRACTED_PROTEIN_SUFFIX))
        system_ligand = load_nparray(os.path.join(extracted_data_folder, system + EXTRACTED_LIGAND_SUFFIX))
    except Exception:
        warning_message = f"Loading protein/ligand failed. Protein folder " + \
                          f"{os.path.join(extracted_data_folder, system + EXTRACTED_PROTEIN_SUFFIX)}, " + \
                          f"Ligand folder {os.path.join(extracted_data_folder, system + EXTRACTED_LIGAND_SUFFIX)}"
        # logger.debug(warning_message)
        print(warning_message)
        raise RuntimeError()

    # Saving positive example
    save_example(examples_folder, system_protein, system_ligand, system, system)

    # Creating false example using nb_neg_ex negatives examples
    other_systems = sorted(list(list_systems.difference({system})))
    some_others_systems_indices = np.random.permutation(len(other_systems))[0:nb_neg]

    for other_system in map(lambda index: other_systems[index], some_others_systems_indices):
        other_ligand = load_nparray(os.path.join(extracted_data_folder, other_system + EXTRACTED_LIGAND_SUFFIX))

        if other_system == system:
            raise RuntimeError(f"other_system = {other_system} shoud be != system = {system}")

        # Saving negative example
        try:
            save_example(examples_folder, system_protein, other_ligand, system, other_system)
        except Exception:
            # logger.debug(f'Save failed to {examples_folder}')
            raise RuntimeError()


def create_examples(from_folder, to_folder, nb_neg: int = -1):
    """
    Create examples using data present in `data_folder` and saves them in files in the `example_folder` folder.

    Here, we create both positive and negative examples.

    We are given `nb_systems` that binds ; so this makes `nb_systems` positive examples.

    We can then each protein and some others ligands to create negative examples (ie examples of systems that don't
    bind with each others). Those examples are created randomly taking some others ligand that are not binding.
    This is made reproducible using a seed.

    If we note `nb_neg` the number of negative examples created per positive examples, we have exactly :

        `nb_systems` * `nb_neg` negatives examples

    Hence this procedure creates `n_systems` * (1 + `nb_neg`) examples, that is at max `nb_systems^2` examples.

    :param from_folder:
    :param to_folder:
    :param nb_neg: the number of negative example to create per positive example. Default -1 means maximum.
    :return:
    """
    logger = logging.getLogger('__main__.create_example')
    fh = logging.FileHandler(os.path.join(LOGS_FOLDER, get_current_timestamp()))
    logger.addHandler(fh)

    # Getting all the systems
    extract_id = lambda x: x.split("_")[0]

    list_systems = set(list(map(extract_id, os.listdir(from_folder))))
    logger.debug(f'Get systems ids from {from_folder}')

    nb_systems = len(list_systems)

    if nb_neg > nb_systems:
        raise RuntimeError(f"Cannot create more than {nb_systems-1} negatives examples per positive examples (actual "
                           f"value = {nb_neg}")

    # We create all the associated negative examples
    elif nb_neg == -1:
        nb_neg = nb_systems - 1

    # Deleting the folders of examples and recreating it
    if os.path.exists(to_folder):
        logger.debug(f'Delete {to_folder} examples files.')
        for file in os.listdir(to_folder):
            os.remove(os.path.join(to_folder, file))
    else:
        os.makedirs(to_folder)
        logger.debug(f'Create new {to_folder} examples folder.')

    # For each system, we create the associated positive example and we generate some negative examples
    logger.debug('Create 1 positive binding and %d random negative protein-ligand bindings.', nb_neg)
    with futures.ProcessPoolExecutor(max_workers=NB_WORKERS) as executor:
        for system in sorted(list_systems):
            executor.submit(save_system_examples, system, list_systems, nb_neg, from_folder, to_folder)

    logger.debug(f'Create {to_folder} examples done.')


if __name__ == "__main__":
    print(f"Creating training examples with {MAX_NB_NEG_PER_POS} negatives examples per positive examples")
    create_examples(from_folder=EXTRACTED_GIVEN_DATA_TRAIN_FOLDER,
                    to_folder=TRAINING_EXAMPLES_FOLDER,
                    nb_neg=MAX_NB_NEG_PER_POS)

    print("Creating all possible validation examples")
    create_examples(from_folder=extracted_given_data_validation_folder,
                    to_folder=VALIDATION_EXAMPLES_FOLDER)

    print("Creating all possible testing examples")
    create_examples(from_folder=extracted_given_data_test_folder,
                    to_folder=TESTING_EXAMPLES_FOLDER)

    print("Creating all possible examples for final predictions")
    create_examples(from_folder=EXTRACTED_PREDICT_DATA_FOLDER,
                    to_folder=PREDICT_EXAMPLES_FOLDER)