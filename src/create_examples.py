import numpy as np
import os
import shutil
import progressbar

from discretization import load_nparray
from settings import examples_data, extracted_data, float_type, protein_suffix, ligand_suffix, comment_delimiter, \
    widgets_progressbar

# We can augment the number of example combining
nb_neg_ex = 10


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
    if os.path.exists(examples_data):
        shutil.rmtree(examples_data)
    os.makedirs(examples_data)

    def drop_suffix(file_name): return file_name.replace(protein_suffix, "").replace(ligand_suffix, "")

    # Getting all the systems
    list_systems = set(list(map(drop_suffix, os.listdir(extracted_data))))

    # For each system, we create the associated positive example and we generate some negative examples
    for system in progressbar.progressbar(sorted(list_systems), widgets=widgets_progressbar,redirect_stdout=True):
        protein = load_nparray(os.path.join(extracted_data, system + protein_suffix))
        ligand = load_nparray(os.path.join(extracted_data, system + ligand_suffix))

        # Saving positive example
        save_example(examples_data, protein, ligand, system, system)

        # Creating false example using nb_neg_ex negatives examples
        others_system = sorted(list(list_systems.difference(set(system))))
        some_others_systems_indices = np.random.randint(0, len(others_system), nb_neg_ex)
        for other_system in map(lambda index: others_system[index], some_others_systems_indices):
            bad_ligand = load_nparray(os.path.join(extracted_data, other_system + ligand_suffix))

            # Saving negative example
            save_example(examples_data, protein, bad_ligand, system, other_system)
