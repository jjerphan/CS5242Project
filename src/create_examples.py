import numpy as np
import os
import itertools
import progressbar

from settings import examples_data, extracted_data, float_type, protein_suffix, ligand_suffix, comment_delimiter, \
    widgets_progressbar

# We can augment the number of example combining
nb_neg_examples_per_pos_example = 10


def save_example(folder: str, protein: np.ndarray, ligand: np.ndarray, system_protein: str, system_ligand: str):
    file_name = system_protein + "_" + system_ligand + ".csv"
    file_path = os.path.join(folder, file_name)

    # We concatenate the molecule vertically
    if len(protein.shape) < 2 or len(ligand.shape) < 2:
        print("One molecule is empty")
        print(f"Protein {system_protein}: {len(protein.shape)}")
        print(f"Ligand  {system_ligand}: {len(ligand.shape)}")
        return

    if protein.shape[1] != ligand.shape[1]:
        print("Different dimension")
        print(f"Protein {system_protein}: {protein.shape}")
        print(f"Ligand  {system_ligand}: {ligand.shape}")
        return

    example = np.concatenate((protein, ligand), axis=0)

    type_example = ("Positive" if system_protein == system_ligand else "Negative") + "example "

    comments = [f"{type_example} of (Protein, Ligand) : ({system_protein},{system_ligand}",
                f"Number of atoms in Protein: {protein.shape[0]}",
                f"Number of atoms in Ligand : {ligand.shape[0]}"]

    comment = comment_delimiter + f"\n{comment_delimiter} ".join(comments)

    with open(file_path, "w") as f:
        f.write(comment)

    np.savetxt(fname=file_path, X=example)


if __name__ == "__main__":
    if not (os.path.exists(examples_data)):
        os.makedirs(examples_data)


    def drop_suffix(file_name):
        return file_name.replace(protein_suffix, "").replace(ligand_suffix, "")


    list_systems = set(list(map(drop_suffix, os.listdir(extracted_data))))

    for system in progressbar.progressbar(sorted(list_systems), widgets=widgets_progressbar,redirect_stdout=True):
        protein = np.loadtxt(fname=os.path.join(extracted_data, system + protein_suffix), dtype=float_type)
        ligand = np.loadtxt(fname=os.path.join(extracted_data, system + ligand_suffix), dtype=float_type)

        save_example(examples_data, protein, ligand, system, system)
        others_system = sorted(list(list_systems.difference(set(system))))

        some_others_systems_indices = np.random.randint(0, len(others_system), nb_neg_examples_per_pos_example)

        for other_system in map(lambda indice: others_system[indice], some_others_systems_indices):
            bad_ligand = np.loadtxt(fname=os.path.join(extracted_data, other_system + ligand_suffix), dtype=float_type)

            save_example(examples_data, protein, bad_ligand, system, other_system)
