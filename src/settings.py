import os

import progressbar
import numpy as np

# Folders
# Global folder for data and logs
absolute_path = os.path.abspath("..")

data_folder = os.path.join(absolute_path, "training_data")
logs_folder = os.path.join(absolute_path, "logs")
job_submissions_folder = os.path.join(absolute_path, "job_submissions")

# Data given (not modified)
original_data_folder = os.path.join(data_folder, "original")

# First extraction of data
extracted_data_folder = os.path.join(data_folder, "extracted")
extracted_data_train_folder = os.path.join(extracted_data_folder, "train")
extracted_data_test_folder = os.path.join(extracted_data_folder, "test")

# Conversion to examples
training_examples_folder = os.path.join(data_folder, "training_examples")

# Suffixes for extracted data files
extracted_protein_suffix = "_pro_cg.csv"
extracted_ligand_suffix = "_lig_cg.csv"

# Persisted models
models_folders = os.path.join("..", "models")

# Some settings for number and persisting tensors
float_type = np.float32
formatter = "%.16f"
comment_delimiter = "#"


# Features used to train:
#  - 3 spatial coordinates : x , y, z (floating values)
#  - 1 features for one hot encoding of atom types (is_hydrophobic)
#  - 1 features for one hot encoding of molecules types (is_from_protein)

features_names = ["x", "y", "z", "is_hydrophobic", "is_from_protein"]
nb_features = len(features_names)
nb_channels = nb_features - 3 # coordinates are not used as features
indices_features = dict(zip(features_names, list(range(nb_features))))

# We have 3000 positives pairs of ligands
nb_systems = 3000

n_training_examples = 2700
train_indices = list(map(int, open(os.path.join(data_folder, "train_indices")).readlines()))
test_indices = list(map(int, open(os.path.join(data_folder, "test_indices")).readlines()))

# Test the consistency : no overlap and all the files are used on the two sets
assert (len(set(train_indices).intersection(set(test_indices))) == 0)
assert (len(test_indices) + len(train_indices) == nb_systems)
assert (len(train_indices) == n_training_examples)

# All the other will be used to construct examples on the go

# We augment the number of examples
nb_neg_ex_per_pos = 10

# To scale protein-ligands system in a cube of shape (resolution_cube,resolution_cube,resolution_cube)
resolution_cube = 20

# Obtained with values_range on the complete original dataset : to be rerun again
hydrophobic_types = {"C"}
polar_types = {'P', 'O', 'TE', 'F', 'N', 'AS', 'O1-', 'MO',
               'B', 'BR', 'SB', 'RU', 'SE', 'HG', 'CL',
               'S', 'FE', 'ZN', 'CU', 'SI', 'V', 'I', 'N+1',
               'N1+', 'CO', 'W', }

x_min = -244.401
x_max = 310.935

y_min = -186.407
y_max = 432.956

z_mix = -177.028
z_max = 432.956
##

# Misc.
widgets_progressbar = [
    ' [', progressbar.Timer(), '] ',
    progressbar.Bar("░", fill="⋅"),
    ' (', progressbar.ETA(), ') ',
]

# Training parameters
nb_epochs_default = 1
batch_size_default = 32
n_gpu_default = 1
optimizer_default = "rmsprop"

def progress(iterable):
    """
    Custom progress bar
    :param iterable: the iterable to wrap
    :return:
    """
    return progressbar.progressbar(iterable, widgets=widgets_progressbar, redirect_stdout=True)


def extract_id(file_name):
    new_name = file_name.replace(extracted_protein_suffix, "").replace(extracted_ligand_suffix, "")
    return new_name
